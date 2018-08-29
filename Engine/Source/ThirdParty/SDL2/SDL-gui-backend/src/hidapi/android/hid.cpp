//=================== Copyright Valve Corporation, All rights reserved. =======
//
// Purpose: A wrapper implementing "HID" API for Android
//
//          This layer glues the hidapi API to Android's USB and BLE stack.
//
//=============================================================================

#include <jni.h>
#include <android/log.h>
#include <pthread.h>
#include <errno.h>	// For ETIMEDOUT and ECONNRESET

#define TAG "hidapi"
#ifdef DEBUG
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#else
#define LOGV(...)
#define LOGD(...)
#endif

#define SDL_JAVA_PREFIX                                 org_libsdl_app
#define CONCAT1(prefix, class, function)                CONCAT2(prefix, class, function)
#define CONCAT2(prefix, class, function)                Java_ ## prefix ## _ ## class ## _ ## function
#define HID_DEVICE_MANAGER_JAVA_INTERFACE(function)     CONCAT1(SDL_JAVA_PREFIX, HIDDeviceManager, function)

#include "../hidapi/hidapi.h"
typedef uint32_t uint32;
typedef uint64_t uint64;


struct hid_device_
{
	int nId;
};

static JavaVM *g_JVM;
static pthread_key_t g_ThreadKey;

template<class T>
class hid_device_ref
{
public:
	hid_device_ref( T *pObject = nullptr ) : m_pObject( nullptr )
	{
		SetObject( pObject );
	}

	hid_device_ref( const hid_device_ref &rhs ) : m_pObject( nullptr )
	{
		SetObject( rhs.GetObject() );
	}

	~hid_device_ref()
	{
		SetObject( nullptr );
	}

	void SetObject( T *pObject )
	{
		if ( m_pObject && m_pObject->DecrementRefCount() == 0 )
		{
			delete m_pObject;
		}

		m_pObject = pObject;

		if ( m_pObject )
		{
			m_pObject->IncrementRefCount();
		}
	}

	hid_device_ref &operator =( T *pObject )
	{
		SetObject( pObject );
		return *this;
	}

	hid_device_ref &operator =( const hid_device_ref &rhs )
	{
		SetObject( rhs.GetObject() );
		return *this;
	}

	T *GetObject() const
	{
		return m_pObject;
	}

	T* operator->() const
	{
		return m_pObject;
	}

	operator bool() const
	{
		return ( m_pObject != nullptr );
	}

private:
	T *m_pObject;
};

class hid_mutex_guard
{
public:
	hid_mutex_guard( pthread_mutex_t *pMutex ) : m_pMutex( pMutex )
	{
		pthread_mutex_lock( m_pMutex );
	}
	~hid_mutex_guard()
	{
		pthread_mutex_unlock( m_pMutex );
	}

private:
	pthread_mutex_t *m_pMutex;
};

class hid_buffer
{
public:
	hid_buffer() : m_pData( nullptr ), m_nSize( 0 ), m_nAllocated( 0 )
	{
	}

	hid_buffer( const uint8_t *pData, size_t nSize ) : m_pData( nullptr ), m_nSize( 0 ), m_nAllocated( 0 )
	{
		assign( pData, nSize );
	}

	~hid_buffer()
	{
		delete[] m_pData;
	}

	void assign( const uint8_t *pData, size_t nSize )
	{
		if ( nSize > m_nAllocated )
		{
			delete[] m_pData;
			m_pData = new uint8_t[ nSize ];
			m_nAllocated = nSize;
		}

		m_nSize = nSize;
		memcpy( m_pData, pData, nSize );
	}

	void clear()
	{
		m_nSize = 0;
	}

	size_t size() const
	{
		return m_nSize;
	}

	const uint8_t *data() const
	{
		return m_pData;
	}

private:
	uint8_t *m_pData;
	size_t m_nSize;
	size_t m_nAllocated;
};

class hid_buffer_pool
{
public:
	hid_buffer_pool() : m_nSize( 0 ), m_pHead( nullptr ), m_pTail( nullptr ), m_pFree( nullptr )
	{
	}

	~hid_buffer_pool()
	{
		clear();

		while ( m_pFree )
		{
			hid_buffer_entry *pEntry = m_pFree;
			m_pFree = m_pFree->m_pNext;
			delete pEntry;
		}
	}

	size_t size() const { return m_nSize; }

	const hid_buffer &front() const { return m_pHead->m_buffer; }

	void pop_front()
	{
		hid_buffer_entry *pEntry = m_pHead;
		if ( pEntry )
		{
			m_pHead = pEntry->m_pNext;
			if ( !m_pHead )
			{
				m_pTail = nullptr;
			}
			pEntry->m_pNext = m_pFree;
			m_pFree = pEntry;
			--m_nSize;
		}
	}

	void emplace_back( const uint8_t *pData, size_t nSize )
	{
		hid_buffer_entry *pEntry;

		if ( m_pFree )
		{
			pEntry = m_pFree;
			m_pFree = m_pFree->m_pNext;
		}
		else
		{
			pEntry = new hid_buffer_entry;
		}
		pEntry->m_pNext = nullptr;

		if ( m_pTail )
		{
			m_pTail->m_pNext = pEntry;
		}
		else
		{
			m_pHead = pEntry;
		}
		m_pTail = pEntry;

		pEntry->m_buffer.assign( pData, nSize );
		++m_nSize;
	}

	void clear()
	{
		while ( size() > 0 )
		{
			pop_front();
		}
	}

private:
	struct hid_buffer_entry
	{
		hid_buffer m_buffer;
		hid_buffer_entry *m_pNext;
	};

	size_t m_nSize;
	hid_buffer_entry *m_pHead;
	hid_buffer_entry *m_pTail;
	hid_buffer_entry *m_pFree;
};

static jbyteArray NewByteArray( JNIEnv* env, const uint8_t *pData, size_t nDataLen )
{
	jbyteArray array = env->NewByteArray( nDataLen );
	jbyte *pBuf = env->GetByteArrayElements( array, NULL );
	memcpy( pBuf, pData, nDataLen );
	env->ReleaseByteArrayElements( array, pBuf, 0 );

	return array;
}

static char *CreateStringFromJString( JNIEnv *env, const jstring &sString )
{
	size_t nLength = env->GetStringUTFLength( sString );
	const char *pjChars = env->GetStringUTFChars( sString, NULL );
	char *psString = (char*)malloc( nLength + 1 );
	memcpy( psString, pjChars, nLength );
	psString[ nLength ] = '\0';
	env->ReleaseStringUTFChars( sString, pjChars );
	return psString;
}

static wchar_t *CreateWStringFromJString( JNIEnv *env, const jstring &sString )
{
	size_t nLength = env->GetStringLength( sString );
	const jchar *pjChars = env->GetStringChars( sString, NULL );
	wchar_t *pwString = (wchar_t*)malloc( ( nLength + 1 ) * sizeof( wchar_t ) );
	wchar_t *pwChars = pwString;
	for ( size_t iIndex = 0; iIndex < nLength; ++iIndex )
	{
		pwChars[ iIndex ] = pjChars[ iIndex ];
	}
	pwString[ nLength ] = '\0';
	env->ReleaseStringChars( sString, pjChars );
	return pwString;
}

static wchar_t *CreateWStringFromWString( const wchar_t *pwSrc )
{
	size_t nLength = wcslen( pwSrc );
	wchar_t *pwString = (wchar_t*)malloc( ( nLength + 1 ) * sizeof( wchar_t ) );
	memcpy( pwString, pwSrc, nLength * sizeof( wchar_t ) );
	pwString[ nLength ] = '\0';
	return pwString;
}

static hid_device_info *CopyHIDDeviceInfo( const hid_device_info *pInfo )
{
	hid_device_info *pCopy = new hid_device_info;
	*pCopy = *pInfo;
	pCopy->path = strdup( pInfo->path );
	pCopy->product_string = CreateWStringFromWString( pInfo->product_string );
	pCopy->manufacturer_string = CreateWStringFromWString( pInfo->manufacturer_string );
	pCopy->serial_number = CreateWStringFromWString( pInfo->serial_number );
	return pCopy;
}

static void FreeHIDDeviceInfo( hid_device_info *pInfo )
{
	free( pInfo->path );
	free( pInfo->serial_number );
	free( pInfo->manufacturer_string );
	free( pInfo->product_string );
	delete pInfo;
}

static jclass  g_HIDDeviceManagerCallbackClass;
static jobject g_HIDDeviceManagerCallbackHandler;
static jmethodID g_midHIDDeviceManagerOpen;
static jmethodID g_midHIDDeviceManagerSendOutputReport;
static jmethodID g_midHIDDeviceManagerSendFeatureReport;
static jmethodID g_midHIDDeviceManagerGetFeatureReport;
static jmethodID g_midHIDDeviceManagerClose;

uint64_t get_timespec_ms( const struct timespec &ts )
{
	return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

class CHIDDevice
{
public:
	CHIDDevice( int nDeviceID, hid_device_info *pInfo )
	{
		m_nId = nDeviceID;
		m_pInfo = pInfo;

		// The Bluetooth Steam Controller needs special handling
		const int VALVE_USB_VID	= 0x28DE;
		const int D0G_BLE2_PID = 0x1106;
		if ( pInfo->vendor_id == VALVE_USB_VID && pInfo->product_id == D0G_BLE2_PID )
		{
			m_bIsBLESteamController = true;
		}
	}

	~CHIDDevice()
	{
		FreeHIDDeviceInfo( m_pInfo );

		// Note that we don't delete m_pDevice, as the app may still have a reference to it
	}

	int IncrementRefCount()
	{
		return ++m_nRefCount;
	}

	int DecrementRefCount()
	{
		return --m_nRefCount;
	}

	int GetId()
	{
		return m_nId;
	}

	const hid_device_info *GetDeviceInfo()
	{
		return m_pInfo;
	}

	hid_device *GetDevice()
	{
		return m_pDevice;
	}

	int GetDeviceRefCount()
	{
		return m_nDeviceRefCount;
	}

	int IncrementDeviceRefCount()
	{
		return ++m_nDeviceRefCount;
	}

	int DecrementDeviceRefCount()
	{
		return --m_nDeviceRefCount;
	}

	bool BOpen()
	{
		// Make sure thread is attached to JVM/env
		JNIEnv *env;
		g_JVM->AttachCurrentThread( &env, NULL );
		pthread_setspecific( g_ThreadKey, (void*)env );

		m_bIsWaitingForOpen = false;
		m_bOpenResult = env->CallBooleanMethod( g_HIDDeviceManagerCallbackHandler, g_midHIDDeviceManagerOpen, m_nId );

		if ( m_bIsWaitingForOpen )
		{
			hid_mutex_guard cvl( &m_cvLock );

			const int OPEN_TIMEOUT_SECONDS = 60;
			struct timespec ts, endtime;
			clock_gettime( CLOCK_REALTIME, &ts );
			endtime = ts;
			endtime.tv_sec += OPEN_TIMEOUT_SECONDS;
			do
			{
				if ( pthread_cond_timedwait( &m_cv, &m_cvLock, &endtime ) != 0 )
				{
					break;
				}
			}
			while ( m_bIsWaitingForOpen && get_timespec_ms( ts ) < get_timespec_ms( endtime ) );
		}

		if ( !m_bOpenResult )
		{
			if ( m_bIsWaitingForOpen )
			{
				LOGV( "Device open failed - timed out waiting for device permission" );
			}
			else
			{
				LOGV( "Device open failed" );
			}
			return false;
		}

		m_pDevice = new hid_device;
		m_pDevice->nId = m_nId;
		m_nDeviceRefCount = 1;
		return true;
	}

	void SetOpenPending()
	{
		m_bIsWaitingForOpen = true;
	}

	void SetOpenResult( bool bResult )
	{
		if ( m_bIsWaitingForOpen )
		{
			m_bOpenResult = bResult;
			m_bIsWaitingForOpen = false;
			pthread_cond_signal( &m_cv );
		}
	}

	void ProcessInput( const uint8_t *pBuf, size_t nBufSize )
	{
		hid_mutex_guard l( &m_dataLock );

		size_t MAX_REPORT_QUEUE_SIZE = 16;
		if ( m_vecData.size() >= MAX_REPORT_QUEUE_SIZE )
		{
			m_vecData.pop_front();
		}
		m_vecData.emplace_back( pBuf, nBufSize );
	}

	int GetInput( unsigned char *data, size_t length )
	{
		hid_mutex_guard l( &m_dataLock );

		if ( m_vecData.size() == 0 )
		{
//			LOGV( "hid_read_timeout no data available" );
			return 0;
		}

		const hid_buffer &buffer = m_vecData.front();
		size_t nDataLen = buffer.size() > length ? length : buffer.size();
		if ( m_bIsBLESteamController )
		{
			data[0] = 0x03;
			memcpy( data + 1, buffer.data(), nDataLen );
			++nDataLen;
		}
		else
		{
			memcpy( data, buffer.data(), nDataLen );
		}
		m_vecData.pop_front();

//		LOGV("Read %u bytes", nDataLen);
//		LOGV("%02x %02x %02x %02x %02x %02x %02x %02x ....",
//			 data[0], data[1], data[2], data[3],
//			 data[4], data[5], data[6], data[7]);

		return nDataLen;
	}

	int SendOutputReport( const unsigned char *pData, size_t nDataLen )
	{
		// Make sure thread is attached to JVM/env
		JNIEnv *env;
		g_JVM->AttachCurrentThread( &env, NULL );
		pthread_setspecific( g_ThreadKey, (void*)env );

		jbyteArray pBuf = NewByteArray( env, pData, nDataLen );
		int nRet = env->CallIntMethod( g_HIDDeviceManagerCallbackHandler, g_midHIDDeviceManagerSendOutputReport, m_nId, pBuf );
		env->DeleteLocalRef( pBuf );
		return nRet;
	}

	int SendFeatureReport( const unsigned char *pData, size_t nDataLen )
	{
		// Make sure thread is attached to JVM/env
		JNIEnv *env;
		g_JVM->AttachCurrentThread( &env, NULL );
		pthread_setspecific( g_ThreadKey, (void*)env );

		jbyteArray pBuf = NewByteArray( env, pData, nDataLen );
		int nRet = env->CallIntMethod( g_HIDDeviceManagerCallbackHandler, g_midHIDDeviceManagerSendFeatureReport, m_nId, pBuf );
		env->DeleteLocalRef( pBuf );
		return nRet;
	}

	void ProcessFeatureReport( const uint8_t *pBuf, size_t nBufSize )
	{
		hid_mutex_guard cvl( &m_cvLock );
		if ( m_bIsWaitingForFeatureReport )
		{
			m_featureReport.assign( pBuf, nBufSize );

			m_bIsWaitingForFeatureReport = false;
			m_nFeatureReportError = 0;
			pthread_cond_signal( &m_cv );
		}
	}

	int GetFeatureReport( unsigned char *pData, size_t nDataLen )
	{
		// Make sure thread is attached to JVM/env
		JNIEnv *env;
		g_JVM->AttachCurrentThread( &env, NULL );
		pthread_setspecific( g_ThreadKey, (void*)env );

		{
			hid_mutex_guard cvl( &m_cvLock );
			if ( m_bIsWaitingForFeatureReport )
			{
				LOGV( "Get feature report already ongoing... bail" );
				return -1; // Read already ongoing, we currently do not serialize, TODO
			}
			m_bIsWaitingForFeatureReport = true;
		}

		jbyteArray pBuf = NewByteArray( env, pData, nDataLen );
		int nRet = env->CallBooleanMethod( g_HIDDeviceManagerCallbackHandler, g_midHIDDeviceManagerGetFeatureReport, m_nId, pBuf ) ? 0 : -1;
		env->DeleteLocalRef( pBuf );
		if ( nRet < 0 )
		{
			LOGV( "GetFeatureReport failed" );
			m_bIsWaitingForFeatureReport = false;
			return -1;
		}

		{
			hid_mutex_guard cvl( &m_cvLock );
			if ( m_bIsWaitingForFeatureReport )
			{
				LOGV("=== Going to sleep" );
				// Wait in CV until we are no longer waiting for a feature report.
				const int FEATURE_REPORT_TIMEOUT_SECONDS = 2;
				struct timespec ts, endtime;
				clock_gettime( CLOCK_REALTIME, &ts );
				endtime = ts;
				endtime.tv_sec += FEATURE_REPORT_TIMEOUT_SECONDS;
				do
				{
					if ( pthread_cond_timedwait( &m_cv, &m_cvLock, &endtime ) != 0 )
					{
						break;
					}
				}
				while ( m_bIsWaitingForFeatureReport && get_timespec_ms( ts ) < get_timespec_ms( endtime ) );

				// We are back
				if ( m_bIsWaitingForFeatureReport )
				{
					m_nFeatureReportError = -ETIMEDOUT;
					m_bIsWaitingForFeatureReport = false;
				}
				LOGV( "=== Got feature report err=%d", m_nFeatureReportError );
				if ( m_nFeatureReportError != 0 )
				{
					return m_nFeatureReportError;
				}
			}

			size_t uBytesToCopy = m_featureReport.size() > nDataLen ? nDataLen : m_featureReport.size();
			memcpy( pData, m_featureReport.data(), uBytesToCopy );
			m_featureReport.clear();
			LOGV( "=== Got %u bytes", uBytesToCopy );

			return uBytesToCopy;
		}
	}

	void Close( bool bDeleteDevice )
	{
		// Make sure thread is attached to JVM/env
		JNIEnv *env;
		g_JVM->AttachCurrentThread( &env, NULL );
		pthread_setspecific( g_ThreadKey, (void*)env );

		env->CallVoidMethod( g_HIDDeviceManagerCallbackHandler, g_midHIDDeviceManagerClose, m_nId );

		hid_mutex_guard dataLock( &m_dataLock );
		m_vecData.clear();

		// Clean and release pending feature report reads
		hid_mutex_guard cvLock( &m_cvLock );
		m_featureReport.clear();
		m_bIsWaitingForFeatureReport = false;
		m_nFeatureReportError = -ECONNRESET;
		pthread_cond_broadcast( &m_cv );

		if ( bDeleteDevice )
		{
			delete m_pDevice;
			m_pDevice = nullptr;
		}
	}

private:
	int m_nRefCount = 0;
	int m_nId = 0;
	hid_device_info *m_pInfo = nullptr;
	hid_device *m_pDevice = nullptr;
	bool m_bIsBLESteamController = false;
	int m_nDeviceRefCount = 0;

	pthread_mutex_t m_dataLock = PTHREAD_MUTEX_INITIALIZER; // This lock has to be held to access m_vecData
	hid_buffer_pool m_vecData;

	// For handling get_feature_report
	pthread_mutex_t m_cvLock = PTHREAD_MUTEX_INITIALIZER; // This lock has to be held to access any variables below
	pthread_cond_t m_cv = PTHREAD_COND_INITIALIZER;
	bool m_bIsWaitingForOpen = false;
	bool m_bOpenResult = false;
	bool m_bIsWaitingForFeatureReport = false;
	int m_nFeatureReportError = 0;
	hid_buffer m_featureReport;

public:
	hid_device_ref<CHIDDevice> next;
};

class CHIDDevice;
static pthread_mutex_t g_DevicesMutex = PTHREAD_MUTEX_INITIALIZER;
static hid_device_ref<CHIDDevice> g_Devices;

static hid_device_ref<CHIDDevice> FindDevice( int nDeviceId )
{
	hid_device_ref<CHIDDevice> pDevice;

	hid_mutex_guard l( &g_DevicesMutex );
	for ( pDevice = g_Devices; pDevice; pDevice = pDevice->next )
	{
		if ( pDevice->GetId() == nDeviceId )
		{
			break;
		}
	}
	return pDevice;
}

static void ThreadDestroyed(void* value)
{
	/* The thread is being destroyed, detach it from the Java VM and set the g_ThreadKey value to NULL as required */
	JNIEnv *env = (JNIEnv*) value;
	if (env != NULL) {
		g_JVM->DetachCurrentThread();
		pthread_setspecific(g_ThreadKey, NULL);
	}
}

extern "C"
JNIEXPORT void JNICALL HID_DEVICE_MANAGER_JAVA_INTERFACE(HIDDeviceRegisterCallback)(JNIEnv *env, jobject thiz, jobject callbackHandler)
{
	LOGV( "HIDDeviceRegisterCallback()");

	env->GetJavaVM( &g_JVM );

	/*
	 * Create mThreadKey so we can keep track of the JNIEnv assigned to each thread
	 * Refer to http://developer.android.com/guide/practices/design/jni.html for the rationale behind this
	 */
	if (pthread_key_create(&g_ThreadKey, ThreadDestroyed) != 0) {
		__android_log_print(ANDROID_LOG_ERROR, TAG, "Error initializing pthread key");
	}

	g_HIDDeviceManagerCallbackHandler = env->NewGlobalRef( callbackHandler );
	jclass objClass = env->GetObjectClass( callbackHandler );
	if ( objClass )
	{
		g_HIDDeviceManagerCallbackClass = reinterpret_cast< jclass >( env->NewGlobalRef(objClass) );
		g_midHIDDeviceManagerOpen = env->GetMethodID( g_HIDDeviceManagerCallbackClass, "openDevice", "(I)Z" );
		if ( !g_midHIDDeviceManagerOpen )
		{
			__android_log_print(ANDROID_LOG_ERROR, TAG, "HIDDeviceRegisterCallback: callback class missing openDevice" );
		}
		g_midHIDDeviceManagerSendOutputReport = env->GetMethodID( g_HIDDeviceManagerCallbackClass, "sendOutputReport", "(I[B)I" );
		if ( !g_midHIDDeviceManagerSendOutputReport )
		{
			__android_log_print(ANDROID_LOG_ERROR, TAG, "HIDDeviceRegisterCallback: callback class missing sendOutputReport" );
		}
		g_midHIDDeviceManagerSendFeatureReport = env->GetMethodID( g_HIDDeviceManagerCallbackClass, "sendFeatureReport", "(I[B)I" );
		if ( !g_midHIDDeviceManagerSendFeatureReport )
		{
			__android_log_print(ANDROID_LOG_ERROR, TAG, "HIDDeviceRegisterCallback: callback class missing sendFeatureReport" );
		}
		g_midHIDDeviceManagerGetFeatureReport = env->GetMethodID( g_HIDDeviceManagerCallbackClass, "getFeatureReport", "(I[B)Z" );
		if ( !g_midHIDDeviceManagerGetFeatureReport )
		{
			__android_log_print(ANDROID_LOG_ERROR, TAG, "HIDDeviceRegisterCallback: callback class missing getFeatureReport" );
		}
		g_midHIDDeviceManagerClose = env->GetMethodID( g_HIDDeviceManagerCallbackClass, "closeDevice", "(I)V" );
		if ( !g_midHIDDeviceManagerClose )
		{
			__android_log_print(ANDROID_LOG_ERROR, TAG, "HIDDeviceRegisterCallback: callback class missing closeDevice" );
		}
		env->DeleteLocalRef( objClass );
	}
}

extern "C"
JNIEXPORT void JNICALL HID_DEVICE_MANAGER_JAVA_INTERFACE(HIDDeviceReleaseCallback)(JNIEnv *env, jobject thiz)
{
	LOGV("HIDDeviceReleaseCallback");
	env->DeleteGlobalRef( g_HIDDeviceManagerCallbackClass );
	env->DeleteGlobalRef( g_HIDDeviceManagerCallbackHandler );
}

extern "C"
JNIEXPORT void JNICALL HID_DEVICE_MANAGER_JAVA_INTERFACE(HIDDeviceConnected)(JNIEnv *env, jobject thiz, int nDeviceID, jstring sIdentifier, int nVendorId, int nProductId, jstring sSerialNumber, int nReleaseNumber, jstring sManufacturer, jstring sProduct, int nInterface )
{
	LOGV( "HIDDeviceConnected() id=%d VID/PID = %.4x/%.4x, interface %d\n", nDeviceID, nVendorId, nProductId, nInterface );

	hid_device_info *pInfo = new hid_device_info;
	memset( pInfo, 0, sizeof( *pInfo ) );
	pInfo->path = CreateStringFromJString( env, sIdentifier );
	pInfo->vendor_id = nVendorId;
	pInfo->product_id = nProductId;
	pInfo->serial_number = CreateWStringFromJString( env, sSerialNumber );
	pInfo->release_number = nReleaseNumber;
	pInfo->manufacturer_string = CreateWStringFromJString( env, sManufacturer );
	pInfo->product_string = CreateWStringFromJString( env, sProduct );
	pInfo->interface_number = nInterface;

	hid_device_ref<CHIDDevice> pDevice( new CHIDDevice( nDeviceID, pInfo ) );

	hid_mutex_guard l( &g_DevicesMutex );
	hid_device_ref<CHIDDevice> pLast, pCurr;
	for ( pCurr = g_Devices; pCurr; pLast = pCurr, pCurr = pCurr->next )
	{
		continue;
	}
	if ( pLast )
	{
		pLast->next = pDevice;
	}
	else
	{
		g_Devices = pDevice;
	}
}

extern "C"
JNIEXPORT void JNICALL HID_DEVICE_MANAGER_JAVA_INTERFACE(HIDDeviceOpenPending)(JNIEnv *env, jobject thiz, int nDeviceID)
{
	LOGV( "HIDDeviceOpenPending() id=%d\n", nDeviceID );
	hid_device_ref<CHIDDevice> pDevice = FindDevice( nDeviceID );
	if ( pDevice )
	{
		pDevice->SetOpenPending();
	}
}

extern "C"
JNIEXPORT void JNICALL HID_DEVICE_MANAGER_JAVA_INTERFACE(HIDDeviceOpenResult)(JNIEnv *env, jobject thiz, int nDeviceID, bool bOpened)
{
	LOGV( "HIDDeviceOpenResult() id=%d, result=%s\n", nDeviceID, bOpened ? "true" : "false" );
	hid_device_ref<CHIDDevice> pDevice = FindDevice( nDeviceID );
	if ( pDevice )
	{
		pDevice->SetOpenResult( bOpened );
	}
}

extern "C"
JNIEXPORT void JNICALL HID_DEVICE_MANAGER_JAVA_INTERFACE(HIDDeviceDisconnected)(JNIEnv *env, jobject thiz, int nDeviceID)
{
	LOGV( "HIDDeviceDisconnected() id=%d\n", nDeviceID );
	hid_device_ref<CHIDDevice> pDevice;
	{
		hid_mutex_guard l( &g_DevicesMutex );
		hid_device_ref<CHIDDevice> pLast, pCurr;
		for ( pCurr = g_Devices; pCurr; pLast = pCurr, pCurr = pCurr->next )
		{
			if ( pCurr->GetId() == nDeviceID )
			{
				pDevice = pCurr;

				if ( pLast )
				{
					pLast->next = pCurr->next;
				}
				else
				{
					g_Devices = pCurr->next;
				}
			}
		}
	}
	if ( pDevice )
	{
		pDevice->Close( false );
	}
}

extern "C"
JNIEXPORT void JNICALL HID_DEVICE_MANAGER_JAVA_INTERFACE(HIDDeviceInputReport)(JNIEnv *env, jobject thiz, int nDeviceID, jbyteArray value)
{
	jbyte *pBuf = env->GetByteArrayElements(value, NULL);
	jsize nBufSize = env->GetArrayLength(value);

//	LOGV( "HIDDeviceInput() id=%d len=%u\n", nDeviceID, nBufSize );
	hid_device_ref<CHIDDevice> pDevice = FindDevice( nDeviceID );
	if ( pDevice )
	{
		pDevice->ProcessInput( reinterpret_cast< const uint8_t* >( pBuf ), nBufSize );
	}

	env->ReleaseByteArrayElements(value, pBuf, 0);
}

extern "C"
JNIEXPORT void JNICALL HID_DEVICE_MANAGER_JAVA_INTERFACE(HIDDeviceFeatureReport)(JNIEnv *env, jobject thiz, int nDeviceID, jbyteArray value)
{
	jbyte *pBuf = env->GetByteArrayElements(value, NULL);
	jsize nBufSize = env->GetArrayLength(value);

	LOGV( "HIDDeviceFeatureReport() id=%d len=%u\n", nDeviceID, nBufSize );
	hid_device_ref<CHIDDevice> pDevice = FindDevice( nDeviceID );
	if ( pDevice )
	{
		pDevice->ProcessFeatureReport( reinterpret_cast< const uint8_t* >( pBuf ), nBufSize );
	}

	env->ReleaseByteArrayElements(value, pBuf, 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C"
{

int hid_init(void)
{
	return 0;
}

struct hid_device_info HID_API_EXPORT * HID_API_CALL hid_enumerate(unsigned short vendor_id, unsigned short product_id)
{
	struct hid_device_info *root = NULL;
	hid_mutex_guard l( &g_DevicesMutex );
	for ( hid_device_ref<CHIDDevice> pDevice = g_Devices; pDevice; pDevice = pDevice->next )
	{
		const hid_device_info *info = pDevice->GetDeviceInfo();
		if ( ( vendor_id == 0 && product_id == 0 ) ||
			 ( vendor_id == info->vendor_id && product_id == info->product_id ) )
		{
			hid_device_info *dev = CopyHIDDeviceInfo( info );
			dev->next = root;
			root = dev;
		}
	}
	return root;
}

void  HID_API_EXPORT HID_API_CALL hid_free_enumeration(struct hid_device_info *devs)
{
	while ( devs )
	{
		struct hid_device_info *next = devs->next;
		FreeHIDDeviceInfo( devs );
		devs = next;
	}
}

HID_API_EXPORT hid_device * HID_API_CALL hid_open(unsigned short vendor_id, unsigned short product_id, const wchar_t *serial_number)
{
	// TODO: Implement
	return NULL;
}

HID_API_EXPORT hid_device * HID_API_CALL hid_open_path(const char *path, int bExclusive)
{
	LOGV( "hid_open_path( %s )", path );

	hid_device_ref< CHIDDevice > pDevice;
	{
		hid_mutex_guard l( &g_DevicesMutex );
		for ( hid_device_ref<CHIDDevice> pCurr = g_Devices; pCurr; pCurr = pCurr->next )
		{
			if ( strcmp( pCurr->GetDeviceInfo()->path, path ) == 0 ) 
			{
				if ( pCurr->GetDevice() ) {
					pCurr->IncrementDeviceRefCount();
					return pCurr->GetDevice();
				}

				// Hold a shared pointer to the controller for the duration
				pDevice = pCurr;
				break;
			}
		}
	}
	if ( pDevice && pDevice->BOpen() )
	{
		return pDevice->GetDevice();
	}
	return NULL;
}

int  HID_API_EXPORT HID_API_CALL hid_write(hid_device *device, const unsigned char *data, size_t length)
{
	LOGV( "hid_write id=%d length=%u", device->nId, length );
	hid_device_ref<CHIDDevice> pDevice = FindDevice( device->nId );
	if ( pDevice )
	{
		return pDevice->SendOutputReport( data, length );
	}
	return -1; // Controller was disconnected
}

// TODO: Implement timeout?
int HID_API_EXPORT HID_API_CALL hid_read_timeout(hid_device *device, unsigned char *data, size_t length, int milliseconds)
{
//	LOGV( "hid_read_timeout id=%d length=%u timeout=%d", device->nId, length, milliseconds );
	hid_device_ref<CHIDDevice> pDevice = FindDevice( device->nId );
	if ( pDevice )
	{
		return pDevice->GetInput( data, length );
	}
	LOGV( "controller was disconnected" );
	return -1; // Controller was disconnected
}

// TODO: Implement blocking
int  HID_API_EXPORT HID_API_CALL hid_read(hid_device *device, unsigned char *data, size_t length)
{
	LOGV( "hid_read id=%d length=%u", device->nId, length );
	return hid_read_timeout( device, data, length, 0 );
}

// TODO: Implement?
int  HID_API_EXPORT HID_API_CALL hid_set_nonblocking(hid_device *device, int nonblock)
{
	return -1;
}

int HID_API_EXPORT HID_API_CALL hid_send_feature_report(hid_device *device, const unsigned char *data, size_t length)
{
	LOGV( "hid_send_feature_report id=%d length=%u", device->nId, length );
	hid_device_ref<CHIDDevice> pDevice = FindDevice( device->nId );
	if ( pDevice )
	{
		return pDevice->SendFeatureReport( data, length );
	}
	return -1; // Controller was disconnected
}


// Synchronous operation. Will block until completed.
int HID_API_EXPORT HID_API_CALL hid_get_feature_report(hid_device *device, unsigned char *data, size_t length)
{
	LOGV( "hid_get_feature_report id=%d length=%u", device->nId, length );
	hid_device_ref<CHIDDevice> pDevice = FindDevice( device->nId );
	if ( pDevice )
	{
		return pDevice->GetFeatureReport( data, length );
	}
	return -1; // Controller was disconnected
}


void HID_API_EXPORT HID_API_CALL hid_close(hid_device *device)
{
	LOGV( "hid_close id=%d", device->nId );
	hid_device_ref<CHIDDevice> pDevice = FindDevice( device->nId );
	if ( pDevice )
	{
		pDevice->DecrementDeviceRefCount();
		if ( pDevice->GetDeviceRefCount() == 0 ) {
			pDevice->Close( true );
		}
	}
	else
	{
		// Couldn't find it, it's already closed
		delete device;
	}

}

int HID_API_EXPORT_CALL hid_get_manufacturer_string(hid_device *device, wchar_t *string, size_t maxlen)
{
	hid_device_ref<CHIDDevice> pDevice = FindDevice( device->nId );
	if ( pDevice )
	{
		wcsncpy( string, pDevice->GetDeviceInfo()->manufacturer_string, maxlen );
		return 0;
	}
	return -1;
}

int HID_API_EXPORT_CALL hid_get_product_string(hid_device *device, wchar_t *string, size_t maxlen)
{
	hid_device_ref<CHIDDevice> pDevice = FindDevice( device->nId );
	if ( pDevice )
	{
		wcsncpy( string, pDevice->GetDeviceInfo()->product_string, maxlen );
		return 0;
	}
	return -1;
}

int HID_API_EXPORT_CALL hid_get_serial_number_string(hid_device *device, wchar_t *string, size_t maxlen)
{
	hid_device_ref<CHIDDevice> pDevice = FindDevice( device->nId );
	if ( pDevice )
	{
		wcsncpy( string, pDevice->GetDeviceInfo()->serial_number, maxlen );
		return 0;
	}
	return -1;
}

int HID_API_EXPORT_CALL hid_get_indexed_string(hid_device *device, int string_index, wchar_t *string, size_t maxlen)
{
	return -1;
}

HID_API_EXPORT const wchar_t* HID_API_CALL hid_error(hid_device *device)
{
	return NULL;
}

int hid_exit(void)
{
	return 0;
}

}
