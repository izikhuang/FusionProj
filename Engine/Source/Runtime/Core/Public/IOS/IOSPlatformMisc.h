// Copyright 1998-2017 Epic Games, Inc. All Rights Reserved.

/*=============================================================================================
	IOSPlatformMisc.h: iOS platform misc functions
==============================================================================================*/

#pragma once
#include "GenericPlatform/GenericPlatformMisc.h"
#include "IOS/IOSSystemIncludes.h"

#ifndef IOS_PROFILING_ENABLED
#define IOS_PROFILING_ENABLED (UE_BUILD_DEBUG | UE_BUILD_DEVELOPMENT)
#endif

#ifdef __OBJC__

class FScopeAutoreleasePool
{
public:

	FScopeAutoreleasePool()
	{
		Pool = [[NSAutoreleasePool alloc] init];
	}

	~FScopeAutoreleasePool()
	{
		[Pool release];
	}

private:

	NSAutoreleasePool*	Pool;
};

#define SCOPED_AUTORELEASE_POOL const FScopeAutoreleasePool PREPROCESSOR_JOIN(Pool,__LINE__);

#endif // __OBJC__

/**
* iOS implementation of the misc OS functions
**/
struct CORE_API FIOSPlatformMisc : public FGenericPlatformMisc
{
    static void PlatformPreInit();
	static void PlatformInit();
    static void PlatformHandleSplashScreen(bool ShowSplashScreen = false);
	static void GetEnvironmentVariable(const TCHAR* VariableName, TCHAR* Result, int32 ResultLength);
	static void* GetHardwareWindow();

	static bool AllowThreadHeartBeat()
	{
		return false;
	}

#if !UE_BUILD_SHIPPING
	static bool IsDebuggerPresent()
	{
		// Based on http://developer.apple.com/library/mac/#qa/qa1361/_index.html

		extern CORE_API bool GIgnoreDebugger;
		if (GIgnoreDebugger)
		{
			return false;
		}

		struct kinfo_proc Info;
		int32 Mib[] = { CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid() };
		SIZE_T Size = sizeof(Info);

		sysctl( Mib, sizeof( Mib ) / sizeof( *Mib ), &Info, &Size, NULL, 0 );

		return ( Info.kp_proc.p_flag & P_TRACED ) != 0;
	}
	FORCEINLINE static void DebugBreak()
	{
		if( IsDebuggerPresent() )
		{
			//Signal interupt to our process, this is caught by the main thread, this is not immediate but you can continue
			//when triggered by check macros you will need to inspect other threads for the appFailAssert call.
			//kill( getpid(), SIGINT );

			//If you want an immediate break use the trap instruction, continued execuction is halted
#if WITH_SIMULATOR
            __asm__ ( "int $3" );
#elif PLATFORM_64BITS
			__asm__ ( "svc 0" );
#else
            __asm__ ( "trap" );
#endif
		}
	}
#endif

	/** Break into debugger. Returning false allows this function to be used in conditionals. */
	FORCEINLINE static bool DebugBreakReturningFalse()
	{
#if !UE_BUILD_SHIPPING
		DebugBreak();
#endif
		return false;
	}

	/** Prompts for remote debugging if debugger is not attached. Regardless of result, breaks into debugger afterwards. Returns false for use in conditionals. */
	FORCEINLINE static bool DebugBreakAndPromptForRemoteReturningFalse(bool bIsEnsure = false)
	{
#if !UE_BUILD_SHIPPING
		if (!IsDebuggerPresent())
		{
			PromptForRemoteDebugging(bIsEnsure);
		}

		DebugBreak();
#endif

		return false;
	}

	FORCEINLINE static void MemoryBarrier()
	{
		__sync_synchronize();
	}

	static void LowLevelOutputDebugString(const TCHAR *Message);
	static const TCHAR* GetSystemErrorMessage(TCHAR* OutBuffer, int32 BufferCount, int32 Error);
	static EAppReturnType::Type MessageBoxExt( EAppMsgType::Type MsgType, const TCHAR* Text, const TCHAR* Caption );
	static int32 NumberOfCores();
	static void LoadPreInitModules();
	static void SetMemoryWarningHandler(void (* Handler)(const FGenericMemoryWarningContext& Context));
	static bool HasPlatformFeature(const TCHAR* FeatureName);
	static FString GetDefaultLanguage();
	static FString GetDefaultLocale();
	static bool SetStoredValue(const FString& InStoreId, const FString& InSectionName, const FString& InKeyName, const FString& InValue);
	static bool GetStoredValue(const FString& InStoreId, const FString& InSectionName, const FString& InKeyName, FString& OutValue);
	static bool DeleteStoredValue(const FString& InStoreId, const FString& InSectionName, const FString& InKeyName);
	static TArray<uint8> GetSystemFontBytes();
	static TArray<FString> GetPreferredLanguages();
	static FString GetLocalCurrencyCode();
	static FString GetLocalCurrencySymbol();
	static void GetValidTargetPlatforms(class TArray<class FString>& TargetPlatformNames);
	static bool HasActiveWiFiConnection();
	static EScreenPhysicalAccuracy ComputePhysicalScreenDensity(int32& ScreenDensity);

	static int GetAudioVolume();
	static bool AreHeadphonesPluggedIn();
	static int GetBatteryLevel();
	static bool IsRunningOnBattery();

	static void RegisterForRemoteNotifications();
	static bool IsRegisteredForRemoteNotifications();
	static void UnregisterForRemoteNotifications();

	static class IPlatformChunkInstall* GetPlatformChunkInstall();
	
#if IOS_PROFILING_ENABLED
	static void BeginNamedEvent(const struct FColor& Color,const TCHAR* Text);
	static void BeginNamedEvent(const struct FColor& Color,const ANSICHAR* Text);
	static void EndNamedEvent();
#endif
	
	//////// Platform specific
	static void* CreateAutoreleasePool();
	static void ReleaseAutoreleasePool(void *Pool);
	static void HandleLowMemoryWarning();
	static bool IsPackagedForDistribution();
	DEPRECATED(4.14, "GetUniqueDeviceId is deprecated. Use GetDeviceId instead.")
	static FString GetUniqueDeviceId();
	/**
	 * Implemented using UIDevice::identifierForVendor,
	 * so all the caveats that apply to that API call apply here.
	 */
	static FString GetDeviceId();
	static FString GetOSVersion();
	static FString GetUniqueAdvertisingId();
	static bool GetDiskTotalAndFreeSpace(const FString& InPath, uint64& TotalNumberOfBytes, uint64& NumberOfFreeBytes);

	// Possible iOS devices
	enum EIOSDevice
	{
		// add new devices to the top, and add to IOSDeviceNames below!
		IOS_IPhone4,
		IOS_IPhone4S,
		IOS_IPhone5, // also the IPhone5C
		IOS_IPhone5S,
		IOS_IPodTouch5,
		IOS_IPodTouch6,
		IOS_IPad2,
		IOS_IPad3,
		IOS_IPad4,
		IOS_IPadMini,
		IOS_IPadMini2, // also the iPadMini3
		IOS_IPadMini4,
		IOS_IPadAir,
		IOS_IPadAir2,
		IOS_IPhone6,
		IOS_IPhone6Plus,
		IOS_IPhone6S,
		IOS_IPhone6SPlus,
        IOS_IPhone7,
        IOS_IPhone7Plus,
		IOS_IPadPro,
		IOS_AppleTV,
		IOS_IPhoneSE,
		IOS_IPadPro_129,
		IOS_IPadPro_97,
		IOS_IPadPro_105,
		IOS_IPadPro2_129,
		IOS_Unknown,
	};

	static EIOSDevice GetIOSDeviceType();

	static FORCENOINLINE const TCHAR* GetDefaultDeviceProfileName()
	{
		static const TCHAR* IOSDeviceNames[] = 
		{
			L"IPhone4",
			L"IPhone4S",
			L"IPhone5",
			L"IPhone5S",
			L"IPodTouch5",
			L"IPodTouch6",
			L"IPad2",
			L"IPad3",
			L"IPad4",
			L"IPadMini",
			L"IPadMini2",
			L"IPadMini4",
			L"IPadAir",
			L"IPadAir2",
			L"IPhone6",
			L"IPhone6Plus",
			L"IPhone6S",
			L"IPhone6SPlus",
            L"IPhone7",
            L"IPhone7Plus",
			L"IPadPro",
			L"AppleTV",
			L"IPhoneSE",
			L"IPadPro129",
			L"IPadPro97",
			L"IPadPro105",
			L"IPadPro2_129",
			L"Unknown",
		};
		static_assert((sizeof(IOSDeviceNames) / sizeof(IOSDeviceNames[0])) == ((int32)IOS_Unknown + 1), "Mismatched IOSDeviceNames and EIOSDevice.");
		
		// look up into the string array by the enum
		return IOSDeviceNames[(int32)GetIOSDeviceType()];
	}

	static FString GetCPUVendor();
	static FString GetCPUBrand();
	static void GetOSVersions(FString& out_OSVersionLabel, FString& out_OSSubVersionLabel);
	static int32 IOSVersionCompare(uint8 Major, uint8 Minor, uint8 Revision);
	
    static void SetGracefulTerminationHandler();
    static void SetCrashHandler(void(*CrashHandler)(const FGenericCrashContext& Context));
};

typedef FIOSPlatformMisc FPlatformMisc;
