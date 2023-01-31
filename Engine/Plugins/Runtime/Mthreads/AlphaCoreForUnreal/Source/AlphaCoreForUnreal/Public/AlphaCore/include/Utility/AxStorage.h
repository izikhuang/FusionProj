#ifndef __ALPHA_CORE_STORAGE_H__
#define __ALPHA_CORE_STORAGE_H__

#include <vector>
#include <iostream>
#include <fstream>
#include "Math/AxVectorBase.h"
#include "AxMacro.h"
#include "AxLog.h"
#include "AxDataType.h"
#include "AxStoragePool.h"
#include <algorithm>
#include <sstream>
//TODO: move to cpp
#include "Math/AxVectorHelper.h"
#include <ostream>
#include "Math/AxMatrixBase.h"
#include "Math/AxMat.h"
#include "AxEngine.h"

namespace AlphaUtility 
{
	static void ReadSTLString(std::ifstream& ifs, std::string& val)
	{
		if (!ifs)
			return;
		int n = 0;
		ifs.read((char*)&n, sizeof(int));
		val.resize(n - 1);
		ifs.read((char*)&(val[0]), sizeof(char)*n);
	}
	static void WriteSTLString(std::ofstream& ofs, std::string& val)
	{
		if (!ofs)
			return;
		int n = val.size() + 1;
		ofs.write((char*)&n, sizeof(int));
		ofs.write((char*)val.c_str(), sizeof(char)*(val.size() + 1));
	}

	static std::string ToUpper(std::string str)
	{
 		str[0] = std::toupper(str[0]);
		return str;
	}
};

class AxStorageBase
{
public:
	AxStorageBase();
	virtual ~AxStorageBase();
	AxUInt64    GetBufferSize() { return m_iBufferSize; };
	AxUInt32    GetVectorSize() { return m_iVecSize; };
	AxUInt32    GetBankWidth()  { return m_iBankWidth; };
	AxUInt32	GetItems() { return m_iBufferSize * m_iVecSize; };
	virtual AxUInt64 GetCapacity()	 { return 0; };
	virtual AxUInt64 GetIOCapacity() { return 0; };
	enum StorageProperty
	{
		kPrivate		= 0b0001,
		kProtected		= 0b0010,
		kRunTimeOnly	= 0b0100
	};

	//TODO : Change to desc
	struct Desc
	{
		bool IsArrayProperty;
	};

	bool IsRTOnlyData();
	bool IsProtectedData();
	bool IsPrivateData();
	void SetPrivateDataMark(bool e);
	void SetProtectedDataMark(bool e);
	void SetRTOnlyDataMark(bool e);
	void SetName(std::string n) { m_sName = n; };
	std::string GetName() { return  m_sName; };

	virtual void ReadRaw(std::ifstream& ifs) {};
	virtual void SaveRaw(std::ofstream& ofs) {};
	virtual void ClearAndDestory() {};
	virtual void Clear() {};
	virtual void Resize(AxUInt64 size) {};
	virtual void ResizeStorage(AxUInt64 size, bool updateDeviceMemory = true) {};
	virtual void ResizeStorageDevice(AxUInt64 size) {};

	virtual void PrintData(const char* head = " ", unsigned int start = 0, int end = -1) {};
	virtual void PrintDataDevice(const char* head = " ", unsigned int start = 0, int end = -1) {};

	virtual AlphaCore::AxDataType GetDataType() { return AlphaCore::AxDataType::kInvalidDataType; };
	virtual bool IsAarrayProperty() { return m_bIsArrayProperty; };
	virtual bool DeviceMalloc(bool loadToDevice = true) { return false; };
	virtual bool HasDeviceData() { return false; };

	virtual void AlignDeviceData() {};
	virtual void JsonDeserialization(std::string jsonRaw) {};
	virtual void JsonSerialization(std::string jsonRaw) {};
	virtual void* GetStorageRawData() { return nullptr; };
	virtual bool LoadToDevice()		  { return false; };
	virtual bool LoadToHost()		  { return false; };

	void SetToZero() { std::memset(this->GetStorageRawData(), 0, this->GetCapacity()); }
	void SetToFill() { std::memset(this->GetStorageRawData(), AX_MAX_BIT, this->GetCapacity()); }

	static void PrintDataFStreamingRAW(void* data, AxUInt32 size);

protected:

	std::string  m_sName;
	AxUInt64	 m_iBufferSize;
	AxUInt32	 m_iVecSize;
	AxUInt32	 m_iBankWidth;
	AxUInt32	 m_iPropertyToken;
	AxUInt32	 m_iBlockSize;
	bool		 m_bIsArrayProperty;
};


struct AxSPMDTick
{
	AlphaCore::AxBackendAPI LockType;
	int* CurrPos;
	void* Locker;
	ALPHA_SHARE_FUNC int AddIndex(int num = 1)
	{
		int last = *CurrPos;
#ifndef __CUDA_ARCH__
		if (LockType == AlphaCore::AxBackendAPI::CPUx86)
			*CurrPos = last + num;
#else
		if (LockType == AlphaCore::AxBackendAPI::CUDA) {
			last = atomicAdd(CurrPos, num);
		}
#endif
		return last;
	}

	inline void Reset()
	{
		if (LockType == AlphaCore::AxBackendAPI::CPUx86)
			*CurrPos = 0;
#ifdef ALPHA_CUDA
		if (LockType == AlphaCore::AxBackendAPI::CUDA)
			cudaMemset(CurrPos, 0, sizeof(int));
#endif
	}

	inline int Get()
	{
		if (LockType == AlphaCore::AxBackendAPI::CPUx86)
			return *CurrPos;
#ifdef ALPHA_CUDA
		if (LockType == AlphaCore::AxBackendAPI::CUDA)
		{
			int curr = -3;
			cudaMemcpy(&curr, CurrPos, sizeof(int), cudaMemcpyDeviceToHost);
			cudaThreadSynchronize();
			return curr;
		}
#endif // 
		return -3;
	}


};

static AxSPMDTick MakeSPMDTick(AlphaCore::AxBackendAPI type)
{
	AxSPMDTick tick;
	tick.LockType = type;
	if (tick.LockType == AlphaCore::AxBackendAPI::CPUx86)
		tick.CurrPos = new int;
#ifdef ALPHA_CUDA
	if (tick.LockType == AlphaCore::AxBackendAPI::CUDA)
		cudaMalloc((void**)&(tick.CurrPos), sizeof(int));
#endif
	tick.Locker = nullptr;
	tick.Reset();
	return tick;
}


template<class T>
class AxStorage :public AxStorageBase
{
public:
	AxStorage(const char* name = "__default")
	{
		m_iBlockSize = 1;
		m_iPropertyToken = 0;
		m_iBankWidth = sizeof(T);
		m_iBufferSize = 0;
		m_sName = name;
		m_iVecSize = AlphaCore::TypeVecSize<T>();
		m_iAccessStorageStart = -1;
		m_iAccessStorageEnd = -1;
#ifdef ALPHA_CUDA
		m_DevicePtr = nullptr;
#endif
	}

	AxStorage(std::string name ,AxUInt64 size)
	{
		m_iBlockSize = 1;
		m_iBufferSize = 0;
		m_iPropertyToken = 0;
		m_iBankWidth = sizeof(T);
		this->Resize(size);
		m_sName = name;
		m_iVecSize = AlphaCore::TypeVecSize<T>();
		m_iAccessStorageStart = -1;
		m_iAccessStorageEnd = -1;
#ifdef ALPHA_CUDA
		m_DevicePtr = nullptr;
#endif 
	}

	virtual ~AxStorage()
	{
		ClearAndDestory();
	}
	std::vector<T> m_Data;

	typedef AxStorage<T> ThisType;
	virtual void Resize(AxUInt64 size)
	{
		if (m_iBufferSize == size)
			return;
		m_iBufferSize = size; // TODO Do not match real storage
		if (m_Data.size() < size)
			m_Data.resize(size);
	}
	virtual AlphaCore::AxDataType GetDataType() 
	{
		return AlphaCore::TypeID<T>();
	};

	virtual void ResizeStorage(AxUInt64 size, bool updateDeviceMemory = true)
	{
		if (m_Data.size() == size)
			return;
		m_Data.resize(size);
		if (this->HasDeviceData() && updateDeviceMemory)
			this->ResizeStorageDevice(size);
	}

	AxUInt32 GetStorageSize()
	{
		return m_Data.size();
	}

	void Init(const T& constant)
	{
		AX_FOR_I(m_iBufferSize)
			m_Data[i] = constant;
	}

	void Push(T src)
	{
		m_Data.push_back(src);
		m_iBufferSize++;// m_Data.size();
	}

	virtual void ClearAndDestory()
	{
#ifdef ALPHA_CUDA
		if (this->HasDeviceData())
		{
			AX_WARN("device storage release  --- --- --- {} : Release {:03.2f} MB", m_sName,
				(float)GetStorageCapacity() / 1024.0f / 1024.0f);
			cudaFree(m_DevicePtr);
			m_DevicePtr = nullptr;
		}
#endif 
		auto _tmp = std::vector<T>();
		m_Data.swap(_tmp);
		m_iBufferSize = 0;
	}

	virtual void* GetStorageRawData() 
	{
		return (void*)m_Data.data();
	};

	virtual void Clear()
	{
		m_iBufferSize = 0;
		m_Data.clear();
	}

	T& operator [](uInt64 index)
	{
		return m_Data[index];
	}

	T operator ()(uInt64 index,AxUInt32 comp)
	{
		return m_Data[index];
	}

	ThisType& operator*=(const T &c)
	{
		for (uInt64 i = 0; i < m_Data.size(); ++i)
			this->m_Data[i] *= c;
 		return *this;
	}

	ThisType& operator/=(const T &c)
	{
		for (uInt64 i = 0; i < m_Data.size(); ++i)
			this->m_Data[i] /= c;
		return *this;
	}

	ThisType& operator+=(const ThisType &x)
	{
		for (uInt64 i = 0; i < m_Data.size(); ++i)
			this->m_Data[i] += x.m_Data[i];
		return *this;
	}

	ThisType& operator-=(const ThisType &x)
	{
		for (uInt64 i = 0; i < m_Data.size(); ++i)
			this->m_Data[i] -= x.m_Data[i];
		return *this;
	}

	void CopyBufferRaw(AxStorage<T>* other, bool forceAlignSize = true)
	{
		if (forceAlignSize)
		{
			if (other->Size() != this->Size())
				this->Resize(other->Size());
		}
		//TODO : std::min(other->GetCapacity(),this->GetCapacity())
		std::memcpy(m_Data.data(), other->m_Data.data(), std::min(other->GetCapacity(), this->GetCapacity()));
	}

	//AxUInt64 BufferSize()			{ return m_Data.size(); }
	AxUInt64 Size()const			
	{
		//std::cout << "S:" << m_iAccessStorageStart << " - " << m_iAccessStorageEnd << std::endl;
		if (m_iAccessStorageStart == -1 && m_iAccessStorageEnd == -1)
		{
			return m_iBufferSize;
		}
		AxInt32 realStart = m_iAccessStorageStart;
		if (realStart < 0)
			realStart = 0;
		AxInt32 realEnd = m_iAccessStorageEnd;
		if (realEnd <= 0)
			realEnd = m_iBufferSize;
		return realEnd - realStart;
	}
	AxUInt64 GetTypeSize()			{ return sizeof(T); }
	virtual AxUInt64 GetCapacity()  { return sizeof(T) * m_iBufferSize; };
	virtual AxUInt64 GetStorageCapacity() { return sizeof(T) * m_Data.size(); };

	virtual AxUInt64 GetIOCapacity()
	{ 
		// sizeof(int)
		// m_iPropertyToken	 AxUInt32 
		// m_iBankWidth		 AxUInt32
		// m_iVecSize		 AxUInt32
		// m_iBufferSize	 AxUInt64
		// size of StorageHeadInfo 
		return sizeof(int) + (m_sName.size() + 1) +
			sizeof(AxUInt32) + sizeof(AxUInt32) + sizeof(AxUInt32) +
			sizeof(AxUInt64) + this->GetCapacity();
	};

	virtual void ReadRaw(std::ifstream& ifs)
	{
		if (!ifs)
			return;
		POS_READ_INFO("NAME", ifs);
		AlphaUtility::ReadSTLString(ifs, m_sName);
 		POS_READ_INFO(m_sName.c_str(), ifs);
 		ifs.read((char*)&m_iPropertyToken,sizeof(AxUInt32));
		POS_READ_INFO("m_iPropertyToken", ifs);
		ifs.read((char*)&m_iBankWidth,	  sizeof(AxUInt32));
		POS_READ_INFO("m_iBankWidth", ifs);
		ifs.read((char*)&m_iVecSize,	  sizeof(AxUInt32));
		POS_READ_INFO("m_iVecSize", ifs);
		ifs.read((char*)&m_iBufferSize,	  sizeof(AxUInt64));
		POS_READ_INFO("m_iBufferSize", ifs);
		this->m_Data.resize(m_iBufferSize);//DO NOT USE Resize Method !!!
		ifs.read((char*)m_Data.data(), this->GetCapacity());
	}

	virtual void SaveRaw(std::ofstream& ofs)
	{
		if (!ofs)
			return;
		POS_WRITE_INFO("NAME", ofs);
		AlphaUtility::WriteSTLString(ofs, m_sName);
		//POS_WRITE_INFO(m_sName.c_str(), ofs);
		ofs.write((char*)&m_iPropertyToken, sizeof(AxUInt32));
		POS_WRITE_INFO("m_iPropertyToken", ofs);
		ofs.write((char*)&m_iBankWidth,		sizeof(AxUInt32));
		POS_WRITE_INFO("m_iBankWidth", ofs);
		ofs.write((char*)&m_iVecSize,		sizeof(AxUInt32));
		POS_WRITE_INFO("m_iVecSize", ofs);
		ofs.write((char*)&m_iBufferSize,	sizeof(AxUInt64));
		POS_WRITE_INFO("m_iBufferSize", ofs);
		char* dataRaw = (char*)m_Data.data();
		POS_WRITE_INFO("m_Data", ofs);
		ofs.write(dataRaw, this->GetCapacity());
	}

	T* Data()	{ return m_Data.data(); }
	T* GetDataRaw() 
	{
		if (m_iAccessStorageStart == -1)
			return m_Data.data(); 
		return m_Data.data() + m_iAccessStorageStart;
	}

	template<typename S>
	S* GetDataRaw() 
	{
		if (m_iAccessStorageStart == -1)
			return (S*)m_Data.data();
		return (S*)(m_Data.data() + m_iAccessStorageStart);
	}

	AxInt32 GetAccessStorageStart()
	{
		return m_iAccessStorageStart;
	}

	AxInt32 GetAccessStorageEnd()
	{
		return m_iAccessStorageEnd;
	}

	void CreateIdenityBuffer()
	{
		AX_FOR_I(this->Size())
			this->Set(i, i);
	}
	typename std::vector<T>::iterator Begin() { return m_Data.begin(); };
	typename std::vector<T>::iterator End() { return m_Data.end(); };
	template<typename S>
	void CopyRaw(S* other, AxUInt32 size)
	{
		this->Resize(size);
		std::memcpy(m_Data.data(), other, size * sizeof(T));
	}

	template<typename S>
	void CopyRawFORCE(S* other, AxUInt32 dstStart,AxUInt32 srcStart, AxUInt32 srcEnd)
	{
		std::memcpy(m_Data.data() + dstStart, other + srcStart, (srcEnd - srcStart) * sizeof(T));
	}

	template<typename S>
	S* GetDataRawDevice() 
	{ 
#ifdef ALPHA_CUDA
 		return (S*)m_DevicePtr;
#else
		return nullptr;
#endif 
 	}

 	T* GetDataRawDevice()
	{
#ifdef ALPHA_CUDA
		if (m_iAccessStorageStart == -1)
	 		return (T*)m_DevicePtr;
		return (T*)m_DevicePtr + m_iAccessStorageStart;
#else
		return nullptr;
#endif 
	}

	void CreateIdenityBufferDevice();

	virtual void PrintData(const char* head = " ", unsigned int start = 0, int end = -1)
	{
		unsigned int _end = end < 0 ? this->Size() : std::min((int)this->Size(), end);
		printf("| %s | Buffer  [%s][%s]<%d>:", head,
			m_sName.c_str(),AlphaCore::DataTypeToString(this->GetDataType()),
			AlphaCore::TypeVecSize<T>());
		printf("Property token [ RTOnly : %s , Protected : %s , Private : %s ] \n", 
			IsRTOnlyData()		? "True" : "False",
			IsProtectedData()	? "True" : "False",
			IsPrivateData()		? "True" : "False");
		for (size_t i = start; i < _end; ++i)
			std::cout << head << "[ " << i << " ]" << m_Data[i] << std::endl;
	}

	void PrintDataABTest(AxStorage<T>* other, AxUInt32 start = 0, int end = -1, int sep = 1)
	{
		AxUInt32 _end = end < 0 ? this->Size() : std::min((int)this->Size(), end + 1);
		for (size_t i = start; i < _end; ++i)
			if (i % sep == 0)
				std::cout << m_sName.c_str() << " : " << other->GetName() << " [ " << i << " ]:" << m_Data[i] << "  " << other->m_Data[i] << std::endl;
	}

	void PrintDataTest3Buffer(AxStorage<T>* other, AxStorage<T>* other2, AxUInt32 start = 0, int end = -1)
	{
		AxUInt32 _end = end < 0 ? this->Size() : std::min((int)this->Size(), end);
		for (size_t i = start; i < _end; ++i)
			std::cout << m_sName.c_str() << "  ,  " << other->GetName() << "  ,  " << other2->GetName()\
					  << " [ " << i << " ] : " << m_Data[i] << "   " << other->m_Data[i] << "   " << other2->m_Data[i] << std::endl;
	}

	void PrintDataTest4Buffer(AxStorage<T>* other1, AxStorage<T>* other2, AxStorage<T>* other3, AxUInt32 start = 0, int end = -1)
	{
		AxUInt32 _end = end < 0 ? this->Size() : std::min((int)this->Size(), end);
		for (size_t i = start; i < _end; ++i)
		{	
			auto temp = other1->m_Data[i];
			std::cout << m_sName.c_str() << "  ,  " \
				<< other1->GetName() << "  ,  " \
				<< other2->GetName() << "  ,  "\
				<< other3->GetName()\
				<< " [  " << i << "  ] : "\
				<< m_Data[i] << "   " \
				<< other1->m_Data[i] << "   " \
				<< other2->m_Data[temp] << "   " \
				<< other3->m_Data[temp] << std::endl;
		}
	}

	//void Append(AxUInt32 num) { m_Data.resize(m_Data.size() + num); };

	void PrintNonZeroData(const char* head = " ", unsigned int start = 0, int end = -1)
	{
		unsigned int _end = end < 0 ? this->Size() : end;
		printf("Buffer [%s]:", m_sName.c_str());
		for (size_t i = start; i < _end; ++i)
		{
			if (m_Data[i] == 0)
				continue;
			std::cout << head << "[ " << i << " ]" << m_Data[i] << std::endl;
		}
	}
	
	inline T	Get(AxUInt64 index) { return m_Data[index]; };
	inline T&	GetValue(AxUInt64 index)
	{ 
		if (index > this->Size())
		{
			AX_ERROR("Out");
		}
		return m_Data[index];
	};
	inline T	Get(AxUInt64 index, AxUInt32 subId) { return m_Data[index * m_iBlockSize + subId]; };

	void Set(T value, AxUInt64 index = 0) { m_Data[index] = value; };
	void SetConstant(T value) 
	{
		AX_FOR_I(this->Size())
			m_Data[i] = value; 
	};

	void PrintDataFStreaming(std::string path = "", const char* head = " ", unsigned int start = 0, int end = -1)
	{
		if (path.size() == 0)
		{
			path = AlphaCoreEngine::GetInstance()->GetPrintDataFStreamPath();
			path += "DOPDebugStreaming.txt";
		}

		std::ofstream ofs(path, std::ios::app);
		if (!ofs)
			return;
 		
		AxUInt32 _end = end < 0 ? this->Size() : end;
		AX_INFO("PrintBuffer As TXT {} Size:{}  Range:[{} - {}]\n", path.c_str(), this->Size(), start, _end);
		ofs << "__BufferTraceStart__#" << m_sName.c_str() << std::endl;;
		for (size_t i = start; i < _end; ++i)
		{
			std::stringstream sstr;
			sstr << m_Data[i];
			ofs << " \"" << head << "\"" << "<" << i << "> " << sstr.str().c_str() << " " << std::endl;
		}
		ofs << "__BufferTraceEnd__#" << m_sName.c_str();
		ofs.flush();
		ofs.close();
		AX_INFO("Save buffer print data @ {} succ", path);
	}
	
	virtual void ResizeStorageDevice(AxUInt64 size)
	{
#ifdef ALPHA_CUDA
		//TODO Next Remove
		this->LoadToHost();
		cudaFree(m_DevicePtr);
		this->m_DevicePtr = nullptr;
		this->DeviceMalloc();
#endif
	}

	AxInt32 m_iAccessStorageStart;
	AxInt32 m_iAccessStorageEnd;
	void SetAvailableRange(AxInt32 start, AxInt32 end)
	{
		m_iAccessStorageStart = start;
		m_iAccessStorageEnd   = end;
	}

	void ClearAccessRange()
	{
		m_iAccessStorageStart = -1;
		m_iAccessStorageEnd	  = -1;
	}

	AxVector2I SetAvailableRange()
	{
		return MakeVector2I(m_iAccessStorageStart, m_iAccessStorageEnd);
	}

#ifdef ALPHA_CUDA

public:

	uInt64 m_iDeviceBufferSize;
	virtual bool DeviceMalloc(bool loadToDevice = true)
	{
		if (m_DevicePtr != nullptr)
		{
			AX_WARN("{0} DeviceRegistered!", m_sName.c_str());
			return false;
		}

		auto cudaRet = cudaMalloc((void**)&m_DevicePtr, this->GetStorageCapacity());
		if (cudaRet != cudaSuccess)
		{
			AX_GET_DEVICE_LAST_ERROR;
			AX_ERROR("{0} CudaMalloc Frailed", m_sName.c_str());
			return false;
		}
		else
		{
 			AX_INFO("Malloc device memory succ use | {:03.2f} MB | {}",
				(float)GetStorageCapacity() / 1024.0f / 1024.0f,m_sName.c_str());
		}

		m_iDeviceBufferSize = m_Data.size();
		if (loadToDevice)
			LoadToDevice();
		return true;
	}

	virtual bool HasDeviceData()
	{
		return m_DevicePtr != nullptr;
	}

	//virtual int GetDataTypeToken() { return AlphaCore::DataTypeToken<T>(); };

	virtual void AlignDeviceData()
	{
		if (m_DevicePtr == nullptr)
			return;

	}

	virtual bool LoadToDevice()
	{
		if (!m_DevicePtr)
		{
			AX_ERROR("{0} Device Not Registered!", m_sName.c_str());
			return false;
		}
		
		if (m_iDeviceBufferSize < m_Data.size())
		{
			cudaFree(m_DevicePtr);
			m_DevicePtr = nullptr;
			DeviceMalloc(false);
			AX_WARN("{0} Device Re-allocation ! ", m_sName.c_str());
		}

		auto cudaRet = cudaMemcpy(this->GetDevicePtr(), m_Data.data(), this->GetCapacity(), cudaMemcpyHostToDevice);
		if (cudaRet != cudaSuccess)
		{
			AX_ERROR(" {0} cudaMemcpy frailed whit size {1}", m_sName.c_str(), this->GetCapacity());
			return false;
		}

		m_iDeviceBufferSize = m_Data.size();
		return true;
	}

	virtual bool LoadToHost()
	{
		if (!m_DevicePtr)
		{
			AX_ERROR("{0} Device Not Registered!", m_sName.c_str());
			return false;
		}
		AX_WARN("Performance Warnning !!! : Loadback {0} form device", m_sName.c_str());
		auto cudaRet = cudaMemcpy(m_Data.data(), m_DevicePtr, this->GetCapacity(), cudaMemcpyDeviceToHost);
		if (cudaRet != cudaSuccess)
		{
			AX_GET_DEVICE_LAST_ERROR;
			AX_ERROR("{0}  cudaMemcpy frailed", m_sName.c_str());
			return false;
		}
		return true;
	}

	T* GetDevicePtr()
	{
		return (T*)m_DevicePtr;
	}

	bool DeviceToDeviceMemcpy(T* dPtr)
	{
		if (!m_DevicePtr || !dPtr)
		{
 			AX_ERROR("{} : Device Not Registered!", m_sName.c_str());
			return false;
		}
 		cudaError_t ret = cudaMemcpy(m_DevicePtr, dPtr,this->GetCapacity(),cudaMemcpyDeviceToDevice);
 		if (ret != cudaSuccess)
		{
			AX_GET_DEVICE_LAST_ERROR;
			AX_ERROR(" \"{}\" : Device 2 Device cudaMemcpy frailed", m_sName.c_str());
			return false;
		}
		return true;
	}

	bool DeviceToDeviceMemcpy(AxStorage<T>* dPtr)
	{
		return DeviceToDeviceMemcpy(dPtr->GetDataRawDevice());
	}

	AxVector2UI GetBlockThreadInfo(AxUInt32 blockSize=512)
	{
		return ThreadBlockInfo(blockSize, this->Size());
	}
	
	virtual void PrintDataDevice(const char* head = " ", unsigned int start = 0, int end = -1)
	{
		this->LoadToHost();
		this->PrintData(head, start, end);
	}

	void SetToZeroDevice()
	{
		if (m_DevicePtr == nullptr)
			return;
		cudaMemset(m_DevicePtr, 0, this->GetCapacity());
	}

	void SetToFillDevice()
	{
		if (m_DevicePtr == nullptr)
			return;
		cudaMemset(m_DevicePtr, AX_INVALID_INT32, this->GetCapacity());
	}

 	void* m_DevicePtr;

	void PrintDataFStreamingDevice(std::string path="", const char* head = " ", unsigned int start = 0, int end = -1)
	{
		this->LoadToHost();
		if (path.size() == 0)
		{
			path = AlphaCoreEngine::GetInstance()->GetPrintDataFStreamPath();
			path += "DOPDebugStreaming.CUDA.txt";
		}
		this->PrintDataFStreaming(path, head, start, end);
	}
#endif
};

typedef AxStorage<float>		AxBufferF;
typedef AxStorage<double>		AxBufferD;
typedef AxStorage<char>			AxBufferC;
typedef AxStorage<int>			AxBufferI;
typedef AxStorage<AxUInt32>		AxBufferUInt32;
typedef AxStorage<AxVector2>	AxBufferV2;
typedef AxStorage<AxVector2I>	AxBuffer2I;
typedef AxStorage<AxVector2UI>	AxBuffer2UI;
typedef AxStorage<AxVector3>	AxBufferV3;
typedef AxStorage<AxVector3I>	AxBuffer3I;
typedef AxStorage<AxVector3UI>	AxBuffer3UI;
typedef AxStorage<AxVector4>	AxBufferV4;
typedef AxStorage<AxVector4I>	AxBuffer4I;
typedef AxStorage<AlphaCore::AxPrimitiveType> AxPrimTypeBuffer;
typedef AxStorage<void*>		AxBufferRAW;
typedef AxStorage<Quat>			AxBufferQuat;
typedef AxStorage<AxMat2x2F>	AxBufferMat2x2F;
typedef AxStorage<AxMat2x2D>	AxBufferMat2x2D;
typedef AxStorage<AxMatrix3x3>	AxBufferMat3x3F;
typedef AxStorage<AxMatrix3x3D>	AxBufferMat3x3D;
typedef AxStorage<AxMatrix4x4>	AxBufferMat4x4F;
typedef AxStorage<AxMatrix3x3D>	AxBufferMat4x4D;

//AxMatrix3x3

//Advance
typedef AxStorage<AxUInt32>			AxBufferUInt32;
typedef AxStorage<AxUInt64>			AxBufferUInt64;
typedef AxStorage<unsigned char>	AxBufferUChar;
class AxBufferS :public AxStorage<std::string>
{
public:
	AxBufferS(std::string name, long int vs)
	{
		m_sName = name;
		this->Resize(vs);
	}

	~AxBufferS()
	{

	}

	virtual void ReadRaw(std::ifstream& ifs)
	{
		if (!ifs)
			return;
		AlphaUtility::ReadSTLString(ifs, m_sName);
		ifs.read((char*)&m_iPropertyToken, sizeof(AxUInt32));
		ifs.read((char*)&m_iBankWidth, sizeof(AxUInt32));
		ifs.read((char*)&m_iVecSize, sizeof(AxUInt32));
		ifs.read((char*)&m_iBufferSize, sizeof(AxUInt64));
		//this->Resize(m_iBufferSize);
		//ifs.read((char*)m_Data.data(), this->GetCapacity());
		this->m_Data.resize(m_iBufferSize);
		for (int i = 0; i < m_Data.size(); ++i)
			AlphaUtility::ReadSTLString(ifs, m_Data[i]);
	}

	virtual void SaveRaw(std::ofstream& ofs)
	{
		if (!ofs)
			return;
		AlphaUtility::WriteSTLString(ofs, m_sName);
		ofs.write((char*)&m_iPropertyToken, sizeof(AxUInt32));
		ofs.write((char*)&m_iBankWidth, sizeof(AxUInt32));
		ofs.write((char*)&m_iVecSize, sizeof(AxUInt32));
		ofs.write((char*)&m_iBufferSize, sizeof(AxUInt64));
		//std::cout << "m_IBufferSize : " << this->m_iBufferSize << "  " << __FUNCTION__ << std::endl;
		//std::cout << " off" << ofs.tellp() << std::endl;
		for (int i = 0; i < m_Data.size(); ++i)
			AlphaUtility::WriteSTLString(ofs, m_Data[i]);
	}

	virtual AxUInt64 GetCapacity()
	{
		AxUInt64 ret = 0;
		AX_FOR_I(m_iBufferSize)
			ret += m_Data[i].size() + 1 + sizeof(int);
		return ret;
	};

	virtual AxUInt64 GetIOCapacity()
	{
		// m_iPropertyToken	 AxUInt32 
		// m_iBankWidth		 AxUInt32
		// m_iVecSize		 AxUInt32
		// m_iBufferSize	 AxUInt64
		return sizeof(int) + m_sName.size() + 1 + sizeof(AxUInt32) + sizeof(AxUInt32) + sizeof(AxUInt32) +
			sizeof(AxUInt64) + this->GetCapacity();
	};
};

template <class T>
struct AxStorageAccessor
{
	T* Data;
};



namespace AlphaCore
{
	namespace StorageHelper
	{
		template<typename T>
		static AxFp32 BufferCompare(T* A, T* B, AxUInt32 bufferSize)
		{
			AX_INFO("Comp: {} Items", bufferSize);
			T ret = 0;
			AX_FOR_I(bufferSize)
				ret += std::fabs(A[i] - B[i]);
			return ret;
		};
	}
}


#endif //

