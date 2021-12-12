#ifndef __ALPHA_CORE_STORAGE_H__
#define __ALPHA_CORE_STORAGE_H__

#include <vector>
#include <iostream>
#include <fstream>
#include <Math/AxVectorBase.h>
#include <AxMacro.h>
#include <AxLog.h>

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
};

template<class T>
class AxStorage
{
public:
	AxStorage() 
	{
		m_iBufferSize = 0;
		m_sName = "__default";
#ifdef ALPHA_CUDA
		m_DevicePtr = nullptr;
#endif // 

	}
	~AxStorage() 
	{
		Release(); 
	}
	std::vector<T> m_Data;

	void Read(std::ifstream& ifs);
	void Save(std::ofstream& ofs);

	typedef AxStorage<T> ThisType;
	void Resize(uInt64 size)
	{
		if (m_iBufferSize == size)
			return;
		m_iBufferSize = size;
		m_Data.resize(size);
 	}
	void Push(T src)
	{
		m_Data.push_back(src);
	}

	void Release()
	{
		auto _tmp = std::vector<T>();
		m_Data.swap(_tmp);
		m_iBufferSize = 0;
	}

	void Clear()
	{
		m_iBufferSize = 0;
		m_Data.clear();
	}

	T& operator [](uInt64 index)
	{
		return m_Data[index];
	}

	T operator ()(uInt64 index,uInt32 comp)
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

	uInt64 BufferSize()		{ return m_Data.size(); }
	uInt64 Size()			{ return m_iBufferSize; }
	uInt64 GetTypeSize()	{ return sizeof(T); }
	T* GetRawPtr()	{ return m_Data.data(); }

	void SetName(std::string n) { m_sName = n; };
	std::string GetName()const { return  m_sName; };

	void SetToZero() { memset(m_Data.data(), 0, sizeof(T)*m_Data.size()); }

	void PrintData(const char* head = " ", unsigned int start = 0, int end = -1)
	{
		unsigned int _end = end < 0 ? m_Data.size() : end;
		printf("Buffer [%s]:", m_sName.c_str());
		for (size_t i = start; i < _end; ++i)
			std::cout << head << "[ " << i << " ]" << m_Data[i] << std::endl;
	}

	void PrintNonZeroData(const char* head = " ", unsigned int start = 0, int end = -1)
	{
		
		unsigned int _end = end < 0 ? m_Data.size() : end;
		printf("Buffer [%s]:", m_sName.c_str());
		for (size_t i = start; i < _end; ++i)
		{
			if (m_Data[i] == 0)
				continue;
			std::cout << head << "[ " << i << " ]" << m_Data[i] << std::endl;
		}
		//*/
	}


	uInt64 GetNumBytes() { return m_Data.size() * sizeof(T); }

private:
	std::string m_sName;
	uInt64 m_iBufferSize;

#ifdef ALPHA_CUDA

public:

	uInt64 m_iDeviceBufferSize;
 
	bool DeviceMalloc(bool loadToDevice = true)
	{
		if (m_DevicePtr != nullptr)
		{
			AX_WARN("{0} DeviceRegistered!", m_sName.c_str());
			return false;
		}

		auto cudaRet = cudaMalloc((void**)&m_DevicePtr, this->GetNumBytes());
		if (cudaRet != cudaSuccess)
		{
			AX_GET_CUDA_LAST_ERROR;

			AX_ERROR("{0} CudaMalloc Frailed", m_sName.c_str());
			return false;
		}
		else
		{
 			AX_INFO("Malloc device memory succ use | {:03.2f} MB | {}",
				(float)GetNumBytes() / 1024.0f / 1024.0f,m_sName.c_str());
		}

		m_iDeviceBufferSize = m_Data.size();
		if (loadToDevice)
			LoadToDevice();

		return true;
	}

	bool HasDeviceData()
	{
		return m_DevicePtr!=nullptr;
	}

	bool LoadToDevice()
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

		auto cudaRet = cudaMemcpy(this->GetDevicePtr(), m_Data.data(), this->GetNumBytes(), cudaMemcpyHostToDevice);
		if (cudaRet != cudaSuccess)
		{
			AX_ERROR(" {0} cudaMemcpy frailed whit size {1}", m_sName.c_str(), this->GetNumBytes());
			return false;
		}

		m_iDeviceBufferSize = m_Data.size();
		return true;
	}

	bool LoadToHost()
	{
		if (!m_DevicePtr)
		{
			AX_ERROR("{0} Device Not Registered!", m_sName.c_str());
			return false;
		}
		auto cudaRet = cudaMemcpy(m_Data.data(), m_DevicePtr, this->GetNumBytes(), cudaMemcpyDeviceToHost);
		if (cudaRet != cudaSuccess)
		{
			AX_GET_CUDA_LAST_ERROR;
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
 		cudaError_t ret = cudaMemcpy(m_DevicePtr, dPtr,this->GetNumBytes(),cudaMemcpyDeviceToDevice);
 		if (ret != cudaSuccess)
		{
			AX_GET_CUDA_LAST_ERROR;
			AX_ERROR(" \"{}\" : Device 2 Device cudaMemcpy frailed", m_sName.c_str());
			return false;
		}
		return true;
	}

	void SetToZeroDevice()
	{
		if (m_DevicePtr == nullptr)
			return;
		cudaMemset(m_DevicePtr, 0, this->GetNumBytes());
	}

 	void* m_DevicePtr;

#endif

};

typedef AxStorage<float>		AxBufferF;
typedef AxStorage<double>		AxBufferD;
typedef AxStorage<char>			AxBufferC;
typedef AxStorage<int>			AxBufferI;
typedef AxStorage<uInt32>		AxBufferUInt32;
typedef AxStorage<AxVector3>	AxBufferV3;
typedef AxStorage<AxVector2I>	AxBuffer2I;



#endif // !__ALPHA_CORE_STORAGE_H__
