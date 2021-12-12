#ifndef __ALPHA_CORE_GRID_3D_H__
#define __ALPHA_CORE_GRID_3D_H__

#include <Utility/AxDescrition.h>
#include <Math/AxVectorBase.h>
#include <Math/AxVectorHelper.h>
#include <vector>
#include <sstream>
#include <cmath>
#include <cfloat>

struct AxFieldHeadInfo
{
	uInt32 nFields;

};

class AxField
{
public:
	AxField() {};
	~AxField() {};

	std::string GetName()
	{
		return m_sName;
	}

#ifdef ALPHA_CUDA
	virtual bool DeviceMalloc(bool loadToDevice = true)
	{

		return false;
	}
#endif 



protected:

	std::string m_sName;


};

template<class T>
class AxField3DBase : public AxField
{
public:

	AxField3DBase(std::string name = AlphaProperty::DensityField)
	{
		SetFieldResolution(10, 10, 10);
		m_sName = name;
		m_VoxelBuffer.SetName(m_sName + ".voxels");
  	}
	AxField3DBase(uInt32 nx, uInt32 ny, uInt32 nz, std::string name = AlphaProperty::DensityField)
	{
		SetFieldResolution(nx, ny, nz);
		m_sName = name;
		m_VoxelBuffer.SetName(m_sName + ".voxels");
	}

	AxField3DBase(const AlphaCore::Desc::AxField3DInfo& info, std::string name = AlphaProperty::DensityField)
	{
		AxVector3 size = MakeVector3(
			(float)info.Resolution.x * info.VoxelSize.x,
			(float)info.Resolution.y * info.VoxelSize.y,
			(float)info.Resolution.z * info.VoxelSize.z);
		Init(info.Pivot, size, info.Resolution);
		m_sName = name;
		m_VoxelBuffer.SetName(m_sName + ".voxels");
	}

	AxField3DBase(AxField3DBase<T>* field,bool cpyVoxelBuf=false)
	{
 		AxVector3 size = MakeVector3(
			field->GetFieldInfo().Resolution.x*field->GetFieldInfo().VoxelSize.x,
			field->GetFieldInfo().Resolution.y*field->GetFieldInfo().VoxelSize.y,
			field->GetFieldInfo().Resolution.z*field->GetFieldInfo().VoxelSize.z);
		Init(field->GetFieldInfo().Pivot, size, field->GetFieldInfo().Resolution);
		if (!cpyVoxelBuf)
			return;
	}

	AxField3DBase(AxVector3 pivot, AxVector3 size, AxVector3UI res, std::string name = AlphaProperty::DensityField, T* dataRaw = nullptr)
	{
		Init(pivot,size,res);
		m_sName = name;
		m_VoxelBuffer.SetName(m_sName + ".voxels");
		if (dataRaw != nullptr)
			this->ReadRawBuffer(dataRaw);
	}

	~AxField3DBase()
	{
		m_VoxelBuffer.Release();
	}

	void Init(AxVector3 pivot, AxVector3 size, AxVector3UI res,bool setToZero = false)
	{
		SetPivot(pivot);
		SetFieldResolution(res);
		SetFieldSize(size);
		if (setToZero)
			m_VoxelBuffer.SetToZero();
	}

	uInt32 GetNX()
	{
		return m_FieldInfo.Resolution.x;
	}

	uInt32 GetNY()
	{
		return m_FieldInfo.Resolution.y;
	}

	uInt32 GetNZ()
	{
		return m_FieldInfo.Resolution.z;
	}

	void Release()
	{
		m_VoxelBuffer.Release();
	}

	void SetFieldResolution(AxVector3UI res)
	{
		SetFieldResolution(res.x, res.y, res.z);
	}

	void SetFieldResolution(uInt32 nx, uInt32 ny, uInt32 nz)
	{
		AxVector3 sizeOld = MakeVector3(
			m_FieldInfo.Resolution.x * m_FieldInfo.VoxelSize.x, 
			m_FieldInfo.Resolution.y * m_FieldInfo.VoxelSize.y,
			m_FieldInfo.Resolution.z * m_FieldInfo.VoxelSize.z);

		m_FieldInfo.Resolution.x = nx;
		m_FieldInfo.Resolution.y = ny;
		m_FieldInfo.Resolution.z = nz;
		
		SetFieldSize(sizeOld);

		m_VoxelBuffer.Resize(nx * ny * nz);
	}

	void SetPivot(AxVector3 pivot)
	{
		m_FieldInfo.Pivot = pivot;
	}

	void SetPivot(float tx,float ty,float tz)
	{
		m_FieldInfo.Pivot.x = tx;
		m_FieldInfo.Pivot.y = ty;
		m_FieldInfo.Pivot.z = tz;
	}

	void SetFieldSize(float sx, float sy, float sz)
	{
		AxVector3 size = MakeVector3(sx, sy, sz);
		m_FieldInfo.VoxelSize = size / m_FieldInfo.Resolution;
		m_FieldInfo.FieldSize = size;
 	}

	void SetFieldSize(AxVector3 size)
	{
		SetFieldSize(size.x, size.y, size.z);
	}

	void SetVoxelSize(AxVector3 size)
	{
		m_FieldInfo.VoxelSize = size;
	}

	uInt64 GetNumVoxels() 
	{ 
		return m_FieldInfo.Resolution.x * m_FieldInfo.Resolution.y * m_FieldInfo.Resolution.z;
	}

	uInt64 GetSliceVoxels_XY()
	{
		return m_FieldInfo.Resolution.x * m_FieldInfo.Resolution.y;
	}

	uInt64 GetSliceVoxels_YZ()
	{
		return  m_FieldInfo.Resolution.y * m_FieldInfo.Resolution.z;
	}

	uInt64 GetSliceVoxels_XZ()
	{
		return m_FieldInfo.Resolution.x * m_FieldInfo.Resolution.z;
	}

	AxVector3UI GetResolution() { return m_FieldInfo.Resolution; };

	AlphaCore::Desc::AxField3DInfo GetFieldInfo() { return m_FieldInfo; }

	T* GetRawData() { return m_VoxelBuffer.GetRawPtr(); };

	void ReadRawBuffer(T* data)
	{
		std::memcpy(this->m_VoxelBuffer.m_Data.data(),data , this->GetNumVoxels() * sizeof(T));
	}

	void PrintInfo()
	{
		std::cout<<" ------------ Field [ "<<m_sName<<" ]----------------" << std::endl;
		std::cout << "Pivot:" <<
			m_FieldInfo.Pivot.x <<" , "<<
			m_FieldInfo.Pivot.y <<" , " << 
			m_FieldInfo.Pivot.z << std::endl;

		std::cout << "Res : [" <<
			m_FieldInfo.Resolution.x << " , " <<
			m_FieldInfo.Resolution.y << " , " <<
			m_FieldInfo.Resolution.z << " ] " << std::endl;

		std::cout << "voxelSize : [" <<
			m_FieldInfo.VoxelSize.x << " , " <<
			m_FieldInfo.VoxelSize.y << " , " <<
			m_FieldInfo.VoxelSize.z << " ] " << std::endl;
		 
		/*
		for (int i = 0; i < m_VoxelBuffer.Size(); i+=10)
		{
			std::cout << m_VoxelBuffer[i] <<  ",";
			if (i % 128 == 0)
				std::cout << "\n";
		}
		*/
	}

	virtual void SetName(std::string name)
	{
		m_sName = name;
		m_VoxelBuffer.SetName(m_sName + ".voxels");
	}

	T GetValue(uInt64 voxelId)
	{
		return m_VoxelBuffer[voxelId];
	}

	T GetValue(uInt32 idx, uInt32 idy, uInt32 idz)
	{
		uInt32 voxelId = idz * m_FieldInfo.Resolution.x *m_FieldInfo.Resolution.y + idy * m_FieldInfo.Resolution.x + idx;
		return m_VoxelBuffer[voxelId];
	}

	void SetValue(uInt64 voxelId, T val)
	{
		m_VoxelBuffer[voxelId] = val;
	}

	void SetValue(uInt32 idx, uInt32 idy, uInt32 idz, T val)
	{
		uInt32 voxelId = idz * m_FieldInfo.Resolution.x *m_FieldInfo.Resolution.y + idy * m_FieldInfo.Resolution.x + idx;
		m_VoxelBuffer[voxelId] = val;
	}

	void AddField(AxField3DBase<T>* field)
	{
 		for (uInt64 i = 0; i < field->GetNumVoxels(); ++i)
 			this->SetValue(i, this->GetValue(i) + field->GetValue(i));
 	}

	void MultiplyConstant(T constant)
	{
		for (uInt64 i = 0; i < this->GetNumVoxels(); ++i)
			this->SetValue(i, this->GetValue(i)*constant);
	}

	void SubtractField(AxField3DBase<T>* field)
	{
		for (uInt64 i = 0; i < field->GetNumVoxels(); ++i)
			this->SetValue(i, this->GetValue(i) - field->GetValue(i));
	}

	AxStorage<T>& GetVoxelStorageBuffer()
	{
		return m_VoxelBuffer;
	}

	void SetToZero()
	{
		memset(m_VoxelBuffer.GetRawPtr(), 0, m_VoxelBuffer.Size()*sizeof(T));
	}

	bool CopyVoxelDataBuffer(AxField3DBase<T>* field)
	{
		if (field->GetNumVoxels() != this->GetNumVoxels())
			return false;
		std::cout << "COPY : " << this->m_sName.c_str() << " ---- " << field->GetName() << std::endl;
		memcpy(m_VoxelBuffer.GetRawPtr(), field->GetRawData(), m_VoxelBuffer.Size() * sizeof(T));
		return true;
	}


	T Different(AxField3DBase<T>* field)
	{
		if (field->GetNumVoxels() != this->GetNumVoxels())
			return -1;
		T total = 0;
		for (uInt64 i = 0; i < field->GetNumVoxels(); ++i) 
			total += abs(this->GetValue(i) - field->GetValue(i));
		return total;
	}

	void TraceData(int start = 0 ,int end =-1, int sep = 1, const char* head = "")
	{
		return;
 #ifdef ALPHA_CUDA
 		if(m_VoxelBuffer.HasDeviceData())
			this->LoadToHost();
#endif 
		end = end < 0 ? m_VoxelBuffer.BufferSize() : end;
		std::stringstream sstr;
		sstr << "[";
		for (size_t i = start; i < end; i+= 1){
			if (std::isnan(m_VoxelBuffer[i]))
				continue;
			sstr << m_VoxelBuffer[i];
			if (i != end - 1)
				sstr << ",";
		}
		sstr << "]";
		AX_TRACE("<{}> Field {} {}", head,m_sName.c_str(), sstr.str().c_str());

	}


private:	

	AlphaCore::Desc::AxField3DInfo m_FieldInfo;
	AxStorage<T> m_VoxelBuffer;


public:

	bool Read(std::string path)
	{
		std::ifstream ifs(path.c_str(), std::ios::binary);
		if (!ifs)
			return false;
		AxFieldHeadInfo headInfo;
		ifs.read((char*)(&headInfo.nFields), sizeof(uInt32));
		this->Read(ifs);
		ifs.close();
		return true;
	}


	bool Read(std::ifstream& ifs)
	{
		AlphaUtility::ReadSTLString(ifs, m_sName);
		m_VoxelBuffer.SetName(m_sName + ".voxels");

		uInt32 dataPrecision = sizeof(T);
		uInt64 nVoxels = 0;

		ifs.read((char*)(&m_FieldInfo.Pivot.x), sizeof(float));
		ifs.read((char*)(&m_FieldInfo.Pivot.y), sizeof(float));
		ifs.read((char*)(&m_FieldInfo.Pivot.z), sizeof(float));
 		ifs.read((char*)(&m_FieldInfo.Resolution.x), sizeof(uInt32));
		ifs.read((char*)(&m_FieldInfo.Resolution.y), sizeof(uInt32));
		ifs.read((char*)(&m_FieldInfo.Resolution.z), sizeof(uInt32));
 		ifs.read((char*)(&m_FieldInfo.FieldSize.x), sizeof(float));
		ifs.read((char*)(&m_FieldInfo.FieldSize.y), sizeof(float));
		ifs.read((char*)(&m_FieldInfo.FieldSize.z), sizeof(float));
 		ifs.read((char*)(&nVoxels), sizeof(uInt64));
		ifs.read((char*)(&dataPrecision), sizeof(uInt32));

		this->Init(m_FieldInfo.Pivot, m_FieldInfo.FieldSize, m_FieldInfo.Resolution);
 		ifs.read((char*)this->GetRawData(), sizeof(T)*nVoxels);

		return true;
	}

	bool Save(std::string path)
	{
		std::ofstream ofs(path.c_str(), std::ios::binary);
		if (!ofs)
			return false;
		AxFieldHeadInfo headInfo;
		headInfo.nFields = 1;
		ofs.write((char*)(&headInfo.nFields), sizeof(uInt32));
 		this->Save(ofs);
		ofs.close();
		return true;
	}

	bool Save(std::ofstream& ofs)
	{
		AlphaUtility::WriteSTLString(ofs, m_sName);

		uInt32 dataPrecision = sizeof(T);
		uInt64 nVoxels = this->GetNumVoxels();

		ofs.write((char*)(&m_FieldInfo.Pivot.x), sizeof(float));
		ofs.write((char*)(&m_FieldInfo.Pivot.y), sizeof(float));
		ofs.write((char*)(&m_FieldInfo.Pivot.z), sizeof(float));

		ofs.write((char*)(&m_FieldInfo.Resolution.x), sizeof(uInt32));
		ofs.write((char*)(&m_FieldInfo.Resolution.y), sizeof(uInt32));
		ofs.write((char*)(&m_FieldInfo.Resolution.z), sizeof(uInt32));

		ofs.write((char*)(&m_FieldInfo.FieldSize.x), sizeof(float));
		ofs.write((char*)(&m_FieldInfo.FieldSize.y), sizeof(float));
		ofs.write((char*)(&m_FieldInfo.FieldSize.z), sizeof(float));

		ofs.write((char*)(&nVoxels), sizeof(uInt64));
		ofs.write((char*)(&dataPrecision), sizeof(uInt32));
		ofs.write((char*)this->GetRawData(),sizeof(T)*nVoxels);
		return true;
	}

#ifdef ALPHA_CUDA

	virtual bool DeviceMalloc(bool loadToDevice = true)
	{
		return m_VoxelBuffer.DeviceMalloc(loadToDevice);
	}

	void LoadToHost()
	{
		m_VoxelBuffer.LoadToHost();
	}

	void LoadToDevice()
	{
		m_VoxelBuffer.LoadToDevice();
	}

	T* GetRawDataDevice()
	{
		return m_VoxelBuffer.GetDevicePtr();
	}

	void DeviceMemCopy(T* deviceFieldRawPtr)
	{
		m_VoxelBuffer.DeviceToDeviceMemcpy(deviceFieldRawPtr);
	}

	void SetToZeroDevice()
	{
		m_VoxelBuffer.SetToZeroDevice();
	}

#endif 

};


typedef AxField3DBase<float>		AxScalarFieldF32;
typedef AxField3DBase<double>		AxScalarFieldF64;
typedef AxField3DBase<uInt32>		AxScalarFieldUInt32;


template<class T>
class AxVectorField3DBase : public AxField
{
public:

	AxVectorField3DBase()
	{

	}

	AxVectorField3DBase(const AlphaCore::Desc::AxField3DInfo& info,std::string name="v")
	{
		FieldX = new AxField3DBase<T>(info, name + std::string(".x"));
		FieldY = new AxField3DBase<T>(info, name + std::string(".y"));
		FieldZ = new AxField3DBase<T>(info, name + std::string(".z"));
		m_sName = name;
	}
	AxVectorField3DBase(AxVector3 pivot, AxVector3 size, AxVector3UI res, std::string name = "v")
	{
		FieldX = new AxField3DBase<T>(pivot, size, res, name + std::string(".x"));
		FieldY = new AxField3DBase<T>(pivot, size, res, name + std::string(".y"));
		FieldZ = new AxField3DBase<T>(pivot, size, res, name + std::string(".z"));
		m_sName = name;
	}

	AxVectorField3DBase(AxField3DBase<T>* x, AxField3DBase<T>* y, AxField3DBase<T>*z)
	{
		FieldX = x;
		FieldY = y;
		FieldZ = z;
	}
	~AxVectorField3DBase()
	{
		FieldX->Release();
		FieldY->Release();
		FieldZ->Release();
	}

	void MultiplyVector3(AxVector3 vec)
	{
		FieldX->MultiplyConstant(vec.x);
		FieldY->MultiplyConstant(vec.y);
		FieldZ->MultiplyConstant(vec.z);
	}
	
	void AddVectorField(AxVectorField3DBase<T>* vecField)
	{
		FieldX->AddField(vecField->FieldX);
		FieldY->AddField(vecField->FieldY);
		FieldZ->AddField(vecField->FieldZ);
	}

	void PrintInfo()
	{
		FieldX->PrintInfo();
		FieldY->PrintInfo();
		FieldZ->PrintInfo();
	}

	void ClearField()
	{
		FieldX->SetToZero();
		FieldY->SetToZero();
		FieldZ->SetToZero();
	} 
	//
	AxField3DBase<T>* FieldX;
	AxField3DBase<T>* FieldY;
	AxField3DBase<T>* FieldZ;

	virtual bool DeviceMalloc(bool loadToDevice = true)
	{
		FieldX->DeviceMalloc(loadToDevice);
		FieldY->DeviceMalloc(loadToDevice);
		FieldZ->DeviceMalloc(loadToDevice);

		//todo : must all malloc succ
		return true;
	}						 

	void TraceData(int start = 0, int end = -1, int sep = 1, const char* head = "")
	{
		FieldX->TraceData(start, end, sep, head);
		FieldY->TraceData(start, end, sep, head);
		FieldZ->TraceData(start, end, sep, head);
	}
};

typedef AxVectorField3DBase<float>	AxVecFieldF32;
typedef AxVectorField3DBase<double>	AxVecFieldF64;
typedef AxVectorField3DBase<uInt32>	AxVecFieldUInt32;



#endif