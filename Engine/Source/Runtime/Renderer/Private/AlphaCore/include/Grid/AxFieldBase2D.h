#ifndef __ALPHA_CORE_GRID_2D_H__
#define __ALPHA_CORE_GRID_2D_H__

#include <Utility/AxDescrition.h>
#include <Math/AxVectorBase.h>
#include <Math/AxVectorHelper.h>

template<class T>
class AxField2DBase
{
public:
	
	AxField2DBase(std::string name = AlphaProperty::DensityField)
	{
		SetFieldResolution(10, 10);
		m_sName = name;
 	}

	AxField2DBase(uInt32 nx, uInt32 ny, std::string name = AlphaProperty::DensityField)
	{
		SetFieldResolution(nx, ny);
		m_sName = name;
	}

	AxField2DBase(const AlphaCore::Desc::AxField2DInfo& info, std::string name = AlphaProperty::DensityField)
	{
		AxVector2 size = MakeVector2(
			(float)info.Resolution.x * info.VoxelSize.x,
			(float)info.Resolution.y * info.VoxelSize.y);
		Init(info.Pivot, size, info.Resolution);
		m_sName = name;
	}

	AxField2DBase(AxField2DBase<T>* field,bool cpyVoxelBuf=false)
	{
 		AxVector2 size = MakeVector2(
			field->GetFieldInfo().Resolution.x*field->GetFieldInfo().VoxelSize.x,
			field->GetFieldInfo().Resolution.y*field->GetFieldInfo().VoxelSize.y);
		Init(field->GetFieldInfo().Pivot, size, field->GetFieldInfo().Resolution);
		if (!cpyVoxelBuf)
			return;
	}

	AxField2DBase(AxVector3 pivot, AxVector2 size, AxVector2UI res, std::string name = AlphaProperty::DensityField)
	{
		Init(pivot,size,res);
		m_sName = name;
	}

	~AxField2DBase()
	{
		m_CellBuffer.Release();
	}

	void Init(AxVector3 pivot, AxVector2 size, AxVector2UI res)
	{
		SetPivot(pivot);
		SetFieldResolution(res);
		SetFieldSize(size);
	}

	void Init(uInt32 nx, uInt32 ny)
	{
		SetPivot(MakeVector3(0,0,0));
		SetFieldResolution(nx,ny);
		SetFieldSize(MakeVector2(nx,ny));
	}

	void Release()
	{
		m_CellBuffer.Release();
	}

	void LoadToHost()
	{
		m_CellBuffer.LoadToHost();
	}

	void DeviceMalloc()
	{
		m_CellBuffer.DeviceMalloc();
	}

	void SetFieldResolution(AxVector2UI res)
	{
		SetFieldResolution(res.x, res.y);
	}

	void SetFieldResolution(uInt32 nx, uInt32 ny)
	{
		if (nx == m_FieldInfo.Resolution.x && m_FieldInfo.Resolution.y == ny)
			return;

		AxVector2 sizeOld = MakeVector2(
			m_FieldInfo.Resolution.x * m_FieldInfo.VoxelSize.x, 
			m_FieldInfo.Resolution.y * m_FieldInfo.VoxelSize.y);
 		m_FieldInfo.Resolution.x = nx;
		m_FieldInfo.Resolution.y = ny;
 		SetFieldSize(sizeOld);
		m_CellBuffer.Resize(nx * ny);
		if (m_CellBuffer.HasDeviceData())
			m_CellBuffer.LoadToDevice();
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

	void SetFieldSize(float sx, float sy)
	{
		AxVector2 size = MakeVector2(sx, sy);
		m_FieldInfo.VoxelSize = size / m_FieldInfo.Resolution;
 	}

	void SetFieldSize(AxVector2 size)
	{
		SetFieldSize(size.x, size.y);
	}

	void SetVoxelSize(AxVector2 size)
	{
		m_FieldInfo.VoxelSize = size;
	}

	uInt64 GetNumVoxels() 
	{ 
		return m_FieldInfo.Resolution.x * m_FieldInfo.Resolution.y;
	}

	AxVector2UI GetResolution() { return m_FieldInfo.Resolution; };

	AlphaCore::Desc::AxField2DInfo GetFieldInfo() { return m_FieldInfo; }

	T* GetRawData() { return m_CellBuffer.GetRawPtr(); };
	T* GetRawDataDevice()
	{
		return m_CellBuffer.GetDevicePtr();
	}

	void PrintNonZeroData(const char* head)
	{
		m_CellBuffer.PrintNonZeroData(head);
	}

	T GetValue(uInt64 voxelId)
	{
		return m_CellBuffer[voxelId];
	}

	void SetValue(uInt64 voxelId, T val)
	{
		m_CellBuffer[voxelId] = val;
	}


	AxStorage<T>& GetVoxelStorageBuffer()
	{
		return m_CellBuffer;
	}

	void SetToZero()
	{
		memset(m_CellBuffer.GetRawPtr(), 0, m_CellBuffer.Size()*sizeof(T));
	}

	void SetName(std::string name)
	{
		m_sName = name;
	}

	std::string GetName()
	{
		return m_sName;
	}

	bool CopyCellDataBuffer(AxField2DBase<T>* field)
	{
		if (field->GetNumVoxels() != this->GetNumVoxels())
			return false;
		memcpy(m_CellBuffer.GetRawPtr(), field->GetRawData(), m_CellBuffer.Size() * sizeof(T));
		return true;
	}

private:

	std::string m_sName;
	AlphaCore::Desc::AxField2DInfo m_FieldInfo;
	AxStorage<T> m_CellBuffer;

};

typedef AxField2DBase<float>		 AxScalarField2DF32;
typedef AxField2DBase<double>		 AxScalarField2DF64;
typedef AxField2DBase<AxColorRGBA8>	 AxImageRGBA8;
typedef AxField2DBase<AxColorRGBA>   AxImageRGBA;


#endif