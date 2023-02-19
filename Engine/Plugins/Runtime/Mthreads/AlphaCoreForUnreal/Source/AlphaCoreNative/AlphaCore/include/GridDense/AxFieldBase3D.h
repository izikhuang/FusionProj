#ifndef __ALPHA_CORE_GRID_3D_H__
#define __ALPHA_CORE_GRID_3D_H__

#include "Utility/AxDescrition.h"
#include "Math/AxVectorBase.h"
#include "Math/AxVectorHelper.h"
#include "AccelTree/AxAccelTree.DataType.h"
#include "GridDense/AxGridDense.DataType.h"
#include <vector>
#include <sstream>
#include <cmath>
#include <cfloat>



#if ALPHA_CUDA
#include "cuda_texture_types.h"
#include "texture_types.h"
#include <surface_types.h>
#include <surface_functions.h>
#endif

struct AxFieldHeadInfo
{
	AxUInt32 nFields;
};
class AxGeometry;
class AxField
{
public:

	virtual AlphaCore::AxDataType GetDataType()
	{
		return AlphaCore::AxDataType::kInvalidDataType;
	}

	AxField() {
		m_iFieldId = -1;
		m_iBindPrimitiveID = -1;
		m_OwnGeometry = nullptr;
	};
	virtual ~AxField() {};

	virtual AxAABB GetAABB()
	{
		return AlphaCore::AccelTree::MakeAABB();
	}

	std::string GetName()
	{
		return m_sName;
	}

	std::string GetName() const
	{
		return m_sName;
	}

	virtual bool Save(std::ofstream& ofs)
	{
		return false;
	}

	virtual bool Read(std::ifstream& ifs)
	{
		return false;
	}

	virtual void Release()
	{
	}

	virtual AxUInt32 GetIOCapacity()
	{
		return 0;
	}

	virtual bool DeviceMalloc(bool loadToDevice = true)
	{
		return false;
	}

	void SetFieldIndex(AxInt32 fieldId)
	{
		m_iFieldId = fieldId;
	};
	AxInt32 GetFieldIndex() { return m_iFieldId; };

	virtual void PrintInfo(bool printVoxelDatas = false, AxUInt32 sep = 128)
	{
	}

	virtual void LoadToDevice()
	{
	}

	virtual bool HasDeviceData()
	{
		return false;
	}

	void SetBindPrimitiveID(AxInt32 primId)
	{
		m_iBindPrimitiveID = primId;
	}

	AxInt32 GetBindPrimitiveID()
	{
		return m_iBindPrimitiveID;
	}

	void SetOwnGeometry(AxGeometry* parentGeo)
	{
		parentGeo = m_OwnGeometry;
	}

	AxGeometry* GetOwnGeometry()
	{
		return m_OwnGeometry;
	}

protected:
	std::string m_sName;
	AxInt32 m_iFieldId;
	AxInt32 m_iBindPrimitiveID;
	AxGeometry* m_OwnGeometry;
};

template <class StorageRawDataT>
struct AxFieldRawDesc
{
	bool IsValid;
	StorageRawDataT* VoxelData;
	// cudaTextureHad
	AxField3DInfo FieldInfo;
};

template <class T>
struct AxFieldVoxelDataHandler
{
#ifdef ALPHA_CUDA
	cudaTextureObject_t CudaTex3D;
#endif
	T* VoxelData;
};

struct AxFieldCacheHeadInfo
{
	AxUInt64 FieldCacheOffset;
	char FieldName[1024];
	AxUInt64 FieldCapacity;
};

#if ALPHA_CUDA
typedef cudaTextureObject_t AxFieldDataHandlerR;
typedef cudaSurfaceObject_t AxFieldDataHandlerRW;
#else
typedef int AxFieldDataHandlerR;
typedef int AxFieldDataHandlerRW;
#endif

struct AxFieldHardwareInterfaceDesc
{
	bool UseAPIStage;
	AlphaCore::AxBackendAPI APIInfo;
	AxFieldDataHandlerR VolumeTexR;
	AxFieldDataHandlerRW VolumeTexRW;
};

inline std::ostream& operator<<(std::ostream& os, const AxFieldHardwareInterfaceDesc& desc)
{
	os << " " << AlphaCore::AxBackendAPIToString(desc.APIInfo) << " | "
		<< (desc.UseAPIStage ? "True" : "False")
		<< " Texture ID : " << desc.VolumeTexR << " - "
		<< " Surface ID : " << desc.VolumeTexRW << "  "
		<< std::endl;
	return os;
}

/*
struct AxField3DRawDesc
{
	bool IsValid;
	void* VoxelData;
	AxField3DInfo FieldInfo;
	AxFieldHardwareInterfaceDesc HardwareInfo;
};
*/

template<class T>
class AxField3DBase : public AxField
{
public:
	struct RAWDesc
	{
		bool IsValid;
		T* VoxelData;
		AxField3DInfo FieldInfo;
		AxFieldHardwareInterfaceDesc HardwareInfo;
	};
	struct RAWDescLOD
	{
		bool IsValid;
		RAWDesc LOD[16];
		AxUInt8 NumLevels;
	};

	RAWDesc GetFiedRAWDesc()
	{
		RAWDesc desc;
		desc.VoxelData = this->GetVoxelStorageBufferPtr()->GetDataRaw();
		desc.FieldInfo = this->GetFieldInfo();
		desc.HardwareInfo = this->GetHardwareInterfaceDesc();
		return desc;
	};

	RAWDesc GetFiedRAWDescDevice()
	{
		RAWDesc desc;
		desc.VoxelData = this->GetVoxelStorageBufferPtr()->GetDataRawDevice();
		desc.FieldInfo = this->GetFieldInfo();
		desc.HardwareInfo = this->GetHardwareInterfaceDesc();
		return desc;
	};

	RAWDescLOD GetRAWDescLOD()
	{
		RAWDescLOD ret;
		ret.IsValid = true;
		ret.LOD[0] = this->GetFiedRAWDesc();
		AX_FOR_I(this->GetNumMultiGridDatas())
		{
			auto mgField = this->GetMultiGridDataByLevelIndex(i);
			ret.LOD[i + 1] = mgField->GetFiedRAWDesc();
		}
		ret.NumLevels = this->GetNumMultiGridDatas() + 1;
		return ret;
	}

	RAWDescLOD GetRAWDescLODDevice()
	{
		RAWDescLOD ret;
		ret.IsValid = true;
		ret.LOD[0] = this->GetFiedRAWDescDevice();
		AX_FOR_I(this->GetNumMultiGridDatas())
		{
			auto mgField = this->GetMultiGridDataByLevelIndex(i);
			ret.LOD[i + 1] = mgField->GetFiedRAWDescDevice();
		}
		ret.NumLevels = this->GetNumMultiGridDatas() + 1;
		return ret;
	}

	static RAWDesc GetRAWDesc(AxField3DBase<T>* field)
	{
		RAWDesc desc;
		desc.IsValid = false;
		desc.VoxelData = nullptr;
		if (field == nullptr)
			return desc;
		return field->GetFiedRAWDesc();
	}

	static RAWDesc GetRAWDescDevice(AxField3DBase<T>* field)
	{
		RAWDesc desc;
		desc.IsValid = false;
		desc.VoxelData = nullptr;
		if (field == nullptr)
			return desc;
		return field->GetFiedRAWDescDevice();
	}

	virtual bool HasDeviceData()
	{
		return this->GetVoxelStorageBufferPtr()->HasDeviceData();
	}

	// AxField3DBase(const AxField3DBase<T>& src)
	//{
	////auto fieldInfo = src.GetFieldInfo();
	////this->Init(fieldInfo.Pivot,fieldInfo.FieldSize,fieldInfo.Resolution);
	////m_VoxelBuffer.SetName(src.GetName() + ".voxels");
	//}

	AxField3DBase(std::string name = AlphaProperty::DensityField)
	{
		this->Init(MakeVector3(0, 0, 0), MakeVector3(10, 10, 10), MakeVector3UI(10, 10, 10), true);
		_initMemberVars();
		m_sName = name;
		m_VoxelBuffer.SetName(m_sName + ".voxels");
	}
	AxField3DBase(AxUInt32 nx, AxUInt32 ny, AxUInt32 nz, std::string name = AlphaProperty::DensityField)
	{
		this->Init(MakeVector3(0, 0, 0), MakeVector3(10, 10, 10), MakeVector3UI(nx, ny, nz), true);
		_initMemberVars();
		m_sName = name;
		m_VoxelBuffer.SetName(m_sName + ".voxels");
	}

	AxField3DBase(const AxField3DInfo& info, std::string name = AlphaProperty::DensityField)
	{
		AxVector3 size = MakeVector3(
			(float)info.Resolution.x * info.VoxelSize.x,
			(float)info.Resolution.y * info.VoxelSize.y,
			(float)info.Resolution.z * info.VoxelSize.z);
		Init(info.Pivot, size, info.Resolution);
		_initMemberVars();
		m_sName = name;
		m_VoxelBuffer.SetName(m_sName + ".voxels");
	}

	AxField3DBase(AxField3DBase<T>* field, bool cpyVoxelBuf = false)
	{
		AxVector3 size = MakeVector3(
			field->GetFieldInfo().Resolution.x * field->GetFieldInfo().VoxelSize.x,
			field->GetFieldInfo().Resolution.y * field->GetFieldInfo().VoxelSize.y,
			field->GetFieldInfo().Resolution.z * field->GetFieldInfo().VoxelSize.z);
		this->Init(field->GetFieldInfo().Pivot, size, field->GetFieldInfo().Resolution);
		_initMemberVars();
		if (!cpyVoxelBuf)
			return;
	}

	AxField3DBase(AxVector3 pivot, AxVector3 size, AxVector3UI res, std::string name = AlphaProperty::DensityField, T* dataRaw = nullptr)
	{
		this->Init(pivot, size, res);
		_initMemberVars();
		m_sName = name;
		m_VoxelBuffer.SetName(m_sName + ".voxels");
		if (dataRaw != nullptr)
			this->ReadRawBuffer(dataRaw);
	}

	~AxField3DBase()
	{
		m_VoxelBuffer.ClearAndDestory();
	}

	void Init(AxVector3 pivot, AxVector3 size, AxVector3UI res, bool setToZero = false)
	{
		this->SetPivot(pivot);
		this->SetFieldResolution(res);
		this->SetFieldSize(size);
		if (setToZero)
			m_VoxelBuffer.SetToZero();
		m_FieldInfo.RotationMatrix = Make3x3Identity();
		m_FieldInfo.InverseRotMatrix = Make3x3Identity();
	}

	void Match(AxField3DBase<T>* src)
	{
		if (src == nullptr)
			return;
		auto info = src->GetFieldInfo();
		this->SetPivot(info.Pivot);
		this->SetFieldResolution(info.Resolution);
		this->SetFieldSize(info.FieldSize);
	}

	void Match(const AxField3DBase<T>& other)
	{
		auto info = other.GetFieldInfo();
		this->SetPivot(info.Pivot);
		this->SetFieldResolution(info.Resolution);
		this->SetFieldSize(info.FieldSize);
	}

	void Match(const AxField3DInfo& otherInfo)
	{
		this->SetPivot(otherInfo.Pivot);
		this->SetFieldResolution(otherInfo.Resolution);
		this->SetFieldSize(otherInfo.FieldSize);
	}

	void SetAllBoundaryExtesion()
	{
		AX_FOR_I(6)
			this->m_FieldInfo.BoundaryInfo[i] = AxGridBoundaryInfo::kExtension;
	}

	void SetAllBoundaryOutsideZero()
	{
		AX_FOR_I(6)
			this->m_FieldInfo.BoundaryInfo[i] = AxGridBoundaryInfo::kOutsideZero;
	}

	void SetAllBoundaryInverse()
	{
		AX_FOR_I(6)
			this->m_FieldInfo.BoundaryInfo[i] = AxGridBoundaryInfo::kInverse;
	}

	void SetBoundaryConditionXPostive(AxGridBoundaryInfo info)
	{
		this->m_FieldInfo.BoundaryInfo[AxGridBoundaryIndex::XPostiveOffset] = info;
	}
	void SetBoundaryConditionXNegtive(AxGridBoundaryInfo info)
	{
		this->m_FieldInfo.BoundaryInfo[AxGridBoundaryIndex::XNegtiveOffset] = info;
	}
	void SetBoundaryConditionYPostive(AxGridBoundaryInfo info)
	{
		this->m_FieldInfo.BoundaryInfo[AxGridBoundaryIndex::YPostiveOffset] = info;
	}
	void SetBoundaryConditionYNegtive(AxGridBoundaryInfo info)
	{
		this->m_FieldInfo.BoundaryInfo[AxGridBoundaryIndex::YNegtiveOffset] = info;
	}
	void SetBoundaryConditionZPostive(AxGridBoundaryInfo info)
	{
		this->m_FieldInfo.BoundaryInfo[AxGridBoundaryIndex::ZPostiveOffset] = info;
	}
	void SetBoundaryConditionZNegtive(AxGridBoundaryInfo info)
	{
		this->m_FieldInfo.BoundaryInfo[AxGridBoundaryIndex::ZNegtiveOffset] = info;
	}

	void CopyBoundaryConditionFromFieldInfo(const AxField3DInfo& info)
	{
		AX_FOR_I(6)
			this->m_FieldInfo.BoundaryInfo[i] = info.BoundaryInfo[i];
	}


	virtual AlphaCore::AxDataType GetDataType()
	{
		return AlphaCore::TypeID<T>();
	}


	AxUInt32 GetNX()
	{
		return m_FieldInfo.Resolution.x;
	}

	AxUInt32 GetNY()
	{
		return m_FieldInfo.Resolution.y;
	}

	AxUInt32 GetNZ()
	{
		return m_FieldInfo.Resolution.z;
	}

	int BuildMultiGridData(int maxLevel = -1, AxUInt32 minRes = 4)
	{
		if (maxLevel < 0)
			maxLevel = 6;
		if (maxLevel == 0)
			return 0;
		maxLevel = std::min(maxLevel, (int)this->_evalLODFieldTitleSize());
		AxUInt32 allLevels = maxLevel + 1; // add self
		if (m_iNumMultiGridDatas != 0)
			m_iNumMultiGridDatas = 0;
		AxInt32 currLevel = allLevels;
		while (currLevel >= 1)
		{
			int levelIndex = allLevels - currLevel;
			AxField3DInfo info;
			if (!_evalLODFieldInfo(maxLevel, levelIndex, info, minRes))
				break;
			std::string currName = this->GetName() + "__LOD" + std::to_string(levelIndex);
			AxField3DBase<T>* currMGField = this->createMultiGridData(levelIndex);
			if (levelIndex != 0)
				currMGField->SetName(currName);
			currMGField->Match(info);
			currLevel--;
		}
		m_iNumMultiGridDatas = allLevels - 1 - currLevel;
		return m_iNumMultiGridDatas;
	}

	virtual AxAABB GetAABB()
	{
		AxAABB box;
		box.Min = this->m_FieldInfo.Pivot - this->m_FieldInfo.FieldSize * 0.5f;
		box.Max = this->m_FieldInfo.Pivot + this->m_FieldInfo.FieldSize * 0.5f;
		return box;
	}

	virtual void Release()
	{
		m_VoxelBuffer.ClearAndDestory();
		AxUInt32 nMG = this->GetNumMultiGridDatas();
		AX_FOR_I(int(nMG))
			m_MultiGridData[i]->Release();
	}

	void SetFieldResolution(AxVector3UI res)
	{
		SetFieldResolution(res.x, res.y, res.z);
	}

	void SetFieldResolution(AxUInt32 nx, AxUInt32 ny, AxUInt32 nz)
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

	void SetPivot(float tx, float ty, float tz)
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

	void SetRotationMatrix(AxVector3 eulerAngles)
	{
		m_FieldInfo.RotationMatrix = EulerToMatrix3x3_XYZ(eulerAngles);
		m_FieldInfo.InverseRotMatrix = Transposed(m_FieldInfo.RotationMatrix);
	}

	void SetRotationMatrix(float rx, float ry, float rz)
	{
		SetRotationMatrix(MakeVector3(rx, ry, rz));
	}

	AxUInt32 GetNumVoxels()
	{
		return m_FieldInfo.Resolution.x * m_FieldInfo.Resolution.y * m_FieldInfo.Resolution.z;
	}

	AxUInt32 GetSliceVoxels_XY()
	{
		return m_FieldInfo.Resolution.x * m_FieldInfo.Resolution.y;
	}

	AxUInt32 GetSliceVoxels_YZ()
	{
		return m_FieldInfo.Resolution.y * m_FieldInfo.Resolution.z;
	}

	AxUInt32 GetSliceVoxels_XZ()
	{
		return m_FieldInfo.Resolution.x * m_FieldInfo.Resolution.z;
	}

	AxVector3UI GetResolution() { return m_FieldInfo.Resolution; };

	AxField3DInfo GetFieldInfo()
	{
		return this->m_FieldInfo;
	}

	AxField3DInfo GetFieldInfo() const
	{
		return this->m_FieldInfo;
	}
	// AxField3DInfo GetFieldInfo() const { return m_FieldInfo; }

	T* GetRawData()
	{
		return m_VoxelBuffer.Data();
	};

	T* GetRawData() const
	{
		return m_VoxelBuffer.Data();
	};

	void ReadRawBuffer(T* data)
	{
		std::memcpy(this->m_VoxelBuffer.m_Data.data(), data, this->GetNumVoxels() * sizeof(T));
	}

	virtual void PrintInfo(bool printVoxelDatas = false, AxUInt32 sep = 128) override
	{
		std::cout << " ------------ Field [ " << m_sName << " ]----------------" << std::endl;
		std::cout << "Pivot:" << m_FieldInfo.Pivot.x << " , " << m_FieldInfo.Pivot.y << " , " << m_FieldInfo.Pivot.z << std::endl;

		std::cout << "Res : [" << m_FieldInfo.Resolution.x << " , " << m_FieldInfo.Resolution.y << " , " << m_FieldInfo.Resolution.z << " ] " << std::endl;

		std::cout << "voxelSize : [" << m_FieldInfo.VoxelSize.x << " , " << m_FieldInfo.VoxelSize.y << " , " << m_FieldInfo.VoxelSize.z << " ] " << std::endl;

		std::cout << "Field Boundary " << std::endl;
		std::cout << " [ + X ] " << AxField3DInfoToString(m_FieldInfo.BoundaryInfo[0]) << "   ";
		std::cout << " [ - X ] " << AxField3DInfoToString(m_FieldInfo.BoundaryInfo[1]) << std::endl;
		std::cout << " [ + Y ] " << AxField3DInfoToString(m_FieldInfo.BoundaryInfo[2]) << "   ";
		std::cout << " [ - Y ] " << AxField3DInfoToString(m_FieldInfo.BoundaryInfo[3]) << std::endl;
		std::cout << " [ + Z ] " << AxField3DInfoToString(m_FieldInfo.BoundaryInfo[4]) << "   ";
		std::cout << " [ - Z ] " << AxField3DInfoToString(m_FieldInfo.BoundaryInfo[5]) << std::endl;

		std::cout << "BindPrimitiveID : " << this->m_iBindPrimitiveID << std::endl;
		if (printVoxelDatas)
		{
			for (int i = 0; i < m_VoxelBuffer.Size(); i += 10)
			{
				std::cout << m_VoxelBuffer[i] << ",";
				if (i % sep == 0)
					std::cout << "\n";
			}
		}

		AX_FOR_I(int(this->m_iNumMultiGridDatas))
			this->m_MultiGridData[i]->PrintInfo(printVoxelDatas, sep);
	}

	void SetName(std::string name)
	{
		m_sName = name;
		m_VoxelBuffer.SetName(m_sName + ".voxels");
	}

	T GetValue(uInt64 voxelId)
	{
		return m_VoxelBuffer[voxelId];
	}

	T GetValue(AxUInt32 idx, AxUInt32 idy, AxUInt32 idz)
	{
		AxUInt32 voxelId = idz * m_FieldInfo.Resolution.x * m_FieldInfo.Resolution.y + idy * m_FieldInfo.Resolution.x + idx;
		return m_VoxelBuffer[voxelId];
	}

	void SetValue(uInt64 voxelId, T val)
	{
		m_VoxelBuffer[voxelId] = val;
	}

	void SetValue(AxUInt32 idx, AxUInt32 idy, AxUInt32 idz, T val)
	{
		AxUInt32 voxelId = idz * m_FieldInfo.Resolution.x * m_FieldInfo.Resolution.y + idy * m_FieldInfo.Resolution.x + idx;
		m_VoxelBuffer[voxelId] = val;
	}

	T GetValueMemory(AxInt32 index)
	{
		return m_VoxelBuffer[index];
	}

	T GetValueMemory(AxUInt32 idx, AxUInt32 idy, AxUInt32 idz)
	{
		AxVector3I index3 = MakeVector3I(idx, idy, idz);
		int voxelID = Index3ToMemoryIndex(index3);
		return m_VoxelBuffer[voxelID];
	}

	T GetValueMemoryBit16(AxUInt32 idx, AxUInt32 idy, AxUInt32 idz)
	{
		AxVector3I index3 = MakeVector3I(idx, idy, idz);
		int voxelID = Index3ToMemoryIndexBit16(index3);
		return m_VoxelBuffer[voxelID];
	}

	void SetValueMemory(AxUInt32 voxelId, T val)
	{
		m_VoxelBuffer[voxelId] = val;
		return;
	}

	void SetValueMemory(AxUInt32 idx, AxUInt32 idy, AxUInt32 idz, T val)
	{
		AxVector3I index3 = MakeVector3I(idx, idy, idz);
		int voxelID = Index3ToMemoryIndex(index3);
		m_VoxelBuffer[voxelID] = val;
		return;
	}

	void SetValueMemoryBit16(AxUInt32 idx, AxUInt32 idy, AxUInt32 idz, T val)
	{
		AxVector3I index3 = MakeVector3I(idx, idy, idz);
		int voxelID = Index3ToMemoryIndexBit16(index3);
		m_VoxelBuffer[voxelID] = val;
		return;
	}

	AxVector3I MemoryIndexToIndex3(AxInt32 index)
	{
		auto res = m_FieldInfo.Resolution;
		int intactX = res.x / 16;
		int intactY = res.y / 16;
		int intactZ = res.z / 16;

		int remainX = res.x % 16;
		int remainY = res.y % 16;
		int remainZ = res.z % 16;

		AxVector3I localRes, localID, localIndex3;

		int localIndex = index;

		localID.z = index / (res.x * res.y * 16);
		localRes.z = localID.z < intactZ ? 16 : remainZ;
		localIndex -= localID.z * res.x * res.y * 16;

		localID.y = localIndex / (res.x * 16 * localRes.z);
		localRes.y = localID.y < intactY ? 16 : remainY;
		localIndex -= localID.y * res.x * 16 * localRes.z;

		localID.x = localIndex / (16 * localRes.y * localRes.z);
		localRes.x = localID.x < intactX ? 16 : remainX;
		localIndex -= localID.x * 16 * localRes.y * localRes.z;

		int nvSlice = localRes.x * localRes.y;
		localIndex3.x = localIndex % localRes.x;
		localIndex3.y = (localIndex % nvSlice) / localRes.x;
		localIndex3.z = localIndex / nvSlice;

		AxVector3I index3;
		index3.x = localID.x * 16 + localIndex3.x;
		index3.y = localID.y * 16 + localIndex3.y;
		index3.z = localID.z * 16 + localIndex3.z;

		return index3;
	}

	AxInt32 Index3ToMemoryIndex(AxVector3I Index3)
	{
		auto res = m_FieldInfo.Resolution;

		int intactX = res.x / 16;
		int intactY = res.y / 16;
		int intactZ = res.z / 16;

		int remainX = res.x % 16;
		int remainY = res.y % 16;
		int remainZ = res.z % 16;

		AxVector3I LocalID;
		LocalID.x = Index3.x / 16;
		LocalID.y = Index3.y / 16;
		LocalID.z = Index3.z / 16;

		AxVector3I LocalRes;
		LocalRes.x = LocalID.x < intactX ? 16 : remainX;
		LocalRes.y = LocalID.y < intactY ? 16 : remainY;
		LocalRes.z = LocalID.z < intactZ ? 16 : remainZ;

		int localStartIndex = res.x * res.y * LocalID.z * 16 +
			res.x * LocalID.y * 16 * LocalRes.z +
			LocalRes.y * LocalID.x * 16 * LocalRes.z;
		AxVector3I LocalIndex3;
		LocalIndex3.x = Index3.x - LocalID.x * 16;
		LocalIndex3.y = Index3.y - LocalID.y * 16;
		LocalIndex3.z = Index3.z - LocalID.z * 16;

		int memoryIndex = localStartIndex
			+ LocalIndex3.z * LocalRes.x * LocalRes.y
			+ LocalIndex3.y * LocalRes.x
			+ LocalIndex3.x;

		return memoryIndex;
	}

	AxInt32 Index3ToMemoryIndexBit16(AxVector3I Index3)
	{
		auto res = m_FieldInfo.Resolution;

		int intactX = res.x >> 4;
		int intactY = res.y >> 4;
		int intactZ = res.z >> 4;


		AxVector3I BlockID;
		BlockID.x = Index3.x >> 4;
		BlockID.y = Index3.y >> 4;
		BlockID.z = Index3.z >> 4;

		AxVector3I LocalIndex3;
		LocalIndex3.x = 0x0000000f & Index3.x;
		LocalIndex3.y = 0x0000000f & Index3.y;
		LocalIndex3.z = 0x0000000f & Index3.z;

		int localStartIndex = (BlockID.z * intactX * intactY + BlockID.y * intactX + BlockID.x) << 12;

		int memoryIndex = localStartIndex
			+ (LocalIndex3.z << 8)
			+ (LocalIndex3.y << 4)
			+ LocalIndex3.x;

		return memoryIndex;
	}

	AxVector3I MemoryIndexToIndex3Bit16(AxUInt32 index)
	{
		auto res = m_FieldInfo.Resolution;

		int intactX = res.x >> 4;
		int intactY = res.y >> 4;
		int intactZ = res.z >> 4;

		int blockID = index >> 12;

		AxVector3I blockID3;
		AxUInt32 nvSlice = intactX * intactY;
		blockID3.x = blockID % intactX;
		blockID3.y = (blockID % nvSlice) / intactX;
		blockID3.z = blockID / nvSlice;

		int localId = index - (blockID << 12);

		AxVector3I localID3;
		localID3.x = localId & 0x0000000f;
		localID3.y = (localId & 0x000000ff) >> 4;
		localID3.z = localId >> 8;

		AxVector3I index3;
		index3.x = (blockID3.x << 4) + localID3.x;
		index3.y = (blockID3.y << 4) + localID3.y;
		index3.z = (blockID3.z << 4) + localID3.z;
		return index3;
	}




	void AddField(AxField3DBase<T>* field)
	{
		for (uInt64 i = 0; i < field->GetNumVoxels(); ++i)
			this->SetValue(i, this->GetValue(i) + field->GetValue(i));
	}

	void MultiplyConstant(T constant)
	{
		for (uInt64 i = 0; i < this->GetNumVoxels(); ++i)
			this->SetValue(i, this->GetValue(i) * constant);
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

	AxStorage<T>* GetVoxelStorageBufferPtr()
	{
		return &m_VoxelBuffer;
	}

	void SetToZero()
	{
		memset(m_VoxelBuffer.Data(), 0, m_VoxelBuffer.Size() * sizeof(T));
	}

	bool CopyVoxelDataBuffer(AxField3DBase<T>* field)
	{
		if (field->GetNumVoxels() != this->GetNumVoxels())
			return false;
		std::cout << "COPY : " << this->m_sName.c_str() << " ---- " << field->GetName() << std::endl;
		memcpy(m_VoxelBuffer.Data(), field->GetRawData(), m_VoxelBuffer.Size() * sizeof(T));
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

	void TraceData(int start = 0, int end = -1, int sep = 1, const char* head = "")
	{
		return;
#ifdef ALPHA_CUDA
		if (m_VoxelBuffer.HasDeviceData())
			this->LoadToHost();
#endif
		end = end < 0 ? m_VoxelBuffer.Size() : end;
		std::stringstream sstr;
		sstr << "[";
		for (size_t i = start; i < end; i += 1)
		{
			if (std::isnan(m_VoxelBuffer[i]))
				continue;
			sstr << m_VoxelBuffer[i];
			if (i != end - 1)
				sstr << ",";
		}
		sstr << "]";
		AX_TRACE("<{}> Field {} {}", head, m_sName.c_str(), sstr.str().c_str());
	}

	AxUInt32 GetNumMultiGridDatas()
	{
		return m_iNumMultiGridDatas;
	}

	void ClearMultiGridDatas()
	{
		m_iNumMultiGridDatas = 0;
	}

	AxField3DBase<T>* GetMultiGridDataByLevelIndex(AxUInt32 levelIdx)
	{
		if (levelIdx < this->m_iNumMultiGridDatas)
			return m_MultiGridData[levelIdx];
		return nullptr;
	}

private:
	AxUInt32 _evalLODFieldTitleSize()
	{
		AxUInt32 maxLevel = 0;
		AxVector3UI res = m_FieldInfo.Resolution;
		while (true)
		{
			AxUInt32 titleResCurr = 2 << maxLevel;
			AxUInt32 nx = (AxUInt32)floor((float)res.x / (float)titleResCurr);
			AxUInt32 ny = (AxUInt32)floor((float)res.y / (float)titleResCurr);
			AxUInt32 nz = (AxUInt32)floor((float)res.z / (float)titleResCurr);
			if (nx == 0 || ny == 0 || nz == 0)
				break;
			std::cout << "titleResCurr:" << titleResCurr << std::endl;
			maxLevel++;
		}
		return maxLevel;
	}

	bool _evalLODFieldInfo(int maxLevel, int curr, AxField3DInfo& dst, AxUInt32 minRes)
	{
		int titleResCurr = 1 << curr;
		int titleRes = 1 << maxLevel;
		if (curr > maxLevel)
			return false;
		AxVector3 vs = m_FieldInfo.VoxelSize;
		AxVector3UI res = m_FieldInfo.Resolution;

		// if (res.x <= titleRes || res.y <= titleRes || res.z <= titleRes)
		//	return false;

		AxUInt32 nx = (AxUInt32)floor((float)res.x / (float)titleRes + 0.5f);
		AxUInt32 ny = (AxUInt32)floor((float)res.y / (float)titleRes + 0.5f);
		AxUInt32 nz = (AxUInt32)floor((float)res.z / (float)titleRes + 0.5f);
		AxVector3UI newRes = MakeVector3UI(nx, ny, nz) * titleRes;
		newRes.x /= titleResCurr;
		newRes.y /= titleResCurr;
		newRes.z /= titleResCurr;

		if (newRes.x < minRes || newRes.y < minRes || newRes.z < minRes)
			return false;

		dst = m_FieldInfo;
		dst.Resolution = newRes;
		dst.VoxelSize *= titleResCurr;
		dst.FieldSize = newRes * dst.VoxelSize;
		std::cout << "newRes:" << dst << std::endl;

		return true;
	}

	AxField3DBase<T>* createMultiGridData(AxUInt32 MGIndex)
	{
		if (MGIndex == 0)
			return this;
		if (m_MultiGridData.size() < MGIndex - 1)
			return m_MultiGridData[MGIndex - 1];
		if (m_OwnGeometry != nullptr)
		{
			// do something
		}
		auto field = new AxField3DBase<T>();
		m_MultiGridData.push_back(field);
		return field;
	}

	void _initMemberVars()
	{
#ifdef ALPHA_CUDA
		m_CUDATexContent = nullptr;
#endif 

		m_bAPIStage = false;
		m_iNumMultiGridDatas = 0;
		std::memset(&m_TexHardwareInterfaceDesc, 0, sizeof(AxFieldHardwareInterfaceDesc));
		this->SetAllBoundaryOutsideZero();
	}

	AxField3DInfo m_FieldInfo;
	AxUInt32 m_iNumMultiGridDatas;

	AxStorage<T> m_VoxelBuffer;
	std::vector<AxField3DBase<T>*> m_MultiGridData;

public:
	bool Read(std::string path)
	{
		std::ifstream ifs(path.c_str(), std::ios::binary);
		if (!ifs)
			return false;
		AxUInt32 nFields;
		ifs.read((char*)(&nFields), sizeof(AxUInt32));
		this->Read(ifs);
		ifs.close();
		return true;
	}

	virtual bool Read(std::ifstream& ifs)
	{
		POS_READ_INFO("m_sName", ifs);
		AlphaUtility::ReadSTLString(ifs, m_sName);
		POS_READ_INFO(m_sName.c_str(), ifs);

		m_VoxelBuffer.SetName(m_sName + ".voxels");

		AxUInt32 dataPrecision = sizeof(T);
		uInt64 nVoxels = 0;

		ifs.read((char*)(&m_iFieldId), sizeof(AxInt32));

		ifs.read((char*)(&m_FieldInfo.Pivot.x), sizeof(float));
		ifs.read((char*)(&m_FieldInfo.Pivot.y), sizeof(float));
		ifs.read((char*)(&m_FieldInfo.Pivot.z), sizeof(float));
		ifs.read((char*)(&m_FieldInfo.Resolution.x), sizeof(AxUInt32));
		ifs.read((char*)(&m_FieldInfo.Resolution.y), sizeof(AxUInt32));
		ifs.read((char*)(&m_FieldInfo.Resolution.z), sizeof(AxUInt32));
		ifs.read((char*)(&m_FieldInfo.FieldSize.x), sizeof(float));
		ifs.read((char*)(&m_FieldInfo.FieldSize.y), sizeof(float));
		ifs.read((char*)(&m_FieldInfo.FieldSize.z), sizeof(float));

		ifs.read((char*)(m_FieldInfo.BoundaryInfo), sizeof(AxGridBoundaryInfo) * 6);
		ifs.read((char*)(&m_iBindPrimitiveID), sizeof(AxUInt32));

		ifs.read((char*)(&nVoxels), sizeof(uInt64));
		ifs.read((char*)(&dataPrecision), sizeof(AxUInt32));

		this->Init(m_FieldInfo.Pivot, m_FieldInfo.FieldSize, m_FieldInfo.Resolution);

		POS_READ_INFO("VOXEL_BUFFER_DATA", ifs);
		ifs.read((char*)(m_VoxelBuffer.Data()), sizeof(T) * nVoxels);

		return true;
	}

	bool Save(std::string path)
	{
		std::ofstream ofs(path.c_str(), std::ios::binary);
		if (!ofs)
			return false;
		AxUInt32 nFields = 1;
		ofs.write((char*)(&nFields), sizeof(AxUInt32));
		this->Save(ofs);
		ofs.close();
		return true;
	}

	virtual bool Save(std::ofstream& ofs)
	{
		POS_WRITE_INFO("m_sName", ofs);
		AlphaUtility::WriteSTLString(ofs, m_sName);
		POS_WRITE_INFO(m_sName.c_str(), ofs);

		AxUInt32 dataPrecision = sizeof(T);
		uInt64 nVoxels = this->GetNumVoxels();

		ofs.write((char*)(&m_iFieldId), sizeof(AxInt32));

		ofs.write((char*)(&m_FieldInfo.Pivot.x), sizeof(float));
		ofs.write((char*)(&m_FieldInfo.Pivot.y), sizeof(float));
		ofs.write((char*)(&m_FieldInfo.Pivot.z), sizeof(float));

		ofs.write((char*)(&m_FieldInfo.Resolution.x), sizeof(AxUInt32));
		ofs.write((char*)(&m_FieldInfo.Resolution.y), sizeof(AxUInt32));
		ofs.write((char*)(&m_FieldInfo.Resolution.z), sizeof(AxUInt32));

		ofs.write((char*)(&m_FieldInfo.FieldSize.x), sizeof(float));
		ofs.write((char*)(&m_FieldInfo.FieldSize.y), sizeof(float));
		ofs.write((char*)(&m_FieldInfo.FieldSize.z), sizeof(float));

		ofs.write((char*)(m_FieldInfo.BoundaryInfo), sizeof(AxGridBoundaryInfo) * 6);
		ofs.write((char*)(&m_iBindPrimitiveID), sizeof(AxUInt32));

		ofs.write((char*)(&nVoxels), sizeof(uInt64));
		ofs.write((char*)(&dataPrecision), sizeof(AxUInt32));

		POS_WRITE_INFO("VOXEL_BUFFER_DATA", ofs);
		ofs.write((char*)(m_VoxelBuffer.Data()), sizeof(T) * nVoxels);

		return true;
	}

	virtual AxUInt32 GetIOCapacity()
	{
		return m_sName.size() + 1 +
			sizeof(AxInt32) +
			sizeof(AxInt32) +
			sizeof(AxFp32) * 6 +
			sizeof(AxUInt32) * 5 +
			sizeof(AxGridBoundaryInfo) * 6 +
			sizeof(AxUInt64) +
			sizeof(T) * this->GetNumVoxels();
	}


	virtual bool DeviceMalloc(bool loadToDevice = true)
	{
		bool ret = true;
		AX_FOR_I(int(this->GetNumMultiGridDatas()))
			ret = ret && this->GetMultiGridDataByLevelIndex(i)->DeviceMalloc(loadToDevice);
		return ret && m_VoxelBuffer.DeviceMalloc(loadToDevice);
	}

	void LoadToHost()
	{
		AX_FOR_I(int(this->GetNumMultiGridDatas()))
			this->GetMultiGridDataByLevelIndex(i)->LoadToHost();
		m_VoxelBuffer.LoadToHost();
	}

	virtual void LoadToDevice()
	{
		AX_FOR_I(int(this->GetNumMultiGridDatas()))
			this->GetMultiGridDataByLevelIndex(i)->LoadToDevice();
		m_VoxelBuffer.LoadToDevice();
	}

	void LoadToDeviceTexture()
	{
#ifdef ALPHA_CUDA
		_createCUDATexture();
		AX_FOR_I(int(this->GetNumMultiGridDatas()))
			this->GetMultiGridDataByLevelIndex(i)->_createCUDATexture();
#endif
	}

	AxFieldHardwareInterfaceDesc m_TexHardwareInterfaceDesc;

#ifdef ALPHA_CUDA
	void LoadToHostTexture()
	{
		if (this->m_CUDATexContent == nullptr)
			return;
		cudaExtent dataSize;
		dataSize.width = m_FieldInfo.Resolution.x;
		dataSize.height = m_FieldInfo.Resolution.y;
		dataSize.depth = m_FieldInfo.Resolution.z;
		cudaMemcpy3DParms copyParams = { 0 };
		/*
		copyParams.srcArray = make_cudaPitchedPtr(
			m_CUDATexContent,
			dataSize.width * sizeof(T),
			dataSize.width,
			dataSize.height);
			*/
		copyParams.srcArray = m_CUDATexContent;
		copyParams.dstPtr = make_cudaPitchedPtr((void*)this->GetRawData(), dataSize.width * sizeof(T), dataSize.width, dataSize.height);
		copyParams.extent = dataSize;
		copyParams.kind = cudaMemcpyDeviceToHost;
		cudaMemcpy3D(&copyParams);
		AX_GET_DEVICE_LAST_ERROR;
		AX_FOR_I(this->GetNumMultiGridDatas())
			this->GetMultiGridDataByLevelIndex(i)->LoadToHostTexture();
	}

	T* GetRawDataDevice()
	{
		return m_VoxelBuffer.GetDevicePtr();
	}

	void DeviceMemCopy(AxField3DBase<T>* deviceFieldRawPtr)
	{
		m_VoxelBuffer.DeviceToDeviceMemcpy(deviceFieldRawPtr->GetVoxelStorageBufferPtr());
	}
	// fix ret is not defined chenhao
	void SetToZeroDevice(bool includeLOD = true)
	{
		AX_FOR_I(this->GetNumMultiGridDatas())
			this->GetMultiGridDataByLevelIndex(i)->SetToZeroDevice();
		m_VoxelBuffer.SetToZeroDevice();
	}

	void CopyToDeviceTexture()
	{
		if (m_CUDATexContent == nullptr || this->HasDeviceData() == false)
		{
			AX_ERROR("Non texture object create it !!!");
			return;
		}
		cudaExtent dataSize;
		dataSize.width = m_FieldInfo.Resolution.x;
		dataSize.height = m_FieldInfo.Resolution.y;
		dataSize.depth = m_FieldInfo.Resolution.z;
		// Memcpy 3D Array
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr(
			this->GetRawDataDevice(),
			dataSize.width * sizeof(T),
			dataSize.width,
			dataSize.height);
		copyParams.dstArray = m_CUDATexContent;
		copyParams.extent = dataSize;
		copyParams.kind = cudaMemcpyDeviceToDevice;
		cudaMemcpy3D(&copyParams);
		AX_GET_DEVICE_LAST_ERROR;
	}

	void ClearCUDATexture();
	AxFieldHardwareInterfaceDesc GetHardwareInterfaceDesc()
	{
		return m_TexHardwareInterfaceDesc;
	}

private:
	cudaArray* m_CUDATexContent;
	void _createCUDATexture(int originDataFlag = 0)
	{
		if (m_CUDATexContent != nullptr)
			return;
		// create 3D array
		AxVector3UI res = m_FieldInfo.Resolution;
		cudaExtent dataSize;
		dataSize.width = res.x;
		dataSize.height = res.y;
		dataSize.depth = res.z;

		auto channelDesc = cudaCreateChannelDesc<T>();
		cudaMalloc3DArray(&m_CUDATexContent, &channelDesc, dataSize, cudaArraySurfaceLoadStore);
		AX_INFO("Malloc device texture memory succ use | {:03.2f} MB | {}",
			(AxFp32)(res.x * res.y * res.z * sizeof(T)) / 1024.0f / 1024.0f,
			this->GetVoxelStorageBufferPtr()->GetName());

		auto copyType = originDataFlag ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
		auto srcPtr = originDataFlag ? this->GetRawDataDevice() : this->GetRawData();
		//Texture : TAG 硬件 Layout 转换  GPU 可以 Copy 出来
		//Layout 转换 DX11 Usage D3D11_TEXTURE2D_DESC 
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr(
			srcPtr,
			dataSize.width * sizeof(T),
			dataSize.width,
			dataSize.height);
		copyParams.dstArray = m_CUDATexContent;
		copyParams.extent = dataSize;
		copyParams.kind = copyType;
		cudaMemcpy3D(&copyParams);
		AX_GET_DEVICE_LAST_ERROR;

		// Create SurfaceObject
		cudaResourceDesc surfRes;
		std::memset(&surfRes, 0, sizeof(cudaResourceDesc));
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = m_CUDATexContent;
		cudaCreateSurfaceObject(&m_TexHardwareInterfaceDesc.VolumeTexRW, &surfRes);

		// other setup
		cudaResourceDesc texRes;
		std::memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = m_CUDATexContent;

		cudaTextureDesc texDescr;
		std::memset(&texDescr, 0, sizeof(cudaTextureDesc));

		texDescr.normalizedCoords = true;
		texDescr.filterMode = cudaFilterModeLinear;
		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;
		// texDescr.readMode = cudaReadModeNormalizedFloat; //VolumeTypeInfo<VolumeType>::readMode;

		texDescr.readMode = cudaReadModeElementType;
		cudaCreateTextureObject(&m_TexHardwareInterfaceDesc.VolumeTexR, &texRes, &texDescr, NULL);
 
		this->m_TexHardwareInterfaceDesc.APIInfo = AlphaCore::AxBackendAPI::CUDA;
		this->m_TexHardwareInterfaceDesc.UseAPIStage = true;
		AX_GET_DEVICE_LAST_ERROR;
	}

#endif

	bool m_bAPIStage;
	/*
			bool IsValid;
			RAWDesc LOD[16];
			AxUInt8 NumLevels;
	*/
};

inline std::ostream& operator<<(std::ostream& os, const AxFieldCacheHeadInfo& fieldCacheInfo)
{
	os << " AxFieldCacheHeadInfo :: FieldName : " << fieldCacheInfo.FieldName << "  CacheOffset : " << fieldCacheInfo.FieldCacheOffset;
	return os;
}


typedef AxField3DBase<AxInt8>		AxScalarFieldI8;
typedef AxField3DBase<AxInt16>		AxScalarFieldI16;
typedef AxField3DBase<AxInt32>		AxScalarFieldI32;
typedef AxField3DBase<AxFp16>		AxScalarFieldF16;
typedef AxField3DBase<float>		AxScalarFieldF32;
typedef AxField3DBase<double>		AxScalarFieldF64;
typedef AxField3DBase<AxUInt32>		AxScalarFieldUInt32;
typedef AxField3DBase<AxUInt8>		AxScalarFieldUInt8;
typedef AxField3DBase<AxVector3>	AxFieldVector3F32;
//to check 李俊峰
typedef AxField3DBase<AxVector3I>	AxFieldVector3I;

inline void PrintInfo(const char* head, const AxScalarFieldF32::RAWDesc& rawDesc)
{
	std::cout << head << "AsRaw@FieldRawDesc : " << rawDesc.FieldInfo
		<< " | RawData PTR : " << rawDesc.VoxelData
		<< rawDesc.HardwareInfo << std::endl;
}

inline void PrintInfo(const AxScalarFieldF32::RAWDescLOD& lod)
{
	std::cout << "AsRaw@FieldRawDescLOD NumLOD:" << (AxUInt32)lod.NumLevels << std::endl;
	AX_FOR_I(lod.NumLevels)
		PrintInfo("   + ", lod.LOD[i]);
}

template <class T>
class AxVectorField3DBase : public AxField
{
public:
	struct RAWDesc
	{
		bool Active;
		T* VoxelDataX;
		T* VoxelDataY;
		T* VoxelDataZ;
		AxField3DInfo FieldInfoX;
		AxField3DInfo FieldInfoY;
		AxField3DInfo FieldInfoZ;
		bool IsStaggeredGrid;
	};

	static RAWDesc GetRAWDesc(AxVectorField3DBase<T>* field)
	{
		RAWDesc desc;
		desc.Active = false;
		desc.VoxelDataX = nullptr;
		desc.VoxelDataY = nullptr;
		desc.VoxelDataZ = nullptr;

		if (field == nullptr)
			return desc;
		return field->GetFiedRAWDesc();
	}

	static RAWDesc GetRAWDescDevice(AxVectorField3DBase<T>* field)
	{
		RAWDesc desc;
		desc.Active = false;
		desc.VoxelDataX = nullptr;
		desc.VoxelDataY = nullptr;
		desc.VoxelDataZ = nullptr;
		if (field == nullptr)
			return desc;
		return field->GetFiedRAWDescDevice();
	}

	void Init()
	{
		AxVector3 pivot = MakeVector3();
		AxVector3 size = MakeVector3(1.0f, 1.0f, 1.0);
		AxVector3UI res = MakeVector3UI(3, 3, 3);
		std::string name = "v";
		if (this->FieldX == nullptr)
			this->FieldX = new AxField3DBase<T>(pivot, size, res, name + std::string(".x"));
		if (this->FieldY == nullptr)
			this->FieldY = new AxField3DBase<T>(pivot, size, res, name + std::string(".y"));
		if (this->FieldZ == nullptr)
			this->FieldZ = new AxField3DBase<T>(pivot, size, res, name + std::string(".z"));
		m_sName = name;
	}

	bool IsStaggeredGrid()
	{
		if (!this->IsValid())
			return false;
		AxVector3UI resX = this->FieldX->GetResolution();
		AxVector3UI resY = this->FieldY->GetResolution();
		AxVector3UI resZ = this->FieldZ->GetResolution();
		resX -= MakeVector3UI(1, 0, 0);
		resY -= MakeVector3UI(0, 1, 0);
		resZ -= MakeVector3UI(0, 0, 1);
		if (IsEqual(resX, resY) && IsEqual(resX, resZ))
			return true;
		return false;
	}
	RAWDesc GetFiedRAWDesc()
	{
		RAWDesc desc;
		desc.VoxelDataX = this->FieldX->GetRawData();
		desc.VoxelDataY = this->FieldY->GetRawData();
		desc.VoxelDataZ = this->FieldZ->GetRawData();
		desc.FieldInfoX = this->FieldX->GetFieldInfo();
		desc.FieldInfoY = this->FieldY->GetFieldInfo();
		desc.FieldInfoZ = this->FieldZ->GetFieldInfo();
		desc.IsStaggeredGrid = IsStaggeredGrid();
		return desc;
	}
	RAWDesc GetFiedRAWDescDevice()
	{
		RAWDesc desc;
		desc.VoxelDataX = this->FieldX->GetRawDataDevice();
		desc.VoxelDataY = this->FieldY->GetRawDataDevice();
		desc.VoxelDataZ = this->FieldZ->GetRawDataDevice();
		desc.FieldInfoX = this->FieldX->GetFieldInfo();
		desc.FieldInfoY = this->FieldY->GetFieldInfo();
		desc.FieldInfoZ = this->FieldZ->GetFieldInfo();
		desc.IsStaggeredGrid = IsStaggeredGrid();
		return desc;
	}
	AxUInt32 GetExecutableVoxelNum(AxInt32 titleSize = -1, AxInt32 shareBoundary = 0)
	{
		AxVector3UI res = this->FieldX->GetResolution();
		if (IsStaggeredGrid())
			res.x -= 1;
		if (titleSize < 8)
			return res.x * res.y * res.z;
		AxVector3UI subRes = MakeVector3UI(res.x / titleSize + 1, res.y / titleSize + 1, res.z / titleSize + 1);
		AxInt32 nx = (titleSize + shareBoundary * 2);
		AxInt32 blockSize = nx * nx * nx;
		return subRes.x * subRes.y * subRes.z * blockSize;
	};

	AxVectorField3DBase()
	{
		this->FieldX = nullptr;
		this->FieldY = nullptr;
		this->FieldZ = nullptr;
	}

	AxVectorField3DBase(const AxField3DInfo& info, std::string name = "v")
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
	AxVectorField3DBase(AxField3DBase<T>* x, AxField3DBase<T>* y, AxField3DBase<T>* z, std::string name = "v")
	{
		FieldX = x;
		FieldY = y;
		FieldZ = z;
		m_sName = name;
	}

	virtual AxAABB GetAABB()
	{
		if (FieldX == nullptr || FieldY == nullptr || FieldZ == nullptr)
			return AlphaCore::AccelTree::MakeAABB();
		return AlphaCore::AccelTree::Merge(AlphaCore::AccelTree::Merge(FieldX->GetAABB(), FieldY->GetAABB()), FieldZ->GetAABB());
	}

	void Set(AxField3DBase<T>* x, AxField3DBase<T>* y, AxField3DBase<T>* z, std::string name = "v")
	{
		FieldX = x;
		FieldY = y;
		FieldZ = z;
		m_sName = name;
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
	bool IsValid() {

		//std::cout << ((this->FieldX != nullptr) && (this->FieldY != nullptr) && (this->FieldZ != nullptr))<< std::endl;
		return ((this->FieldX != nullptr) || (this->FieldY != nullptr) || (this->FieldZ != nullptr));
	}

	void PrintInfo()
	{
		FieldX->PrintInfo();
		FieldY->PrintInfo();
		FieldZ->PrintInfo();

		AxVector3UI res = FieldX->GetFieldInfo().Resolution;
		if (this->IsStaggeredGrid())
			res.x -= 1;

		AX_FOR_K(res.z)
		{
			AX_FOR_J(res.y)
			{
				AX_FOR_I(res.x)
				{
					std::cout << "AsRaw@Voxel_Vec3:" << i << "," << j << "," << k << "|"
						<< FieldX->GetValue(i, j, k) << ","
						<< FieldY->GetValue(i, j, k) << ","
						<< FieldZ->GetValue(i, j, k) << std::endl;
				}
			}
		}
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

	bool AllFieldHashDeviceData()
	{
		if (!this->IsValid())
			return false;
		return FieldX->HasDeviceData() && FieldY->HasDeviceData() && FieldZ->HasDeviceData();
	}

	void LoadToHost()
	{
		FieldX->LoadToHost();
		FieldY->LoadToHost();
		FieldZ->LoadToHost();
	}

	virtual bool DeviceMalloc(bool loadToDevice = true)
	{
		if (!this->IsValid())
			return false;
#ifdef ALPHA_CUDA
		FieldX->DeviceMalloc(loadToDevice);
		FieldY->DeviceMalloc(loadToDevice);
		FieldZ->DeviceMalloc(loadToDevice);
#endif
		return true;
	}

	void TraceData(int start = 0, int end = -1, int sep = 1, const char* head = "")
	{
	}

	bool Save(std::string path)
	{
		std::ofstream ofs(path.c_str(), std::ios::binary);
		if (!ofs)
			return false;
		this->Save(ofs);
		ofs.close();
		return true;
	}

	virtual bool Save(std::ofstream& ofs)
	{
		bool rx = this->FieldX->Save(ofs);
		bool ry = this->FieldY->Save(ofs);
		bool rz = this->FieldZ->Save(ofs);
		return (rx && ry && rz);
	}

	bool Read(std::string path)
	{
		std::ifstream ifs(path.c_str(), std::ios::binary);
		if (!ifs)
			return false;
		this->Read(ifs);
		ifs.close();
		return true;
	}

	virtual bool Read(std::ifstream& ifs)
	{
		bool rx = this->FieldX->Read(ifs);
		bool ry = this->FieldY->Read(ifs);
		bool rz = this->FieldZ->Read(ifs);
		return (rx && ry && rz);
	}

	void SetAllBoundaryExtesion()
	{
		FieldX->SetAllBoundaryExtesion();
		FieldY->SetAllBoundaryExtesion();
		FieldZ->SetAllBoundaryExtesion();
	}

	void SetAllBoundaryOpen()
	{
		FieldX->SetAllBoundaryOutsideZero();
		FieldY->SetAllBoundaryOutsideZero();
		FieldZ->SetAllBoundaryOutsideZero();
	}

	void SetAllBoundaryInverse()
	{
		FieldX->SetAllBoundaryInverse();
		FieldY->SetAllBoundaryInverse();
		FieldZ->SetAllBoundaryInverse();
	}

	void SetBoundaryConditionXPostive(AxGridBoundaryInfo info)
	{
		FieldX->SetBoundaryConditionXPostive(info);
		FieldY->SetBoundaryConditionXPostive(info);
		FieldZ->SetBoundaryConditionXPostive(info);
	}
	void SetBoundaryConditionXNegtive(AxGridBoundaryInfo info)
	{
		FieldX->SetBoundaryConditionXNegtive(info);
		FieldY->SetBoundaryConditionXNegtive(info);
		FieldZ->SetBoundaryConditionXNegtive(info);
	}
	void SetBoundaryConditionYPostive(AxGridBoundaryInfo info)
	{
		FieldX->SetBoundaryConditionYPostive(info);
		FieldY->SetBoundaryConditionYPostive(info);
		FieldZ->SetBoundaryConditionYPostive(info);
	}
	void SetBoundaryConditionYNegtive(AxGridBoundaryInfo info)
	{
		FieldX->SetBoundaryConditionYNegtive(info);
		FieldY->SetBoundaryConditionYNegtive(info);
		FieldZ->SetBoundaryConditionYNegtive(info);
	}
	void SetBoundaryConditionZPostive(AxGridBoundaryInfo info)
	{
		FieldX->SetBoundaryConditionZPostive(info);
		FieldY->SetBoundaryConditionZPostive(info);
		FieldZ->SetBoundaryConditionZPostive(info);
	}
	void SetBoundaryConditionZNegtive(AxGridBoundaryInfo info)
	{
		FieldX->SetBoundaryConditionZNegtive(info);
		FieldY->SetBoundaryConditionZNegtive(info);
		FieldZ->SetBoundaryConditionZNegtive(info);
	}

	void CopyBoundaryConditionFromFieldInfo(const AxField3DInfo& info)
	{
		FieldX->CopyBoundaryConditionFromFieldInfo(info);
		FieldY->CopyBoundaryConditionFromFieldInfo(info);
		FieldZ->CopyBoundaryConditionFromFieldInfo(info);
	}
};

typedef AxVectorField3DBase<float> AxVecFieldF32;
typedef AxVectorField3DBase<double> AxVecFieldF64;
typedef AxVectorField3DBase<AxUInt32> AxVecFieldUInt32;
typedef AxVectorField3DBase<AxFp16> AxVecFieldF16;
//to  check 李俊峰
typedef AxVectorField3DBase<AxInt32> AxVecFieldInt32;

namespace AlphaCore
{
	template <>
	inline AxString TypeName<AxVectorField3DBase<float>>() { return "AxVecFieldFP32"; }
	template <>
	inline AxString TypeName<AxVectorField3DBase<float>*>() { return "AxVecFieldFP32"; }

	template <>
	inline AxString TypeName<AxVectorField3DBase<double>>() { return "AxVecFieldFP64"; }
	template <>
	inline AxString TypeName<AxVectorField3DBase<double>*>() { return "AxVecFieldFP64"; }

	template <>
	inline AxString TypeName<AxVectorField3DBase<AxUInt32>>() { return "AxVecFieldUI32"; }
	template <>
	inline AxString TypeName<AxVectorField3DBase<AxUInt32>*>() { return "AxVecFieldUI32"; }

	template <>
	inline AxString TypeName<AxVectorField3DBase<AxFp16>>() { return "AxVecFieldFP16"; }
	template <>
	inline AxString TypeName<AxVectorField3DBase<AxFp16>*>() { return "AxVecFieldFP16"; }

	template <>
	inline AxString TypeName<AxField3DBase<AxInt32>>() { return "AxFieldI32"; }
	template <>
	inline AxString TypeName<AxField3DBase<AxInt32>*>() { return "AxFieldI32"; }

	template <>
	inline AxString TypeName<AxField3DBase<float>>() { return "AxFieldFP32"; }
	template <>
	inline AxString TypeName<AxField3DBase<float>*>() { return "AxFieldFP32"; }

	template <>
	inline AxString TypeName<AxField3DBase<double>>() { return "AxFieldFP64"; }
	template <>
	inline AxString TypeName<AxField3DBase<double>*>() { return "AxFieldFP64"; }

	template <>
	inline AxString TypeName<AxField3DBase<AxUInt32>>() { return "AxFieldUI32"; }
	template <>
	inline AxString TypeName<AxField3DBase<AxUInt32>*>() { return "AxFieldUI32"; }

	template <>
	inline AxString TypeName<AxField3DBase<AxVector3>>() { return "AxFieldVec3"; }
	template <>
	inline AxString TypeName<AxField3DBase<AxVector3>*>() { return "AxFieldVec3"; }
}


template <typename T>
struct AxVDBInfo
{

};

typedef void VDBPoolData;

template <typename T>
struct ScalarFieldRAWDesc
{
	bool IsValid;
	AxField3DInfo FieldInfo;
	T* RawDataPtr = nullptr;
	AxFieldHardwareInterfaceDesc HardwareInfo;
	VDBPoolData* PoolData = nullptr;
	AxVDBInfo<T> VDBInfo;
};

template<class T>
struct VectorFieldRawDesc {
	bool Active;
	bool IsStaggeredGrid;
	T* RawDataPtrX;
	T* RawDataPtrY;
	T* RawDataPtrZ;
	VDBPoolData* PoolDataX;
	VDBPoolData* PoolDataY;
	VDBPoolData* PoolDataZ;
	AxField3DInfo   FieldInfoX;
	AxField3DInfo   FieldInfoY;
	AxField3DInfo   FieldInfoZ;
	AxVDBInfo<T>       VDBInfoX;
	AxVDBInfo<T>       VDBInfoY;
	AxVDBInfo<T>       VDBInfoZ;
};

typedef ScalarFieldRAWDesc<AxInt8>     AxScalarFieldHandleI8;
typedef ScalarFieldRAWDesc<AxInt16>    AxScalarFieldHandleI16;
typedef ScalarFieldRAWDesc<AxInt32>    AxScalarFieldHandleI32;
typedef ScalarFieldRAWDesc<AxFp32>     AxScalarFieldHandleF32;
typedef ScalarFieldRAWDesc<AxFp64>     AxScalarFieldHandleF64;
typedef ScalarFieldRAWDesc<AxUInt8>    AxScalarFieldHandleU8;
typedef ScalarFieldRAWDesc<AxUInt32>   AxScalarFieldHandleU32;
typedef ScalarFieldRAWDesc<AxVector3>  AxScalarFieldHandleV3F32;


typedef VectorFieldRawDesc<AxInt8>		AxVectorFieldHandleI8;
typedef VectorFieldRawDesc<AxInt16>		AxVectorFieldHandleI16;
typedef VectorFieldRawDesc<AxInt32>		AxVectorFieldHandleI32;
typedef VectorFieldRawDesc<AxFp32>		AxVectorFieldHandleF32;
typedef VectorFieldRawDesc<AxFp64>		AxVectorFieldHandleF64;
typedef VectorFieldRawDesc<AxUInt8>		AxVectorFieldHandleU8;
typedef VectorFieldRawDesc<AxUInt32>	AxVectorFieldHandleU32;


template<typename T>
static VectorFieldRawDesc<T> MakeDefaultVectorFieldHandle()
{
	VectorFieldRawDesc<T> ret;
	ret.Active = false;
	ret.RawDataPtrX = nullptr;
	ret.RawDataPtrY = nullptr;
	ret.RawDataPtrZ = nullptr;
	return ret;
}

template<typename T>
static ScalarFieldRAWDesc<T> MakeDefaultScalarFieldHandle()
{
	ScalarFieldRAWDesc<T> ret;
	ret.IsValid = false;
	ret.RawDataPtr = nullptr;
	return ret;
}


#endif
