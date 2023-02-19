#ifndef __AX_SPATIALHASH_H__
#define __AX_SPATIALHASH_H__

#include "AxAccelTree.DataType.h"
#include "GridDense/AxFieldBase3D.h"
#include "AxGeo.h"

class AxSpatialHash : public AxField
{
public:

	struct RAWDesc
	{
 		AxInt32* PtCellIdBuffer;
		AxInt32* PtSortedIdBuffer;
		AxInt32* StartRaw;
		AxInt32* EndRaw;
		AxVector3* TargetPositionRaw;
		AxField3DInfo FieldInfo;
	};
	AxSpatialHash()
	{
		this->m_sName = "HashGrid";
		this->m_TargetPosBuffer = nullptr;
		this->m_PtCellIdBuffer = nullptr;
		this->m_PtSortedIdBuffer = nullptr;
		this->FieldStart = nullptr;
		this->FieldEnd = nullptr;
	}


	AxSpatialHash::RAWDesc GetRAWDesc(AlphaCore::AxBackendAPI device);

	virtual ~AxSpatialHash()
	{

	}
	
	//TODO : Rename LoadToDevice
	virtual bool DeviceMalloc(bool loadToDevice = true);

	void ClearField()
	{
		FieldStart->GetVoxelStorageBuffer().SetToFill();
		FieldEnd->GetVoxelStorageBuffer().SetToFill();
	}

	void ClearFieldDevice()
	{
		FieldStart->GetVoxelStorageBuffer().SetToFillDevice();
		FieldEnd->GetVoxelStorageBuffer().SetToFillDevice();
	}

	bool UpdateHash();

	bool UpdateHashDevice();

	//Hash Module 1.0
	void SetNeighborSearch3x3Callback();
	void SetNeighborSearch3x3CallbackDevice();

	void Build(AxGeometry* geo,const char* targetPName = AlphaProperty::PrdP);
	//ws temporary
	void BuildPrim(AxGeometry* geo, const char* targetPName = AlphaProperty::PrdP);

	void Init(AxVector3 pivot, AxVector3 size, AxVector3UI res, bool setToZero = false)
	{
		if (this->FieldStart == nullptr)
			this->FieldStart = new AxScalarFieldI32(pivot, size, res, AlphaProperty::AccelTree::Cell2PtStart);
		if (this->FieldEnd == nullptr)
			this->FieldEnd = new AxScalarFieldI32(pivot, size, res, AlphaProperty::AccelTree::Cell2PtEnd);
		FieldStart->Init(pivot, size, res, setToZero);
		FieldEnd->Init(pivot, size, res, setToZero);
	}

	void Init(AxVector3 pivot, AxVector3 size, AxFp32 cellSize, bool setToZero = false)
	{
		AxVector3UI res = MakeVector3UI(
			std::floor(size.x / cellSize),
			std::floor(size.y / cellSize),
			std::floor(size.z / cellSize));
		size.x = res.x * cellSize;
		size.y = res.y * cellSize;
		size.z = res.z * cellSize;
		this->Init(pivot, size, res, setToZero);
	}
	//to check 李俊峰 偶尔会出现由于精度问题导致的res改变，如原有res = (3,4,3)计算后变成（2，4，3）
	void InitForFlip(AxVector3 pivot, AxVector3 size, AxVector3 cellSize, bool setToZero = false)
	{
		AxVector3UI res = MakeVector3UI(
			std::floor(size.x / cellSize.x),
			std::floor(size.y / cellSize.y),
			std::floor(size.z / cellSize.z));
		size.x = res.x * cellSize.x;
		size.y = res.y * cellSize.y;
		size.z = res.z * cellSize.z;
		this->Init(pivot, size, res, setToZero);
	}

	AxUInt32 GetNX()
	{
		return FieldStart->GetNX();
	}

	AxUInt32 GetNY()
	{
		return FieldStart->GetNY();
	}

	AxUInt32 GetNZ()
	{
		return FieldStart->GetNZ();
	}

	AxUInt32 GetNumCells()
	{
		return FieldStart->GetNumVoxels();
	}

	void Release()
	{
		FieldStart->Release();
		FieldEnd->Release();
	}

	void PrintData()
	{
		
	}

	AxScalarFieldI32* GetStartGrid() { return FieldStart; }
	AxScalarFieldI32* GetEndGrid() { return FieldEnd; }

	AxBufferI* GetSortedPtID() { return  m_PtSortedIdBuffer; };
	AxBufferI* GetPtCellIdBuffer() { return  m_PtCellIdBuffer; };


protected:

	void calcParticleHash();
	void sortCellID(); 
	void findStarEnd();

	void calcParticleHashDevice();
	void sortCellIDDevice();
	void findStarEndDevice();


private:
	
	AxScalarFieldI32* FieldStart;
	AxScalarFieldI32* FieldEnd;

	AxBufferV3* m_TargetPosBuffer;
	AxBufferI* m_PtSortedIdBuffer;
	AxBufferI* m_PtCellIdBuffer;

};


namespace AlphaCore
{
	namespace AccelTree
	{
		ALPHA_SPMD_FUNC void CalcParticleHash(
			AxField3DInfo info,
			AxBufferV3* targetPosBuffer,
			AxBufferI* ptCellIdBuffer);

		ALPHA_SPMD_FUNC void FindStarEnd(
			AxBufferI* ptCellIdBuffer,
			AxBufferI* cellStartBuffer,
			AxBufferI* cellEndBuffer);

		namespace CUDA
		{
			ALPHA_SPMD_FUNC void CalcParticleHash(
				AxField3DInfo info,
				AxBufferV3* targetPosBuffer,
				AxBufferI* ptCellIdBuffer,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void FindStarEnd(
				AxBufferI* ptCellIdBuffer,
				AxBufferI* cellStartBuffer,
				AxBufferI* cellEndBuffer,
				AxUInt32 blockSize = 512);
		}
	}
}

#endif // !__AX_SPATIAL_HASH_H__
