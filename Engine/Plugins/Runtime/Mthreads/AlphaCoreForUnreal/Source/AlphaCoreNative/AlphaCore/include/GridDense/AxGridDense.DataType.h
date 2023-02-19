
#ifndef __AX_GRIDDENSE__DATATYPE__H__
#define __AX_GRIDDENSE__DATATYPE__H__


#include "AxDataType.h"
#include "Utility/AxDescrition.h"
#include "AccelTree/AxAccelTree.DataType.h"

enum AxGridBoundaryInfo
{
	kOutsideZero = 0,
	kExtension = 1,
	kInverse = 2
};

enum AxVoxelTypeInfo
{
	kGas   = 0b000001,
	kLquid = 0b000010,
	kSolid = 0b000100,
	kSDF   = 0b001000
};

enum AxGridBoundaryIndex
{
	XPostiveOffset = 0,
	XNegtiveOffset = 1,
	YPostiveOffset = 2,
	YNegtiveOffset = 3,
	ZPostiveOffset = 4,
	ZNegtiveOffset = 5
};

struct AxField3DInfo
{
	AxVector3	Pivot;
	AxVector3	VoxelSize;
	//AxVector3	InvHalfVoxelSize;  //Todo[x]:need implement
	AxVector3	FieldSize;         //Todo[x]:need implement
	AxVector3UI	Resolution;
 	AxGridBoundaryInfo BoundaryInfo[6];
	AlphaCore::AxBackendAPI APIInfo;
	bool UseAPIStage;
	//bool UseRotation;
	AxMatrix3x3 RotationMatrix;
	AxMatrix3x3 InverseRotMatrix;
	//AxAABB FieldBoundingBox;
	//FieldObject:CUDA,DX11
};

static const char* AxField3DInfoToString(const AxGridBoundaryInfo& info)
{
	if (info == AxGridBoundaryInfo::kOutsideZero)
		return "Open";
	if (info == AxGridBoundaryInfo::kExtension)
		return "kExtension";
	if (info == AxGridBoundaryInfo::kInverse)
		return "kInverse";
	return "InvalidBoundary";
}

inline std::ostream& operator << (std::ostream& out, const AxField3DInfo& info)
{
	out << " Field Size:" << info.FieldSize;
	out << " Field Pivot:" << info.Pivot;
	out << " Res :" << info.Resolution;
	out << " VoxelSize :" << info.VoxelSize;
	return out;
}

ALPHA_SHARE_FUNC AxField3DInfo MakeDefaultFieldInfo(
	AxUInt32 resX,
	AxUInt32 resY, 
	AxUInt32 resZ, 
	AxGridBoundaryInfo boundaryInfo = AxGridBoundaryInfo::kOutsideZero)
{
	AxField3DInfo defaultFieldInfo;
	defaultFieldInfo.Resolution.x = resX;
	defaultFieldInfo.Resolution.y = resY;
	defaultFieldInfo.Resolution.z = resZ;
	defaultFieldInfo.VoxelSize = MakeVector3(1.0f, 1.0f, 1.0f);

	defaultFieldInfo.BoundaryInfo[0] = boundaryInfo;
	defaultFieldInfo.BoundaryInfo[1] = boundaryInfo;
	defaultFieldInfo.BoundaryInfo[2] = boundaryInfo;
	defaultFieldInfo.BoundaryInfo[3] = boundaryInfo;
	defaultFieldInfo.BoundaryInfo[4] = boundaryInfo;
	defaultFieldInfo.BoundaryInfo[5] = boundaryInfo;

	return defaultFieldInfo;
}

ALPHA_SHARE_FUNC AxField3DInfo MakeDefaultFieldInfo(
	AxVector3UI res,
	AxGridBoundaryInfo boundaryInfo = AxGridBoundaryInfo::kOutsideZero)
{
	return MakeDefaultFieldInfo(res.x, res.y, res.z, boundaryInfo);
}

ALPHA_SHARE_FUNC AxField3DInfo MakeDefaultFieldInfo(
	AxUInt32 resX,
	AxUInt32 resY,
	AxUInt32 resZ,
	AxGridBoundaryInfo* boundaryInfo )
{
	AxField3DInfo defaultFieldInfo;
	defaultFieldInfo.Resolution.x = resX;
	defaultFieldInfo.Resolution.y = resY;
	defaultFieldInfo.Resolution.z = resZ;
	defaultFieldInfo.VoxelSize = MakeVector3(1.0f, 1.0f, 1.0f);

	defaultFieldInfo.BoundaryInfo[0] = boundaryInfo[0];
	defaultFieldInfo.BoundaryInfo[1] = boundaryInfo[1];
	defaultFieldInfo.BoundaryInfo[2] = boundaryInfo[2];
	defaultFieldInfo.BoundaryInfo[3] = boundaryInfo[3];
	defaultFieldInfo.BoundaryInfo[4] = boundaryInfo[4];
	defaultFieldInfo.BoundaryInfo[5] = boundaryInfo[5];

	return defaultFieldInfo;
}
#endif // !__AX_GRIDDENSE__DATATYPE__H__


