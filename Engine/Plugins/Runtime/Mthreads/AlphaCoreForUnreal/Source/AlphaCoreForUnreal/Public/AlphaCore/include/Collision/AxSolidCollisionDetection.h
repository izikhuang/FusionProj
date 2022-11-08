#ifndef __AX_SOLIDCOLLISIONDETECTION_H__
#define __AX_SOLIDCOLLISIONDETECTION_H__


#include "AxDataType.h"
#include "Math/AxVectorBase.h"
#include "Collision/AxCollision.DataType.h"
#include "SolidUtility/AxSolidUtility.DataType.h"
#include "AccelTree/AxBVHTree.h"


namespace AlphaCore {
	namespace Collision {
		ALPHA_SPMD_FUNC void FCCDWithSortedBVH(
			AlphaCore::SolidUtility::SimData::AxSolidData* geoA,
			AlphaCore::SolidUtility::SimData::AxSolidData* geoB,
			AxBVHTree* bvhTreeA,
			AxBVHTree* bvhTreeB,
			AxBufferContact* contactBuf,
			AxSPMDTick SPMDTick);

		ALPHA_SPMD_FUNC void FCCDWithSortedBVH_O1(
			AlphaCore::SolidUtility::SimData::AxSolidData* geoA,
			AlphaCore::SolidUtility::SimData::AxSolidData* geoB,
			AxBVHTree* bvhTreeA,
			AxBVHTree* bvhTreeB,
			AxBufferContact* contactBuf,
			AxSPMDTick SPMDTick);

		ALPHA_SPMD_FUNC void GroundCollision(
			AxBufferV3* posBuf,
			AxFp32 height);

		ALPHA_SPMD_FUNC void CapsuleCollision(
			AxBufferV3* posBuf,
			AxCapsuleCollider capsuleCollider);

		ALPHA_SPMD_FUNC void OBBCollision(
			AxBufferV3* posBuf,
			AxOBBCollider obbCollider);
		namespace CUDA 
		{
			ALPHA_SPMD_FUNC void FCCDWithSortedBVH(
				AlphaCore::SolidUtility::SimData::AxSolidData* geoA,
				AlphaCore::SolidUtility::SimData::AxSolidData* geoB,
				AxBVHTree* bvhTreeA,
				AxBVHTree* bvhTreeB,
				AxBufferContact* contactBuf,
				AxSPMDTick SPMDTick,
				AxUInt32 blockSize = 256);

			ALPHA_SPMD_FUNC void GroundCollision(
				AxBufferV3* posBuf,
				AxFp32 height,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void CapsuleCollision(
				AxBufferV3* posBuf,
				AxCapsuleCollider capsuleCollider,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void OBBCollision(
				AxBufferV3* posBuf,
				AxOBBCollider obbCollider,
				AxUInt32 blockSize = 512);
		}
	}
}//@namespace end of : Collision
#endif