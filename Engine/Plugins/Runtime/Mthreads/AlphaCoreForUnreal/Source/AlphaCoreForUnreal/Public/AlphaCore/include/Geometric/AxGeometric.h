#ifndef __AX_GEOMETRIC_H__
#define __AX_GEOMETRIC_H__

#include "AxMacro.h"
#include "Utility/AxStorage.h"


namespace AlphaCore {
	namespace Geometric {

		ALPHA_SPMD_FUNC void MakeDeformerXform(
			AxBufferMat3x3F* xformsBuf,
			AxBufferV3* restPosBuf,
			AxBufferV3* dynamicPosBuf,
			AxBufferArray* p2pMapBuf,
			AxBufferUInt32* p2pIndicesBuf);

		ALPHA_SPMD_FUNC void PointDeformer(AxBufferV3* targetPosBuf,
			AxBufferV3* restPosBuf,
			AxBufferV3* dynamicPosBuf,
			AxBufferArray* linkedPtsMapBuf,
			AxBufferUInt32* linkedPtsIndicesBuf,
			AxBufferF* linkedWeightsBuf,
			AxBufferMat3x3F* xformsBuf,
			AxBufferArray* p2pMapBuf,
			AxBufferUInt32* p2pIndicesBuf);

		ALPHA_SPMD_FUNC void DeformerByMatrix(AxBufferV3* targetPosBuf,
			AxBufferV3* restPosBuf,
			AxBufferV3* dynamicPosBuf,
			AxBufferArray* linkedPtsMapBuf,
			AxBufferUInt32* linkedPtsIndicesBuf,
			AxBufferF* linkedWeightsBuf,
			AxBufferMat3x3F* xformsBuf);

		ALPHA_SPMD_FUNC void ComputePrimNormal(
			AxBufferUInt32* topologyIndicesBuf,
			AxBufferV3* posBuf,
			AxBuffer2UI* prim2IBuf,
			AxBufferV3* primNormalBuf);

		ALPHA_SPMD_FUNC void IntegratePointNormal(
			AxBufferArray* point2PrimMapBuf,
			AxBufferUInt32* point2PrimIndicesBuf,
			AxBufferV3* primNormalBuf,
			AxBufferV3* pointNormalBuf);

		namespace CUDA {

			ALPHA_SPMD_FUNC void MakeDeformerXform(
				AxBufferMat3x3F* xformsBuf,
				AxBufferV3* restPosBuf,
				AxBufferV3* dynamicPosBuf,
				AxBufferArray* p2pMapBuf,
				AxBufferUInt32* p2pIndicesBuf,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void PointDeformer(
				AxBufferV3* targetPosBuf,
				AxBufferV3* restPosBuf,
				AxBufferV3* dynamicPosBuf,
				AxBufferArray* linkedPtsMapBuf,
				AxBufferUInt32* linkedPtsIndicesBuf,
				AxBufferF* linkedWeightsBuf,
				AxBufferMat3x3F* xformsBuf,
				AxBufferArray* p2pMapBuf,
				AxBufferUInt32* p2pIndicesBuf,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void DeformerByMatrix(AxBufferV3* targetPosBuf,
				AxBufferV3* restPosBuf,
				AxBufferV3* dynamicPosBuf,
				AxBufferArray* linkedPtsMapBuf,
				AxBufferUInt32* linkedPtsIndicesBuf,
				AxBufferF* linkedWeightsBuf,
				AxBufferMat3x3F* xformsBuf,
				AxUInt32 blockSize = 512);


			ALPHA_SPMD_FUNC void ComputePrimNormal(
				AxBufferUInt32* topologyIndicesBuf,
				AxBufferV3* posBuf,
				AxBuffer2UI* prim2IBuf,
				AxBufferV3* primNormalBuf,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void IntegratePointNormal(
				AxBufferArray* point2PrimMapBuf,
				AxBufferUInt32* point2PrimIndicesBuf,
				AxBufferV3* primNormalBuf,
				AxBufferV3* pointNormalBuf,
				AxUInt32 blockSize = 512);

		}
	}
}//@namespace end of : Geometric
#endif