#ifndef __AX_COLLISIONRESOLVE_H__
#define __AX_COLLISIONRESOLVE_H__

#include "AxDataType.h"
#include "AxMacro.h"
#include "AxCollision.DataType.h"
#include "Math/AxVectorHelper.h"
#include "AxDescrition.h"
#include "GridDense/AxGridDense.DataType.h"

namespace AlphaCore 
{
	namespace Collision 
	{
		ALPHA_SPMD_FUNC void PBDCollisionResolveJacobi(
			AxBufferContact* contactBuf,
			AxUInt32 realCtxNum,
			AxBufferV3* selfPosBuf,
			AxBufferV3* selfPrdPBuf,
			AxBufferV3* colliderPrdPBuf,
			AxBufferF* pointThicknessBuf,
			const AlphaCore::Collision::SimData::AxPBDCollisionResolveData& resolveData,
			AxUInt32 iterations,
			AxFp32 coffe);

		ALPHA_SPMD_FUNC void P2POverlapFixJacobiConstRadius(
			AxField3DInfo fieldInfo,
			AxBufferI* sortedPtIdBuf,
			AxBufferI* ptCellIdBuf,
			AxBufferI* cell2PtStartBuf,
			AxBufferI* cell2PtEndBuf,
			AxBufferV3* prdPBuf,
			AxUInt32 iterations,
			AxFp32 pscale);

		namespace Internal 
		{
			ALPHA_SPMD_FUNC void pbdContactResolveJacobi(
				AxBufferContact* contactBuf,
				AxUInt32 realCtxNum,
				AxBufferV3* selfPosBuf,
				AxBufferV3* selfPrdPBuf,
				AxBufferV3* colliderPrdPBuf,
				AxBufferF* pointThicknessBuf,
				AlphaCore::Collision::SimData::AxPBDCollisionResolveData::RawData resolveData,
				bool updateIndicesBuffer = false);

			ALPHA_SPMD_FUNC void evaluatePoint2ContactVertexMap(
				AxBufferUInt32* outPt2ContactVertexStartBuf,
				AxBufferUInt32* outPt2ContactVertexEndBuf,
				AxBufferUInt32* contactIndicesBuf,
				AxUInt32 contactIndicesSize);

			ALPHA_SPMD_FUNC void jacobiMovePoint(
				AxBufferV4* contactFixVec4Buf,
				AxBufferUInt32* outPt2ContactVertexStartBuf,
				AxBufferUInt32* outPt2ContactVertexEndBuf,
				AxBufferUInt32* sortedCtxIndicesIDBuf,
				AxBufferV3* selfPrdPBuf,
				AxBufferV3* collisionNormalDstBuf,
				int numPoints,
				AxFp32 coeff);
		}

		namespace CUDA 
		{
			ALPHA_SPMD_FUNC void PBDCollisionResolveJacobi(
				AxBufferContact* contactBuf,
				AxUInt32 realCtxNum,
				AxBufferV3* selfPosBuf,
				AxBufferV3* selfPrdPBuf,
				AxBufferV3* colliderPrdPBuf,
				AxBufferF* pointThicknessBuf,
				const AlphaCore::Collision::SimData::AxPBDCollisionResolveData& resolveData,
				AxUInt32 iterations,
				AxFp32 coffe,
				AxUInt32 blockSize = 512);


			ALPHA_SPMD_FUNC void P2POverlapFixJacobiConstRadius(
				AxField3DInfo fieldInfo,
				AxBufferI* sortedPtIdBuf,
				AxBufferI* ptCellIdBuf,
				AxBufferI* cell2PtStartBuf,
				AxBufferI* cell2PtEndBuf,
				AxBufferV3* prdPBuf,
				AxUInt32 iterations,
				AxFp32 pscale,
				AxUInt32 blockSize = 512);


			namespace Internal
			{ 
				ALPHA_SPMD_FUNC void pbdContactResolveJacobi(
					AxBufferContact* contactBuf,
					AxUInt32 realCtxNum,
					AxBufferV3* selfPosBuf,
					AxBufferV3* selfPrdPBuf,
					AxBufferV3* colliderPrdPBuf,
					AxBufferF* pointThicknessBuf,
					AlphaCore::Collision::SimData::AxPBDCollisionResolveData::RawData resolveData,
					bool updateIndicesBuffer,
					AxUInt32 blockSize = 512);

				ALPHA_SPMD_FUNC void evaluatePoint2ContactVertexMap(
					AxBufferUInt32* outPt2ContactVertexStartBuf,
					AxBufferUInt32* outPt2ContactVertexEndBuf,
					AxBufferUInt32* contactIndicesBuf,
					AxUInt32 contactIndicesSize,
					AxUInt32 blockSize = 512);


				ALPHA_SPMD_FUNC void jacobiMovePoint(
					AxBufferV4* contactFixVec4Buf,
					AxBufferUInt32* outPt2ContactVertexStartBuf,
					AxBufferUInt32* outPt2ContactVertexEndBuf,
					AxBufferUInt32* sortedCtxIndicesIDBuf,
					AxBufferV3* selfPrdPBuf,
					AxBufferV3* collisionNormalDstBuf,
					int numPoints,
					AxFp32 coeff,
					AxUInt32 blockSize = 512);
			}
		}
	}
}//@namespace end of : Collision

#endif