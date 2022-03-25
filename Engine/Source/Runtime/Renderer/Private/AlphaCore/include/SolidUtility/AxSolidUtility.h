#ifndef __AX_SOLIDUTILITY_H__
#define __AX_SOLIDUTILITY_H__

#include <Utility/AxStorage.h>
#include "AxSolidUtility.DataType.h"
#include <AxGeo.h>

#define ALPHA_ENGINE_ARCH_AX_SOLIDUTILITY 1
#if ALPHA_ENGINE_ARCH_AX_SOLIDUTILITY

namespace AlphaCore
{
	namespace SolidUtility
	{
		ALPHA_SPMD_FUNC void AdvectOrd1(
			AxBufferV3* accelBuf,
			AxBufferV3* velBuf,
			AxBufferV3* prdPBuf,
			float dt);

		ALPHA_SPMD_FUNC void AdvectBDF2(
			AxBufferV3* prdPosBuf,
			AxBufferV3* prevPosBuf,
			AxBufferV3* lastPosBuf,
			AxBufferV3* velBuf,
			AxBufferV3* prevVelBuf,
			AxBufferV3* lastVelBuf,
			AxBufferV3* accelBuf,
			float dt);

		ALPHA_SPMD_FUNC void IntergratorOrd1(
			AxBufferV3* posBuf,
			AxBufferV3* prdPBuf,
			AxBufferV3* hitNormalBuf,
			AxBufferV3* velBuf,
			AxBufferV3* accelBuf,
			float dampRange,
			float dampRate,
			float ctxNormalDrag,
			float ctxTangentDrag,
			float dt,
			float invDt);

		ALPHA_SPMD_FUNC void IntergratorBDF2(
			AxBufferV3* prdPBuf,
			AxBufferV3* prevPBuf,
			AxBufferV3* lastPBuf,
			AxBufferV3* velBuf,
			AxFp32 dt);

		ALPHA_SPMD_FUNC void PredictOrient1Ord(
			AxBufferF* orientBuf,
			AxBufferF* inertiaBuf,
			AxBufferV3* omegaBuf,
			AxFp32 dt);

		ALPHA_SPMD_FUNC void IntegrateOmegaBDF1(
			AxBufferF* orientBuf,
			AxBufferF* orientpreviousBuf,
			AxBufferV3* omegaBuf,
			AxFp32 dt);

		namespace CUDA 
		{
			ALPHA_SPMD_FUNC void AdvectOrd1(
				AxBufferV3* accelBuf,
				AxBufferV3* velBuf,
				AxBufferV3* prdPBuf,
				float dt,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void AdvectBDF2(
				AxBufferV3* prdPosBuf,
				AxBufferV3* prevPosBuf,
				AxBufferV3* lastPosBuf,
				AxBufferV3* velBuf,
				AxBufferV3* prevVelBuf,
				AxBufferV3* lastVelBuf,
				AxBufferV3* accelBuf,
				float dt,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void IntergratorOrd1(
				AxBufferV3* posBuf,
				AxBufferV3* prdPBuf,
				AxBufferV3* hitNormalBuf,
				AxBufferV3* velBuf,
				AxBufferV3* accelBuf,
				float dampRange,
				float dampRate,
				float ctxNormalDrag,
				float ctxTangentDrag,
				float dt,
				float invDt,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void IntergratorBDF2(
				AxBufferV3* prdPBuf,
				AxBufferV3* prevPBuf,
				AxBufferV3* lastPBuf,
				AxBufferV3* velBuf,
				AxFp32 dt,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void PredictOrient1Ord(
				AxBufferF* orientBuf,
				AxBufferF* inertiaBuf,
				AxBufferV3* omegaBuf,
				AxFp32 dt,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void IntegrateOmegaBDF1(
				AxBufferF* orientBuf,
				AxBufferF* orientpreviousBuf,
				AxBufferV3* omegaBuf,
				AxFp32 dt,
				AxUInt32 blockSize = 512);
		}
	}
}//@namespace end of : SolidUtility

#endif//@AX_END_OF ALPHA_ENGINE_ARCH_AX_SOLIDUTILITY

#endif