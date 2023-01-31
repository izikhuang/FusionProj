#ifndef __AX_FORCE_H__
#define __AX_FORCE_H__

#include "Utility/AxStorage.h"
#include "AxMacro.h"
#include "ProceduralContent/AxNoise.DataType.h"

namespace AlphaCore {
	namespace Particle {
		ALPHA_SPMD_FUNC void ImplicitVelocityDamping(
			AxBufferV3* posBuf,
			AxBufferV3* velBuf,
			AxBuffer2UI* p2pMap2IBuf,
			AxBufferUInt32* p2pMapIndicesBuf,
			AxBufferV3* targetVelocityBuf,
			AxBufferF* airResistBuf,
			AxBufferF* normalDragBuf,
			AxBufferF* targetDragBuf,
			AxBufferF* massBuf,
			AxFp32 dt);

		ALPHA_SPMD_FUNC void WindForce(
			AxCurlNoiseParam curlNoiseParam,
			AxBufferV3* posBuf,
			AxBufferV3* windVelBuf,
			AxBufferF* windAirresistBuf,
			AxBufferF* massBuf,
			AxBufferF* windIntensityBuf,
			AxVector3 windVelocityParam,
			AxFp32 windAirresistParam,
			bool ignoreMass,
			AxFp32 time,
			AxFp32 dt);


		namespace CUDA 
		{
			ALPHA_SPMD_FUNC void ImplicitVelocityDamping(
				AxBufferV3* posBuf,
				AxBufferV3* velBuf,
				AxBuffer2UI* p2pMap2IBuf,
				AxBufferUInt32* p2pMapIndicesBuf,
				AxBufferV3* targetVelocityBuf,
				AxBufferF* airResistBuf,
				AxBufferF* normalDragBuf,
				AxBufferF* targetDragBuf,
				AxBufferF* massBuf,
				AxFp32 dt,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void WindForce(
				AxCurlNoiseParam curlNoiseParam,
				AxBufferV3* posBuf,
				AxBufferV3* windVelBuf,
				AxBufferF* windAirresistBuf,
				AxBufferF* massBuf,
				AxBufferF* windIntensityBuf,
				AxVector3 windVelocityParam,
				AxFp32 windAirresistParam,
				bool ignoreMass,
				AxFp32 time,
				AxFp32 dt,
				AxUInt32 blockSize = 256);
		}
	}
}//@namespace end of : Particle
#endif