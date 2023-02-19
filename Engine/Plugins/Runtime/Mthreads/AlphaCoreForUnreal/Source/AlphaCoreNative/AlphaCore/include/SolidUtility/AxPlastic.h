#ifndef __AX_PLASTIC_H__
#define __AX_PLASTIC_H__

#include "AxDataType.h"
#include "AxMacro.h"
#include "Utility/AxStorage.h"

namespace AlphaCore {
	namespace SolidUtility {
		ALPHA_SPMD_FUNC void ApplyFastPlasticDeform(
			AxBuffer2UI* primList2IBuf,
			AxBufferUInt32* topologyIndicesBuf,
			AxBufferI* primTypeBuf,
			AxBufferF* orientBuf,
			AxBufferV4* restvectorBuf,
			AxBufferF* restlengthBuf,
			AxBufferF* stiffnessBuf,
			AxBufferF* plastichardeningBuf,
			AxBufferF* plasticthresholdBuf,
			AxBufferF* plasticrateBuf,
			AxBufferF* flowBuf,
			AxFp32 dt);

		namespace CUDA {
			ALPHA_SPMD_FUNC void ApplyFastPlasticDeform(AxBuffer2UI* primList2IBuf,
				AxBufferUInt32* topologyIndicesBuf,
				AxBufferI* primTypeBuf,
				AxBufferF* orientBuf,
				AxBufferV4* restvectorBuf,
				AxBufferF* restlengthBuf,
				AxBufferF* stiffnessBuf,
				AxBufferF* plastichardeningBuf,
				AxBufferF* plasticthresholdBuf,
				AxBufferF* plasticrateBuf,
				AxBufferF* flowBuf,
				AxFp32 dt,
				AxUInt32 blockSize = 512);

		}
	}
}//@namespace end of : SolidUtility
#endif