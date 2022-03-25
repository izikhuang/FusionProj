#ifndef __AX_VOLUMERENDER_H__
#define __AX_VOLUMERENDER_H__

#include <AxMacro.h>
#include "AxVolumeRender.DataType.h"

namespace AlphaCore {
	namespace VolumeRender {
		ALPHA_SPMD_FUNC void GasVolumeRender(
			AxVolumeRenderObject gasVolumeRenderObject,
			AxSceneRenderDesc sceneDesc,
			AxTextureRGBA8* outputTexBuf,
			AxTextureR32* depthTexBuf,
			AxFp32 stepSize,
			int width,
			int height,
			AxMatrix4x4 postXform);

		namespace CUDA {
			ALPHA_SPMD_FUNC void GasVolumeRender(
				AxVolumeRenderObject gasVolumeRenderObject,
				AxSceneRenderDesc sceneDesc,
				AxTextureRGBA8* outputTexBuf,
				AxTextureR32* depthTexBuf,
				AxFp32 stepSize,
				int width,
				int height,
				AxMatrix4x4 postXform,
				AxUInt32 blockSize = 512);

		}
	}
}//@namespace end of : VolumeRender
#endif