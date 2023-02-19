#ifndef __AX_VOLUMERENDER_H__
#define __AX_VOLUMERENDER_H__

#include "AxMacro.h"
#include "AxVolumeRender.DataType.h"

namespace AlphaCore
{
	namespace VolumeRender
	{
		ALPHA_SPMD_FUNC void GasVolumeRender(
			AxVolumeRenderObject gasVolumeRenderObject,
			AxSceneRenderDesc sceneDesc,
			AxTextureRGBA8 *outputTexBuf,
			AxTextureR32 *depthTexBuf,
			AxFp32 stepSize,
			int width,
			int height,
			AxMatrix4x4 postXform = {1.0f, 0.0f, 0.0f, 0.0f,
									 0.0f, 1.0f, 0.0f, 0.0f,
									 0.0f, 0.0f, 1.0f, 0.0f,
									 0.0f, 0.0f, 0.0f, 1.0f});

		ALPHA_SPMD_FUNC void GasVolumeRenderInUE(
			AxVolumeRenderObject gasVolumeRenderObject,
			AxSceneRenderDesc sceneDesc,
			AxTextureRGBA* outputTexBuf,
			AxTextureR32* depthTexBuf,
			//AxFp32 stepSize,
			int width,
			int height);

		ALPHA_SPMD_FUNC void ConvertScalarFieldF32ToInt8(
			AxScalarFieldF32 *scalarFiled, AxUInt8 *dstData,
			AxVector2 minMaxInput);

		namespace CUDA
		{
			ALPHA_SPMD_FUNC void GasVolumeRender(
				AxVolumeRenderObject gasVolumeRenderObject,
				AxSceneRenderDesc sceneDesc,
				AxTextureRGBA8 *outputTexBuf,
				AxTextureR32 *depthTexBuf,
				AxFp32 stepSize,
				int width,
				int height,
				AxMatrix4x4 postXform = {1.0f, 0.0f, 0.0f, 0.0f,
										 0.0f, 1.0f, 0.0f, 0.0f,
										 0.0f, 0.0f, 1.0f, 0.0f,
										 0.0f, 0.0f, 0.0f, 1.0f},
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void GasVolumeRenderInUE(
				AxVolumeRenderObject gasVolumeRenderObject,
				AxSceneRenderDesc sceneDesc,
				AxTextureRGBA* outputTexBuf,
				AxTextureR32* depthTexBuf,
				//AxFp32 stepSize,
				int width,
				int height,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void ConvertScalarFieldF32ToInt8(
				AxScalarFieldF32 *scalarFiled, AxUInt8 *dstData,
				AxVector2 minMaxInput, AxUInt32 blockSize = 512);

		}
	}
} //@namespace end of : VolumeRender
#endif