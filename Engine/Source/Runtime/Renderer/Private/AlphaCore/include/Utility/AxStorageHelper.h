#ifndef __AX_STORAGEHELPER_H__
#define __AX_STORAGEHELPER_H__


#include <AxMacro.h>
#include <AxDataType.h>
#include <Math/AxVectorHelper.h>
#include <Utility/AxStorage.h>

namespace AlphaCore {
	namespace StorageHelper 
	{
		template <typename T>
		ALPHA_SPMD_FUNC void MakeIdenityBufferT(AxStorage<T>* idenityInputBuf, AxUInt64 start = 0, AxInt32 end = -1);
		template <typename T>
		ALPHA_SPMD_FUNC void MakeIdenityBuffer(AxStorage<T>* idenityInputBuf, AxUInt64 start = 0, AxInt32 end = -1);
		template <>
		ALPHA_SPMD_FUNC void MakeIdenityBuffer(AxStorage<AxUInt32>* idenityInputBuf, AxUInt64 start , AxInt32 end);
		template <>
		ALPHA_SPMD_FUNC void MakeIdenityBuffer(AxStorage<AxInt32>* idenityInputBuf, AxUInt64 start, AxInt32 end);

		ALPHA_SPMD_FUNC void LerpV3(AxBufferV3* outBuffer, AxBufferV3* a, AxBufferV3* b,AxFp32 t);

		ALPHA_SPMD_FUNC void Vec3AddConstant(AxBufferV3* outBuffer, AxVector3 v3Constant, AxFp32 coeff);

		namespace CUDA 
		{
			template <typename T>
			ALPHA_SPMD_FUNC void MakeIdenityBufferT(AxStorage<T>* idenityInputBuf, AxUInt64 start = 0, AxInt32 end = -1,AxUInt32 blockSize = 512);
			template <typename T>
			ALPHA_SPMD_FUNC void MakeIdenityBuffer(AxStorage<T>* idenityInputBuf, AxUInt64 start = 0, AxInt32 end = -1, AxUInt32 blockSize = 512);
			template <>
			ALPHA_SPMD_FUNC void MakeIdenityBuffer(AxStorage<AxUInt32>* idenityInputBuf, AxUInt64 start, AxInt32 end, AxUInt32 blockSize);
			template <>
			ALPHA_SPMD_FUNC void MakeIdenityBuffer(AxStorage<AxInt32>* idenityInputBuf, AxUInt64 start, AxInt32 end, AxUInt32 blockSize);
		
			ALPHA_SPMD_FUNC void LerpV3(AxBufferV3* outBuffer, AxBufferV3* a, AxBufferV3* b, AxFp32 t, AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void Vec3AddConstant(AxBufferV3* outBuffer, AxVector3 v3Constant, AxFp32 coeff, AxUInt32 blockSize = 512);

		}

	}
}//@namespace end of : StorageHelper
#endif