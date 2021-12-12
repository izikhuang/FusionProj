#pragma once
#include <cstdint>
#include <vector_types.h>
#include <Grid/AxFieldBase3D.h>
#include <Utility/AxDescrition.h>

#ifdef DEBUG
#define CUDA_CHECK(val) checkCuda((val), #val, __FILE__, __LINE__)
#else
#define CUDA_CHECK
#endif

template <typename T>
void checkCuda(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result),
                cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

extern "C" void copy_kernel(dim3 blockNum, dim3 threadNum, uchar4* src, uchar4* dst, uint32_t width, uint32_t height);

void volume_kernel(AxScalarFieldF32* field, uchar4* output, AlphaCore::Desc::AxCameraInfo& cam, AlphaCore::Desc::AxPointLightInfo& lightInfo, float stepSize, unsigned int width, unsigned int height);
__host__ __device__ int rayBox(
    const AxVector3 &pivot,
    const AxVector3 &dir,
    const AxVector3 &boxmin,
    const AxVector3 &boxmax,
    float &tnear,
    float &tfar);
__host__ __device__ AxColorRGBA lightMarching(float* fieldRaw, const AlphaCore::Desc::AxField3DInfo& fieldInfo, const AlphaCore::Desc::AxPointLightInfo& lightInfo, const AxVector3& rayPos, const AxVector3& boxMin, const AxVector3& boxMax, float shadowFactor = 1.f);