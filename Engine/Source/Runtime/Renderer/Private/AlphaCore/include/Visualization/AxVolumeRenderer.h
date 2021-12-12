#pragma once
#include <cstdint>
#include <vector_types.h>
#include <Grid/AxFieldBase3D.h>

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

extern "C" void volume_kernel(AxScalarFieldF32 *field, uchar4 *output, AlphaCore::Desc::AxCameraInfo &camInfo, AlphaCore::Desc::AxPointLightInfo &lightInfo, float stepSize, unsigned int width, unsigned int height);