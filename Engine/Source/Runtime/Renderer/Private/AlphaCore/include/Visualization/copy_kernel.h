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

struct AxVolumeMaterial
{
    AxVector2 minMaxInputDensity;// 0~2
    AxVector2 minMaxOuputDensity;// 0~1
    float LookUpTableDensity[128]; 
    AxColorRGBA8 LookUpTableDensityColor[128];

    AxVector2 minMaxInputHeat;// 0~2
    AxVector2 minMaxOuputHeat;// 0~1
    float LookUpTableHeat[128];

    AxVector2 minMaxInputTemperature;// 0~2
    AxVector2 minMaxOuputTemperature;// 0~1
    AxColorRGBA8 LookUpTableTemperature[128];
};

struct AxVolumeRenderObjectRawData
{
    float* density = nullptr;
    float* heat = nullptr;
    float* temp = nullptr;
    AlphaCore::Desc::AxField3DInfo densityInfo;
    AlphaCore::Desc::AxField3DInfo heatInfo;
    AlphaCore::Desc::AxField3DInfo tempInfo;
};

void volume_kernel(const AxVolumeRenderObjectRawData& volume, const AxVolumeMaterial& material, float4 *worldPos, uchar4 *output, AlphaCore::Desc::AxCameraInfo &camInfo, AlphaCore::Desc::AxPointLightInfo &lightInfo, float stepSize, unsigned int width, unsigned int height);
__host__ __device__ int rayBox(
    const AxVector3 &pivot,
    const AxVector3 &dir,
    const AxVector3 &boxmin,
    const AxVector3 &boxmax,
    float &tnear,
    float &tfar);
__host__ __device__ AxColorRGBA lightMarching(float* fieldRaw, const AlphaCore::Desc::AxField3DInfo& fieldInfo, const AlphaCore::Desc::AxPointLightInfo& lightInfo, const AxVector3& rayPos, const AxVector3& boxMin, const AxVector3& boxMax, float shadowFactor = 1.f);
__host__ __device__ void pixel_render(int x, int y, const AxVolumeRenderObjectRawData& volume, const AxVolumeMaterial& material, float4 *worldPosTex, uchar4 *output, const AlphaCore::Desc::AxCameraInfo& camInfo, const AlphaCore::Desc::AxPointLightInfo& lightInfo, float stepSize, unsigned int width, unsigned int height);
