/*
* Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "DDGIVolumeUpdate.h"

// UE4 public interfaces
#include "CoreMinimal.h"
#include "Engine/TextureRenderTarget2D.h"
#include "SceneView.h"
#include "RenderGraph.h"
#include "RayGenShaderUtils.h"
#include "ShaderParameterStruct.h"
#include "GlobalShader.h"
#include "RTXGIPluginSettings.h"

// UE4 private interfaces
#include "ReflectionEnvironment.h"
#include "FogRendering.h"
#include "SceneRendering.h"
#include "SceneTextureParameters.h"
#include "RayTracing/RayTracingLighting.h"
#include "DeferredShadingRenderer.h"
#include "ScenePrivate.h"

#include <cmath>

// local includes
#include "DDGIVolumeComponent.h"

#define LOCTEXT_NAMESPACE "FRTXGIPlugin"

#if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)
static TAutoConsoleVariable<int> CVarDDGIProbeVis(
    TEXT("r.RTXGI.DDGIProbeVis"),
    0,
    TEXT("If 1, will render what the probes see. If 2, will show misses (blue), hits (green), backfaces (red). \'vis DDGIProbeVis\' to see the output.\n"),
    ECVF_RenderThreadSafe);
#endif

// So we have something to give the "AddPass" calls that we add for transitions
BEGIN_SHADER_PARAMETER_STRUCT(FDummyShaderParameters, )
    SHADER_PARAMETER(int, Dummy)
END_SHADER_PARAMETER_STRUCT()

static FMatrix ComputeRandomRotation()
{
    // This approach is based on James Arvo's implementation from Graphics Gems 3 (pg 117-120).
    // Also available at: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.53.1357&rep=rep1&type=pdf

    // Setup a random rotation matrix using 3 uniform RVs
    float u1 = 2.f * 3.14159265359 * FMath::FRand();
    float cos1 = std::cosf(u1);
    float sin1 = std::sinf(u1);

    float u2 = 2.f * 3.14159265359 * FMath::FRand();
    float cos2 = std::cosf(u2);
    float sin2 = std::sinf(u2);

    float u3 = FMath::FRand();
    float sq3 = 2.f * std::sqrtf(u3 * (1.f - u3));

    float s2 = 2.f * u3 * sin2 * sin2 - 1.f;
    float c2 = 2.f * u3 * cos2 * cos2 - 1.f;
    float sc = 2.f * u3 * sin2 * cos2;

    // Create the random rotation matrix
    float _11 = cos1 * c2 - sin1 * sc;
    float _12 = sin1 * c2 + cos1 * sc;
    float _13 = sq3 * cos2;

    float _21 = cos1 * sc - sin1 * s2;
    float _22 = sin1 * sc + cos1 * s2;
    float _23 = sq3 * sin2;

    float _31 = cos1 * (sq3 * cos2) - sin1 * (sq3 * sin2);
    float _32 = sin1 * (sq3 * cos2) + cos1 * (sq3 * sin2);
    float _33 = 1.f - 2.f * u3;

    return FMatrix(
        FPlane( _11, _12, _13, 0.f ),
        FPlane(_21, _22, _23, 0.f ),
        FPlane(_31, _32, _33, 0.f ),
        FPlane(0.f, 0.f, 0.f, 1.f )
    );
}

class FRayTracingRTXGIProbeUpdateRGS : public FGlobalShader
{
    DECLARE_GLOBAL_SHADER(FRayTracingRTXGIProbeUpdateRGS)
    SHADER_USE_ROOT_PARAMETER_STRUCT(FRayTracingRTXGIProbeUpdateRGS, FGlobalShader)

    class FEnableTwoSidedGeometryDim : SHADER_PERMUTATION_BOOL("ENABLE_TWO_SIDED_GEOMETRY"); // If false, it will cull back face triangles. We want this on for probe relocation and to stop light leak.
    class FEnableMaterialsDim : SHADER_PERMUTATION_BOOL("ENABLE_MATERIALS");                 // If false, forces the geo to opaque (no alpha test). We want this off for speed.
    class FDecoupleSampleGeneration : SHADER_PERMUTATION_BOOL("DECOUPLE_SAMPLE_GENERATION"); // Something for the skylight. We will turn this off
    class FEnableRelocation : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_PROBE_RELOCATION");
    class FDebugFormatRadiance : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_DEBUG_FORMAT_RADIANCE");
    class FEnableScrolling : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_PROBE_SCROLL");

    class FSkyLight : SHADER_PERMUTATION_BOOL("DDGI_SKYLIGHT");

    using FPermutationDomain = TShaderPermutationDomain<FEnableTwoSidedGeometryDim, FEnableMaterialsDim, FDecoupleSampleGeneration, FEnableRelocation, FDebugFormatRadiance, FEnableScrolling, FSkyLight>;

    static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
    {
        FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

        OutEnvironment.SetDefine(TEXT("RTXGI_DDGI_PROBE_STATE_CLASSIFIER"), FDDGIVolumeSceneProxy::FComponentData::c_RTXGI_DDGI_PROBE_STATE_CLASSIFIER ? 1 : 0);

        // Set to 1 to be able to visualize this in the editor by typing "vis DDGIVolumeUpdateDebug" and later "vis none" to make it go away.
        // Set to 0 to disable and deadstrip everything related
        OutEnvironment.SetDefine(TEXT("DDGIVolumeUpdateDebug"), 0);
    }

    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
    {
        return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
    }

    BEGIN_SHADER_PARAMETER_STRUCT(FVolumeData, )
        SHADER_PARAMETER_TEXTURE(Texture2D, ProbeIrradiance)
        SHADER_PARAMETER_TEXTURE(Texture2D, ProbeDistance)
        SHADER_PARAMETER_UAV(RWTexture2D<float4>, ProbeOffsets)
        SHADER_PARAMETER_TEXTURE(Texture2D<uint>, ProbeStates)
        SHADER_PARAMETER(FVector, Origin)
        SHADER_PARAMETER(FVector4, Rotation)
        SHADER_PARAMETER(FVector, ProbeGridSpacing)
        SHADER_PARAMETER(float, ProbeMaxRayDistance)
        SHADER_PARAMETER(FIntVector, ProbeGridCounts)
        SHADER_PARAMETER(FMatrix, ProbeRayRotationTransform)
        SHADER_PARAMETER(int, NumRaysPerProbe)
        SHADER_PARAMETER(FVector, Radius)
        SHADER_PARAMETER(int, ProbeNumIrradianceTexels)
        SHADER_PARAMETER(int, ProbeNumDistanceTexels)
        SHADER_PARAMETER(float, ProbeIrradianceEncodingGamma)
        SHADER_PARAMETER(float, NormalBias)
        SHADER_PARAMETER(float, ViewBias)
        SHADER_PARAMETER(FIntVector, ProbeScrollOffsets)
        SHADER_PARAMETER_SAMPLER(SamplerState, LinearClampSampler)
        SHADER_PARAMETER(float, IrradianceScalar)
        SHADER_PARAMETER(float, EmissiveMultiplier)
        SHADER_PARAMETER(int, ProbeIndexStart)
        SHADER_PARAMETER(int, ProbeIndexCount)
    END_SHADER_PARAMETER_STRUCT()

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
        SHADER_PARAMETER_SRV(RaytracingAccelerationStructure, TLAS)

        SHADER_PARAMETER(uint32, FrameRandomSeed)

        SHADER_PARAMETER_STRUCT(FVolumeData, DDGIVolume)

        SHADER_PARAMETER(FVector, Sky_Color)
        SHADER_PARAMETER_TEXTURE(Texture2D, Sky_Texture)
        SHADER_PARAMETER_SAMPLER(SamplerState, Sky_TextureSampler)

        SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, RadianceOutput)
        SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, DebugOutput)  // Per unreal RDG presentation, this is deadstripped if the shader doesn't write to it

        // assorted things needed by material resolves, even though some don't make sense outside of screenspace
        SHADER_PARAMETER_RDG_TEXTURE(Texture2D, SSProfilesTexture)
        SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, ViewUniformBuffer)
        SHADER_PARAMETER_STRUCT_REF(FRaytracingLightDataPacked, LightDataPacked)
    END_SHADER_PARAMETER_STRUCT()
};

IMPLEMENT_GLOBAL_SHADER(FRayTracingRTXGIProbeUpdateRGS, "/Plugin/RTXGI/Private/ProbeUpdateRGS.usf", "ProbeUpdateRGS", SF_RayGen);

#if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)

class FRayTracingRTXGIProbeViewRGS : public FGlobalShader
{
    DECLARE_GLOBAL_SHADER(FRayTracingRTXGIProbeViewRGS)
    SHADER_USE_ROOT_PARAMETER_STRUCT(FRayTracingRTXGIProbeViewRGS, FGlobalShader)

    class FEnableTwoSidedGeometryDim : SHADER_PERMUTATION_BOOL("ENABLE_TWO_SIDED_GEOMETRY"); // If false, it will cull back face triangles. We want this on for probe relocation and to stop light leak.
    class FEnableMaterialsDim : SHADER_PERMUTATION_BOOL("ENABLE_MATERIALS");                 // If false, forces the geo to opaque (no alpha test). We want this off for speed.
    class FDecoupleSampleGeneration : SHADER_PERMUTATION_BOOL("DECOUPLE_SAMPLE_GENERATION"); // Something for the skylight. We will turn this off
    class FVolumeDebugView : SHADER_PERMUTATION_INT("VOLUME_DEBUG_VIEW", 2);

    using FPermutationDomain = TShaderPermutationDomain<FEnableTwoSidedGeometryDim, FEnableMaterialsDim, FDecoupleSampleGeneration, FVolumeDebugView>;

    static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
    {
        FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

        OutEnvironment.SetDefine(TEXT("RTXGI_DDGI_PROBE_STATE_CLASSIFIER"), FDDGIVolumeSceneProxy::FComponentData::c_RTXGI_DDGI_PROBE_STATE_CLASSIFIER ? 1 : 0);
        OutEnvironment.SetDefine(TEXT("RTXGI_DDGI_PROBE_RELOCATION"), 0);
    }

    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
    {
        return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
    }

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
        SHADER_PARAMETER_SRV(RaytracingAccelerationStructure, TLAS)

        SHADER_PARAMETER(uint32, FrameRandomSeed)

        SHADER_PARAMETER(FVector, CameraPos)
        SHADER_PARAMETER(FMatrix, CameraMatrix)

        SHADER_PARAMETER(float, DDGIVolume_PreExposure)
        SHADER_PARAMETER(int32, DDGIVolume_ShouldUsePreExposure)

        SHADER_PARAMETER(FVector, Sky_Color)
        SHADER_PARAMETER_TEXTURE(Texture2D, Sky_Texture)
        SHADER_PARAMETER_SAMPLER(SamplerState, Sky_TextureSampler)

        SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, RadianceOutput)

        // assorted things needed by material resolves, even though some don't make sense outside of screenspace
        SHADER_PARAMETER_RDG_TEXTURE(Texture2D, SSProfilesTexture)
        SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, ViewUniformBuffer)
        SHADER_PARAMETER_STRUCT_REF(FRaytracingLightDataPacked, LightDataPacked)
    END_SHADER_PARAMETER_STRUCT()
};

IMPLEMENT_GLOBAL_SHADER(FRayTracingRTXGIProbeViewRGS, "/Plugin/RTXGI/Private/ProbeViewRGS.usf", "ProbeViewRGS", SF_RayGen);

#endif // #if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)

class FDDGIIrradianceBlend : public FGlobalShader
{
    DECLARE_GLOBAL_SHADER(FDDGIIrradianceBlend)
    SHADER_USE_PARAMETER_STRUCT(FDDGIIrradianceBlend, FGlobalShader)

    class FRaysPerProbeEnum : SHADER_PERMUTATION_INT("RAYS_PER_PROBE_ENUM", int32(EDDGIRaysPerProbe::Count));
    class FEnableRelocation : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_PROBE_RELOCATION");
    class FDebugFormatRadiance : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_DEBUG_FORMAT_RADIANCE");
    class FEnableScrolling : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_PROBE_SCROLL");

    using FPermutationDomain = TShaderPermutationDomain<FRaysPerProbeEnum, FEnableRelocation, FDebugFormatRadiance, FEnableScrolling>;

    static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
    {
        FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

        OutEnvironment.SetDefine(TEXT("RTXGI_DDGI_PROBE_STATE_CLASSIFIER"), FDDGIVolumeSceneProxy::FComponentData::c_RTXGI_DDGI_PROBE_STATE_CLASSIFIER ? 1 : 0);

        OutEnvironment.SetDefine(TEXT("PROBE_NUM_TEXELS"), FDDGIVolumeSceneProxy::FComponentData::c_NumTexelsIrradiance);
        OutEnvironment.SetDefine(TEXT("RTXGI_DDGI_BLEND_RADIANCE"), 1);

        // Set to 1 to be able to visualize this in the editor by typing "vis DDGIIrradianceBlendDebug" and later "vis none" to make it go away.
        // Set to 0 to disable and deadstrip everything related
        OutEnvironment.SetDefine(TEXT("DDGIIrradianceBlendDebug"), 0);

        // needed for a typed UAV load. This already assumes we are raytracing, so should be fine.
        OutEnvironment.CompilerFlags.Add(CFLAG_AllowTypedUAVLoads);
    }

    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
    {
        return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
    }

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )

        // DDGI volume parameters
        // {
            SHADER_PARAMETER(FIntVector, DDGIVolume_probeGridCounts)
            SHADER_PARAMETER(int, DDGIVolume_numRaysPerProbe)
            SHADER_PARAMETER(FMatrix, DDGIVolume_probeRayRotationTransform)
            SHADER_PARAMETER(FVector, DDGIVolume_probeGridSpacing)
            SHADER_PARAMETER(float, DDGIVolume_probeDistanceExponent)
            SHADER_PARAMETER(float, DDGIVolume_probeInverseIrradianceEncodingGamma)
            SHADER_PARAMETER(float, DDGIVolume_probeHysteresis)
            SHADER_PARAMETER(float, DDGIVolume_probeChangeThreshold)
            SHADER_PARAMETER(float, DDGIVolume_probeBrightnessThreshold)
            SHADER_PARAMETER(FIntVector, DDGIVolume_probeScrollOffsets)
            SHADER_PARAMETER(int, DDGIVolume_ProbeIndexStart)
            SHADER_PARAMETER(int, DDGIVolume_ProbeIndexCount)
        // }

        SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, DDGIProbeRTRadianceUAV)
        SHADER_PARAMETER_UAV(RWTexture2D<uint>, DDGIProbeStates)
        SHADER_PARAMETER_UAV(RWTexture2D<float4>, DDGIProbeUAV)

        SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, DebugOutput)  // Per unreal RDG presentation, this is deadstripped if the shader doesn't write to it

    END_SHADER_PARAMETER_STRUCT()
};

IMPLEMENT_GLOBAL_SHADER(FDDGIIrradianceBlend, "/Plugin/RTXGI/Private/SDK/ddgi/ProbeBlendingCS.usf", "DDGIProbeBlendingCS", SF_Compute);

class FDDGIDistanceBlend : public FGlobalShader
{
    DECLARE_GLOBAL_SHADER(FDDGIDistanceBlend)
    SHADER_USE_PARAMETER_STRUCT(FDDGIDistanceBlend, FGlobalShader)

    class FRaysPerProbeEnum : SHADER_PERMUTATION_INT("RAYS_PER_PROBE_ENUM", int32(EDDGIRaysPerProbe::Count));
    class FEnableRelocation : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_PROBE_RELOCATION");
    class FDebugFormatRadiance : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_DEBUG_FORMAT_RADIANCE");
    class FEnableScrolling : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_PROBE_SCROLL");

    using FPermutationDomain = TShaderPermutationDomain<FRaysPerProbeEnum, FEnableRelocation, FDebugFormatRadiance, FEnableScrolling>;

    static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
    {
        FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

        OutEnvironment.SetDefine(TEXT("RTXGI_DDGI_PROBE_STATE_CLASSIFIER"), FDDGIVolumeSceneProxy::FComponentData::c_RTXGI_DDGI_PROBE_STATE_CLASSIFIER ? 1 : 0);
        OutEnvironment.SetDefine(TEXT("PROBE_NUM_TEXELS"), FDDGIVolumeSceneProxy::FComponentData::c_NumTexelsDistance);
        OutEnvironment.SetDefine(TEXT("RTXGI_DDGI_BLEND_RADIANCE"), 0);

        // Set to 1 to be able to visualize this in the editor by typing "vis DDGIDistanceBlendDebug" and later "vis none" to make it go away.
        // Set to 0 to disable and deadstrip everything related
        OutEnvironment.SetDefine(TEXT("DDGIDistanceBlendDebug"), 0);

        // needed for a typed UAV load. This already assumes we are raytracing, so should be fine.
        OutEnvironment.CompilerFlags.Add(CFLAG_AllowTypedUAVLoads);
    }

    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
    {
        return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
    }

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )

        // DDGI volume parameters
        // {
            SHADER_PARAMETER(FIntVector, DDGIVolume_probeGridCounts)
            SHADER_PARAMETER(int, DDGIVolume_numRaysPerProbe)
            SHADER_PARAMETER(FMatrix, DDGIVolume_probeRayRotationTransform)
            SHADER_PARAMETER(FVector, DDGIVolume_probeGridSpacing)
            SHADER_PARAMETER(float, DDGIVolume_probeDistanceExponent)
            SHADER_PARAMETER(float, DDGIVolume_probeInverseIrradianceEncodingGamma)
            SHADER_PARAMETER(float, DDGIVolume_probeHysteresis)
            SHADER_PARAMETER(float, DDGIVolume_probeChangeThreshold)
            SHADER_PARAMETER(float, DDGIVolume_probeBrightnessThreshold)
            SHADER_PARAMETER(FIntVector, DDGIVolume_probeScrollOffsets)
            SHADER_PARAMETER(int, DDGIVolume_ProbeIndexStart)
            SHADER_PARAMETER(int, DDGIVolume_ProbeIndexCount)
        // }

        SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, DDGIProbeRTRadianceUAV)
        SHADER_PARAMETER_UAV(RWTexture2D<uint>, DDGIProbeStates)
        SHADER_PARAMETER_UAV(RWTexture2D<float4>, DDGIProbeUAV)

        SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, DebugOutput)  // Per unreal RDG presentation, this is deadstripped if the shader doesn't write to it

    END_SHADER_PARAMETER_STRUCT()
};

IMPLEMENT_GLOBAL_SHADER(FDDGIDistanceBlend, "/Plugin/RTXGI/Private/SDK/ddgi/ProbeBlendingCS.usf", "DDGIProbeBlendingCS", SF_Compute);

class FDDGIBorderRowUpdate : public FGlobalShader
{
    DECLARE_GLOBAL_SHADER(FDDGIBorderRowUpdate)
    SHADER_USE_PARAMETER_STRUCT(FDDGIBorderRowUpdate, FGlobalShader)

        static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
    {
        FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

        // needed for a typed UAV load. This already assumes we are raytracing, so should be fine.
        OutEnvironment.CompilerFlags.Add(CFLAG_AllowTypedUAVLoads);
    }

    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
    {
        return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
    }

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
        SHADER_PARAMETER(uint32, ProbeNumIrradianceOrDistanceTexels)
        SHADER_PARAMETER_UAV(RWTexture2D<float4>, DDGIProbeUAV)
    END_SHADER_PARAMETER_STRUCT()
};

IMPLEMENT_GLOBAL_SHADER(FDDGIBorderRowUpdate, "/Plugin/RTXGI/Private/SDK/ddgi/ProbeBorderUpdateCS.usf", "DDGIProbeBorderRowUpdateCS", SF_Compute);

class FDDGIBorderColumnUpdate : public FGlobalShader
{
    DECLARE_GLOBAL_SHADER(FDDGIBorderColumnUpdate)
    SHADER_USE_PARAMETER_STRUCT(FDDGIBorderColumnUpdate, FGlobalShader)

        static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
    {
        FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

        // needed for a typed UAV load. This already assumes we are raytracing, so should be fine.
        OutEnvironment.CompilerFlags.Add(CFLAG_AllowTypedUAVLoads);
    }

    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
    {
        return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
    }

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
        SHADER_PARAMETER(uint32, ProbeNumIrradianceOrDistanceTexels)
        SHADER_PARAMETER_UAV(RWTexture2D<float4>, DDGIProbeUAV)
    END_SHADER_PARAMETER_STRUCT()
};

IMPLEMENT_GLOBAL_SHADER(FDDGIBorderColumnUpdate, "/Plugin/RTXGI/Private/SDK/ddgi/ProbeBorderUpdateCS.usf", "DDGIProbeBorderColumnUpdateCS", SF_Compute);

class FDDGIProbesRelocate : public FGlobalShader
{
    DECLARE_GLOBAL_SHADER(FDDGIProbesRelocate)
    SHADER_USE_PARAMETER_STRUCT(FDDGIProbesRelocate, FGlobalShader)

    class FDebugFormatRadiance : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_DEBUG_FORMAT_RADIANCE");
    class FEnableScrolling : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_PROBE_SCROLL");

    using FPermutationDomain = TShaderPermutationDomain<FDebugFormatRadiance, FEnableScrolling>;

    static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
    {
        FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

        OutEnvironment.SetDefine(TEXT("RTXGI_DDGI_PROBE_STATE_CLASSIFIER"), FDDGIVolumeSceneProxy::FComponentData::c_RTXGI_DDGI_PROBE_STATE_CLASSIFIER ? 1 : 0);
        OutEnvironment.SetDefine(TEXT("RTXGI_DDGI_PROBE_RELOCATION"), 1);

        // needed for a typed UAV load. This already assumes we are raytracing, so should be fine.
        OutEnvironment.CompilerFlags.Add(CFLAG_AllowTypedUAVLoads);
    }

    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
    {
        return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
    }

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
        SHADER_PARAMETER(float, ProbeDistanceScale)
        SHADER_PARAMETER(FVector, DDGIVolume_probeGridSpacing)
        SHADER_PARAMETER(FIntVector, DDGIVolume_probeGridCounts)
        SHADER_PARAMETER(int, DDGIVolume_numRaysPerProbe)
        SHADER_PARAMETER(float, DDGIVolume_probeBackfaceThreshold)
        SHADER_PARAMETER(FMatrix, DDGIVolume_probeRayRotationTransform)
        SHADER_PARAMETER(float, DDGIVolume_probeMinFrontfaceDistance)
        SHADER_PARAMETER(FIntVector, DDGIVolume_ProbeScrollOffsets)
        SHADER_PARAMETER(int, DDGIVolume_ProbeIndexStart)
        SHADER_PARAMETER(int, DDGIVolume_ProbeIndexCount)

        SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, DDGIProbeRTRadianceUAV)
        SHADER_PARAMETER_UAV(RWTexture2D<float4>, DDGIProbeOffsets)
    END_SHADER_PARAMETER_STRUCT()
};

IMPLEMENT_GLOBAL_SHADER(FDDGIProbesRelocate, "/Plugin/RTXGI/Private/SDK/ddgi/ProbeRelocationCS.usf", "DDGIProbeRelocationCS", SF_Compute);

class FDDGIProbesClassify : public FGlobalShader
{
    DECLARE_GLOBAL_SHADER(FDDGIProbesClassify)
    SHADER_USE_PARAMETER_STRUCT(FDDGIProbesClassify, FGlobalShader)

    class FEnableRelocation : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_PROBE_RELOCATION");
    class FDebugFormatRadiance : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_DEBUG_FORMAT_RADIANCE");
    class FEnableScrolling : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_PROBE_SCROLL");

    using FPermutationDomain = TShaderPermutationDomain<FEnableRelocation, FDebugFormatRadiance, FEnableScrolling>;

    static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
    {
        FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

        OutEnvironment.SetDefine(TEXT("RTXGI_DDGI_PROBE_STATE_CLASSIFIER"), 1);

        // needed for a typed UAV load. This already assumes we are raytracing, so should be fine.
        OutEnvironment.CompilerFlags.Add(CFLAG_AllowTypedUAVLoads);
    }

    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
    {
        return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
    }

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
        SHADER_PARAMETER(FVector, DDGIVolume_probeGridSpacing)
        SHADER_PARAMETER(FIntVector, DDGIVolume_probeGridCounts)
        SHADER_PARAMETER(int, DDGIVolume_numRaysPerProbe)
        SHADER_PARAMETER(float, DDGIVolume_probeBackfaceThreshold)
        SHADER_PARAMETER(FIntVector, DDGIVolume_probeScrollOffsets)
        SHADER_PARAMETER(int, DDGIVolume_ProbeIndexStart)
        SHADER_PARAMETER(int, DDGIVolume_ProbeIndexCount)

        SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, DDGIProbeRTRadianceUAV)
        SHADER_PARAMETER_UAV(RWTexture2D<uint>, DDGIProbeStates)
    END_SHADER_PARAMETER_STRUCT()
};

IMPLEMENT_GLOBAL_SHADER(FDDGIProbesClassify, "/Plugin/RTXGI/Private/SDK/ddgi/ProbeStateClassifierCS.usf", "DDGIProbeStateClassifierCS", SF_Compute);

namespace DDGIVolumeUpdate
{
    FDelegateHandle AnyRayTracingPassEnabledHandle;
    FDelegateHandle PrepareRayTracingHandle;

    void DDGIUpdateVolume_RenderThread(const FScene& Scene, const FViewInfo& View, FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy);

#if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)
    void DDGIUpdateVolume_RenderThread_DDGIProbeVis(const FScene& Scene, const FViewInfo& View, FRDGBuilder& GraphBuilder);
#endif 

    void DDGIUpdateVolume_RenderThread_RTRadiance(const FScene& Scene, const FViewInfo& View, FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy, const FMatrix& ProbeRayRotationTransform, FRDGTextureRef ProbesRadianceTex, FRDGTextureUAVRef ProbesRadianceUAV, bool highBitCount);
    void DDGIUpdateVolume_RenderThread_IrradianceBlend(const FViewInfo& View, FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy, const FMatrix& ProbeRayRotationTransform, FRDGTextureUAVRef ProbesRadianceUAV, bool highBitCount);
    void DDGIUpdateVolume_RenderThread_DistanceBlend(const FViewInfo& View, FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy, const FMatrix& ProbeRayRotationTransform, FRDGTextureUAVRef ProbesRadianceUAV, bool highBitCount);
    void DDGIUpdateVolume_RenderThread_IrradianceBorderUpdate(const FViewInfo& View, FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy);
    void DDGIUpdateVolume_RenderThread_DistanceBorderUpdate(const FViewInfo& View, FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy);
    void DDGIUpdateVolume_RenderThread_RelocateProbes(FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy, const FMatrix& ProbeRayRotationTransform, FRDGTextureUAVRef ProbesRadianceUAV, bool highBitCount);
    void DDGIUpdateVolume_RenderThread_ClassifyProbes(FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy, FRDGTextureUAVRef ProbesRadianceUAV, bool highBitCount);

    void PrepareRayTracingShaders(const FViewInfo& View, TArray<FRHIRayTracingShader*>& OutRayGenShaders);

    bool ShouldRenderRayTracingEffect(bool bEffectEnabled)
    {
        if (!IsRayTracingEnabled())
        {
            return false;
        }

        static auto CVar = IConsoleManager::Get().FindConsoleVariable(TEXT("r.RayTracing.ForceAllRayTracingEffects"));
        const int32 OverrideMode = CVar != nullptr ? CVar->GetInt() : -1;

        if (OverrideMode >= 0)
        {
            return OverrideMode > 0;
        }
        else
        {
            return bEffectEnabled;
        }
    }

    // ---------------------- IMPLEMENTATION ------------------

    bool ShouldDynamicUpdate(const FViewInfo& View)
    {
        return ShouldRenderRayTracingEffect(true) && View.RayTracingScene.RayTracingSceneRHI != nullptr;
    }

    void Startup()
    {
        FGlobalIlluminationExperimentalPluginDelegates::FPrepareRayTracing& PRTDelegate = FGlobalIlluminationExperimentalPluginDelegates::PrepareRayTracing();
        PrepareRayTracingHandle = PRTDelegate.AddStatic(PrepareRayTracingShaders);

        FGlobalIlluminationExperimentalPluginDelegates::FAnyRayTracingPassEnabled& ARTPEDelegate = FGlobalIlluminationExperimentalPluginDelegates::AnyRayTracingPassEnabled();
        AnyRayTracingPassEnabledHandle = ARTPEDelegate.AddStatic(
            [](bool& anyEnabled)
            {
                anyEnabled |= true;
            }
        );
    }

    void Shutdown()
    {
        FGlobalIlluminationExperimentalPluginDelegates::FPrepareRayTracing& PRTDelegate = FGlobalIlluminationExperimentalPluginDelegates::PrepareRayTracing();
        check(PrepareRayTracingHandle.IsValid());
        PRTDelegate.Remove(PrepareRayTracingHandle);

        FGlobalIlluminationExperimentalPluginDelegates::FAnyRayTracingPassEnabled& ARTPEDelegate = FGlobalIlluminationExperimentalPluginDelegates::AnyRayTracingPassEnabled();
        check(AnyRayTracingPassEnabledHandle.IsValid());
        ARTPEDelegate.Remove(AnyRayTracingPassEnabledHandle);
    }

    static void CopyTexture_RenderThread(FRHICommandList& RHICmdList, FTexture2DRHIRef dest, FDDGITexturePixels src)
    {
        // Early out if the textures are not compatible
        if (!dest || !src.texture || dest->GetSizeX() != src.w || dest->GetSizeY() != src.h) return;

        // Transition the source texture to a copy source and destination texture to a copy destination
        RHICmdList.Transition(FRHITransitionInfo(src.texture, ERHIAccess::WritableMask, ERHIAccess::CopySrc));
        RHICmdList.Transition(FRHITransitionInfo(dest, ERHIAccess::WritableMask, ERHIAccess::CopyDest));

        RHICmdList.CopyTexture(src.texture, dest, FRHICopyTextureInfo{});

        // Transition the source and destination textures back to general
        RHICmdList.Transition(FRHITransitionInfo(src.texture, ERHIAccess::CopySrc, ERHIAccess::WritableMask));
        RHICmdList.Transition(FRHITransitionInfo(dest, ERHIAccess::CopyDest, ERHIAccess::WritableMask));
    }

    void DDGIUpdatePerFrame_RenderThread(const FScene& Scene, const FViewInfo& View, FRDGBuilder& GraphBuilder)
    {
        // ASSUMES RENDERTHREAD
        check(IsInRenderingThread() || IsInParallelRenderingThread());

#if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)
        DDGIUpdateVolume_RenderThread_DDGIProbeVis(Scene, View, GraphBuilder);
#endif

        // Gather the list of volumes to update
        TArray<FDDGIVolumeSceneProxy*> sceneVolumes;
        float totalPriority = 0.0f;
        for (FDDGIVolumeSceneProxy* proxy : FDDGIVolumeSceneProxy::AllProxiesReadyForRender_RenderThread)
        {
            // Copy the volume's texture data to the GPU, if loading from disk has finished
            if (proxy->TextureLoadContext.ReadyForLoad)
            {
                GraphBuilder.AddPass(
                    RDG_EVENT_NAME("DDGI Volume Texture Loading"),
                    GraphBuilder.AllocParameters<FDummyShaderParameters>(),
                    ERDGPassFlags::Compute | ERDGPassFlags::NeverCull,
                    [proxy](FRHICommandList& RHICmdList)
                    {
                        if (!proxy->TextureLoadContext.ReadyForLoad) return;

                        CopyTexture_RenderThread(RHICmdList, proxy->ProbesIrradianceTex, proxy->TextureLoadContext.Irradiance);
                        CopyTexture_RenderThread(RHICmdList, proxy->ProbesDistanceTex, proxy->TextureLoadContext.Distance);
                        CopyTexture_RenderThread(RHICmdList, proxy->ProbesOffsetsTex, proxy->TextureLoadContext.Offsets);
                        CopyTexture_RenderThread(RHICmdList, proxy->ProbesStatesTex, proxy->TextureLoadContext.States);
                        proxy->TextureLoadContext.Clear();
                    }
                );
            }

            // Don't update the volume if it isn't part of the current scene
            if (proxy->OwningScene != &Scene) continue;

            // Don't update static runtime volumes during gameplay
            if (View.bIsGameView && proxy->ComponentData.RuntimeStatic) continue;

            // Don't update the volume if it is disabled
            if (!proxy->ComponentData.EnableVolume) continue;

            sceneVolumes.Add(proxy);
            totalPriority += proxy->ComponentData.UpdatePriority;
        }

        // Advance the scene's round robin value by the golden ratio (conjugate) and use that 
        // as a "random number" to give each volume a fair turn at recieving an update.
        float& value = FDDGIVolumeSceneProxy::SceneRoundRobinValue.FindOrAdd(&Scene);
        value += 0.61803398875f;
        value -= floor(value);

        // Update the relevant volumes with ray tracing
        float desiredPriority = totalPriority * value;
        for (int index = 0; index < sceneVolumes.Num(); ++index)
        {
            desiredPriority -= sceneVolumes[index]->ComponentData.UpdatePriority;
            if (desiredPriority <= 0.0f || index == sceneVolumes.Num() - 1)
            {
                DDGIUpdateVolume_RenderThread(Scene, View, GraphBuilder, sceneVolumes[index]);
                break;
            }
        }
    }

    void PrepareRayTracingShaders(const FViewInfo& View, TArray<FRHIRayTracingShader*>& OutRayGenShaders)
    {
        const auto FeatureLevel = GMaxRHIFeatureLevel;
        auto ShaderMap = GetGlobalShaderMap(FeatureLevel);

        for (int i = 0; i < 16; ++i)
        {
            FRayTracingRTXGIProbeUpdateRGS::FPermutationDomain PermutationVector;
            PermutationVector.Set<FRayTracingRTXGIProbeUpdateRGS::FEnableTwoSidedGeometryDim>(true);
            PermutationVector.Set<FRayTracingRTXGIProbeUpdateRGS::FEnableMaterialsDim>(false);
            PermutationVector.Set<FRayTracingRTXGIProbeUpdateRGS::FDecoupleSampleGeneration>(false);
            PermutationVector.Set<FRayTracingRTXGIProbeUpdateRGS::FEnableRelocation>((i & 1) != 0 ? true : false);
            PermutationVector.Set<FRayTracingRTXGIProbeUpdateRGS::FDebugFormatRadiance>((i & 2) != 0 ? true : false);
            PermutationVector.Set<FRayTracingRTXGIProbeUpdateRGS::FEnableScrolling>((i & 4) != 0 ? true : false);
            PermutationVector.Set<FRayTracingRTXGIProbeUpdateRGS::FSkyLight>((i & 8) != 0 ? true : false);
            TShaderMapRef<FRayTracingRTXGIProbeUpdateRGS> RayGenerationShader(ShaderMap, PermutationVector);

            OutRayGenShaders.Add(RayGenerationShader.GetRayTracingShader());
        }

        #if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)
        {
            FRayTracingRTXGIProbeViewRGS::FPermutationDomain PermutationVector;
            PermutationVector.Set<FRayTracingRTXGIProbeViewRGS::FEnableTwoSidedGeometryDim>(true);
            PermutationVector.Set<FRayTracingRTXGIProbeViewRGS::FEnableMaterialsDim>(false);
            PermutationVector.Set<FRayTracingRTXGIProbeViewRGS::FDecoupleSampleGeneration>(false);
            PermutationVector.Set<FRayTracingRTXGIProbeViewRGS::FVolumeDebugView>(0);
            TShaderMapRef<FRayTracingRTXGIProbeViewRGS> RayGenerationShader(ShaderMap, PermutationVector);
            OutRayGenShaders.Add(RayGenerationShader.GetRayTracingShader());
        }

        {
            FRayTracingRTXGIProbeViewRGS::FPermutationDomain PermutationVector;
            PermutationVector.Set<FRayTracingRTXGIProbeViewRGS::FEnableTwoSidedGeometryDim>(true);
            PermutationVector.Set<FRayTracingRTXGIProbeViewRGS::FEnableMaterialsDim>(false);
            PermutationVector.Set<FRayTracingRTXGIProbeViewRGS::FDecoupleSampleGeneration>(false);
            PermutationVector.Set<FRayTracingRTXGIProbeViewRGS::FVolumeDebugView>(1);
            TShaderMapRef<FRayTracingRTXGIProbeViewRGS> RayGenerationShader(ShaderMap, PermutationVector);
            OutRayGenShaders.Add(RayGenerationShader.GetRayTracingShader());
        }
        #endif //!(UE_BUILD_SHIPPING || UE_BUILD_TEST)
    }

    void DDGIUpdateVolume_RenderThread(const FScene& Scene, const FViewInfo& View, FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy)
    {
        // Early out if ray tracing is not enabled
        if (!ShouldDynamicUpdate(View)) return;

        bool highBitCount = (GetDefault<URTXGIPluginSettings>()->IrradianceBits == EDDGIIrradianceBits::n32);

        // ASSUMES RENDERTHREAD
        check(IsInRenderingThread() || IsInParallelRenderingThread());
        check(VolProxy);

        FMatrix ProbeRayRotationTransform = ComputeRandomRotation();

        // Create the temporary radiance texture & UAV
        FRDGTextureRef ProbesRadianceTex;
        FRDGTextureUAVRef ProbesRadianceUAV;
        {
            const FDDGIVolumeSceneProxy::FComponentData& ComponentData = VolProxy->ComponentData;
            FRDGTextureDesc DDGIDebugOutputDesc = FRDGTextureDesc::Create2D(
                FIntPoint
                {
                    (int32)ComponentData.GetNumRaysPerProbe(),
                    (int32)ComponentData.ProbeCounts.X * ComponentData.ProbeCounts.Y * ComponentData.ProbeCounts.Z,
                },
                // This texture stores both color and distance
                highBitCount ? FDDGIVolumeSceneProxy::FComponentData::c_pixelFormatRadianceHighBitDepth : FDDGIVolumeSceneProxy::FComponentData::c_pixelFormatRadianceLowBitDepth,
                FClearValueBinding::None,
                TexCreate_ShaderResource | TexCreate_UAV
            );

            ProbesRadianceTex = GraphBuilder.CreateTexture(DDGIDebugOutputDesc, TEXT("DDGIVolumeRadiance"));
            ProbesRadianceUAV = GraphBuilder.CreateUAV(ProbesRadianceTex);
        }

        DDGIUpdateVolume_RenderThread_RTRadiance(Scene, View, GraphBuilder, VolProxy, ProbeRayRotationTransform, ProbesRadianceTex, ProbesRadianceUAV, highBitCount);
        DDGIUpdateVolume_RenderThread_IrradianceBlend(View, GraphBuilder, VolProxy, ProbeRayRotationTransform, ProbesRadianceUAV, highBitCount);
        DDGIUpdateVolume_RenderThread_DistanceBlend(View, GraphBuilder, VolProxy, ProbeRayRotationTransform, ProbesRadianceUAV, highBitCount);
        DDGIUpdateVolume_RenderThread_IrradianceBorderUpdate(View, GraphBuilder, VolProxy);
        DDGIUpdateVolume_RenderThread_DistanceBorderUpdate(View, GraphBuilder, VolProxy);

        if (VolProxy->ComponentData.EnableProbeRelocation)
        {
            DDGIUpdateVolume_RenderThread_RelocateProbes(GraphBuilder, VolProxy, ProbeRayRotationTransform, ProbesRadianceUAV, highBitCount);
        }

        if (FDDGIVolumeSceneProxy::FComponentData::c_RTXGI_DDGI_PROBE_STATE_CLASSIFIER)
        {
            DDGIUpdateVolume_RenderThread_ClassifyProbes(GraphBuilder, VolProxy, ProbesRadianceUAV, highBitCount);
        }
    }

#if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)
    void DDGIUpdateVolume_RenderThread_DDGIProbeVis(const FScene& Scene, const FViewInfo& View, FRDGBuilder& GraphBuilder)
    {
        // Early out if not visualizing probes
        int DDGIProbeVis = CVarDDGIProbeVis.GetValueOnRenderThread();
        if (DDGIProbeVis == 0 || View.RayTracingScene.RayTracingSceneRHI == nullptr) return;

        static const int c_probeVisWidth = 800;
        static const int c_probeVisHeight = 600;

        // create the texture and uav being rendered to
        FRDGTextureDesc ProbeVisTex = FRDGTextureDesc::Create2D(
            FIntPoint(c_probeVisWidth, c_probeVisHeight),
            EPixelFormat::PF_A32B32G32R32F,
            FClearValueBinding::None,
            TexCreate_ShaderResource | TexCreate_UAV
        );
        FRDGTextureUAVRef ProbeVisUAV = GraphBuilder.CreateUAV(GraphBuilder.CreateTexture(ProbeVisTex, TEXT("DDGIProbeVis")));

        // get the shader
        const auto FeatureLevel = GMaxRHIFeatureLevel;
        auto ShaderMap = GetGlobalShaderMap(FeatureLevel);
        FRayTracingRTXGIProbeViewRGS::FPermutationDomain PermutationVector;
        PermutationVector.Set<FRayTracingRTXGIProbeViewRGS::FEnableTwoSidedGeometryDim>(true);
        PermutationVector.Set<FRayTracingRTXGIProbeViewRGS::FEnableMaterialsDim>(false);
        PermutationVector.Set<FRayTracingRTXGIProbeViewRGS::FDecoupleSampleGeneration>(false);
        PermutationVector.Set<FRayTracingRTXGIProbeViewRGS::FVolumeDebugView>(DDGIProbeVis - 1);
        TShaderMapRef<FRayTracingRTXGIProbeViewRGS> RayGenerationShader(ShaderMap, PermutationVector);

        // fill out shader parameters
        FRayTracingRTXGIProbeViewRGS::FParameters DefaultPassParameters;
        FRayTracingRTXGIProbeViewRGS::FParameters* PassParameters = GraphBuilder.AllocParameters<FRayTracingRTXGIProbeViewRGS::FParameters>();
        *PassParameters = DefaultPassParameters;

        PassParameters->DDGIVolume_PreExposure = View.PreExposure;
        PassParameters->DDGIVolume_ShouldUsePreExposure = View.Family->EngineShowFlags.Tonemapper;

        PassParameters->CameraPos = View.ViewMatrices.GetViewOrigin();
        PassParameters->CameraMatrix = View.ViewMatrices.GetViewMatrix().Inverse();

        PassParameters->TLAS = View.RayTracingScene.RayTracingSceneRHI->GetShaderResourceView();
        check(PassParameters->TLAS);
        PassParameters->RadianceOutput = ProbeVisUAV;
        PassParameters->FrameRandomSeed = GFrameNumber;

        // skylight parameters
        if (Scene.SkyLight && Scene.SkyLight->ProcessedTexture)
        {
            PassParameters->Sky_Color = FVector(Scene.SkyLight->GetEffectiveLightColor());
            PassParameters->Sky_Texture = Scene.SkyLight->ProcessedTexture->TextureRHI;
            PassParameters->Sky_TextureSampler = Scene.SkyLight->ProcessedTexture->SamplerStateRHI;
        }
        else
        {
            PassParameters->Sky_Color = FVector(0.0);
            PassParameters->Sky_Texture = GBlackTextureCube->TextureRHI;
            PassParameters->Sky_TextureSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
        }

        PassParameters->SSProfilesTexture = GraphBuilder.RegisterExternalTexture(View.RayTracingSubSurfaceProfileTexture);
        PassParameters->ViewUniformBuffer = View.ViewUniformBuffer;
        PassParameters->LightDataPacked = View.RayTracingLightData.UniformBuffer;

        FIntPoint DispatchSize(c_probeVisWidth, c_probeVisHeight);

        GraphBuilder.AddPass(
            RDG_EVENT_NAME("DDGI RTRadiance %dx%d", DispatchSize.X, DispatchSize.Y),
            PassParameters,
            ERDGPassFlags::Compute,
            [PassParameters, &View, RayGenerationShader, DispatchSize](FRHICommandList& RHICmdList)
            {
                FRayTracingShaderBindingsWriter GlobalResources;
                SetShaderParameters(GlobalResources, RayGenerationShader, *PassParameters);

                FRHIRayTracingScene* RayTracingSceneRHI = View.RayTracingScene.RayTracingSceneRHI;
                RHICmdList.RayTraceDispatch(View.RayTracingMaterialPipeline, RayGenerationShader.GetRayTracingShader(), RayTracingSceneRHI, GlobalResources, DispatchSize.X, DispatchSize.Y);
            }
        );
    }
#endif //!(UE_BUILD_SHIPPING || UE_BUILD_TEST)

    void DDGIUpdateVolume_RenderThread_RTRadiance(const FScene& Scene, const FViewInfo& View, FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy, const FMatrix& ProbeRayRotationTransform, FRDGTextureRef ProbesRadianceTex, FRDGTextureUAVRef ProbesRadianceUAV, bool highBitCount)
    {
        // Deal with probe ray budgets, and updating probes in a round robin fashion within the volume
        int ProbeUpdateRayBudget = GetDefault<URTXGIPluginSettings>()->ProbeUpdateRayBudget;
        if (ProbeUpdateRayBudget == 0)
        {
            VolProxy->ProbeIndexStart = 0;
            VolProxy->ProbeIndexCount = VolProxy->ComponentData.GetProbeCount();
        }
        else
        {
            int ProbeCount = VolProxy->ComponentData.GetProbeCount();
            int ProbeUpdateBudget = ProbeUpdateRayBudget / VolProxy->ComponentData.GetNumRaysPerProbe();
            if (ProbeUpdateBudget < 1)
                ProbeUpdateBudget = 1;
            if (ProbeUpdateBudget > ProbeCount)
                ProbeUpdateBudget = ProbeCount;
            VolProxy->ProbeIndexStart += ProbeUpdateBudget;
            VolProxy->ProbeIndexStart = VolProxy->ProbeIndexStart % ProbeCount;
            VolProxy->ProbeIndexCount = ProbeUpdateBudget;
        }

        const auto FeatureLevel = GMaxRHIFeatureLevel;
        auto ShaderMap = GetGlobalShaderMap(FeatureLevel);

        FRayTracingRTXGIProbeUpdateRGS::FPermutationDomain PermutationVector;
        PermutationVector.Set<FRayTracingRTXGIProbeUpdateRGS::FEnableTwoSidedGeometryDim>(true);
        PermutationVector.Set<FRayTracingRTXGIProbeUpdateRGS::FEnableMaterialsDim>(false);
        PermutationVector.Set<FRayTracingRTXGIProbeUpdateRGS::FDecoupleSampleGeneration>(false);
        PermutationVector.Set<FRayTracingRTXGIProbeUpdateRGS::FEnableRelocation>(VolProxy->ComponentData.EnableProbeRelocation);
        PermutationVector.Set<FRayTracingRTXGIProbeUpdateRGS::FDebugFormatRadiance>(highBitCount);
        PermutationVector.Set<FRayTracingRTXGIProbeUpdateRGS::FEnableScrolling>(VolProxy->ComponentData.EnableProbeScrolling);
        PermutationVector.Set<FRayTracingRTXGIProbeUpdateRGS::FSkyLight>(VolProxy->ComponentData.SkyLight);
        TShaderMapRef<FRayTracingRTXGIProbeUpdateRGS> RayGenerationShader(ShaderMap, PermutationVector);

        FRayTracingRTXGIProbeUpdateRGS::FParameters DefaultPassParameters;
        FRayTracingRTXGIProbeUpdateRGS::FParameters* PassParameters = GraphBuilder.AllocParameters<FRayTracingRTXGIProbeUpdateRGS::FParameters>();
        *PassParameters = DefaultPassParameters;

        PassParameters->TLAS = View.RayTracingScene.RayTracingSceneRHI->GetShaderResourceView();
        check(PassParameters->TLAS);
        PassParameters->RadianceOutput = ProbesRadianceUAV;
        PassParameters->FrameRandomSeed = GFrameNumber;

        // skylight parameters
        if (Scene.SkyLight && Scene.SkyLight->ProcessedTexture)
        {
            PassParameters->Sky_Color = FVector(Scene.SkyLight->GetEffectiveLightColor());
            PassParameters->Sky_Texture = Scene.SkyLight->ProcessedTexture->TextureRHI;
            PassParameters->Sky_TextureSampler = Scene.SkyLight->ProcessedTexture->SamplerStateRHI;
        }
        else
        {
            PassParameters->Sky_Color = FVector(0.0);
            PassParameters->Sky_Texture = GBlackTextureCube->TextureRHI;
            PassParameters->Sky_TextureSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
        }

        // DDGI Volume Parameters
        {
            PassParameters->DDGIVolume.ProbeIrradiance = VolProxy->ProbesIrradianceTex;
            PassParameters->DDGIVolume.ProbeDistance = VolProxy->ProbesDistanceTex;
            PassParameters->DDGIVolume.ProbeOffsets = VolProxy->ProbesOffsetsUAV ? VolProxy->ProbesOffsetsUAV : GBlackTextureWithUAV->UnorderedAccessViewRHI;
            PassParameters->DDGIVolume.ProbeStates = VolProxy->ProbesStatesTex;

            PassParameters->DDGIVolume.Origin = VolProxy->ComponentData.Origin;
            FQuat rotation = VolProxy->ComponentData.Transform.GetRotation();
            PassParameters->DDGIVolume.Rotation = FVector4{ rotation.X, rotation.Y, rotation.Z, rotation.W };
            PassParameters->DDGIVolume.ProbeMaxRayDistance = VolProxy->ComponentData.ProbeMaxRayDistance;
            PassParameters->DDGIVolume.ProbeGridCounts = VolProxy->ComponentData.ProbeCounts;
            PassParameters->DDGIVolume.ProbeRayRotationTransform = ProbeRayRotationTransform;
            PassParameters->DDGIVolume.NumRaysPerProbe = VolProxy->ComponentData.GetNumRaysPerProbe();

            // calculate grid spacing based on size (scale) and probe count
            // regarding the *200: the scale is the radius so we need to double it. There is also an implict * 100 of the basic box.
            FVector volumeSize = VolProxy->ComponentData.Transform.GetScale3D() * 200.0f;
            FVector probeGridSpacing;
            probeGridSpacing.X = volumeSize.X / float(VolProxy->ComponentData.ProbeCounts.X);
            probeGridSpacing.Y = volumeSize.Y / float(VolProxy->ComponentData.ProbeCounts.Y);
            probeGridSpacing.Z = volumeSize.Z / float(VolProxy->ComponentData.ProbeCounts.Z);
            PassParameters->DDGIVolume.ProbeGridSpacing = probeGridSpacing;

            PassParameters->DDGIVolume.Radius = VolProxy->ComponentData.Transform.GetScale3D() * 100.0f;
            PassParameters->DDGIVolume.ProbeNumIrradianceTexels = FDDGIVolumeSceneProxy::FComponentData::c_NumTexelsIrradiance;
            PassParameters->DDGIVolume.ProbeNumDistanceTexels = FDDGIVolumeSceneProxy::FComponentData::c_NumTexelsDistance;
            PassParameters->DDGIVolume.ProbeIrradianceEncodingGamma = VolProxy->ComponentData.ProbeIrradianceEncodingGamma;
            PassParameters->DDGIVolume.NormalBias = VolProxy->ComponentData.NormalBias;
            PassParameters->DDGIVolume.ViewBias = VolProxy->ComponentData.ViewBias;
            PassParameters->DDGIVolume.ProbeScrollOffsets = VolProxy->ComponentData.ProbeScrollOffsets;
            PassParameters->DDGIVolume.IrradianceScalar = VolProxy->ComponentData.IrradianceScalar;
            PassParameters->DDGIVolume.EmissiveMultiplier = VolProxy->ComponentData.EmissiveMultiplier;
            PassParameters->DDGIVolume.ProbeIndexStart = VolProxy->ProbeIndexStart;
            PassParameters->DDGIVolume.ProbeIndexCount = VolProxy->ProbeIndexCount;
            PassParameters->DDGIVolume.LinearClampSampler = TStaticSamplerState<SF_Trilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
        }

        FRDGTextureDesc DDGIDebugOutputDesc = FRDGTextureDesc::Create2D(
            ProbesRadianceTex->Desc.Extent,
            ProbesRadianceTex->Desc.Format,
            FClearValueBinding::None,
            TexCreate_ShaderResource | TexCreate_UAV
        );
        PassParameters->DebugOutput = GraphBuilder.CreateUAV(GraphBuilder.CreateTexture(DDGIDebugOutputDesc, TEXT("DDGIVolumeUpdateDebug")));

        PassParameters->SSProfilesTexture = GraphBuilder.RegisterExternalTexture(View.RayTracingSubSurfaceProfileTexture);
        PassParameters->ViewUniformBuffer = View.ViewUniformBuffer;
        PassParameters->LightDataPacked = View.RayTracingLightData.UniformBuffer;

        FIntPoint DispatchSize = ProbesRadianceTex->Desc.Extent;

        FUnorderedAccessViewRHIRef irradianceUAV = VolProxy->ProbesIrradianceUAV;
        FUnorderedAccessViewRHIRef distanceUAV = VolProxy->ProbesDistanceUAV;

        GraphBuilder.AddPass(
            RDG_EVENT_NAME("DDGI RTRadiance %dx%d", DispatchSize.X, DispatchSize.Y),
            PassParameters,
            ERDGPassFlags::Compute,
            [PassParameters, &View, RayGenerationShader, DispatchSize, irradianceUAV, distanceUAV, ProbesRadianceTex](FRHICommandList& RHICmdList)
            {
                FRayTracingShaderBindingsWriter GlobalResources;
                SetShaderParameters(GlobalResources, RayGenerationShader, *PassParameters);

                FRHIRayTracingScene* RayTracingSceneRHI = View.RayTracingScene.RayTracingSceneRHI;
                RHICmdList.RayTraceDispatch(View.RayTracingMaterialPipeline, RayGenerationShader.GetRayTracingShader(), RayTracingSceneRHI, GlobalResources, DispatchSize.X, DispatchSize.Y);

                // Irradiance and distance are read during the probe update rgs above to give infinite bounce lighting.
                // They are going to be written next based on these results.
                RHICmdList.Transition(FRHITransitionInfo(irradianceUAV, ERHIAccess::ERWBarrier, ERHIAccess::ERWBarrier));
                RHICmdList.Transition(FRHITransitionInfo(distanceUAV, ERHIAccess::ERWBarrier, ERHIAccess::ERWBarrier));
            }
        );
    }

    void DDGIUpdateVolume_RenderThread_IrradianceBlend(const FViewInfo& View, FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy, const FMatrix& ProbeRayRotationTransform, FRDGTextureUAVRef ProbesRadianceUAV, bool highBitCount)
    {
        FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(ERHIFeatureLevel::SM5);
        FDDGIIrradianceBlend::FPermutationDomain PermutationVector;
        PermutationVector.Set<FDDGIIrradianceBlend::FRaysPerProbeEnum>(int(VolProxy->ComponentData.RaysPerProbe));
        PermutationVector.Set<FDDGIIrradianceBlend::FEnableRelocation>(VolProxy->ComponentData.EnableProbeRelocation);
        PermutationVector.Set<FDDGIIrradianceBlend::FDebugFormatRadiance>(highBitCount);
        PermutationVector.Set<FDDGIIrradianceBlend::FEnableScrolling>(VolProxy->ComponentData.EnableProbeScrolling);
        TShaderMapRef<FDDGIIrradianceBlend> ComputeShader(ShaderMap, PermutationVector);

        FDDGIIrradianceBlend::FParameters DefaultPassParameters;
        FDDGIIrradianceBlend::FParameters* PassParameters = GraphBuilder.AllocParameters<FDDGIIrradianceBlend::FParameters>();
        *PassParameters = DefaultPassParameters;

        PassParameters->DDGIVolume_probeGridCounts = VolProxy->ComponentData.ProbeCounts;
        PassParameters->DDGIVolume_numRaysPerProbe = VolProxy->ComponentData.GetNumRaysPerProbe();
        PassParameters->DDGIVolume_probeRayRotationTransform = ProbeRayRotationTransform;
        PassParameters->DDGIVolume_probeDistanceExponent = VolProxy->ComponentData.ProbeDistanceExponent;
        PassParameters->DDGIVolume_probeInverseIrradianceEncodingGamma = 1.0f / VolProxy->ComponentData.ProbeIrradianceEncodingGamma;
        PassParameters->DDGIVolume_probeHysteresis = VolProxy->ComponentData.ProbeHysteresis;
        PassParameters->DDGIVolume_probeChangeThreshold = VolProxy->ComponentData.ProbeChangeThreshold;
        PassParameters->DDGIVolume_probeBrightnessThreshold = VolProxy->ComponentData.ProbeBrightnessThreshold;
        PassParameters->DDGIVolume_probeScrollOffsets = VolProxy->ComponentData.ProbeScrollOffsets;
        PassParameters->DDGIVolume_ProbeIndexStart = VolProxy->ProbeIndexStart;
        PassParameters->DDGIVolume_ProbeIndexCount = VolProxy->ProbeIndexCount;

        // calculate grid spacing based on size (scale) and probe count
        // regarding the *200: the scale is the radius so we need to double it. There is also an implict * 100 of the basic box.
        FVector volumeSize = VolProxy->ComponentData.Transform.GetScale3D() * 200.0f;
        FVector probeGridSpacing;
        probeGridSpacing.X = volumeSize.X / float(VolProxy->ComponentData.ProbeCounts.X);
        probeGridSpacing.Y = volumeSize.Y / float(VolProxy->ComponentData.ProbeCounts.Y);
        probeGridSpacing.Z = volumeSize.Z / float(VolProxy->ComponentData.ProbeCounts.Z);
        PassParameters->DDGIVolume_probeGridSpacing = probeGridSpacing;

        PassParameters->DDGIProbeRTRadianceUAV = ProbesRadianceUAV;
        PassParameters->DDGIProbeUAV = VolProxy->ProbesIrradianceUAV;
        PassParameters->DDGIProbeStates = VolProxy->ProbesStatesUAV ? VolProxy->ProbesStatesUAV : GBlackTextureWithUAV->UnorderedAccessViewRHI;

        FRDGTextureDesc DDGIDebugOutputDesc = FRDGTextureDesc::Create2D(
            VolProxy->ProbesIrradianceTex->GetSizeXY(),
            VolProxy->ProbesIrradianceTex->GetFormat(),
            FClearValueBinding::None,
            TexCreate_ShaderResource | TexCreate_UAV
        );
        PassParameters->DebugOutput = GraphBuilder.CreateUAV(GraphBuilder.CreateTexture(DDGIDebugOutputDesc, TEXT("DDGIIrradianceBlendDebug")));

        FIntPoint ProbeCount2D = VolProxy->ComponentData.Get2DProbeCount();
        FComputeShaderUtils::AddPass(
            GraphBuilder,
            RDG_EVENT_NAME("DDGI Radiance Blend"),
            ComputeShader,
            PassParameters,
            FIntVector(ProbeCount2D.X, ProbeCount2D.Y, 1)
        );
    }

    void DDGIUpdateVolume_RenderThread_DistanceBlend(const FViewInfo& View, FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy, const FMatrix& ProbeRayRotationTransform, FRDGTextureUAVRef ProbesRadianceUAV, bool highBitCount)
    {
        FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(ERHIFeatureLevel::SM5);
        FDDGIDistanceBlend::FPermutationDomain PermutationVector;
        PermutationVector.Set<FDDGIDistanceBlend::FRaysPerProbeEnum>(int(VolProxy->ComponentData.RaysPerProbe));
        PermutationVector.Set<FDDGIDistanceBlend::FEnableRelocation>(int(VolProxy->ComponentData.EnableProbeRelocation));
        PermutationVector.Set<FDDGIDistanceBlend::FDebugFormatRadiance>(highBitCount);
        PermutationVector.Set<FDDGIDistanceBlend::FEnableScrolling>(int(VolProxy->ComponentData.EnableProbeScrolling));
        TShaderMapRef<FDDGIDistanceBlend> ComputeShader(ShaderMap, PermutationVector);

        FDDGIDistanceBlend::FParameters DefaultPassParameters;
        FDDGIDistanceBlend::FParameters* PassParameters = GraphBuilder.AllocParameters<FDDGIDistanceBlend::FParameters>();
        *PassParameters = DefaultPassParameters;

        PassParameters->DDGIVolume_probeGridCounts = VolProxy->ComponentData.ProbeCounts;
        PassParameters->DDGIVolume_numRaysPerProbe = VolProxy->ComponentData.GetNumRaysPerProbe();
        PassParameters->DDGIVolume_probeRayRotationTransform = ProbeRayRotationTransform;
        PassParameters->DDGIVolume_probeDistanceExponent = VolProxy->ComponentData.ProbeDistanceExponent;
        PassParameters->DDGIVolume_probeInverseIrradianceEncodingGamma = 1.0f / VolProxy->ComponentData.ProbeIrradianceEncodingGamma;
        PassParameters->DDGIVolume_probeHysteresis = VolProxy->ComponentData.ProbeHysteresis;
        PassParameters->DDGIVolume_probeChangeThreshold = VolProxy->ComponentData.ProbeChangeThreshold;
        PassParameters->DDGIVolume_probeBrightnessThreshold = VolProxy->ComponentData.ProbeBrightnessThreshold;
        PassParameters->DDGIVolume_probeScrollOffsets = VolProxy->ComponentData.ProbeScrollOffsets;
        PassParameters->DDGIVolume_ProbeIndexStart = VolProxy->ProbeIndexStart;
        PassParameters->DDGIVolume_ProbeIndexCount = VolProxy->ProbeIndexCount;

        // calculate grid spacing based on size (scale) and probe count
        // regarding the *200: the scale is the radius so we need to double it. There is also an implict * 100 of the basic box.
        FVector volumeSize = VolProxy->ComponentData.Transform.GetScale3D() * 200.0f;
        FVector probeGridSpacing;
        probeGridSpacing.X = volumeSize.X / float(VolProxy->ComponentData.ProbeCounts.X);
        probeGridSpacing.Y = volumeSize.Y / float(VolProxy->ComponentData.ProbeCounts.Y);
        probeGridSpacing.Z = volumeSize.Z / float(VolProxy->ComponentData.ProbeCounts.Z);
        PassParameters->DDGIVolume_probeGridSpacing = probeGridSpacing;

        PassParameters->DDGIProbeRTRadianceUAV = ProbesRadianceUAV;
        PassParameters->DDGIProbeUAV = VolProxy->ProbesDistanceUAV;
        PassParameters->DDGIProbeStates = VolProxy->ProbesStatesUAV ? VolProxy->ProbesStatesUAV : GBlackTextureWithUAV->UnorderedAccessViewRHI;

        FRDGTextureDesc DDGIDebugOutputDesc = FRDGTextureDesc::Create2D(
            VolProxy->ProbesDistanceTex->GetSizeXY(),
            VolProxy->ProbesDistanceTex->GetFormat(),
            FClearValueBinding::None,
            TexCreate_ShaderResource | TexCreate_UAV
        );
        PassParameters->DebugOutput = GraphBuilder.CreateUAV(GraphBuilder.CreateTexture(DDGIDebugOutputDesc, TEXT("DDGIDistanceBlendDebug")));

        FIntPoint ProbeCount2D = VolProxy->ComponentData.Get2DProbeCount();
        FComputeShaderUtils::AddPass(
            GraphBuilder,
            RDG_EVENT_NAME("DDGI Distance Blend"),
            ComputeShader,
            PassParameters,
            FIntVector(ProbeCount2D.X, ProbeCount2D.Y, 1)
        );

        // UAV barrier for irradiance and distance uavs so the blends are done before doing border updates
        FUnorderedAccessViewRHIRef irradianceUAV = VolProxy->ProbesIrradianceUAV;
        FUnorderedAccessViewRHIRef distanceUAV = VolProxy->ProbesDistanceUAV;
        GraphBuilder.AddPass(
            RDG_EVENT_NAME("DDGI Pre Border Update UAV Barrier"),
            GraphBuilder.AllocParameters<FDummyShaderParameters>(),
            ERDGPassFlags::Compute | ERDGPassFlags::NeverCull,
            [irradianceUAV, distanceUAV](FRHICommandList& RHICmdList)
            {
                RHICmdList.Transition(FRHITransitionInfo(irradianceUAV, ERHIAccess::ERWBarrier, ERHIAccess::ERWBarrier));
                RHICmdList.Transition(FRHITransitionInfo(distanceUAV, ERHIAccess::ERWBarrier, ERHIAccess::ERWBarrier));
            }
        );
    }

    void DDGIUpdateVolume_RenderThread_IrradianceBorderUpdate(const FViewInfo& View, FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy)
    {
        float groupSize = 8.0f;
        FIntPoint ProbeCount2D = VolProxy->ComponentData.Get2DProbeCount();

        // Row
        {
            FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(ERHIFeatureLevel::SM5);
            TShaderMapRef<FDDGIBorderRowUpdate> ComputeShader(ShaderMap);

            FDDGIBorderRowUpdate::FParameters DefaultPassParameters;
            FDDGIBorderRowUpdate::FParameters* PassParameters = GraphBuilder.AllocParameters<FDDGIBorderRowUpdate::FParameters>();
            *PassParameters = DefaultPassParameters;

            PassParameters->ProbeNumIrradianceOrDistanceTexels = FDDGIVolumeSceneProxy::FComponentData::c_NumTexelsIrradiance;
            PassParameters->DDGIProbeUAV = VolProxy->ProbesIrradianceUAV;

            uint32 numThreadsX = (ProbeCount2D.X * (PassParameters->ProbeNumIrradianceOrDistanceTexels + 2));
            uint32 numThreadsY = ProbeCount2D.Y;
            uint32 numGroupsX = (uint32)ceil((float)numThreadsX / groupSize);
            uint32 numGroupsY = (uint32)ceil((float)numThreadsY / groupSize);

            FComputeShaderUtils::AddPass(
                GraphBuilder,
                RDG_EVENT_NAME("DDGI Irradiance Border Update Row"),
                ComputeShader,
                PassParameters,
                FIntVector(numGroupsX, numGroupsY, 1)
            );
        }

        // Column
        {
            FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(ERHIFeatureLevel::SM5);
            TShaderMapRef<FDDGIBorderColumnUpdate> ComputeShader(ShaderMap);

            FDDGIBorderColumnUpdate::FParameters DefaultPassParameters;
            FDDGIBorderColumnUpdate::FParameters* PassParameters = GraphBuilder.AllocParameters<FDDGIBorderColumnUpdate::FParameters>();
            *PassParameters = DefaultPassParameters;

            PassParameters->ProbeNumIrradianceOrDistanceTexels = FDDGIVolumeSceneProxy::FComponentData::c_NumTexelsIrradiance;
            PassParameters->DDGIProbeUAV = VolProxy->ProbesIrradianceUAV;

            uint32 numThreadsX = (ProbeCount2D.X * 2);
            uint32 numThreadsY = (ProbeCount2D.Y * (PassParameters->ProbeNumIrradianceOrDistanceTexels + 2));
            uint32 numGroupsX = (uint32)ceil((float)numThreadsX / groupSize);
            uint32 numGroupsY = (uint32)ceil((float)numThreadsY / groupSize);

            FComputeShaderUtils::AddPass(
                GraphBuilder,
                RDG_EVENT_NAME("DDGI Irradiance Border Update Column"),
                ComputeShader,
                PassParameters,
                FIntVector(numGroupsX, numGroupsY, 1)
            );
        }
    }

    void DDGIUpdateVolume_RenderThread_DistanceBorderUpdate(const FViewInfo& View, FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy)
    {
        float groupSize = 8.0f;
        FIntPoint ProbeCount2D = VolProxy->ComponentData.Get2DProbeCount();

        // Row
        {
            FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(ERHIFeatureLevel::SM5);
            TShaderMapRef<FDDGIBorderRowUpdate> ComputeShader(ShaderMap);

            FDDGIBorderRowUpdate::FParameters DefaultPassParameters;
            FDDGIBorderRowUpdate::FParameters* PassParameters = GraphBuilder.AllocParameters<FDDGIBorderRowUpdate::FParameters>();
            *PassParameters = DefaultPassParameters;

            PassParameters->ProbeNumIrradianceOrDistanceTexels = FDDGIVolumeSceneProxy::FComponentData::c_NumTexelsDistance;
            PassParameters->DDGIProbeUAV = VolProxy->ProbesDistanceUAV;

            uint32 numThreadsX = (ProbeCount2D.X * (PassParameters->ProbeNumIrradianceOrDistanceTexels + 2));
            uint32 numThreadsY = ProbeCount2D.Y;
            uint32 numGroupsX = (uint32)ceil((float)numThreadsX / groupSize);
            uint32 numGroupsY = (uint32)ceil((float)numThreadsY / groupSize);

            FComputeShaderUtils::AddPass(
                GraphBuilder,
                RDG_EVENT_NAME("DDGI Distance Border Update Row"),
                ComputeShader,
                PassParameters,
                FIntVector(numGroupsX, numGroupsY, 1)
            );
        }

        // Column
        {
            FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(ERHIFeatureLevel::SM5);
            TShaderMapRef<FDDGIBorderColumnUpdate> ComputeShader(ShaderMap);

            FDDGIBorderColumnUpdate::FParameters DefaultPassParameters;
            FDDGIBorderColumnUpdate::FParameters* PassParameters = GraphBuilder.AllocParameters<FDDGIBorderColumnUpdate::FParameters>();
            *PassParameters = DefaultPassParameters;

            PassParameters->ProbeNumIrradianceOrDistanceTexels = FDDGIVolumeSceneProxy::FComponentData::c_NumTexelsDistance;
            PassParameters->DDGIProbeUAV = VolProxy->ProbesDistanceUAV;

            uint32 numThreadsX = (ProbeCount2D.X * 2);
            uint32 numThreadsY = (ProbeCount2D.Y * (PassParameters->ProbeNumIrradianceOrDistanceTexels + 2));
            uint32 numGroupsX = (uint32)ceil((float)numThreadsX / groupSize);
            uint32 numGroupsY = (uint32)ceil((float)numThreadsY / groupSize);

            FComputeShaderUtils::AddPass(
                GraphBuilder,
                RDG_EVENT_NAME("DDGI Distance Border Update Column"),
                ComputeShader,
                PassParameters,
                FIntVector(numGroupsX, numGroupsY, 1)
            );
        }

        // UAV barrier for irradiance and distance uavs so the border updates are done before they are read by the next processes
        FUnorderedAccessViewRHIRef irradianceUAV = VolProxy->ProbesIrradianceUAV;
        FUnorderedAccessViewRHIRef distanceUAV = VolProxy->ProbesDistanceUAV;
        GraphBuilder.AddPass(
            RDG_EVENT_NAME("DDGI Post Border Update UAV Barrier"),
            GraphBuilder.AllocParameters<FDummyShaderParameters>(),
            ERDGPassFlags::Compute | ERDGPassFlags::NeverCull,
            [irradianceUAV, distanceUAV](FRHICommandList& RHICmdList)
            {
                RHICmdList.Transition(FRHITransitionInfo(irradianceUAV, ERHIAccess::ERWBarrier, ERHIAccess::ERWBarrier));
                RHICmdList.Transition(FRHITransitionInfo(distanceUAV, ERHIAccess::ERWBarrier, ERHIAccess::ERWBarrier));
            }
        );
    }

    void DDGIUpdateVolume_RenderThread_RelocateProbes(FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy, const FMatrix& ProbeRayRotationTransform, FRDGTextureUAVRef ProbesRadianceUAV, bool highBitCount)
    {
        FDDGIProbesRelocate::FPermutationDomain PermutationVector;
        PermutationVector.Set<FDDGIProbesRelocate::FDebugFormatRadiance>(highBitCount);
        PermutationVector.Set<FDDGIProbesRelocate::FEnableScrolling>(VolProxy->ComponentData.EnableProbeScrolling);
        FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(ERHIFeatureLevel::SM5);
        TShaderMapRef<FDDGIProbesRelocate> ComputeShader(ShaderMap, PermutationVector);

        FDDGIProbesRelocate::FParameters DefaultPassParameters;
        FDDGIProbesRelocate::FParameters* PassParameters = GraphBuilder.AllocParameters<FDDGIProbesRelocate::FParameters>();
        *PassParameters = DefaultPassParameters;

        // run every frame with full distance scale value for continuous relocation
        PassParameters->ProbeDistanceScale = 1.0f;

        // calculate grid spacing based on size (scale) and probe count
        // regarding the *200: the scale is the radius so we need to double it. There is also an implict * 100 of the basic box.
        FVector volumeSize = VolProxy->ComponentData.Transform.GetScale3D() * 200.0f;
        FVector probeGridSpacing;
        probeGridSpacing.X = volumeSize.X / float(VolProxy->ComponentData.ProbeCounts.X);
        probeGridSpacing.Y = volumeSize.Y / float(VolProxy->ComponentData.ProbeCounts.Y);
        probeGridSpacing.Z = volumeSize.Z / float(VolProxy->ComponentData.ProbeCounts.Z);
        PassParameters->DDGIVolume_probeGridSpacing = probeGridSpacing;

        PassParameters->DDGIVolume_probeGridCounts = VolProxy->ComponentData.ProbeCounts;
        PassParameters->DDGIVolume_numRaysPerProbe = VolProxy->ComponentData.GetNumRaysPerProbe();

        PassParameters->DDGIVolume_ProbeScrollOffsets = VolProxy->ComponentData.ProbeScrollOffsets;
        PassParameters->DDGIVolume_ProbeIndexStart = VolProxy->ProbeIndexStart;
        PassParameters->DDGIVolume_ProbeIndexCount = VolProxy->ProbeIndexCount;

        PassParameters->DDGIVolume_probeBackfaceThreshold = VolProxy->ComponentData.ProbeBackfaceThreshold;
        PassParameters->DDGIVolume_probeRayRotationTransform = ProbeRayRotationTransform;
        PassParameters->DDGIVolume_probeMinFrontfaceDistance = VolProxy->ComponentData.ProbeMinFrontfaceDistance;

        PassParameters->DDGIProbeRTRadianceUAV = ProbesRadianceUAV;
        PassParameters->DDGIProbeOffsets = VolProxy->ProbesOffsetsUAV ? VolProxy->ProbesOffsetsUAV : GBlackTextureWithUAV->UnorderedAccessViewRHI;

        float groupSizeX = 8.f;
        float groupSizeY = 4.f;

        FIntPoint ProbeCount2D = VolProxy->ComponentData.Get2DProbeCount();
        uint32 numThreadsX = ProbeCount2D.X;
        uint32 numThreadsY = ProbeCount2D.Y;
        uint32 numGroupsX = (uint32)ceil((float)numThreadsX / groupSizeX);
        uint32 numGroupsY = (uint32)ceil((float)numThreadsY / groupSizeY);

        FComputeShaderUtils::AddPass(
            GraphBuilder,
            RDG_EVENT_NAME("DDGI Probe Relocation"),
            ComputeShader,
            PassParameters,
            FIntVector(numGroupsX, numGroupsY, 1)
        );

        // UAV barrier for ProbesOffsets
        FUnorderedAccessViewRHIRef offsetsUAV = VolProxy->ProbesOffsetsUAV;
        GraphBuilder.AddPass(
            RDG_EVENT_NAME("DDGI Probe Relocation UAV Barrier"),
            GraphBuilder.AllocParameters<FDummyShaderParameters>(),
            ERDGPassFlags::Compute | ERDGPassFlags::NeverCull,
            [offsetsUAV](FRHICommandList& RHICmdList)
            {
                RHICmdList.Transition(FRHITransitionInfo(offsetsUAV, ERHIAccess::ERWBarrier, ERHIAccess::ERWBarrier));
            }
        );
    }

    void DDGIUpdateVolume_RenderThread_ClassifyProbes(FRDGBuilder& GraphBuilder, FDDGIVolumeSceneProxy* VolProxy, FRDGTextureUAVRef ProbesRadianceUAV, bool highBitCount)
    {
        // get the permuted shader
        FDDGIProbesClassify::FPermutationDomain PermutationVector;
        PermutationVector.Set<FDDGIProbesClassify::FEnableRelocation>(VolProxy->ComponentData.EnableProbeRelocation);
        PermutationVector.Set <FDDGIProbesClassify::FDebugFormatRadiance>(highBitCount);
        PermutationVector.Set <FDDGIProbesClassify::FEnableScrolling>(VolProxy->ComponentData.EnableProbeScrolling);
        FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(ERHIFeatureLevel::SM5);
        TShaderMapRef<FDDGIProbesClassify> ComputeShader(ShaderMap, PermutationVector);

        // set up the shader parameters
        FDDGIProbesClassify::FParameters DefaultPassParameters;
        FDDGIProbesClassify::FParameters* PassParameters = GraphBuilder.AllocParameters<FDDGIProbesClassify::FParameters>();
        *PassParameters = DefaultPassParameters;

        // calculate grid spacing based on size (scale) and probe count
        // regarding the *200: the scale is the radius so we need to double it. There is also an implict * 100 of the basic box.
        FVector volumeSize = VolProxy->ComponentData.Transform.GetScale3D() * 200.0f;
        FVector probeGridSpacing;
        probeGridSpacing.X = volumeSize.X / float(VolProxy->ComponentData.ProbeCounts.X);
        probeGridSpacing.Y = volumeSize.Y / float(VolProxy->ComponentData.ProbeCounts.Y);
        probeGridSpacing.Z = volumeSize.Z / float(VolProxy->ComponentData.ProbeCounts.Z);
        PassParameters->DDGIVolume_probeGridSpacing = probeGridSpacing;

        PassParameters->DDGIVolume_probeGridCounts = VolProxy->ComponentData.ProbeCounts;
        PassParameters->DDGIVolume_numRaysPerProbe = VolProxy->ComponentData.GetNumRaysPerProbe();
        PassParameters->DDGIVolume_probeBackfaceThreshold = VolProxy->ComponentData.ProbeBackfaceThreshold;
        PassParameters->DDGIVolume_probeScrollOffsets = VolProxy->ComponentData.ProbeScrollOffsets;
        PassParameters->DDGIVolume_ProbeIndexStart = VolProxy->ProbeIndexStart;
        PassParameters->DDGIVolume_ProbeIndexCount = VolProxy->ProbeIndexCount;

        PassParameters->DDGIProbeRTRadianceUAV = ProbesRadianceUAV;
        PassParameters->DDGIProbeStates = VolProxy->ProbesStatesUAV;

        // Dispatch the compute shader
        float groupSizeX = 8.f;
        float groupSizeY = 4.f;

        FIntPoint ProbeCount2D = VolProxy->ComponentData.Get2DProbeCount();
        uint32 numThreadsX = ProbeCount2D.X;
        uint32 numThreadsY = ProbeCount2D.Y;
        uint32 numGroupsX = (uint32)ceil((float)numThreadsX / groupSizeX);
        uint32 numGroupsY = (uint32)ceil((float)numThreadsY / groupSizeY);

        FComputeShaderUtils::AddPass(
            GraphBuilder,
            RDG_EVENT_NAME("DDGI Probe Classification"),
            ComputeShader,
            PassParameters,
            FIntVector(numGroupsX, numGroupsY, 1)
        );

        // UAV barrier for ProbesStates
        FUnorderedAccessViewRHIRef statesUAV = VolProxy->ProbesStatesUAV;
        GraphBuilder.AddPass(
            RDG_EVENT_NAME("DDGI Probe Classification UAV Barrier"),
            GraphBuilder.AllocParameters<FDummyShaderParameters>(),
            ERDGPassFlags::Compute|ERDGPassFlags::NeverCull,
            [statesUAV](FRHICommandList& RHICmdList)
            {
                RHICmdList.Transition(FRHITransitionInfo(statesUAV, ERHIAccess::ERWBarrier, ERHIAccess::ERWBarrier));
            }
        );
    }

} // namespace DDGIVolumeUpdate

#undef LOCTEXT_NAMESPACE
