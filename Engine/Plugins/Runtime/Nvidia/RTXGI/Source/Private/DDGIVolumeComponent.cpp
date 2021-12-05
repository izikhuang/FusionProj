/*
* Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "DDGIVolumeComponent.h"
#include "DDGIVolume.h"
#include "DDGIVolumeUpdate.h"

#include "RTXGIPluginSettings.h"

// UE4 Public Interfaces
#include "ConvexVolume.h"
#include "RenderGraphBuilder.h"
#include "ShaderParameterStruct.h"
#include "ShaderParameterUtils.h"
#include "SystemTextures.h"

// UE4 Private Interfaces
#include "PostProcess/SceneRenderTargets.h"
#include "SceneRendering.h"
#include "DeferredShadingRenderer.h"
#include "ScenePrivate.h"

DECLARE_GPU_STAT_NAMED(RTXGI_Update, TEXT("RTXGI Update"));
DECLARE_GPU_STAT_NAMED(RTXGI_ApplyLighting, TEXT("RTXGI Apply Lighting"));

static TAutoConsoleVariable<bool> CVarUseDDGI(
    TEXT("r.RTXGI.DDGI"),
    true,
    TEXT("If false, this will disable the lighting contribution and functionality of DDGI volumes.\n"),
    ECVF_RenderThreadSafe);

BEGIN_SHADER_PARAMETER_STRUCT(FVolumeData, )
    SHADER_PARAMETER_TEXTURE(Texture2D, ProbeIrradiance)
    SHADER_PARAMETER_TEXTURE(Texture2D, ProbeDistance)
    SHADER_PARAMETER_UAV(RWTexture2D<float4>, ProbeOffsets)
    SHADER_PARAMETER_TEXTURE(Texture2D<uint>, ProbeStates)
    SHADER_PARAMETER(FVector, Position)
    SHADER_PARAMETER(FVector4, Rotation)
    SHADER_PARAMETER(FVector, Radius)
    SHADER_PARAMETER(FVector, ProbeGridSpacing)
    SHADER_PARAMETER(FIntVector, ProbeGridCounts)
    SHADER_PARAMETER(FIntVector, ProbeScrollOffsets)
    SHADER_PARAMETER(uint32, LightingChannelMask)
    SHADER_PARAMETER(int, ProbeNumIrradianceTexels)
    SHADER_PARAMETER(int, ProbeNumDistanceTexels)
    SHADER_PARAMETER(float, ProbeIrradianceEncodingGamma)
    SHADER_PARAMETER(float, NormalBias)
    SHADER_PARAMETER(float, ViewBias)
    SHADER_PARAMETER(float, BlendDistance)
    SHADER_PARAMETER(float, BlendDistanceBlack)
    SHADER_PARAMETER(float, ApplyLighting)
    SHADER_PARAMETER(float, IrradianceScalar)
END_SHADER_PARAMETER_STRUCT()

BEGIN_SHADER_PARAMETER_STRUCT(FApplyLightingDeferredShaderParameters, )
    SHADER_PARAMETER_RDG_TEXTURE(Texture2D, Normals)
    SHADER_PARAMETER_RDG_TEXTURE(Texture2D, Depth)
    SHADER_PARAMETER_RDG_TEXTURE(Texture2D, BaseColor)
    SHADER_PARAMETER_RDG_TEXTURE(Texture2D, Metallic)
    SHADER_PARAMETER_RDG_TEXTURE(Texture2D, LightingChannelsTexture)
    SHADER_PARAMETER(FMatrix, ScreenToTranslatedWorld)
    SHADER_PARAMETER(FVector, WorldCameraOrigin)
    SHADER_PARAMETER(float, PreExposure)
    SHADER_PARAMETER(FVector4, InvDeviceZToWorldZTransform)
    SHADER_PARAMETER(int32, ShouldUsePreExposure)
    SHADER_PARAMETER(int32, NumVolumes)
    // Volumes are sorted from densest probes to least dense probes
    SHADER_PARAMETER_STRUCT_ARRAY(FVolumeData, DDGIVolume, [FDDGIVolumeSceneProxy::FComponentData::c_RTXGI_DDGI_MAX_SHADING_VOLUMES])
    SHADER_PARAMETER_SAMPLER(SamplerState, LinearClampSampler)
    RENDER_TARGET_BINDING_SLOTS()
END_SHADER_PARAMETER_STRUCT()

class FApplyLightingDeferredShaderVS : public FGlobalShader
{
public:
    DECLARE_GLOBAL_SHADER(FApplyLightingDeferredShaderVS);
    SHADER_USE_PARAMETER_STRUCT(FApplyLightingDeferredShaderVS, FGlobalShader);

    using FParameters = FApplyLightingDeferredShaderParameters;

    class FEnableRelocation : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_PROBE_RELOCATION");
    class FEnableScrolling : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_PROBE_SCROLL");

    using FPermutationDomain = TShaderPermutationDomain<FEnableRelocation, FEnableScrolling>;

    static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
    {
        FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

        FString volumeMacroList;
        for (int i = 0; i < FDDGIVolumeSceneProxy::FComponentData::c_RTXGI_DDGI_MAX_SHADING_VOLUMES; ++i)
            volumeMacroList += FString::Printf(TEXT(" VOLUME_ENTRY(%i)"), i);
        OutEnvironment.SetDefine(TEXT("VOLUME_LIST"), volumeMacroList.GetCharArray().GetData());

        OutEnvironment.SetDefine(TEXT("RTXGI_DDGI_PROBE_STATE_CLASSIFIER"), FDDGIVolumeSceneProxy::FComponentData::c_RTXGI_DDGI_PROBE_STATE_CLASSIFIER ? 1 : 0);

        // needed for a typed UAV load. This already assumes we are raytracing, so should be fine.
        OutEnvironment.CompilerFlags.Add(CFLAG_AllowTypedUAVLoads);
    }

    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
    {
        return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
    }
};

class FApplyLightingDeferredShaderPS : public FGlobalShader
{
public:
    DECLARE_GLOBAL_SHADER(FApplyLightingDeferredShaderPS);
    SHADER_USE_PARAMETER_STRUCT(FApplyLightingDeferredShaderPS, FGlobalShader);

    using FParameters = FApplyLightingDeferredShaderParameters;

    class FLightingChannelsDim : SHADER_PERMUTATION_BOOL("USE_LIGHTING_CHANNELS");
    class FEnableRelocation : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_PROBE_RELOCATION");
    class FEnableScrolling : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_PROBE_SCROLL");
    class FDebugFormatRadiance : SHADER_PERMUTATION_BOOL("RTXGI_DDGI_DEBUG_FORMAT_RADIANCE");

    using FPermutationDomain = TShaderPermutationDomain<FLightingChannelsDim, FEnableRelocation, FEnableScrolling, FDebugFormatRadiance>;

    static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
    {
        FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

        FString volumeMacroList;
        for (int i = 0; i < FDDGIVolumeSceneProxy::FComponentData::c_RTXGI_DDGI_MAX_SHADING_VOLUMES; ++i)
            volumeMacroList += FString::Printf(TEXT(" VOLUME_ENTRY(%i)"), i);
        OutEnvironment.SetDefine(TEXT("VOLUME_LIST"), volumeMacroList.GetCharArray().GetData());

        OutEnvironment.SetDefine(TEXT("RTXGI_DDGI_PROBE_STATE_CLASSIFIER"), FDDGIVolumeSceneProxy::FComponentData::c_RTXGI_DDGI_PROBE_STATE_CLASSIFIER ? 1 : 0);

        // needed for a typed UAV load. This already assumes we are raytracing, so should be fine.
        OutEnvironment.CompilerFlags.Add(CFLAG_AllowTypedUAVLoads);
    }

    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
    {
        return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
    }
};

IMPLEMENT_GLOBAL_SHADER(FApplyLightingDeferredShaderVS, "/Plugin/RTXGI/Private/ApplyLightingDeferred.usf", "MainVS", SF_Vertex);
IMPLEMENT_GLOBAL_SHADER(FApplyLightingDeferredShaderPS, "/Plugin/RTXGI/Private/ApplyLightingDeferred.usf", "MainPS", SF_Pixel);

// Delegate Handles
FDelegateHandle FDDGIVolumeSceneProxy::RenderDiffuseIndirectVisualizationsHandle;
FDelegateHandle FDDGIVolumeSceneProxy::RenderDiffuseIndirectLightHandle;

TSet<FDDGIVolumeSceneProxy*> FDDGIVolumeSceneProxy::AllProxiesReadyForRender_RenderThread;
TMap<const FSceneInterface*, float> FDDGIVolumeSceneProxy::SceneRoundRobinValue;

bool FDDGIVolumeSceneProxy::IntersectsViewFrustum(const FViewInfo& View)
{
    // Get the volume position and scale
    FVector ProxyPosition = ComponentData.Origin;
    FQuat   ProxyRotation = ComponentData.Transform.GetRotation();
    FVector ProxyScale = ComponentData.Transform.GetScale3D();
    FVector ProxyExtent = ProxyScale * 100.0f;

    if (ProxyRotation.IsIdentity())
    {
        // This volume is not rotated, test it against the view frustum
        // Skip this volume if it doesn't intersect the view frustum
        return View.ViewFrustum.IntersectBox(ProxyPosition, ProxyExtent);
    }
    else
    {
        // TODO: optimize CPU performance for many volumes (100s to 1000s)

        // This volume is rotated, transform the view frustum so the volume's
        // oriented bounding box becomes an axis-aligned bounding box.
        FConvexVolume TransformedViewFrustum;
        FMatrix FrustumTransform = FTranslationMatrix::Make(-ProxyPosition)
            * FRotationMatrix::Make(ProxyRotation)
            * FTranslationMatrix::Make(ProxyPosition);

        // Based on SetupViewFrustum()
        if (View.SceneViewInitOptions.OverrideFarClippingPlaneDistance > 0.0f)
        {
            FVector PlaneBasePoint = FrustumTransform.TransformPosition(View.ViewMatrices.GetViewOrigin() + View.GetViewDirection() * View.SceneViewInitOptions.OverrideFarClippingPlaneDistance);
            FVector PlaneNormal = FrustumTransform.TransformVector(View.GetViewDirection());

            const FPlane FarPlane(PlaneBasePoint, PlaneNormal);
            // Derive the view frustum from the view projection matrix, overriding the far plane
            GetViewFrustumBounds(TransformedViewFrustum, FrustumTransform * View.ViewMatrices.GetViewProjectionMatrix(), FarPlane, true, false);
        }
        else
        {
            // Derive the view frustum from the view projection matrix.
            GetViewFrustumBounds(TransformedViewFrustum, FrustumTransform * View.ViewMatrices.GetViewProjectionMatrix(), false);
        }

        // Test the transformed view frustum against the volume
        // Skip this volume if it doesn't intersect the view frustum
        return TransformedViewFrustum.IntersectBox(ProxyPosition, ProxyExtent);
    }
}

void FDDGIVolumeSceneProxy::OnIrradianceOrDistanceBitsChange()
{
    EDDGIIrradianceBits IrradianceBits = GetDefault<URTXGIPluginSettings>()->IrradianceBits;
    EDDGIDistanceBits DistanceBits = GetDefault<URTXGIPluginSettings>()->DistanceBits;

    // tell all the proxies about the change
    ENQUEUE_RENDER_COMMAND(DDGIOnIrradianceBitsChange)(
        [IrradianceBits, DistanceBits](FRHICommandListImmediate& RHICmdList)
        {
            for (FDDGIVolumeSceneProxy* DDGIProxy : AllProxiesReadyForRender_RenderThread)
            {
                DDGIProxy->ReallocateSurfaces_RenderThread(RHICmdList, IrradianceBits, DistanceBits);
                DDGIProxy->ResetTextures_RenderThread(RHICmdList);
            }
        }
    );
}

void FDDGIVolumeSceneProxy::ReallocateSurfaces_RenderThread(FRHICommandListImmediate& RHICmdList, EDDGIIrradianceBits IrradianceBits, EDDGIDistanceBits DistanceBits)
{
    FIntPoint ProxyDims = ComponentData.Get2DProbeCount();

    // Irradiance
    {
        int numTexels = FDDGIVolumeSceneProxy::FComponentData::c_NumTexelsIrradiance;
        FIntPoint ProxyTexDims = ProxyDims * (numTexels + 2);
        FRHIResourceCreateInfo createInfo(TEXT("DDGIIrradiance"));
        ProbesIrradianceTex = RHICreateTexture2D(ProxyTexDims.X, ProxyTexDims.Y, (IrradianceBits == EDDGIIrradianceBits::n32 ) ? FDDGIVolumeSceneProxy::FComponentData::c_pixelFormatIrradianceHighBitDepth : FDDGIVolumeSceneProxy::FComponentData::c_pixelFormatIrradianceLowBitDepth, 1, 1, TexCreate_ShaderResource | TexCreate_UAV, createInfo);
        ProbesIrradianceUAV = RHICreateUnorderedAccessView(ProbesIrradianceTex, 0);
    }

    // Distance
    {
        int numTexels = FDDGIVolumeSceneProxy::FComponentData::c_NumTexelsDistance;
        FIntPoint ProxyTexDims = ProxyDims * (numTexels + 2);
        FRHIResourceCreateInfo createInfo(TEXT("DDGIDistance"));
        ProbesDistanceTex = RHICreateTexture2D(ProxyTexDims.X, ProxyTexDims.Y, (DistanceBits == EDDGIDistanceBits::n32) ? FDDGIVolumeSceneProxy::FComponentData::c_pixelFormatDistanceHighBitDepth : FDDGIVolumeSceneProxy::FComponentData::c_pixelFormatDistanceLowBitDepth, 1, 1, TexCreate_ShaderResource | TexCreate_UAV, createInfo);
        ProbesDistanceUAV = RHICreateUnorderedAccessView(ProbesDistanceTex, 0);
    }

    // Offsets - only pay the cost of this resource if this volume is actually doing relocation
    if (ComponentData.EnableProbeRelocation)
    {
        FRHIResourceCreateInfo createInfo(TEXT("DDGIOffsets"));
        ProbesOffsetsTex = RHICreateTexture2D(ProxyDims.X, ProxyDims.Y, FDDGIVolumeSceneProxy::FComponentData::c_pixelFormatOffsets, 1, 1, TexCreate_ShaderResource | TexCreate_UAV, createInfo);
        ProbesOffsetsUAV = RHICreateUnorderedAccessView(ProbesOffsetsTex, 0);
    }
    else
    {
        ProbesOffsetsTex = nullptr;
        ProbesOffsetsUAV = nullptr;
    }

    // probe classifications
    if (FDDGIVolumeSceneProxy::FComponentData::c_RTXGI_DDGI_PROBE_STATE_CLASSIFIER)
    {
        FRHIResourceCreateInfo createInfo(TEXT("DDGIStates"));
        ProbesStatesTex = RHICreateTexture2D(ProxyDims.X, ProxyDims.Y, FDDGIVolumeSceneProxy::FComponentData::c_pixelFormatStates, 1, 1, TexCreate_ShaderResource | TexCreate_UAV, createInfo);
        ProbesStatesUAV = RHICreateUnorderedAccessView(ProbesStatesTex, 0);
    }
    else
    {
        ProbesStatesTex = nullptr;
        ProbesStatesUAV = nullptr;
    }
}

void FDDGIVolumeSceneProxy::ResetTextures_RenderThread(FRHICommandListImmediate& RHICmdList)
{
    // reset textures to pristine initial state
    RHICmdList.ClearUAVFloat(ProbesIrradianceUAV, FVector4{ 0.0f, 0.0f, 0.0f, 0.0f });
    RHICmdList.ClearUAVFloat(ProbesDistanceUAV, FVector4{ 0.0f, 0.0f, 0.0f, 0.0f });

    if (ProbesOffsetsUAV)
        RHICmdList.ClearUAVFloat(ProbesOffsetsUAV, FVector4{ 0.0f, 0.0f, 0.0f, 0.0f });

    if (ProbesStatesUAV)
        RHICmdList.ClearUAVUint(ProbesStatesUAV, FUintVector4{ 0, 0, 0, 0 });
}

void FDDGIVolumeSceneProxy::RenderDiffuseIndirectLight_RenderThread(
    const FScene& Scene,
    const FViewInfo& View,
    FRDGBuilder& GraphBuilder,
    FGlobalIlluminationExperimentalPluginResources& Resources)
{
    // Early out if DDGI is disabled
    if (!CVarUseDDGI.GetValueOnRenderThread()) return;

    // Update DDGIVolumes when rendering a main view and when ray tracing is available.
    // Other views can use DDGIVolumes for lighting, but don't need to update the volumes.
    // This is especially true for situations like bIsSceneCapture, when bSceneCaptureUsesRayTracing is false, and it can make incorrect probe update results.
    if (!View.bIsSceneCapture && !View.bIsReflectionCapture && !View.bIsPlanarReflection)
    {
        RDG_GPU_STAT_SCOPE(GraphBuilder, RTXGI_Update);
        RDG_EVENT_SCOPE(GraphBuilder, "RTXGI Update");
        DDGIVolumeUpdate::DDGIUpdatePerFrame_RenderThread(Scene, View, GraphBuilder);
    }

    {
        RDG_GPU_STAT_SCOPE(GraphBuilder, RTXGI_ApplyLighting);
        RDG_EVENT_SCOPE(GraphBuilder, "RTXGI Apply Lighting");

        // DDGIVolume and useful metadata
        struct FProxyEntry
        {
            FVector Position;
            FQuat Rotation;
            FVector Scale;
            float Density;
            uint32 lightingChannelMask;
            const FDDGIVolumeSceneProxy* proxy;
        };

        // Find all the volumes that intersect the view frustum
        TArray<FProxyEntry> volumes;
        for (FDDGIVolumeSceneProxy* volumeProxy : AllProxiesReadyForRender_RenderThread)
        {
            // Skip this volume if it belongs to another scene
            if (volumeProxy->OwningScene != &Scene) continue;

            // Skip this volume if it is not enabled
            if (!volumeProxy->ComponentData.EnableVolume) continue;

            // Skip this volume if it doesn't intersect the view frustum
            if (!volumeProxy->IntersectsViewFrustum(View)) continue;

            // Get the volume position, rotation, and scale
            FVector ProxyPosition = volumeProxy->ComponentData.Origin;
            FQuat   ProxyRotation = volumeProxy->ComponentData.Transform.GetRotation();
            FVector ProxyScale = volumeProxy->ComponentData.Transform.GetScale3D();

            float ProxyDensity = float(volumeProxy->ComponentData.ProbeCounts.X * volumeProxy->ComponentData.ProbeCounts.Y * volumeProxy->ComponentData.ProbeCounts.Z) / (ProxyScale.X * ProxyScale.Y * ProxyScale.Z);
            uint32 ProxyLightingChannelMask =
                (volumeProxy->ComponentData.LightingChannels.bChannel0 ? 1 : 0) |
                (volumeProxy->ComponentData.LightingChannels.bChannel1 ? 2 : 0) |
                (volumeProxy->ComponentData.LightingChannels.bChannel2 ? 4 : 0);

            // Add the current volume to the list of in-frustum volumes
            volumes.Add(FProxyEntry{ ProxyPosition, ProxyRotation, ProxyScale, ProxyDensity, ProxyLightingChannelMask, volumeProxy });
        }

        // Early out if no volumes contribute light to the current view
        if (volumes.Num() == 0) return;

        // TODO: manage in-frustum volumes in a more sophisticated way
        // Support a large number of volumes by culling volumes based on spatial data, projected view area, and/or other heuristics

        // Sort the in-frustum volumes by user specified priority
        Algo::Sort(volumes, [](const FProxyEntry& A, const FProxyEntry& B)
        {
            if (A.proxy->ComponentData.LightingPriority < B.proxy->ComponentData.LightingPriority) return true;
            if ((A.proxy->ComponentData.LightingPriority == B.proxy->ComponentData.LightingPriority) && (A.Density > B.Density)) return true;
            return false;
        });

        // Get the number of relevant in-frustum volumes
        int32 numVolumes = FMath::Min(volumes.Num(), FDDGIVolumeSceneProxy::FComponentData::c_RTXGI_DDGI_MAX_SHADING_VOLUMES);

        // Truncate the in-frustum volumes list to the maximum number of volumes supported
        volumes.SetNum(numVolumes, true);

        // Sort the final volume list by descending probe density
        Algo::Sort(volumes, [](const FProxyEntry& A, const FProxyEntry& B)
        {
            return (A.Density > B.Density);
        });

        // Register the GBuffer textures with the render graph
        FRDGTextureRef GBufferATexture = GraphBuilder.RegisterExternalTexture(Resources.GBufferA);
        FRDGTextureRef GBufferBTexture = GraphBuilder.RegisterExternalTexture(Resources.GBufferB);
        FRDGTextureRef GBufferCTexture = GraphBuilder.RegisterExternalTexture(Resources.GBufferC);
        FRDGTextureRef SceneDepthTexture = GraphBuilder.RegisterExternalTexture(Resources.SceneDepthZ);
        FRDGTextureRef SceneColorTexture = GraphBuilder.RegisterExternalTexture(Resources.SceneColor);
        if (!View.bUsesLightingChannels) Resources.LightingChannelsTexture = nullptr;

        // Precompute the inverse view-projection transformation matrix
        FMatrix ScreenToTranslatedWorld = FMatrix(
            FPlane(1, 0, 0, 0),
            FPlane(0, 1, 0, 0),
            FPlane(0, 0, View.ProjectionMatrixUnadjustedForRHI.M[2][2], 1),
            FPlane(0, 0, View.ProjectionMatrixUnadjustedForRHI.M[3][2], 0))
            * View.ViewMatrices.GetInvTranslatedViewProjectionMatrix();

        // Loop over the shader permutations to render indirect light from relevant volumes
        for (int permutationIndex = 0; permutationIndex < 4; ++permutationIndex)
        {
            // Render with the current shader permutation if there one (or more) volume that matches the permutation settings
            bool enableRelocation = (permutationIndex & 1) != 0;
            bool enableScrolling = (permutationIndex & 2) != 0;
            bool foundAMatch = false;
            for (int32 i = 0; i < volumes.Num(); ++i)
            {
                foundAMatch = true;
                foundAMatch = foundAMatch && (enableRelocation == volumes[i].proxy->ComponentData.EnableProbeRelocation);
                foundAMatch = foundAMatch && (enableScrolling == volumes[i].proxy->ComponentData.EnableProbeScrolling);
                if (foundAMatch) break;
            }

            // Skip this shader permutation if there are no volumes that match its feature set
            if (!foundAMatch) continue;

            // Get the vertex shader permutation
            FGlobalShaderMap* GlobalShaderMap = GetGlobalShaderMap(ERHIFeatureLevel::SM5);
            FApplyLightingDeferredShaderVS::FPermutationDomain PermutationVectorVS;
            PermutationVectorVS.Set<FApplyLightingDeferredShaderVS::FEnableRelocation>(enableRelocation);
            PermutationVectorVS.Set<FApplyLightingDeferredShaderVS::FEnableScrolling>(enableScrolling);
            TShaderMapRef<FApplyLightingDeferredShaderVS> VertexShader(GlobalShaderMap, PermutationVectorVS);

            // Get the pixel shader permutation
            bool highBitCount = (GetDefault<URTXGIPluginSettings>()->IrradianceBits == EDDGIIrradianceBits::n32);
            FApplyLightingDeferredShaderPS::FPermutationDomain PermutationVectorPS;
            PermutationVectorPS.Set<FApplyLightingDeferredShaderPS::FLightingChannelsDim>(Resources.LightingChannelsTexture != nullptr);
            PermutationVectorPS.Set<FApplyLightingDeferredShaderPS::FEnableRelocation>(enableRelocation);
            PermutationVectorPS.Set<FApplyLightingDeferredShaderPS::FEnableScrolling>(enableScrolling);
            PermutationVectorPS.Set<FApplyLightingDeferredShaderPS::FDebugFormatRadiance>(highBitCount);
            TShaderMapRef<FApplyLightingDeferredShaderPS> PixelShader(GlobalShaderMap, PermutationVectorPS);

            // Set the shader parameters
            FApplyLightingDeferredShaderParameters DefaultPassParameters;
            FApplyLightingDeferredShaderParameters* PassParameters = GraphBuilder.AllocParameters<FApplyLightingDeferredShaderParameters>();
            *PassParameters = DefaultPassParameters;
            PassParameters->Normals = GBufferATexture;
            PassParameters->Depth = SceneDepthTexture;
            PassParameters->BaseColor = GBufferCTexture;
            PassParameters->Metallic = GBufferBTexture;
            PassParameters->LightingChannelsTexture = Resources.LightingChannelsTexture;
            PassParameters->ScreenToTranslatedWorld = ScreenToTranslatedWorld;
            PassParameters->WorldCameraOrigin = View.ViewMatrices.GetViewOrigin();
            PassParameters->InvDeviceZToWorldZTransform = View.InvDeviceZToWorldZTransform;
            PassParameters->RenderTargets[0] = FRenderTargetBinding(SceneColorTexture, ERenderTargetLoadAction::ELoad);
            PassParameters->LinearClampSampler = TStaticSamplerState<SF_Trilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
            PassParameters->ShouldUsePreExposure = View.Family->EngineShowFlags.Tonemapper;
            PassParameters->PreExposure = View.PreExposure;
            PassParameters->NumVolumes = numVolumes;

            // Set the shader parameters for the relevant volumes
            for (int32 volumeIndex = 0; volumeIndex < numVolumes; ++volumeIndex)
            {
                FProxyEntry volume = volumes[volumeIndex];
                const FDDGIVolumeSceneProxy* volumeProxy = volume.proxy;

                // Set the volume textures
                PassParameters->DDGIVolume[volumeIndex].ProbeIrradiance = volumeProxy->ProbesIrradianceTex;
                PassParameters->DDGIVolume[volumeIndex].ProbeDistance = volumeProxy->ProbesDistanceTex;
                PassParameters->DDGIVolume[volumeIndex].ProbeOffsets = volumeProxy->ProbesOffsetsUAV ? volumeProxy->ProbesOffsetsUAV : GBlackTextureWithUAV->UnorderedAccessViewRHI;
                PassParameters->DDGIVolume[volumeIndex].ProbeStates = volumeProxy->ProbesStatesTex;

                // Set the volume parameters
                PassParameters->DDGIVolume[volumeIndex].Position = volumeProxy->ComponentData.Origin;
                PassParameters->DDGIVolume[volumeIndex].Rotation = FVector4(volume.Rotation.X, volume.Rotation.Y, volume.Rotation.Z, volume.Rotation.W);
                PassParameters->DDGIVolume[volumeIndex].Radius = volume.Scale * 100.0f;
                PassParameters->DDGIVolume[volumeIndex].LightingChannelMask = volume.lightingChannelMask;

                FVector volumeSize = volumeProxy->ComponentData.Transform.GetScale3D() * 200.0f;
                FVector probeGridSpacing;
                probeGridSpacing.X = volumeSize.X / float(volumeProxy->ComponentData.ProbeCounts.X);
                probeGridSpacing.Y = volumeSize.Y / float(volumeProxy->ComponentData.ProbeCounts.Y);
                probeGridSpacing.Z = volumeSize.Z / float(volumeProxy->ComponentData.ProbeCounts.Z);

                PassParameters->DDGIVolume[volumeIndex].ProbeGridSpacing = probeGridSpacing;
                PassParameters->DDGIVolume[volumeIndex].ProbeGridCounts = volumeProxy->ComponentData.ProbeCounts;
                PassParameters->DDGIVolume[volumeIndex].ProbeNumIrradianceTexels = FDDGIVolumeSceneProxy::FComponentData::c_NumTexelsIrradiance;
                PassParameters->DDGIVolume[volumeIndex].ProbeNumDistanceTexels = FDDGIVolumeSceneProxy::FComponentData::c_NumTexelsDistance;
                PassParameters->DDGIVolume[volumeIndex].ProbeIrradianceEncodingGamma = volumeProxy->ComponentData.ProbeIrradianceEncodingGamma;
                PassParameters->DDGIVolume[volumeIndex].NormalBias = volumeProxy->ComponentData.NormalBias;
                PassParameters->DDGIVolume[volumeIndex].ViewBias = volumeProxy->ComponentData.ViewBias;
                PassParameters->DDGIVolume[volumeIndex].BlendDistance = volumeProxy->ComponentData.BlendDistance;
                PassParameters->DDGIVolume[volumeIndex].BlendDistanceBlack = volumeProxy->ComponentData.BlendDistanceBlack;
                PassParameters->DDGIVolume[volumeIndex].ProbeScrollOffsets = volumeProxy->ComponentData.ProbeScrollOffsets;

                // Only apply lighting if this is the pass it should be applied in
                // The shader needs data for all of the volumes for blending purposes
                bool applyLighting = true;
                applyLighting = applyLighting && (enableRelocation == volumeProxy->ComponentData.EnableProbeRelocation);
                applyLighting = applyLighting && (enableScrolling == volumeProxy->ComponentData.EnableProbeScrolling);
                PassParameters->DDGIVolume[volumeIndex].ApplyLighting = applyLighting;
                PassParameters->DDGIVolume[volumeIndex].IrradianceScalar = volumeProxy->ComponentData.IrradianceScalar;

                // Apply the lighting multiplier to artificially lighten or darken the indirect light from the volume
                PassParameters->DDGIVolume[volumeIndex].IrradianceScalar /= volumeProxy->ComponentData.LightingMultiplier;
            }

            // When there are fewer relevant volumes than the maximum supported, set the empty volume texture slots to dummy values
            for (int32 volumeIndex = numVolumes; volumeIndex < FDDGIVolumeSceneProxy::FComponentData::c_RTXGI_DDGI_MAX_SHADING_VOLUMES; ++volumeIndex)
            {
                PassParameters->DDGIVolume[volumeIndex].ProbeIrradiance = GSystemTextures.BlackDummy->GetRenderTargetItem().TargetableTexture;
                PassParameters->DDGIVolume[volumeIndex].ProbeDistance = GSystemTextures.BlackDummy->GetRenderTargetItem().TargetableTexture;
                PassParameters->DDGIVolume[volumeIndex].ProbeOffsets = GBlackTextureWithUAV->UnorderedAccessViewRHI;
                PassParameters->DDGIVolume[volumeIndex].ProbeStates = GSystemTextures.BlackDummy->GetRenderTargetItem().TargetableTexture;
            }

            // Dispatch the fullscreen shader
            FIntRect ViewRect = View.ViewRect;
            GraphBuilder.AddPass(
                Forward<FRDGEventName>(RDG_EVENT_NAME("DDGI Apply Lighting")),
                PassParameters,
                ERDGPassFlags::Raster,
                [PassParameters, GlobalShaderMap, VertexShader, PixelShader, ViewRect](FRHICommandList& RHICmdList)
                {
                    RHICmdList.SetViewport(ViewRect.Min.X, ViewRect.Min.Y, 0.0f, ViewRect.Max.X, ViewRect.Max.Y, 1.0f);

                    FGraphicsPipelineStateInitializer GraphicsPSOInit;
                    RHICmdList.ApplyCachedRenderTargets(GraphicsPSOInit);
                    GraphicsPSOInit.BlendState = TStaticBlendState<CW_RGB, BO_Add, BF_One, BF_One, BO_Add, BF_One, BF_One>::GetRHI();
                    GraphicsPSOInit.RasterizerState = TStaticRasterizerState<>::GetRHI();
                    GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<false, CF_Always>::GetRHI();

                    GraphicsPSOInit.BoundShaderState.VertexDeclarationRHI = GetVertexDeclarationFVector4();
                    GraphicsPSOInit.BoundShaderState.VertexShaderRHI = VertexShader.GetVertexShader();
                    GraphicsPSOInit.BoundShaderState.PixelShaderRHI = PixelShader.GetPixelShader();
                    GraphicsPSOInit.PrimitiveType = PT_TriangleList;

                    SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit);
                    RHICmdList.SetStencilRef(0);

                    SetShaderParameters(RHICmdList, VertexShader, VertexShader.GetVertexShader(), *PassParameters);
                    SetShaderParameters(RHICmdList, PixelShader, PixelShader.GetPixelShader(), *PassParameters);

                    RHICmdList.DrawPrimitive(0, 1, 1);
                }
            );
        }
    }
}

UDDGIVolumeComponent::UDDGIVolumeComponent(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer)
{
    bWantsInitializeComponent = true;
}

void UDDGIVolumeComponent::InitializeComponent()
{
    Super::InitializeComponent();

    UpdateRenderThreadData();

    TransformUpdated.AddLambda(
        [this](USceneComponent* /*UpdatedComponent*/, EUpdateTransformFlags /*UpdateTransformFlags*/, ETeleportType /*Teleport*/)
        {
            UpdateRenderThreadData();
        }
    );
}

// Serialization version for stored DDGIVolume data
struct RTXGI_API FDDGICustomVersion
{
    enum Type
    {
        AddingCustomVersion = 1,
        SaveLoadProbeTextures,     // save pixels and width/height
        SaveLoadProbeTexturesFmt,  // save texel format since the format can change in the project settings
    };

    // The GUID for this custom version number
    const static FGuid GUID;

private:
    FDDGICustomVersion() {}
};
const FGuid FDDGICustomVersion::GUID(0xc12f0537, 0x7346d9c5, 0x336fbba3, 0x738ab145);

// Register the custom version with core
FCustomVersionRegistration GRegisterCustomVersion(FDDGICustomVersion::GUID, FDDGICustomVersion::SaveLoadProbeTexturesFmt, TEXT("DDGIVolCompVer"));

// Create a CPU accessible GPU texture and copy the provided GPU texture's contents to it
static FDDGITexturePixels GetTexturePixelsStep1_RenderThread(FRHICommandListImmediate& RHICmdList, FTexture2DRHIRef textureGPU)
{
    FDDGITexturePixels ret;

    // Early out if a GPU texture is not provided
    if (!textureGPU) return ret;

    ret.w = textureGPU->GetSizeX();
    ret.h = textureGPU->GetSizeY();
    ret.pixelFormat = (int32)textureGPU->GetFormat();

    // Create the texture
    FRHIResourceCreateInfo createInfo(TEXT("DDGIGetTexturePixelsSave"));
    ret.texture = RHICreateTexture2D(
        textureGPU->GetSizeX(),
        textureGPU->GetSizeY(),
        textureGPU->GetFormat(),
        1,
        1,
        TexCreate_ShaderResource | TexCreate_Transient,
        ERHIAccess::CopyDest,
        createInfo);

    // Transition the GPU texture to a copy source
    RHICmdList.Transition(FRHITransitionInfo(textureGPU, ERHIAccess::WritableMask, ERHIAccess::CopySrc));

    // Schedule a copy of the GPU texture to the CPU accessible GPU texture
    RHICmdList.CopyTexture(textureGPU, ret.texture, FRHICopyTextureInfo{});

    // Transition the GPU texture back to general
    RHICmdList.Transition(FRHITransitionInfo(textureGPU, ERHIAccess::CopySrc, ERHIAccess::WritableMask));

    return ret;
}

// Read the CPU accessible GPU texture data into CPU memory
static void GetTexturePixelsStep2_RenderThread(FRHICommandListImmediate& RHICmdList, FDDGITexturePixels& texturePixels)
{
    // Early out if no texture is provided
    if (!texturePixels.texture) return;

    // Get a pointer to the CPU memory
    uint8* mappedTextureMemory = (uint8*)RHILockTexture2D(texturePixels.texture, 0, RLM_ReadOnly, texturePixels.stride, false);

    // Copy the texture data to CPU memory
    texturePixels.pixels.AddZeroed(texturePixels.h * texturePixels.stride);
    FMemory::Memcpy(&texturePixels.pixels[0], mappedTextureMemory, texturePixels.h * texturePixels.stride);

    RHIUnlockTexture2D(texturePixels.texture, 0, false);
}

static void SaveFDDGITexturePixels(FArchive& Ar, FDDGITexturePixels& texturePixels, bool saveFormat)
{
    check(Ar.IsSaving());

    Ar << texturePixels.w;
    Ar << texturePixels.h;
    Ar << texturePixels.stride;
    Ar << texturePixels.pixels;

    if (saveFormat) Ar << texturePixels.pixelFormat;
}

static void LoadFDDGITexturePixels(FArchive& Ar, FDDGITexturePixels& texturePixels, EPixelFormat expectedPixelFormat, bool loadFormat)
{
    check(Ar.IsLoading());

    // Load the texture data
    Ar << texturePixels.w;
    Ar << texturePixels.h;
    Ar << texturePixels.stride;
    Ar << texturePixels.pixels;

    if (loadFormat)
    {
        Ar << texturePixels.pixelFormat;

        // Early out if the loaded pixel format doesn't match our expected format
        if (texturePixels.pixelFormat != expectedPixelFormat) return;
    }

    // Early out if no data was loaded
    if (texturePixels.w == 0 || texturePixels.h == 0 || texturePixels.stride == 0) return;

    // Create the texture resource
    FRHIResourceCreateInfo createInfo(TEXT("DDGITextureLoad"));
    texturePixels.texture = RHICreateTexture2D(
        texturePixels.w,
        texturePixels.h,
        expectedPixelFormat,
        1,
        1,
        TexCreate_ShaderResource | TexCreate_Transient,
        createInfo);

    // Copy the texture's data to the staging buffer
    ENQUEUE_RENDER_COMMAND(DDGILoadTex)(
        [&texturePixels](FRHICommandListImmediate& RHICmdList)
        {
            if (texturePixels.pixels.Num() == texturePixels.h * texturePixels.stride)
            {
                uint32 destStride;
                uint8* mappedTextureMemory = (uint8*)RHILockTexture2D(texturePixels.texture, 0, RLM_WriteOnly, destStride, false);
                if (texturePixels.stride == destStride)
                {
                    // Loaded data has the same stride as expected by the runtime
                    // Copy the entire texture at once
                    FMemory::Memcpy(mappedTextureMemory, &texturePixels.pixels[0], texturePixels.h * texturePixels.stride);
                }
                else
                {
                    // Loaded data has a different stride than expected by the runtime
                    // Texture data was stored with a different API than what is running now (D3D12->VK, VK->D3D12)
                    // Copy each row of the source data to the texture
                    const uint8* SourceBuffer = &texturePixels.pixels[0];
                    for (uint32 Row = 0; Row < texturePixels.h; ++Row)
                    {
                        FMemory::Memcpy(mappedTextureMemory, SourceBuffer, FMath::Min(texturePixels.stride, destStride));

                        mappedTextureMemory += destStride;
                        SourceBuffer += texturePixels.stride;
                    }
                }
                RHIUnlockTexture2D(texturePixels.texture, 0, false);
            }

            // Only clear the texels when in a game.
            // Cooking needs this data to write textures to disk on save, after load, when headless etc.
        #if !WITH_EDITOR
            texturePixels.pixels.Reset();
        #endif
        }
    );
}

void UDDGIVolumeComponent::Serialize(FArchive& Ar)
{
    Super::Serialize(Ar);

    Ar.UsingCustomVersion(FDDGICustomVersion::GUID);
    if(Ar.CustomVer(FDDGICustomVersion::GUID) < FDDGICustomVersion::AddingCustomVersion)
    {
        if (Ar.IsLoading())
        {
            uint32 w, h;
            TArray<float> pixels;
            Ar << w;
            Ar << h;
            Ar << pixels;
        }
    }
    else if (Ar.CustomVer(FDDGICustomVersion::GUID) >= FDDGICustomVersion::SaveLoadProbeTextures)
    {
        // Save and load DDGIVolume texture resources when entering a level
        // Also applicable when ray tracing is not available (DX11 and Vulkan RHI).
        bool saveFormat = Ar.CustomVer(FDDGICustomVersion::GUID) >= FDDGICustomVersion::SaveLoadProbeTexturesFmt;

        FDDGIVolumeSceneProxy* proxy = SceneProxy;

        if (Ar.IsSaving())
        {
            FDDGITexturePixels Irradiance, Distance, Offsets, States;

            // When we are *not* cooking and ray tracing is available, copy the DDGIVolume probe texture resources
            // to CPU memory otherwise, write out the DDGIVolume texture resources acquired at load time
            if (!Ar.IsCooking() && IsRayTracingEnabled() && proxy)
            {
                // Copy textures to CPU accessible texture resources
                ENQUEUE_RENDER_COMMAND(DDGISaveTexStep1)(
                    [&Irradiance, &Distance, &Offsets, &States, proxy](FRHICommandListImmediate& RHICmdList)
                    {
                        Irradiance = GetTexturePixelsStep1_RenderThread(RHICmdList, proxy->ProbesIrradianceTex);
                        Distance = GetTexturePixelsStep1_RenderThread(RHICmdList, proxy->ProbesDistanceTex);
                        Offsets = GetTexturePixelsStep1_RenderThread(RHICmdList, proxy->ProbesOffsetsTex);
                        States = GetTexturePixelsStep1_RenderThread(RHICmdList, proxy->ProbesStatesTex);
                    }
                );
                FlushRenderingCommands();

                // Read the GPU texture data to CPU memory
                ENQUEUE_RENDER_COMMAND(DDGISaveTexStep2)(
                    [&Irradiance, &Distance, &Offsets, &States](FRHICommandListImmediate& RHICmdList)
                    {
                        GetTexturePixelsStep2_RenderThread(RHICmdList, Irradiance);
                        GetTexturePixelsStep2_RenderThread(RHICmdList, Distance);
                        GetTexturePixelsStep2_RenderThread(RHICmdList, Offsets);
                        GetTexturePixelsStep2_RenderThread(RHICmdList, States);
                    }
                );
                FlushRenderingCommands();
            }
            else
            {
                Irradiance = LoadContext.Irradiance;
                Distance = LoadContext.Distance;
                Offsets = LoadContext.Offsets;
                States = LoadContext.States;
            }

            // Write the volume data
            SaveFDDGITexturePixels(Ar, Irradiance, saveFormat);
            SaveFDDGITexturePixels(Ar, Distance, saveFormat);
            SaveFDDGITexturePixels(Ar, Offsets, saveFormat);
            SaveFDDGITexturePixels(Ar, States, saveFormat);
        }
        else if (Ar.IsLoading())
        {
            EDDGIIrradianceBits IrradianceBits = GetDefault<URTXGIPluginSettings>()->IrradianceBits;
            EDDGIDistanceBits DistanceBits = GetDefault<URTXGIPluginSettings>()->DistanceBits;
            bool loadFormat = Ar.CustomVer(FDDGICustomVersion::GUID) >= FDDGICustomVersion::SaveLoadProbeTexturesFmt;

            // Read the volume texture data in and note that it's ready for load
            LoadFDDGITexturePixels(Ar, LoadContext.Irradiance, (IrradianceBits == EDDGIIrradianceBits::n32) ? FDDGIVolumeSceneProxy::FComponentData::c_pixelFormatIrradianceHighBitDepth : FDDGIVolumeSceneProxy::FComponentData::c_pixelFormatIrradianceLowBitDepth, loadFormat);
            LoadFDDGITexturePixels(Ar, LoadContext.Distance, (DistanceBits == EDDGIDistanceBits::n32) ? FDDGIVolumeSceneProxy::FComponentData::c_pixelFormatDistanceHighBitDepth : FDDGIVolumeSceneProxy::FComponentData::c_pixelFormatDistanceLowBitDepth, loadFormat);
            LoadFDDGITexturePixels(Ar, LoadContext.Offsets, FDDGIVolumeSceneProxy::FComponentData::c_pixelFormatOffsets, loadFormat);
            LoadFDDGITexturePixels(Ar, LoadContext.States, FDDGIVolumeSceneProxy::FComponentData::c_pixelFormatStates, loadFormat);

            bool& ReadyForLoad = LoadContext.ReadyForLoad;
            ENQUEUE_RENDER_COMMAND(DDGILoadReady)(
                [&ReadyForLoad](FRHICommandListImmediate& RHICmdList)
                {
                    ReadyForLoad = true;
                }
            );
        }
    }
}

void UDDGIVolumeComponent::UpdateRenderThreadData()
{
    // Send command to the rendering thread to update the transform and other parameters
    if (SceneProxy)
    {
        // Update the volume component's data
        FDDGIVolumeSceneProxy::FComponentData ComponentData;
        ComponentData.RaysPerProbe = RaysPerProbe;
        ComponentData.ProbeMaxRayDistance = ProbeMaxRayDistance;
        ComponentData.LightingChannels = LightingChannels;
        ComponentData.ProbeCounts = ProbeCounts;
        ComponentData.ProbeDistanceExponent = probeDistanceExponent;
        ComponentData.ProbeIrradianceEncodingGamma = probeIrradianceEncodingGamma;
        ComponentData.LightingPriority = LightingPriority;
        ComponentData.UpdatePriority = UpdatePriority;
        ComponentData.ProbeHysteresis = ProbeHysteresis;
        ComponentData.ProbeChangeThreshold = probeChangeThreshold;
        ComponentData.ProbeBrightnessThreshold = probeBrightnessThreshold;
        ComponentData.NormalBias = NormalBias;
        ComponentData.ViewBias = ViewBias;
        ComponentData.BlendDistance = VolumeBlendDistance;
        ComponentData.BlendDistanceBlack = VolumeBlendDistanceBlack;
        ComponentData.ProbeBackfaceThreshold = ProbeBackfaceThreshold;
        ComponentData.ProbeMinFrontfaceDistance = ProbeMinFrontfaceDistance;
        ComponentData.EnableProbeRelocation = EnableProbeRelocation;
        ComponentData.EnableProbeScrolling = EnableProbeScrolling;
        ComponentData.EnableVolume = EnableVolume;
        ComponentData.IrradianceScalar = IrradianceScalar;
        ComponentData.EmissiveMultiplier = EmissiveMultiplier;
        ComponentData.LightingMultiplier = LightMultiplier;
        ComponentData.RuntimeStatic = RuntimeStatic;
        ComponentData.SkyLight = SkyLight;

        if (EnableProbeScrolling)
        {
            // Infinite Scrolling Volume
            // Disable volume transformations and instead move the volume by "scrolling" the probes over an infinite space.
            // Offset "planes" of probes from one end of the volume to the other (in the direction  of movement).
            // Useful for computing GI around a moving object, e.g. characters.
            // NB: scrolling probes can be disruptive when recursive probe sampling is enabled and the volume is small. Sudden changes in scrolled probes will propogate to nearby probes!
            FVector CurrentOrigin = GetOwner()->GetTransform().GetLocation();
            FVector MovementDelta = CurrentOrigin - LastOrigin;

            FVector ProbeGridSpacing;
            FVector VolumeSize = GetOwner()->GetTransform().GetScale3D() * 200.f;
            ProbeGridSpacing.X = VolumeSize.X / float(ProbeCounts.X);
            ProbeGridSpacing.Y = VolumeSize.Y / float(ProbeCounts.Y);
            ProbeGridSpacing.Z = VolumeSize.Z / float(ProbeCounts.Z);

            if(FMath::Abs(MovementDelta.X) >= ProbeGridSpacing.X || FMath::Abs(MovementDelta.Y) >= ProbeGridSpacing.Y || FMath::Abs(MovementDelta.Z) >= ProbeGridSpacing.Z)
            {
                auto absFloor = [](float f)
                {
                    return f >= 0.f ? int(floor(f)) : int(ceil(f));
                };

                // Calculate the number of grid cells that have been moved
                FIntVector Translation;
                Translation.X = int(absFloor(MovementDelta.X / ProbeGridSpacing.X));
                Translation.Y = int(absFloor(MovementDelta.Y / ProbeGridSpacing.Y));
                Translation.Z = int(absFloor(MovementDelta.Z / ProbeGridSpacing.Z));

                // Move the volume origin the number of grid cells * the distance between cells
                LastOrigin.X += float(Translation.X) * ProbeGridSpacing.X;
                LastOrigin.Y += float(Translation.Y) * ProbeGridSpacing.Y;
                LastOrigin.Z += float(Translation.Z) * ProbeGridSpacing.Z;

                // Update the probe scroll offset count
                ProbeScrollOffset.X += Translation.X;
                ProbeScrollOffset.Y += Translation.Y;
                ProbeScrollOffset.X += Translation.Z;
            }

            // Set the probe scroll offsets
            ComponentData.ProbeScrollOffsets.X = ((ProbeScrollOffset.X % ProbeCounts.X) + ProbeCounts.X) % ProbeCounts.X;
            ComponentData.ProbeScrollOffsets.Y = ((ProbeScrollOffset.Y % ProbeCounts.Y) + ProbeCounts.Y) % ProbeCounts.Y;
            ComponentData.ProbeScrollOffsets.Z = ((ProbeScrollOffset.Z % ProbeCounts.Z) + ProbeCounts.Z) % ProbeCounts.Z;

            // Set the volume origin and scale (rotation not allowed)
            ComponentData.Origin = LastOrigin;
            ComponentData.Transform.SetScale3D(GetOwner()->GetTransform().GetScale3D());
        }
        else
        {
            // Finite moveable volume
            // Transform the volume to stay aligned with its parent.
            // Useful for spaces that move, e.g. a ship or train car.
            ComponentData.Transform = GetOwner()->GetTransform();
            ComponentData.Origin = LastOrigin = GetOwner()->GetTransform().GetLocation();
            ComponentData.ProbeScrollOffsets = FIntVector{ 0, 0, 0 };
        }

        // If the ProbeCounts are too large to make textures, let's not update the render thread data to avoid a crash.
        // Everything is ok with not getting an update, ever, so this is safe.
        {
            volatile uint32 maxTextureSize = GetMax2DTextureDimension();

            // DDGIRadiance
            if (uint32(ProbeCounts.X * ProbeCounts.Y * ProbeCounts.Z) > maxTextureSize)
                return;

            FIntPoint ProxyDims = ComponentData.Get2DProbeCount();

            // DDGIIrradiance
            {
                int numTexels = FDDGIVolumeSceneProxy::FComponentData::c_NumTexelsIrradiance;
                FIntPoint ProxyTexDims = ProxyDims * (numTexels + 2);
                if (uint32(ProxyTexDims.X) > maxTextureSize || uint32(ProxyTexDims.Y) > maxTextureSize)
                    return;
            }

            // DDGIDistance
            {
                int numTexels = FDDGIVolumeSceneProxy::FComponentData::c_NumTexelsDistance;
                FIntPoint ProxyTexDims = ProxyDims * (numTexels + 2);
                if (uint32(ProxyTexDims.X) > maxTextureSize || uint32(ProxyTexDims.Y) > maxTextureSize)
                    return;
            }
        }

        FDDGIVolumeSceneProxy* DDGIProxy = SceneProxy;
        EDDGIIrradianceBits IrradianceBits = GetDefault<URTXGIPluginSettings>()->IrradianceBits;
        EDDGIDistanceBits DistanceBits = GetDefault<URTXGIPluginSettings>()->DistanceBits;

        FDDGITextureLoadContext TextureLoadContext = LoadContext;
        LoadContext.ReadyForLoad = false;

        ENQUEUE_RENDER_COMMAND(UpdateGIVolumeTransformCommand)(
            [DDGIProxy, ComponentData, TextureLoadContext, IrradianceBits, DistanceBits](FRHICommandListImmediate& RHICmdList)
            {
                bool needReallocate =
                    DDGIProxy->ComponentData.ProbeCounts != ComponentData.ProbeCounts ||
                    DDGIProxy->ComponentData.RaysPerProbe != ComponentData.RaysPerProbe ||
                    DDGIProxy->ComponentData.EnableProbeRelocation != ComponentData.EnableProbeRelocation;

                // set the data
                DDGIProxy->ComponentData = ComponentData;

                // handle state textures ready to load from serialization
                if (TextureLoadContext.ReadyForLoad)
                    DDGIProxy->TextureLoadContext = TextureLoadContext;

                if (needReallocate)
                {
                    DDGIProxy->ReallocateSurfaces_RenderThread(RHICmdList, IrradianceBits, DistanceBits);
                    DDGIProxy->ResetTextures_RenderThread(RHICmdList);
                    FDDGIVolumeSceneProxy::AllProxiesReadyForRender_RenderThread.Add(DDGIProxy);
                }
            }
        );
    }
}

void UDDGIVolumeComponent::EnableVolumeComponent(bool enabled)
{
    EnableVolume = enabled;
    UpdateRenderThreadData();
}

void UDDGIVolumeComponent::Startup()
{
#if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)
	FGlobalIlluminationExperimentalPluginDelegates::FRenderDiffuseIndirectVisualizations& RVDelegate = FGlobalIlluminationExperimentalPluginDelegates::RenderDiffuseIndirectVisualizations();
    FDDGIVolumeSceneProxy::RenderDiffuseIndirectVisualizationsHandle = RVDelegate.AddStatic(FDDGIVolumeSceneProxy::RenderDiffuseIndirectVisualizations_RenderThread);
#endif

    FGlobalIlluminationExperimentalPluginDelegates::FRenderDiffuseIndirectLight& RDILDelegate = FGlobalIlluminationExperimentalPluginDelegates::RenderDiffuseIndirectLight();
    FDDGIVolumeSceneProxy::RenderDiffuseIndirectLightHandle = RDILDelegate.AddStatic(FDDGIVolumeSceneProxy::RenderDiffuseIndirectLight_RenderThread);
}

void UDDGIVolumeComponent::Shutdown()
{
#if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)
    FGlobalIlluminationExperimentalPluginDelegates::FRenderDiffuseIndirectVisualizations& RVDelegate = FGlobalIlluminationExperimentalPluginDelegates::RenderDiffuseIndirectVisualizations();
    check(FDDGIVolumeSceneProxy::RenderDiffuseIndirectVisualizationsHandle.IsValid());
    RVDelegate.Remove(FDDGIVolumeSceneProxy::RenderDiffuseIndirectVisualizationsHandle);
#endif

    FGlobalIlluminationExperimentalPluginDelegates::FRenderDiffuseIndirectLight& RDILDelegate = FGlobalIlluminationExperimentalPluginDelegates::RenderDiffuseIndirectLight();
    check(FDDGIVolumeSceneProxy::RenderDiffuseIndirectLightHandle.IsValid());
    RDILDelegate.Remove(FDDGIVolumeSceneProxy::RenderDiffuseIndirectLightHandle);
}

bool UDDGIVolumeComponent::Exec(UWorld* InWorld, const TCHAR* Cmd, FOutputDevice& Ar)
{
    return ProcessConsoleExec(Cmd, Ar, NULL);
}

void UDDGIVolumeComponent::DDGIClearVolumes()
{
    ENQUEUE_RENDER_COMMAND(DDGIClearVolumesCommand)(
        [](FRHICommandListImmediate& RHICmdList)
        {
            for (FDDGIVolumeSceneProxy* DDGIProxy : FDDGIVolumeSceneProxy::AllProxiesReadyForRender_RenderThread)
            {
                DDGIProxy->ResetTextures_RenderThread(RHICmdList);
            }
        }
    );
}

void UDDGIVolumeComponent::CreateRenderState_Concurrent(FRegisterComponentContext* Context)
{
    Super::CreateRenderState_Concurrent(Context);
    check(SceneProxy == nullptr);

#if WITH_EDITOR
    if (!GetOwner()->IsTemporarilyHiddenInEditor())
#endif
    {
        SceneProxy = new FDDGIVolumeSceneProxy(GetScene());
        UpdateRenderThreadData();
    }
}

void UDDGIVolumeComponent::DestroyRenderState_Concurrent()
{
    Super::DestroyRenderState_Concurrent();

    if (SceneProxy)
    {
        FDDGITextureLoadContext& ComponentLoadContext = LoadContext;

        FDDGIVolumeSceneProxy* DDGIProxy = SceneProxy;
        ENQUEUE_RENDER_COMMAND(DeleteProxy)(
            [DDGIProxy, &ComponentLoadContext](FRHICommandListImmediate& RHICmdList)
            {
                // If the component has textures pending load, nothing to do here. Those are the most authoritative.
                if (!ComponentLoadContext.ReadyForLoad)
                {
                    // If the proxy has textures pending load which haven't been serviced yet, the component should take those
                    // in case it creates another proxy.
                    if (DDGIProxy->TextureLoadContext.ReadyForLoad)
                    {
                        ComponentLoadContext = DDGIProxy->TextureLoadContext;
                    }
                    // otherwise, we should copy the textures from this proxy into textures for the TextureLoadContext
                    // to make them survive to the next proxy for this component if one is created.
                    else
                    {
                        ComponentLoadContext.ReadyForLoad = true;
                        ComponentLoadContext.Irradiance = GetTexturePixelsStep1_RenderThread(RHICmdList, DDGIProxy->ProbesIrradianceTex);
                        ComponentLoadContext.Distance = GetTexturePixelsStep1_RenderThread(RHICmdList, DDGIProxy->ProbesDistanceTex);
                        ComponentLoadContext.Offsets = GetTexturePixelsStep1_RenderThread(RHICmdList, DDGIProxy->ProbesOffsetsTex);
                        ComponentLoadContext.States = GetTexturePixelsStep1_RenderThread(RHICmdList, DDGIProxy->ProbesStatesTex);
                    }
                }

                delete DDGIProxy;
            }
        );

        // wait for the above command to finish, so we know we got the load context if present
        FlushRenderingCommands();

        SceneProxy = nullptr;
    }
}
