// Copyright Epic Games, Inc. All Rights Reserved.

#include "DeferredShadingRenderer.h"

#if RHI_RAYTRACING

#include "ClearQuad.h"
#include "SceneRendering.h"
#include "SceneRenderTargets.h"
#include "SceneUtils.h"
#include "RenderTargetPool.h"
#include "RHIResources.h"
#include "UniformBuffer.h"
#include "RHI/Public/PipelineStateCache.h"
#include "Raytracing/RaytracingOptions.h"
#include "RayTracingMaterialHitShaders.h"
#include "SceneTextureParameters.h"
#include "RendererModule.h"

#include "PostProcess/PostProcessing.h"
#include "PostProcess/SceneFilterRendering.h"

#endif // RHI_RAYTRACING

static TAutoConsoleVariable<int32> CVarRayTracingRenderSim(
	TEXT("r.RayTracing.Simulation"),
	0,
	TEXT("Enables simulation (default = 0)"),
	ECVF_RenderThreadSafe
);

class FRetchWorldPosCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FRetchWorldPosCS)
	SHADER_USE_PARAMETER_STRUCT(FRetchWorldPosCS, FGlobalShader)

	using FPermutationDomain = TShaderPermutationDomain<>;

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
	}

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		//SHADER_PARAMETER(float, FloatParameter)

		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, ViewUniformBuffer)
		SHADER_PARAMETER_STRUCT_INCLUDE(FSceneTextureParameters, SceneTextures)

		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, OutWorldPos)
		END_SHADER_PARAMETER_STRUCT()
};

IMPLEMENT_GLOBAL_SHADER(FRetchWorldPosCS, "/Engine/Private/RayTracing/RayTracingSimulation.usf", "MainWorldPosCS", SF_Compute);

BEGIN_SHADER_PARAMETER_STRUCT(FRenderSimulationParameters, )
	SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneTextureUniformParameters, SceneTextures)
	RDG_TEXTURE_ACCESS(WorldPos, ERHIAccess::SRVGraphics)
	//RDG_TEXTURE_ACCESS(ShadowMaskTexture, ERHIAccess::SRVGraphics)
	//RDG_TEXTURE_ACCESS(LightingChannelsTexture, ERHIAccess::SRVGraphics)
	RENDER_TARGET_BINDING_SLOTS()
END_SHADER_PARAMETER_STRUCT()

void FDeferredShadingSceneRenderer::RenderSimulation(
	FRDGBuilder& GraphBuilder,
	FRDGTextureRef SceneColorTexture)
#if RHI_RAYTRACING
{
	if (CVarRayTracingRenderSim.GetValueOnRenderThread() == 0)
	{
		return;
	}
	FSceneTextureParameters SceneTextures = GetSceneTextureParameters(GraphBuilder);

	ESceneTextureSetupMode SceneTexturesSetupMode = ESceneTextureSetupMode::SceneDepth;
	TRDGUniformBufferRef<FSceneTextureUniformParameters> SceneTextureBuffers = CreateSceneTextureUniformBuffer(GraphBuilder, FeatureLevel, SceneTexturesSetupMode);

	FRDGTextureRef WorldPosTexture = nullptr;
	{
		FRDGTextureDesc Desc = FRDGTextureDesc::Create2D(
			SceneTextures.SceneDepthTexture->Desc.Extent,
			PF_FloatRGBA,
			FClearValueBinding::None,
			TexCreate_ShaderResource | TexCreate_RenderTargetable | TexCreate_UAV);

		WorldPosTexture = GraphBuilder.CreateTexture(Desc, TEXT("WorldPosTexture"));
	}

	for (int viewIndex = 0; viewIndex < Views.Num(); viewIndex++)
	{
		const FViewInfo& View = Views[viewIndex];

		FRetchWorldPosCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FRetchWorldPosCS::FParameters>();
		PassParameters->SceneTextures = SceneTextures;
		PassParameters->ViewUniformBuffer = View.ViewUniformBuffer;
		PassParameters->OutWorldPos = GraphBuilder.CreateUAV(WorldPosTexture);

		TShaderMapRef<FRetchWorldPosCS> ComputeShader(GetGlobalShaderMap(FeatureLevel));

		FIntPoint Resolution = FIntPoint::DivideAndRoundUp(View.ViewRect.Size(), 16);
		FIntVector DisPatchSize = FIntVector(Resolution.X, Resolution.Y, 1);
		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("FRetchWorldPosCS"),
			ComputeShader,
			PassParameters,
			DisPatchSize);

		//ClearUnusedGraphResources(ComputeShader, PassParameters);
		//GraphBuilder.AddPass(
		//	RDG_EVENT_NAME("FRetchWorldPosCS"),// Forward<FRDGEventName>(PassName),
		//	PassParameters,
		//	ERDGPassFlags::Compute,
		//	[PassParameters, ComputeShader, DisPatchSize](FRHIComputeCommandList& RHICmdList)
		//{
		//	FComputeShaderUtils::Dispatch(RHICmdList, ComputeShader, *PassParameters, DisPatchSize);
		//});

		FLightSceneInfo* DirectionalLightSceneInfo = NULL;

		for (TSparseArray<FLightSceneInfoCompact>::TConstIterator LightIt(Scene->Lights); LightIt; ++LightIt)
		{
			const FLightSceneInfoCompact& LightSceneInfoCompact = *LightIt;
			FLightSceneInfo* LightSceneInfo = LightSceneInfoCompact.LightSceneInfo;

			if (ViewFamily.EngineShowFlags.LightFunctions
				&& LightSceneInfo->Proxy->GetLightType() == LightType_Directional
				// Band-aid fix for extremely rare case that light scene proxy contains NaNs.
				&& !LightSceneInfo->Proxy->GetDirection().ContainsNaN()
				&& LightSceneInfo->ShouldRenderLightViewIndependent()
				&& LightSceneInfo->ShouldRenderLight(View))
			{
				DirectionalLightSceneInfo = LightSceneInfo;
			}
		}

		FRenderSimulationParameters* SimParameters = GraphBuilder.AllocParameters<FRenderSimulationParameters>();
		SimParameters->SceneTextures = SceneTextureBuffers;
		SimParameters->WorldPos = WorldPosTexture;
		ERDGPassFlags PassFlags = ERDGPassFlags::Raster;

		GraphBuilder.AddPass(
			RDG_EVENT_NAME("SimParameters"),
			SimParameters,
			PassFlags,
			[this, WorldPosTexture, &View, DirectionalLightSceneInfo](FRHICommandListImmediate& RHICmdList)
		{
			FRHITexture* WorldPosTex = TryGetRHI(WorldPosTexture);

			int ImageWidth = View.ViewRect.Width();
			int ImageHeight = View.ViewRect.Height();
			FReadSurfaceDataFlags readPixelFlags(RCM_UNorm);
			FIntRect IntRect(View.ViewRect.Min.X, View.ViewRect.Min.Y, ImageWidth, ImageHeight);
			TArray<FFloat16Color> WorldPosData;
			//RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
			GDynamicRHI->RHIReadSurfaceFloatData_RenderThread(RHICmdList, WorldPosTex, IntRect, WorldPosData, CubeFace_PosX, 0, 0);

			FViewUniformShaderParameters ViewVoxelizeParameters = *View.CachedViewUniformShaderParameters;
			FLinearColor TempWorldPos = FLinearColor(WorldPosData[100]);
			UE_LOG(LogRenderer, Log, TEXT("WorldPosData %s"), *TempWorldPos.ToString());
			UE_LOG(LogRenderer, Log, TEXT("camera pos: %s"), *ViewVoxelizeParameters.WorldCameraOrigin.ToString());
			UE_LOG(LogRenderer, Log, TEXT("camera forword: %s"), *ViewVoxelizeParameters.ViewForward.ToString());

			if (DirectionalLightSceneInfo)
			{
				const FVector LightDirection = DirectionalLightSceneInfo->Proxy->GetDirection().GetSafeNormal();
				UE_LOG(LogRenderer, Log, TEXT("DirectionLight LightDirection: %s"), *LightDirection.ToString());
			}
		});
	}
}
#else
{
	unimplemented();
}
#endif
