/*
* Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "RenderAlphaCore.h"

// UE4 public interfaces
#include "CoreMinimal.h"
#include "Engine/TextureRenderTarget2D.h"
#include "SceneView.h"
#include "RenderGraph.h"
#include "RayGenShaderUtils.h"
#include "ShaderParameterStruct.h"
#include "GlobalShader.h"
//#include "RTXGIPluginSettings.h"

// UE4 private interfaces
//#include "ReflectionEnvironment.h"
//#include "FogRendering.h"
#include "SceneRendering.h"
#include "SceneTextureParameters.h"
#include "DeferredShadingRenderer.h"
#include "ScenePrivate.h"

#include <cmath>

#define LOCTEXT_NAMESPACE "FRenderAlphaCorePlugin"

static TAutoConsoleVariable<int32> CVarRayTracingRenderSim(
	TEXT("r.RayTracing.Simulation"),
	0,
	TEXT("Enables simulation (default = 0)"),
	ECVF_RenderThreadSafe
);

// So we have something to give the "AddPass" calls that we add for transitions
BEGIN_SHADER_PARAMETER_STRUCT(FDummyShaderParameters, )
    SHADER_PARAMETER(int, Dummy)
END_SHADER_PARAMETER_STRUCT()


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
		//SHADER_PARAMETER_STRUCT_INCLUDE(FSceneTextureParameters, SceneTextures)
		//SHADER_PARAMETER_RDG_TEXTURE(Texture2D, Normals)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, DepthTexture)

		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, OutWorldPos)
	END_SHADER_PARAMETER_STRUCT()
};

IMPLEMENT_GLOBAL_SHADER(FRetchWorldPosCS, "/Plugin/AlphaCore/Private/SimAlphaCore.usf", "MainWorldPosCS", SF_Compute);

BEGIN_SHADER_PARAMETER_STRUCT(FRenderSimulationParameters, )
	//SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneTextureUniformParameters, SceneTextures)
	RDG_TEXTURE_ACCESS(BufferATexture, ERHIAccess::SRVGraphics)
	RDG_TEXTURE_ACCESS(SceneDepthTexture, ERHIAccess::SRVGraphics)
	RDG_TEXTURE_ACCESS(WorldPos, ERHIAccess::SRVGraphics)
	RDG_TEXTURE_ACCESS(SimOutput, ERHIAccess::SRVGraphics)
	RENDER_TARGET_BINDING_SLOTS()
END_SHADER_PARAMETER_STRUCT()

namespace RenderAlphaCore
{
    FDelegateHandle RenderAlphaCoreHandle;

    void Startup()
    {
		FAlphaCorePluginDelegates::FRenderAlphaCore& RACDelegate = FAlphaCorePluginDelegates::RenderAlphaCoreEffect();
		RenderAlphaCoreHandle = RACDelegate.AddStatic(RenderAlphaCore::RenderAlphaCore_RenderThread);
    }

    void Shutdown()
    {
		FAlphaCorePluginDelegates::FRenderAlphaCore& RACDelegate = FAlphaCorePluginDelegates::RenderAlphaCoreEffect();
		RACDelegate.Remove(RenderAlphaCoreHandle);
    }

    void RenderAlphaCore_RenderThread(
		const FScene& Scene,
		const FViewInfo& View,
		FRDGBuilder& GraphBuilder,
		FGlobalIlluminationExperimentalPluginResources& Resources)
    {
		if (CVarRayTracingRenderSim.GetValueOnRenderThread() == 0)
		{
			return;
		}

		FRDGTextureRef WorldPosTexture = nullptr;
		FRDGTextureRef SimOutput = nullptr;
		FRDGTextureRef DepthTexture = nullptr;

		{
			FRDGTextureDesc Desc = FRDGTextureDesc::Create2D(
				Resources.SceneDepthZ->GetDesc().Extent,
				PF_FloatRGBA,
				FClearValueBinding::None,
				TexCreate_ShaderResource | TexCreate_RenderTargetable | TexCreate_UAV);

			WorldPosTexture = GraphBuilder.CreateTexture(Desc, TEXT("WorldPosTexture"));

			Desc = FRDGTextureDesc::Create2D(
				Resources.SceneDepthZ->GetDesc().Extent,
				PF_A32B32G32R32F,
				FClearValueBinding::None,
				TexCreate_ShaderResource | TexCreate_RenderTargetable | TexCreate_UAV);
			SimOutput = GraphBuilder.CreateTexture(Desc, TEXT("SimOutput"));
			AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(SimOutput), FLinearColor(0.0f, 0.0f, 0.0f, 0.0f));
		}

		FRetchWorldPosCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FRetchWorldPosCS::FParameters>();
		PassParameters->DepthTexture = GraphBuilder.RegisterExternalTexture(Resources.SceneDepthZ);
		PassParameters->ViewUniformBuffer = View.ViewUniformBuffer;
		PassParameters->OutWorldPos = GraphBuilder.CreateUAV(WorldPosTexture);
		TShaderMapRef<FRetchWorldPosCS> ComputeShader(GetGlobalShaderMap(View.FeatureLevel));

		FIntPoint Resolution = FIntPoint::DivideAndRoundUp(View.ViewRect.Size(), 16);
		FIntVector DisPatchSize = FIntVector(Resolution.X, Resolution.Y, 1);
		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("FRetchWorldPosCS"),
			ComputeShader,
			PassParameters,
			DisPatchSize);

		ClearUnusedGraphResources(ComputeShader, PassParameters);
		GraphBuilder.AddPass(
			RDG_EVENT_NAME("FRetchWorldPosCS"),// Forward<FRDGEventName>(PassName),
			PassParameters,
			ERDGPassFlags::Compute,
			[PassParameters, ComputeShader, DisPatchSize](FRHIComputeCommandList& RHICmdList)
		{
			FComputeShaderUtils::Dispatch(RHICmdList, ComputeShader, *PassParameters, DisPatchSize);
		});

    }

} // namespace RenderAlphaCore

#undef LOCTEXT_NAMESPACE
