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
#include "AlphaCore.h"
#endif // RHI_RAYTRACING




static TAutoConsoleVariable<int32> CVarRayTracingRenderSim(
	TEXT("r.RayTracing.Simulation"),
	0,
	TEXT("Enables simulation (default = 0)"),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarRayTracingSimColorR(
	TEXT("r.RayTracing.SimColorR"),
	0,
	TEXT("Enables simulation (default = 0)"),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarRayTracingSimColorG(
	TEXT("r.RayTracing.SimColorG"),
	0,
	TEXT("Enables simulation (default = 0)"),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarRayTracingSimColorB(
	TEXT("r.RayTracing.SimColorB"),
	0,
	TEXT("Enables simulation (default = 0)"),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarRayTracingSimColorA(
	TEXT("r.RayTracing.SimColorA"),
	1.0f,
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
RDG_TEXTURE_ACCESS(SimOutput, ERHIAccess::SRVGraphics)
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
	FRDGTextureRef SimOutput = nullptr;
	FRDGTextureRef DepthTexture = nullptr;

	{
		FRDGTextureDesc Desc = FRDGTextureDesc::Create2D(
			SceneTextures.SceneDepthTexture->Desc.Extent,
			PF_FloatRGBA,
			FClearValueBinding::None,
			TexCreate_ShaderResource | TexCreate_RenderTargetable | TexCreate_UAV);

		WorldPosTexture = GraphBuilder.CreateTexture(Desc, TEXT("WorldPosTexture"));

		Desc = FRDGTextureDesc::Create2D(
			SceneTextures.SceneDepthTexture->Desc.Extent,
			PF_A32B32G32R32F,
			FClearValueBinding::None,
			TexCreate_ShaderResource | TexCreate_RenderTargetable | TexCreate_UAV);
		SimOutput = GraphBuilder.CreateTexture(Desc, TEXT("SimOutput"));
		AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(SimOutput), FLinearColor(0.0f, 0.0f, 0.0f, 0.0f));
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

		ClearUnusedGraphResources(ComputeShader, PassParameters);
		GraphBuilder.AddPass(
			RDG_EVENT_NAME("FRetchWorldPosCS"),// Forward<FRDGEventName>(PassName),
			PassParameters,
			ERDGPassFlags::Compute,
			[PassParameters, ComputeShader, DisPatchSize](FRHIComputeCommandList& RHICmdList)
		{
			FComputeShaderUtils::Dispatch(RHICmdList, ComputeShader, *PassParameters, DisPatchSize);
		});


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
		SimParameters->SimOutput = SimOutput;
		ERDGPassFlags PassFlags = ERDGPassFlags::Raster;

		GraphBuilder.AddPass(
			RDG_EVENT_NAME("SimParameters"),
			SimParameters,
			PassFlags,
			[this, WorldPosTexture, SimOutput, &View, DirectionalLightSceneInfo](FRHICommandListImmediate& RHICmdList)
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


				FVector LightDirection;

				if (DirectionalLightSceneInfo)
				{
					LightDirection = DirectionalLightSceneInfo->Proxy->GetDirection().GetSafeNormal();
					UE_LOG(LogRenderer, Log, TEXT("DirectionLight LightDirection: %s"), *LightDirection.ToString());
				}


				///AlphaCore Content

				float greyFloatRamp[128] = { 0.000000,0.007874,0.015748,0.023622,0.031496,0.039370,0.047244,0.055118,0.062992,0.070866,
											 0.078740,0.086614,0.094488,0.102362,0.110236,0.118110,0.125984,0.133858,0.141732,0.149606,
											 0.157480,0.165354,0.173228,0.181102,0.188976,0.196850,0.204724,0.212598,0.220472,0.228346,
											 0.236220,0.244094,0.251969,0.259843,0.267717,0.275591,0.283465,0.291339,0.299213,0.307087,
											 0.314961,0.322835,0.330709,0.338583,0.346457,0.354331,0.362205,0.370079,0.377953,0.385827,
											 0.393701,0.401575,0.409449,0.417323,0.425197,0.433071,0.440945,0.448819,0.456693,0.464567,
											 0.472441,0.480315,0.488189,0.496063,0.503937,0.511811,0.519685,0.527559,0.535433,0.543307,
											 0.551181,0.559055,0.566929,0.574803,0.582677,0.590551,0.598425,0.606299,0.614173,0.622047,
											 0.629921,0.637795,0.645669,0.653543,0.661417,0.669291,0.677165,0.685039,0.692913,0.700787,
											 0.708661,0.716535,0.724409,0.732283,0.740157,0.748031,0.755906,0.763780,0.771654,0.779528,
											 0.787402,0.795276,0.803150,0.811024,0.818898,0.826772,0.834646,0.842520,0.850394,0.858268,
											 0.866142,0.874016,0.881890,0.889764,0.897638,0.905512,0.913386,0.921260,0.929134,0.937008,
											 0.944882,0.952756,0.960630,0.968504,0.976378,0.984252,0.992126,1.000000 };

				float constFloatRamp[128] = { 1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,
											  1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,
											  1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,
											  1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,
											  1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,
											  1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,
											  1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,
											  1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,
											  1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,
											  1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,
											  1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,
											  1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,
											  1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000 };


				AxColorRGBA8 rgbColorRamp[128] = { {255,0,0,255},{250,4,0,255},{246,8,0,255},{242,12,0,255},{238,16,0,255},
					{234,20,0,255},{230,24,0,255},{226,28,0,255},{222,32,0,255},{218,36,0,255},{214,40,0,255},{210,44,0,255},
					{206,48,0,255},{202,52,0,255},{198,56,0,255},{194,60,0,255},{190,64,0,255},{186,68,0,255},{182,72,0,255},
					{178,76,0,255},{174,80,0,255},{170,84,0,255},{166,88,0,255},{162,92,0,255},{158,96,0,255},{154,100,0,255},
					{150,104,0,255},{146,108,0,255},{142,112,0,255},{138,116,0,255},{134,120,0,255},{130,124,0,255},
					{126,128,0,255},{122,132,0,255},{118,136,0,255},{114,140,0,255},{110,144,0,255},{106,148,0,255},
					{102,152,0,255},{98,156,0,255},{94,160,0,255},{90,164,0,255},{86,168,0,255},{82,172,0,255},{78,176,0,255},
					{74,180,0,255},{70,184,0,255},{66,188,0,255},{62,192,0,255},{58,196,0,255},{54,200,0,255},{50,204,0,255},
					{46,208,0,255},{42,212,0,255},{38,216,0,255},{34,220,0,255},{30,224,0,255},{26,228,0,255},{22,232,0,255},
					{18,236,0,255},{14,240,0,255},{10,244,0,255},{6,248,0,255},{2,252,0,255},{0,252,2,255},{0,248,6,255},
					{0,244,10,255},{0,240,14,255},{0,236,18,255},{0,232,22,255},{0,228,26,255},{0,224,30,255},{0,220,34,255},
					{0,216,38,255},{0,212,42,255},{0,208,46,255},{0,204,50,255},{0,200,54,255},{0,196,58,255},{0,192,62,255},
					{0,188,66,255},{0,184,70,255},{0,180,74,255},{0,176,78,255},{0,172,82,255},{0,168,86,255},{0,164,90,255},
					{0,160,94,255},{0,156,98,255},{0,152,102,255},{0,148,106,255},{0,144,110,255},{0,140,114,255},{0,136,118,255},
					{0,132,122,255},{0,128,126,255},{0,124,130,255},{0,120,134,255},{0,116,138,255},{0,112,142,255},{0,108,146,255},
					{0,104,150,255},{0,100,154,255},{0,96,158,255},{0,92,162,255},{0,88,166,255},{0,84,170,255},{0,80,174,255},
					{0,76,178,255},{0,72,182,255},{0,68,186,255},{0,64,190,255},{0,60,194,255},{0,56,198,255},{0,52,202,255},
					{0,48,206,255},{0,44,210,255},{0,40,214,255},{0,36,218,255},{0,32,222,255},{0,28,226,255},{0,24,230,255},
					{0,20,234,255},{0,16,238,255},{0,12,242,255},{0,8,246,255},{0,4,250,255},{0,0,255,255} };

				AxColorRGBA8 greyColorRamp[128] = { {0,0,0,255},{2,2,2,255},{4,4,4,255},{6,6,6,255},{8,8,8,255},{10,10,10,255},{12,12,12,255},{14,14,14,255},
					{16,16,16,255},{18,18,18,255},{20,20,20,255},{22,22,22,255},{24,24,24,255},{26,26,26,255},{28,28,28,255},{30,30,30,255},{32,32,32,255},
					{34,34,34,255},{36,36,36,255},{38,38,38,255},{40,40,40,255},{42,42,42,255},{44,44,44,255},{46,46,46,255},{48,48,48,255},{50,50,50,255},
					{52,52,52,255},{54,54,54,255},{56,56,56,255},{58,58,58,255},{60,60,60,255},{62,62,62,255},{64,64,64,255},{66,66,66,255},{68,68,68,255},
					{70,70,70,255},{72,72,72,255},{74,74,74,255},{76,76,76,255},{78,78,78,255},{80,80,80,255},{82,82,82,255},{84,84,84,255},{86,86,86,255},
					{88,88,88,255},{90,90,90,255},{92,92,92,255},{94,94,94,255},{96,96,96,255},{98,98,98,255},{100,100,100,255},{102,102,102,255},
					{104,104,104,255},{106,106,106,255},{108,108,108,255},{110,110,110,255},{112,112,112,255},{114,114,114,255},{116,116,116,255},
					{118,118,118,255},{120,120,120,255},{122,122,122,255},{124,124,124,255},{126,126,126,255},{128,128,128,255},{130,130,130,255},
					{132,132,132,255},{134,134,134,255},{136,136,136,255},{138,138,138,255},{140,140,140,255},{142,142,142,255},{144,144,144,255},
					{146,146,146,255},{148,148,148,255},{150,150,150,255},{152,152,152,255},{154,154,154,255},{156,156,156,255},{158,158,158,255},
					{160,160,160,255},{162,162,162,255},{164,164,164,255},{166,166,166,255},{168,168,168,255},{170,170,170,255},{172,172,172,255},
					{174,174,174,255},{176,176,176,255},{178,178,178,255},{180,180,180,255},{182,182,182,255},{184,184,184,255},{186,186,186,255},
					{188,188,188,255},{190,190,190,255},{192,192,192,255},{194,194,194,255},{196,196,196,255},{198,198,198,255},{200,200,200,255},
					{202,202,202,255},{204,204,204,255},{206,206,206,255},{208,208,208,255},{210,210,210,255},{212,212,212,255},{214,214,214,255},
					{216,216,216,255},{218,218,218,255},{220,220,220,255},{222,222,222,255},{224,224,224,255},{226,226,226,255},{228,228,228,255},
					{230,230,230,255},{232,232,232,255},{234,234,234,255},{236,236,236,255},{238,238,238,255},{240,240,240,255},{242,242,242,255},
					{244,244,244,255},{246,246,246,255},{248,248,248,255},{250,250,250,255},{252,252,252,255},{255,255,255,255} };

				AxColorRGBA8 customColorRamp[128] = { {0,0,0,255},{0,0,0,255},{0,0,0,255},{0,0,0,255},{0,0,0,255},{0,0,0,255},{0,0,0,255},{0,0,0,255},
					{0,0,0,255},{0,0,0,255},{0,0,0,255},{0,0,0,255},{0,0,0,255},{1,0,0,255},{1,0,0,255},{1,0,0,255},{1,0,0,255},{1,0,0,255},{1,0,0,255},
					{1,0,0,255},{1,0,0,255},{1,0,0,255},{1,0,0,255},{1,0,0,255},{1,0,0,255},{1,0,0,255},{2,0,0,255},{2,0,0,255},{2,0,0,255},{2,0,0,255},
					{2,0,0,255},{2,0,0,255},{2,0,0,255},{2,0,0,255},{3,0,0,255},{4,0,0,255},{5,0,0,255},{5,0,0,255},{6,0,0,255},{7,0,0,255},{8,0,0,255},
					{9,0,0,255},{10,0,0,255},{11,0,0,255},{12,0,0,255},{13,0,0,255},{13,0,0,255},{14,0,0,255},{15,0,0,255},{16,0,0,255},{17,0,0,255},
					{24,0,0,255},{33,1,0,255},{41,1,0,255},{50,1,0,255},{58,2,0,255},{67,2,0,255},{76,2,0,255},{84,3,0,255},{93,3,0,255},{101,3,0,255},
					{110,4,0,255},{119,4,0,255},{127,4,0,255},{136,5,0,255},{145,5,0,255},{153,5,0,255},{162,6,0,255},{170,6,0,255},{179,6,0,255},
					{188,7,0,255},{196,7,0,255},{205,7,0,255},{213,8,0,255},{222,8,0,255},{231,8,0,255},{239,9,0,255},{248,9,0,255},{255,9,0,255},
					{255,9,0,255},{255,8,0,255},{255,8,0,255},{255,8,0,255},{255,7,0,255},{255,7,0,255},{255,6,0,255},{255,6,0,255},{255,6,0,255},
					{255,5,0,255},{255,5,0,255},{255,5,0,255},{255,4,0,255},{255,4,0,255},{255,3,0,255},{255,3,0,255},{255,3,0,255},{255,2,0,255},
					{255,2,0,255},{255,2,0,255},{255,1,0,255},{255,1,0,255},{255,8,1,255},{255,15,3,255},{255,22,5,255},{255,29,7,255},{255,36,9,255},
					{255,43,10,255},{255,50,12,255},{255,57,14,255},{255,64,16,255},{255,71,17,255},{255,78,19,255},{255,85,21,255},{255,91,23,255},
					{255,98,25,255},{255,105,26,255},{255,112,28,255},{255,119,30,255},{255,126,32,255},{255,125,29,255},{255,120,26,255},{255,115,22,255},
					{255,110,18,255},{255,105,14,255},{255,100,11,255},{255,96,7,255},{255,91,3,255},{255,86,0,255} };



				AxSceneRenderDesc  sceneRenderDesc;
				
				UE_LOG(LogRenderer, Log, TEXT("LightDirection: %s"), *LightDirection.ToString());

				sceneRenderDesc.lightInfo[0].Pivot = MakeVector3(LightDirection.X*(-1000), LightDirection.Y * (-1000), LightDirection.Z * (-1000));
				sceneRenderDesc.lightInfo[0].Intensity = 10.f;
				sceneRenderDesc.lightInfo[0].LightColor = { 1.f, 1.f, 1.f, 1.f };
				sceneRenderDesc.lightNum = 1;

				sceneRenderDesc.camInfo.Pivot = AxVector3{ 
					ViewVoxelizeParameters.WorldCameraOrigin.X, 
					ViewVoxelizeParameters.WorldCameraOrigin.Y, 
					ViewVoxelizeParameters.WorldCameraOrigin.Z };
				sceneRenderDesc.camInfo.Forward = AxVector3{ 
					ViewVoxelizeParameters.ViewForward.X, 
					ViewVoxelizeParameters.ViewForward.Y, 
					ViewVoxelizeParameters.ViewForward.Z };
				sceneRenderDesc.camInfo.UpVector = AxVector3{ 
					ViewVoxelizeParameters.ViewUp.X, 
					ViewVoxelizeParameters.ViewUp.Y, 
					ViewVoxelizeParameters.ViewUp.Z };
				sceneRenderDesc.camInfo.Near = ViewVoxelizeParameters.NearPlane;
				sceneRenderDesc.camInfo.Fov = ViewVoxelizeParameters.FieldOfViewWideAngles.X * 180.f / 3.1415926f;


				//UE_LOG(LogRenderer, Log, TEXT("Image %d,%d\n"), 
				//	ImageWidth, ImageHeight);
				//UE_LOG(LogRenderer, Log, TEXT("cam fov: %f"), 
				//	sceneRenderDesc.camInfo.Fov);
				//UE_LOG(LogRenderer, Log, TEXT("cam near: %f"), 
				//	sceneRenderDesc.camInfo.Near);
				//UE_LOG(LogRenderer, Log, TEXT("cam Pivot: %f, %f, %f"),
				//	sceneRenderDesc.camInfo.Pivot.x,
				//	sceneRenderDesc.camInfo.Pivot.y,
				//	sceneRenderDesc.camInfo.Pivot.z);
				//UE_LOG(LogRenderer, Log, TEXT("cam forward: %f, %f, %f"), 
				//	sceneRenderDesc.camInfo.Forward.x, 
				//	sceneRenderDesc.camInfo.Forward.y, 
				//	sceneRenderDesc.camInfo.Forward.z);
				//UE_LOG(LogRenderer, Log, TEXT("cam up: %f, %f, %f"), 
				//	sceneRenderDesc.camInfo.UpVector.x, 
				//	sceneRenderDesc.camInfo.UpVector.y, 
				//	sceneRenderDesc.camInfo.UpVector.z);


				// 文件路径需要修改下
				std::string smokeFieldsPath = "C:/gitrepo/FusionProj-git/FusionProj/Engine/Source/Runtime/Renderer/Private/AlphaCore/boxVolume.json";
				

				std::vector<AxScalarFieldF32*> houdiniVolumes;
				AlphaCore::GridDense::ReadFields(smokeFieldsPath, houdiniVolumes);

				AxTextureR32 depthTexBuf(ImageWidth, ImageHeight);
				for (int r = 0; r < ImageHeight; ++r)
				{
					for (int c = 0; c < ImageWidth; ++c)
					{
						
						depthTexBuf.SetValue(
							(ImageHeight - 1 - r) * ImageWidth + ImageWidth - c - 1, 
							float(WorldPosData[r * ImageWidth + c].A));
					}
				}

				depthTexBuf.DeviceMalloc();

				
				AxTextureRGBA8 image(ImageWidth, ImageHeight);
				image.SetFieldResolution(ImageWidth, ImageHeight);
				image.SetToZero();
				image.DeviceMalloc();

				float stepSize = 5.f;

				AxVolumeRenderObject VolumeData;

				VolumeData.material.minMaxInputDensity = { 0, 50 };
				VolumeData.material.densityScale = 1.f;
				VolumeData.material.shadowScale = 1.f;
				std::memcpy(VolumeData.material.lookUpTableDensity, greyFloatRamp, 128 * sizeof(float));
				std::memcpy(VolumeData.material.lookUpTableDensityColor, greyColorRamp, 128 * sizeof(AxColorRGBA8));
				std::memcpy(VolumeData.material.LookUpTableHeat, greyFloatRamp, 128 * sizeof(float));
				std::memcpy(VolumeData.material.LookUpTableTemperature, customColorRamp, 128 * sizeof(AxColorRGBA8));
				VolumeData.material.minMaxInputHeat = { 0,10 };
				VolumeData.material.minMaxInputTemperature = { 0,5 };
				VolumeData.material.minMaxOuputTemperature = { 0,1 };

				for (auto vol : houdiniVolumes) {
					std::cout << vol->GetName() << std::endl;
					if (vol->GetName() == "density") {
						VolumeData.densityInfo = vol->GetFieldInfo();
						//UE_LOG(LogRenderer, Log, TEXT("density"));
						vol->DeviceMalloc();
						VolumeData.density = vol->GetRawDataDevice();
					}
					else if (vol->GetName() == "heat") {
						VolumeData.heatInfo = vol->GetFieldInfo();
						//UE_LOG(LogRenderer, Log, TEXT("heat"));
						vol->DeviceMalloc();
						VolumeData.heat = vol->GetRawDataDevice();
					}
					else if (vol->GetName() == "temperature") {
						VolumeData.tempInfo = vol->GetFieldInfo();
						//UE_LOG(LogRenderer, Log, TEXT("temperature"));
							vol->DeviceMalloc();
							VolumeData.temp = vol->GetRawDataDevice();
					}
				}

				
				AxMatrix4x4 xform = {
									0.f,-1.f, 0.f, 0.f,
									0.f, 0.f, 1.f, 0.f,
									1.f, 0.f, 0.f, 0.f,
									0.f, 0.f, 0.f, 1.f };

				AlphaCore::VolumeRender::CUDA::GasVolumeRender(VolumeData, sceneRenderDesc, &image, &depthTexBuf,
					stepSize, ImageWidth, ImageHeight, xform);
				image.LoadToHost();


				//AlphaCore::Image::SaveAsTga("C:/Users/hao.chen/Desktop/VolumeRender/depth/ue4.tga", &image);

				TArray<FLinearColor> OutputData;
				OutputData.AddDefaulted(ImageWidth * ImageHeight);
				AxColorRGBA8* ImageData = image.GetRawData();
				
				for (int r = 0; r < ImageHeight; ++r)
				{
					for (int c = 0; c < ImageWidth; ++c)
					{
						FLinearColor tmpData = FLinearColor(
							ImageData[r * ImageWidth + c].r / 255.f, 
							ImageData[r * ImageWidth + c].g / 255.f,
							ImageData[r * ImageWidth + c].b / 255.f, 
							ImageData[r * ImageWidth + c].a / 255.f);

						OutputData[(ImageHeight - r - 1) * ImageWidth + ImageWidth - 1 - c] = tmpData;
					}
				}
				
				FRHITexture* SimOutputTex = TryGetRHI(SimOutput);
				FUpdateTextureRegion2D TempRegion(0, 0, 0, 0, ImageWidth, ImageHeight);
				RHIUpdateTexture2D(SimOutputTex->GetTexture2D(), 0, TempRegion, ImageWidth * 4 * sizeof(float), (uint8*)OutputData.GetData());
				/**/
				
			});

		const FScreenPassRenderTarget Output(SceneColorTexture, View.ViewRect, ERenderTargetLoadAction::ELoad);

		const FScreenPassTexture SceneColor(SimOutput, View.ViewRect);
		AddDrawTexturePass(GraphBuilder, View, SceneColor, Output, true);
	}
}
#else
{
	unimplemented();
}
#endif
