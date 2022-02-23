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
#include "../AlphaCore/include/AlphaCore.h"
#include "../AlphaCore/include/Visualization/copy_kernel.h"
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
		SimParameters->SimOutput = SimOutput;
		ERDGPassFlags PassFlags = ERDGPassFlags::Raster;
		//UE_LOG(LogRenderer, Warning, TEXT("--------------------------------------------------------"));
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
			FLinearColor TempWorldPos = FLinearColor(WorldPosData[100]);
			//UE_LOG(LogRenderer, Log, TEXT("WorldPosData %s"), *TempWorldPos.ToString());
			//UE_LOG(LogRenderer, Log, TEXT("camera pos: %s"), *ViewVoxelizeParameters.WorldCameraOrigin.ToString());
			//UE_LOG(LogRenderer, Log, TEXT("camera forword: %s"), *ViewVoxelizeParameters.ViewForward.ToString());
			//UE_LOG(LogRenderer, Log, TEXT("camera up: %s"), *ViewVoxelizeParameters.ViewUp.ToString());

			auto size = WorldPosTex->GetSizeXYZ();
			auto tex3D = WorldPosTex->GetTexture3D();
			tex3D->GetSizeZ();
			size.Num();


			if (DirectionalLightSceneInfo)
			{ 
				const FVector LightDirection = DirectionalLightSceneInfo->Proxy->GetDirection().GetSafeNormal();
				UE_LOG(LogRenderer, Log, TEXT("DirectionLight LightDirection: %s"), *LightDirection.ToString());
			}

			AlphaCore::Desc::AxPointLightInfo lightInfo;
			// lightInfo.Pivot = MakeVector3(900.f, 900.f, 1200.f);
			lightInfo.Pivot = AxVector3{ 0.f, 150.f, 0.f };
			lightInfo.Intensity = 30;
			lightInfo.LightColor = { 1.f, 1.f, 1.f, 1.f };
			AlphaCore::Desc::AxCameraInfo camInfo;


			camInfo.Pivot = AxVector3{ ViewVoxelizeParameters.WorldCameraOrigin.X, ViewVoxelizeParameters.WorldCameraOrigin.Y, ViewVoxelizeParameters.WorldCameraOrigin.Z };
			camInfo.Forward = AxVector3{ ViewVoxelizeParameters.ViewForward.X, ViewVoxelizeParameters.ViewForward.Y, ViewVoxelizeParameters.ViewForward.Z };
			camInfo.UpVector = AxVector3{ ViewVoxelizeParameters.ViewUp.X, ViewVoxelizeParameters.ViewUp.Y, ViewVoxelizeParameters.ViewUp.Z };
			camInfo.Near = ViewVoxelizeParameters.NearPlane;
			camInfo.Fov = ViewVoxelizeParameters.FieldOfViewWideAngles.X * 180.f / 3.1415926f;

			//UE_LOG(LogRenderer, Log, TEXT("AlphaCore Cam Fov: %f"), camInfo.Fov);
			//UE_LOG(LogRenderer, Log, TEXT("AlphaCore Cam Near: %f"), camInfo.Near);
			//UE_LOG(LogRenderer, Log, TEXT("AlphaCore Cam Pos: %f, %f, %f"), camInfo.Pivot.x, camInfo.Pivot.y, camInfo.Pivot.z);
			//UE_LOG(LogRenderer, Log, TEXT("AlphaCore Cam Forward: %f, %f, %f"), camInfo.Forward.x, camInfo.Forward.y, camInfo.Forward.z);
			//UE_LOG(LogRenderer, Log, TEXT("AlphaCore Cam Up: %f, %f, %f"), camInfo.UpVector.x, camInfo.UpVector.y, camInfo.UpVector.z);
			constexpr float densityFactor = 0.05f;
			std::string smokeFieldsPath = "C:/Users/hao.chen/Desktop/output_t.axc";
			std::vector<AxScalarFieldF32*> houdiniVolumes;
			AlphaCore::GridDense::ReadFields(smokeFieldsPath, houdiniVolumes);

			float stepSize = 5.f;

			AxImageRGBA worldPosImg(ImageWidth, ImageHeight);
			auto worldPosImgData = (float*)worldPosImg.GetRawData();
			for (int r = 0; r < ImageHeight; ++r)
			{
				for (int c = 0; c < ImageWidth; ++c)
				{
					worldPosImgData[r * ImageWidth * 4 + c * 4] = 0.f;
					worldPosImgData[r * ImageWidth * 4 + c * 4 + 1] = 0.f;
					worldPosImgData[r * ImageWidth * 4 + c * 4 + 2] = 0.f;
					worldPosImgData[r * ImageWidth * 4 + c * 4 + 3] = 0.f;
				}
			}

			AxImageRGBA8 renderImage(ImageWidth, ImageHeight);

			// GPU
			houdiniVolumes[0]->DeviceMalloc();
			AxVolumeRenderObjectRawData data;
			data.densityInfo = houdiniVolumes[0]->GetFieldInfo();
			data.density = houdiniVolumes[0]->GetRawDataDevice();
			AxVolumeMaterial material;
			material.minMaxInputDensity = { 0, 600 };
			material.densityScale = 1.f;
			material.shadowScale = 1.f;
			renderImage.DeviceMalloc();
			worldPosImg.DeviceMalloc();
			uchar4* cudaOutput = (uchar4*)renderImage.GetRawDataDevice();
			float4* worldPosTex = (float4*)worldPosImg.GetRawDataDevice();

			CUDA_CHECK(cudaMemset(cudaOutput, 128, ImageWidth * ImageHeight * 4));
			volume_kernel(data, material, worldPosTex, cudaOutput, camInfo, lightInfo, stepSize, ImageWidth, ImageHeight);
			renderImage.LoadToHost();


			//AlphaCore::Image::SaveAsTga("C:/Users/hao.chen/Desktop/ue4.tga", &renderImage);
			//UE_LOG(LogRenderer, Log, TEXT("save volume render result"));
			
			TArray<FLinearColor> OutputData;

			OutputData.AddDefaulted(ImageWidth * ImageHeight);
			auto output = renderImage.GetRawData();
			//for (int i = 0; i < OutputData.Num(); ++i) {
			//	FLinearColor TempColor = FLinearColor(output[i].r / 255.f, output[i].g / 255.f,
			//		output[i].b / 255.f, output[i].a / 255.f);
			//	OutputData[i] = TempColor;
			//}
			for (int r = 0; r < ImageHeight; ++r)
			{
				for (int c = 0; c < ImageWidth; ++c)
				{
					FLinearColor TempColor = FLinearColor(output[r * ImageWidth + c].r / 255.f, output[r * ImageWidth + c].g / 255.f,
						output[r * ImageWidth + c].b / 255.f, output[r * ImageWidth + c].a / 255.f);
					OutputData[(ImageHeight - r - 1) * ImageWidth + ImageWidth -1 - c] = TempColor;
				}
			}

			FRHITexture* SimOutputTex = TryGetRHI(SimOutput);
			FUpdateTextureRegion2D TempRegion(0, 0, 0, 0, ImageWidth, ImageHeight);
			RHIUpdateTexture2D(SimOutputTex->GetTexture2D(), 0, TempRegion, ImageWidth * 4 * sizeof(float), (uint8*)OutputData.GetData());
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
