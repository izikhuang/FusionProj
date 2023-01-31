/*
* Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Render/RenderAlphaCore.h"
#include "UAxSceneManager.h"
#include "AlphaCoreForUnreal.h"
#include <Utility/AxTimeTick.h>
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
#include "RendererModule.h"

#include <cmath>

#define LOCTEXT_NAMESPACE "FRenderAlphaCorePlugin"

static TAutoConsoleVariable<int32> CVarRayTracingRenderSim(
	TEXT("r.RayTracing.Simulation"),
	1,
	TEXT("Enables simulation (default = 0)"),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarRayTracingSimColorR(
	TEXT("r.RayTracing.SimColorR"),
	0,
	TEXT("Enables simulation (default = 0)"),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarRayTracingSimColorG(
	TEXT("r.RayTracing.SimColorG"),
	0,
	TEXT("Enables simulation (default = 0)"),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarRayTracingSimColorB(
	TEXT("r.RayTracing.SimColorB"),
	0,
	TEXT("Enables simulation (default = 0)"),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<float> CVarRayTracingSimColorA(
	TEXT("r.RayTracing.SimColorA"),
	1.0f,
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
		SHADER_PARAMETER_STRUCT_INCLUDE(FSceneTextureParameters, SceneTextures)
		//SHADER_PARAMETER_RDG_TEXTURE(Texture2D, Normals)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, DepthTexture)

		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, OutWorldPos)
	END_SHADER_PARAMETER_STRUCT()
};

IMPLEMENT_GLOBAL_SHADER(FRetchWorldPosCS, "/Plugin/AlphaCoreForUnreal/Private/SimAlphaCore.usf", "MainWorldPosCS", SF_Compute);

BEGIN_SHADER_PARAMETER_STRUCT(FApplyLightingAlphaCoreShaderParameters, )
	SHADER_PARAMETER_RDG_TEXTURE(Texture2D, AlphaCoreColor)
	SHADER_PARAMETER(FMatrix44f, ScreenToTranslatedWorld)
	RDG_TEXTURE_ACCESS(WorldPos, ERHIAccess::SRVGraphics)
	RDG_TEXTURE_ACCESS(SimOutput, ERHIAccess::SRVGraphics)
	RENDER_TARGET_BINDING_SLOTS()
END_SHADER_PARAMETER_STRUCT()


//
class FApplyLightingAlphaCoreShaderVS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FApplyLightingAlphaCoreShaderVS);
	SHADER_USE_PARAMETER_STRUCT(FApplyLightingAlphaCoreShaderVS, FGlobalShader);

	using FParameters = FApplyLightingAlphaCoreShaderParameters;

	using FPermutationDomain = TShaderPermutationDomain<>;

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

class FApplyLightingAlphaCoreShaderPS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FApplyLightingAlphaCoreShaderPS);
	SHADER_USE_PARAMETER_STRUCT(FApplyLightingAlphaCoreShaderPS, FGlobalShader);

	using FParameters = FApplyLightingAlphaCoreShaderParameters;

	using FPermutationDomain = TShaderPermutationDomain<>;

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

IMPLEMENT_GLOBAL_SHADER(FApplyLightingAlphaCoreShaderVS, "/Plugin/AlphaCoreForUnreal/Private/ApplyLightingAlphaCore.usf", "MainVS", SF_Vertex);
IMPLEMENT_GLOBAL_SHADER(FApplyLightingAlphaCoreShaderPS, "/Plugin/AlphaCoreForUnreal/Private/ApplyLightingAlphaCore.usf", "MainPS", SF_Pixel);



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
		FGlobalIlluminationPluginResources& Resources)
    {
		
		if (CVarRayTracingRenderSim.GetValueOnRenderThread() == 0)
		{
			return;
		}

		///////////////////////////////
		// Get AxRenderData
		///////////////////////////////
		auto SceneManager = AxSceneManager::GetInstance();
		if (SceneManager->GetWorld()->GetSimObjectNum() == 0) return;


		FRDGTextureRef WorldPosTexture = nullptr;
		FRDGTextureRef SimOutput = nullptr;
		FRDGTextureRef DepthTexture = nullptr;
		FRDGTextureRef SceneColorTexture = Resources.SceneColor;

		{
			FRDGTextureDesc Desc = FRDGTextureDesc::Create2D(
				Resources.SceneDepthZ->Desc.Extent,
				PF_FloatRGBA,
				FClearValueBinding::None,
				TexCreate_ShaderResource | TexCreate_RenderTargetable | TexCreate_UAV);

			WorldPosTexture = GraphBuilder.CreateTexture(Desc, TEXT("WorldPosTexture"));

			Desc = FRDGTextureDesc::Create2D(
				Resources.SceneDepthZ->Desc.Extent,
				PF_A32B32G32R32F,
				FClearValueBinding::None,
				TexCreate_ShaderResource | TexCreate_RenderTargetable | TexCreate_UAV);
			SimOutput = GraphBuilder.CreateTexture(Desc, TEXT("SimOutput"));
			AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(SimOutput), FLinearColor(0.0f, 0.0f, 0.0f, 0.0f));
		}

		FRetchWorldPosCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FRetchWorldPosCS::FParameters>();
		PassParameters->DepthTexture = Resources.SceneDepthZ;
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

		FLightSceneInfo* DirectionalLightSceneInfo = NULL;

		for (auto LightIt = Scene.Lights.CreateConstIterator(); LightIt; ++LightIt)
		{
			const FLightSceneInfoCompact& LightSceneInfoCompact = *LightIt;
			FLightSceneInfo* LightSceneInfo = LightSceneInfoCompact.LightSceneInfo;

			if (LightSceneInfo->Proxy->GetLightType() == LightType_Directional
				// Band-aid fix for extremely rare case that light scene proxy contains NaNs.
				&& !LightSceneInfo->Proxy->GetDirection().ContainsNaN()
				/*&& LightSceneInfo->ShouldRenderLightViewIndependent()
				&& LightSceneInfo->ShouldRenderLight(View)*/)
			{
				DirectionalLightSceneInfo = LightSceneInfo;
			}
		}

		FGlobalShaderMap* GlobalShaderMap = GetGlobalShaderMap(ERHIFeatureLevel::SM5);
		TShaderMapRef<FApplyLightingAlphaCoreShaderVS> VertexShader(GlobalShaderMap);
		TShaderMapRef<FApplyLightingAlphaCoreShaderPS> PixelShader(GlobalShaderMap);

		FApplyLightingAlphaCoreShaderParameters* ApplyLightingPassParameters = GraphBuilder.AllocParameters<FApplyLightingAlphaCoreShaderParameters>();
		ApplyLightingPassParameters->AlphaCoreColor = SimOutput;
		ApplyLightingPassParameters->WorldPos = WorldPosTexture;
		ApplyLightingPassParameters->SimOutput = SimOutput;
		ApplyLightingPassParameters->RenderTargets[0] = FRenderTargetBinding(SceneColorTexture, ERenderTargetLoadAction::ELoad);
		//FRenderSimulationParameters* SimParameters = GraphBuilder.AllocParameters<FRenderSimulationParameters>();
		//SimParameters->SceneTextures = SceneTextureBuffers;
		//SimParameters->WorldPos = WorldPosTexture;
		//SimParameters->SimOutput = SimOutput;
		ERDGPassFlags PassFlags = ERDGPassFlags::Raster;

		GraphBuilder.AddPass(
			RDG_EVENT_NAME("SimParameters"),
			ApplyLightingPassParameters,
			PassFlags,
			[ApplyLightingPassParameters, GlobalShaderMap, VertexShader, PixelShader, WorldPosTexture, SimOutput, &View, DirectionalLightSceneInfo]
		(FRHICommandListImmediate& RHICmdList)
			{
				bool debug = false;
				FRHITexture* WorldPosTex = TryGetRHI(WorldPosTexture);

				int ImageWidth = View.ViewRect.Width();
				int ImageHeight = View.ViewRect.Height();
				FReadSurfaceDataFlags readPixelFlags(RCM_UNorm);
				FIntRect IntRect(View.ViewRect.Min.X, View.ViewRect.Min.Y, ImageWidth, ImageHeight);
				TArray<FFloat16Color> WorldPosData;
				//RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
				GDynamicRHI->RHIReadSurfaceFloatData_RenderThread(RHICmdList, WorldPosTex, IntRect, WorldPosData, CubeFace_PosX, 0, 0);
				FViewUniformShaderParameters ViewVoxelizeParameters = *View.CachedViewUniformShaderParameters;
				

				/////////////////////////////////
				/// AlphaCore Sim world
				/////////////////////////////////
				auto SceneManager = AxSceneManager::GetInstance();
				if (SceneManager->GetSimWorldStatus() == 1) return;

				AxSimWorld* world = SceneManager->GetWorld();
				if (!world) return;
				if (world->GetSimObjectNum() == 0) return;

				SceneManager->SetSimWorldStepStarted();
				/////////////////////////////////
				/// Get Light Info and Camera
				/////////////////////////////////

				FVector UELightDirection;
				FLinearColor UELightColor;
				if (DirectionalLightSceneInfo)
				{
					UELightDirection = DirectionalLightSceneInfo->Proxy->GetDirection().GetSafeNormal();
					UELightColor = DirectionalLightSceneInfo->Proxy->GetColor();
				}

				// Get UE Parameters
				FVector LightDir = UELightDirection.RotateAngleAxis(-90, FVector::ZAxisVector).
					RotateAngleAxis(90, FVector::XAxisVector);
				FVector UELightDir = FVector(-LightDir.X, -LightDir.Y, -LightDir.Z);

				FVector3f CameraOrigin = ViewVoxelizeParameters.TranslatedWorldCameraOrigin.RotateAngleAxis(-90, FVector3f::ZAxisVector).
					RotateAngleAxis(90, FVector3f::XAxisVector);
				FVector UECameraOrigin = FVector(-CameraOrigin.X, -CameraOrigin.Y, -CameraOrigin.Z);

				FVector3f CamForw = ViewVoxelizeParameters.ViewForward.RotateAngleAxis(-90, FVector3f::ZAxisVector).
					RotateAngleAxis(90, FVector3f::XAxisVector);
				FVector UECamForward = FVector(-CamForw.X, -CamForw.Y, -CamForw.Z);

				FVector3f CamUp = ViewVoxelizeParameters.ViewUp.RotateAngleAxis(-90, FVector3f::ZAxisVector).
					RotateAngleAxis(90, FVector3f::XAxisVector);
				FVector UECamUp = FVector(-CamUp.X, -CamUp.Y, -CamUp.Z);
				//auto a = 
				// Convert to AxParameters
				AxVector3 LightPivot	= { UELightDir.X * (-1000.f),UELightDir.Y * (-1000.f),UELightDir.Z * (-1000.f) };
				AxVector3 CamPivot		= { UECameraOrigin.X,UECameraOrigin.Y,UECameraOrigin.Z };
				AxVector3 CamForward	= { UECamForward.X,UECamForward.Y,UECamForward.Z };
				AxVector3 UpVector		= { UECamUp.X,UECamUp.Y,UECamUp.Z };

				AxColorRGBA LightColor = { UELightColor.R / UELightColor.A, UELightColor.G / UELightColor.A, UELightColor.B / UELightColor.A, 1.f };
				float LightIntensity = UELightColor.A;

				float Near = ViewVoxelizeParameters.NearPlane;
				float Fov = ViewVoxelizeParameters.FieldOfViewWideAngles.X * 180.f / 3.1415926f;
				
				AlphaCore::Desc::AxPointLightInfo pointLight;
				pointLight.Pivot		= LightPivot;
				pointLight.Intensity	= LightIntensity;
				pointLight.LightColor	= LightColor;
				pointLight.Active		= true;

				AlphaCore::Desc::AxCameraInfo camInfo;
				camInfo.Pivot		= CamPivot;
				camInfo.Forward		= CamForward;
				camInfo.UpVector	= UpVector;
				camInfo.Near		= Near;
				camInfo.Fov			= Fov;
				
				
				world->SetSceneCamera(camInfo);
				world->SetSceneLight(0, &pointLight);

				//AX_WARN("LightPivot      : {}, {}, {}", LightPivot.x, LightPivot.y, LightPivot.z);
				//AX_WARN("LightColor      : {}, {}, {}, {}", UELightColor.R, UELightColor.G, UELightColor.B, UELightColor.A);
				//AX_WARN("LightIntensity  : {}", LightIntensity);
				//AX_WARN("camInfo.Pivot   : {}, {}, {}", CamPivot.x, CamPivot.y, CamPivot.z);
				//AX_WARN("camInfo.Forward : {}, {}, {}", CamForward.x, CamForward.y, CamForward.z);
				//AX_WARN("camInfo.UpVector: {}, {}, {}", UpVector.x, UpVector.y, UpVector.z);
				//AX_WARN("camInfo.Near    : {}", Near);
				//AX_WARN("camInfo.Fov     : {}", Fov);


				/////////////////////////////////
				/// Render in AlphaCore
				/////////////////////////////////
				if (debug)
				{
					AxTextureRGBA8 depthTex(ImageWidth, ImageHeight);
					for (int r = 0; r < ImageHeight; ++r)
					{
						for (int c = 0; c < ImageWidth; ++c)
						{
							depthTex.SetValue(r * ImageWidth + c, MakeColorRGBA8(Byte(WorldPosData[r * ImageWidth + c].A),
								Byte(WorldPosData[r * ImageWidth + c].A),
								Byte(WorldPosData[r * ImageWidth + c].A),
								Byte(WorldPosData[r * ImageWidth + c].A)));
						}
					}
					AlphaCore::Image::SaveAsTga("D:/assets/boxdepth.tga", &depthTex);
				}

				// image resize
				if (!world->GetRenderImage()) { world->RegisterRenderImage(ImageWidth, ImageHeight);}
				world->ResizeRenderImage(ImageWidth, ImageHeight);

				// depthTexture
				if (!world->GetDepthImage()) { world->RegisterDepthImage(ImageWidth, ImageHeight); }
				world->ResizeDepthImage(ImageWidth, ImageHeight);
				for (int r = 0; r < ImageHeight; ++r)
				{
					for (int c = 0; c < ImageWidth; ++c)
					{
						world->GetDepthImage()->SetValue(
							r * ImageWidth + c,
							float(WorldPosData[r * ImageWidth + c].A));
					}
				}
				world->GetDepthImage()->LoadToDevice();

				// step and render
				world->StepAndRender();
				// post render 
				AxTextureRGBA* image = world->GetRenderImage();

				if (image->GetResolution().x != ImageWidth || image->GetResolution().y != ImageHeight) return;
				image->LoadToHost();

				TArray<FLinearColor> OutputData;
				OutputData.AddDefaulted(ImageWidth * ImageHeight);
				AxColorRGBA* ImageData = image->GetRawData();

				if(debug)
				{
					AxTextureRGBA8 out(ImageWidth, ImageHeight);
					for (int r = 0; r < ImageHeight; ++r)
					{
						for (int c = 0; c < ImageWidth; ++c)
						{
							out.SetValue((ImageHeight - r-1) * ImageWidth + ImageWidth - c - 1, MakeColorRGBA8(
								Byte(ImageData[r * ImageWidth + c].r * 255.0f),
								Byte(ImageData[r * ImageWidth + c].g * 255.0f),
								Byte(ImageData[r * ImageWidth + c].b * 255.0f),
								Byte(ImageData[r * ImageWidth + c].a * 255.0f)
							));
						}
					}
					AlphaCore::Image::SaveAsTga("D:/assets/box.tga", &out);
				}

				FMemory::Memcpy(OutputData.GetData(), ImageData, ImageHeight * ImageWidth * 4 * sizeof(float));
				SceneManager->SetSimWorldStepFinished();

				FRHITexture* SimOutputTex = TryGetRHI(SimOutput);
				FUpdateTextureRegion2D TempRegion(View.ViewRect.Min.X, View.ViewRect.Min.Y, View.ViewRect.Min.X, View.ViewRect.Min.Y, ImageWidth, ImageHeight);
				RHIUpdateTexture2D(SimOutputTex->GetTexture2D(), 0, TempRegion, ImageWidth * 4 * sizeof(float), (uint8*)OutputData.GetData());

				RHICmdList.SetViewport(IntRect.Min.X, IntRect.Min.Y, 0.0f, IntRect.Max.X, IntRect.Max.Y, 1.0f);

				FGraphicsPipelineStateInitializer GraphicsPSOInit;
				RHICmdList.ApplyCachedRenderTargets(GraphicsPSOInit);
				GraphicsPSOInit.BlendState = TStaticBlendState<CW_RGBA, BO_Add, BF_SourceAlpha, BF_InverseSourceAlpha, BO_Add, BF_Zero, BF_One>::GetRHI();
				//TStaticBlendState<CW_RGBA, BO_Add, BF_One, BF_One, BO_Add, BF_One, BF_One>::GetRHI();
				GraphicsPSOInit.RasterizerState = TStaticRasterizerState<>::GetRHI();
				GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<false, CF_Always>::GetRHI();

				GraphicsPSOInit.BoundShaderState.VertexDeclarationRHI = GetVertexDeclarationFVector4();
				GraphicsPSOInit.BoundShaderState.VertexShaderRHI = VertexShader.GetVertexShader();
				GraphicsPSOInit.BoundShaderState.PixelShaderRHI = PixelShader.GetPixelShader();
				GraphicsPSOInit.PrimitiveType = PT_TriangleList;

				SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit);
				RHICmdList.SetStencilRef(0);

				SetShaderParameters(RHICmdList, VertexShader, VertexShader.GetVertexShader(), *ApplyLightingPassParameters);
				SetShaderParameters(RHICmdList, PixelShader, PixelShader.GetPixelShader(), *ApplyLightingPassParameters);

				RHICmdList.DrawPrimitive(0, 1, 1);
			});
	}

} // namespace RenderAlphaCore

#undef LOCTEXT_NAMESPACE
