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

// So we have something to give the "AddPass" calls that we add for transitions
BEGIN_SHADER_PARAMETER_STRUCT(FDummyShaderParameters, )
    SHADER_PARAMETER(int, Dummy)
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
    }

} // namespace RenderAlphaCore

#undef LOCTEXT_NAMESPACE
