// Copyright Epic Games, Inc. All Rights Reserved.

#include "AlphaCoreRuntimeModule.h"
#include "Modules/ModuleManager.h"
#include "Interfaces/IPluginManager.h"
#include "Render/RenderAlphaCore.h"
#include "Misc/Paths.h"
#include "ShaderCore.h"

#include "Modules/ModuleInterface.h"
#include <AlphaCore.h>
#include <AxUE4Log.h>


#define LOCTEXT_NAMESPACE "FAlphaCoreRuntimeModule"

void FAlphaCoreRuntimeModule::StartupAlphaCore()
{
	AlphaCore::ActiveUELog();
	AlphaCoreEngine::GetInstance()->LaunchEngine();
	////AlphaCore::Logger::GetInstance()->SetLogPath("D:/log/");
	AX_WARN("AlphaCoreForUnreal Launched!");
	RenderAlphaCore::Startup();
}

void FAlphaCoreRuntimeModule::ShutdownAlphaCore()
{
	RenderAlphaCore::Shutdown();
}

void FAlphaCoreRuntimeModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file per-module
	// Get the base directory of this plugin
	/*
	FString BaseDir = IPluginManager::Get().FindPlugin(GetModularFeatureName())->GetBaseDir();

	// Register the shader directory
	FString PluginShaderDir = FPaths::Combine(BaseDir, TEXT("Shaders"));
	FString PluginMapping = TEXT("/Plugin/") + GetModularFeatureName();
	UE_LOG(LogTemp, Warning, TEXT("PluginMapping PluginMapping %s"), *FString(PluginMapping));
	UE_LOG(LogTemp, Warning, TEXT("PluginShaderDir PluginShaderDir %s"), *FString(PluginShaderDir));

	AddShaderSourceDirectoryMapping(PluginMapping, PluginShaderDir);
	*/

	TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("AlphaCoreForUnreal"));
	FString PluginMapping = TEXT("/Plugin/AlphaCoreForUnreal");
	FString PluginShaderDir = FPaths::Combine(Plugin->GetBaseDir(), TEXT("Shaders"));
	AddShaderSourceDirectoryMapping(PluginMapping, PluginShaderDir);

	StartupAlphaCore();
}

void FAlphaCoreRuntimeModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that support dynamic reloading,
	// we call this function before unloading the module.
	ShutdownAlphaCore();
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FAlphaCoreRuntimeModule, AlphaCoreRuntime)