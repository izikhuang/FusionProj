// Copyright Epic Games, Inc. All Rights Reserved.

#include "AlphaCore.h"
#include "Modules/ModuleManager.h"
#include "Interfaces/IPluginManager.h"

#include "RenderAlphaCore.h"
#include "Misc/Paths.h"
#include "ShaderCore.h"

#define LOCTEXT_NAMESPACE "FAlphaCoreModule"

void FAlphaCoreModule::StartupAlphaCore()
{
	RenderAlphaCore::Startup();
}

void FAlphaCoreModule::ShutdownAlphaCore()
{
	RenderAlphaCore::Shutdown();
}

void FAlphaCoreModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file per-module

	 // Get the base directory of this plugin
	FString BaseDir = IPluginManager::Get().FindPlugin(GetModularFeatureName())->GetBaseDir();

	// Register the shader directory
	FString PluginShaderDir = FPaths::Combine(BaseDir, TEXT("Shaders"));
	FString PluginMapping = TEXT("/Plugin/") + GetModularFeatureName();
	AddShaderSourceDirectoryMapping(PluginMapping, PluginShaderDir);

	StartupAlphaCore();
}

void FAlphaCoreModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that support dynamic reloading,
	// we call this function before unloading the module.
	ShutdownAlphaCore();
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FAlphaCoreModule, AlphaCore)