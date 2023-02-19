// Copyright Epic Games, Inc. All Rights Reserved.

#include "AlphaCoreForUnreal.h"
#include "Modules/ModuleManager.h"
#include "Interfaces/IPluginManager.h"
#include "Render/RenderAlphaCore.h"
#include "Misc/Paths.h"
#include "ShaderCore.h"

#include "Storm/StormDetailsProp.h"
#include "Storm/StormActor.h"
#include "Modules/ModuleInterface.h"

#include "AxUE4Log.h"

#define LOCTEXT_NAMESPACE "FAlphaCoreForUnrealModule"

void FAlphaCoreForUnrealModule::StartupAlphaCore()
{
	AlphaCore::ActiveUELog();
	AlphaCoreEngine::GetInstance()->LaunchEngine();
	////AlphaCore::Logger::GetInstance()->SetLogPath("D:/log/");
	AX_WARN("AlphaCoreForUnreal Launched!");
	RenderAlphaCore::Startup();





}

void FAlphaCoreForUnrealModule::ShutdownAlphaCore()
{
	RenderAlphaCore::Shutdown();




}

void FAlphaCoreForUnrealModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file per-module

	// Register Custom Detail Layout
	FPropertyEditorModule& PropertyModule = FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");
	PropertyModule.RegisterCustomClassLayout(
		AStormActor::StaticClass()->GetFName(),
		FOnGetDetailCustomizationInstance::CreateStatic(&FStormDetailsProp::MakeInstance));
	PropertyModule.NotifyCustomizationModuleChanged();


	// Get the base directory of this plugin
	FString BaseDir = IPluginManager::Get().FindPlugin(GetModularFeatureName())->GetBaseDir();

	// Register the shader directory
	FString PluginShaderDir = FPaths::Combine(BaseDir, TEXT("Shaders"));
	FString PluginMapping = TEXT("/Plugin/") + GetModularFeatureName();
	UE_LOG(LogTemp, Warning, TEXT("PluginMapping PluginMapping %s"), *FString(PluginMapping));
	UE_LOG(LogTemp, Warning, TEXT("PluginShaderDir PluginShaderDir %s"), *FString(PluginShaderDir));

	AddShaderSourceDirectoryMapping(PluginMapping, PluginShaderDir);

	StartupAlphaCore();




}

void FAlphaCoreForUnrealModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that support dynamic reloading,
	// we call this function before unloading the module.
	ShutdownAlphaCore();


	// UnRegister Custom Detail Layout
	if (FModuleManager::Get().IsModuleLoaded("PropertyEditor"))
	{
		FPropertyEditorModule& PropertyModule = FModuleManager::GetModuleChecked<FPropertyEditorModule>("PropertyEditor");
		PropertyModule.UnregisterCustomClassLayout("StormActor");
		PropertyModule.NotifyCustomizationModuleChanged();
	}
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FAlphaCoreForUnrealModule, AlphaCoreForUnreal)