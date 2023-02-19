// Copyright Epic Games, Inc. All Rights Reserved.

#include "AlphaCoreStormSystemModule.h"
#include "Modules/ModuleManager.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "ShaderCore.h"

#include "Modules/ModuleInterface.h"

#include "StormActor.h"
#include "StormOperations/OPFieldSourceComponent.h"
#include "StormOperations/OPVerticityConfinementComponent.h"

#include "StormDetailsProp.h"
#include "StormOperations/StormOperationDetailsProp.h"

#define LOCTEXT_NAMESPACE "FAlphaCoreStormSystemModule"

void FAlphaCoreStormSystemModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file per-module
	
	// StartUp AlphaCore Engine
	AlphaCoreEngine::GetInstance()->LaunchEngine();


	// Register Custom Detail Layout
	FPropertyEditorModule& PropertyModule = FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");
	PropertyModule.RegisterCustomClassLayout(
		AStormActor::StaticClass()->GetFName(),
		FOnGetDetailCustomizationInstance::CreateStatic(&FStormDetailsProp::MakeInstance));
	//PropertyModule.NotifyCustomizationModuleChanged();



	PropertyModule.RegisterCustomClassLayout(
		UOPFieldSourceComponent::StaticClass()->GetFName(),
		FOnGetDetailCustomizationInstance::CreateStatic(&FFieldSourceDetailsProp::MakeInstance));
	//PropertyModule.NotifyCustomizationModuleChanged();

	PropertyModule.RegisterCustomClassLayout(
		UOPVerticityConfinementComponent::StaticClass()->GetFName(),
		FOnGetDetailCustomizationInstance::CreateStatic(&FVerticityConfinementDetailsProp::MakeInstance));

	PropertyModule.NotifyCustomizationModuleChanged();
}

void FAlphaCoreStormSystemModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that support dynamic reloading,
	// we call this function before unloading the module.

	// UnRegister Custom Detail Layout
	if (FModuleManager::Get().IsModuleLoaded("PropertyEditor"))
	{
		FPropertyEditorModule& PropertyModule = FModuleManager::GetModuleChecked<FPropertyEditorModule>("PropertyEditor");
		PropertyModule.UnregisterCustomClassLayout(AStormActor::StaticClass()->GetFName());
		PropertyModule.UnregisterCustomClassLayout(UOPFieldSourceComponent::StaticClass()->GetFName());
		PropertyModule.UnregisterCustomClassLayout(UOPVerticityConfinementComponent::StaticClass()->GetFName());
		PropertyModule.NotifyCustomizationModuleChanged();
	}
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FAlphaCoreStormSystemModule, AlphaCoreStormSystem)