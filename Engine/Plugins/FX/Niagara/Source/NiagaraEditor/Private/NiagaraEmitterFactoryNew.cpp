// Copyright 1998-2018 Epic Games, Inc. All Rights Reserved.

#include "NiagaraEmitterFactoryNew.h"
#include "NiagaraEmitter.h"
#include "NiagaraEditorModule.h"
#include "NiagaraScriptFactoryNew.h"
#include "NiagaraGraph.h"
#include "NiagaraScriptSource.h"
#include "NiagaraSpriteRendererProperties.h"
#include "NiagaraNodeOutput.h"
#include "NiagaraEditorSettings.h"
#include "AssetData.h"
#include "ViewModels/Stack/NiagaraStackGraphUtilities.h"
#include "NiagaraConstants.h"
#include "NiagaraNodeAssignment.h"
#include "SNewEmitterDialog.h"
#include "Misc/MessageDialog.h"
#include "Modules/ModuleManager.h"
#include "Interfaces/IMainFrameModule.h"
#include "Framework/Application/SlateApplication.h"

#include "Misc/ConfigCacheIni.h"

#define LOCTEXT_NAMESPACE "NiagaraEmitterFactory"

UNiagaraEmitterFactoryNew::UNiagaraEmitterFactoryNew(const FObjectInitializer& ObjectInitializer)
: Super(ObjectInitializer)
{
	SupportedClass = UNiagaraEmitter::StaticClass();
	bCreateNew = false;
	bEditAfterNew = true;
	bCreateNew = true;
	EmitterToCopy = nullptr;
}

bool UNiagaraEmitterFactoryNew::ConfigureProperties()
{
	IMainFrameModule& MainFrame = FModuleManager::LoadModuleChecked<IMainFrameModule>("MainFrame");
	TSharedPtr<SWindow>	ParentWindow = MainFrame.GetParentWindow();

	TSharedRef<SNewEmitterDialog> NewEmitterDialog = SNew(SNewEmitterDialog);
	FSlateApplication::Get().AddModalWindow(NewEmitterDialog, ParentWindow);

	if (NewEmitterDialog->GetUserConfirmedSelection() == false)
	{
		// User cancelled or closed the dialog so abort asset creation.
		return false;
	}

	TOptional<FAssetData> SelectedEmitterAsset = NewEmitterDialog->GetSelectedEmitterAsset();
	if (SelectedEmitterAsset.IsSet())
	{
		EmitterToCopy = Cast<UNiagaraEmitter>(SelectedEmitterAsset->GetAsset());
		if (EmitterToCopy == nullptr)
		{
			FText Title = LOCTEXT("FailedToLoadTitle", "Create Default?");
			EAppReturnType::Type DialogResult = FMessageDialog::Open(EAppMsgType::OkCancel, EAppReturnType::Cancel,
				LOCTEXT("FailedToLoadMessage", "The selected emitter failed to load\nWould you like to create a default emitter?"),
				&Title);
			if (DialogResult == EAppReturnType::Cancel)
			{
				return false;
			}
			else
			{
				// The selected emitter couldn't be loaded but the user still wants to create a default emitter so 
				// well need to create a new empty emitter and add some default modules to it.
				EmitterToCopy = nullptr;
				bAddDefaultModulesAndRenderersToEmptyEmitter = true;
			}
		}
	}
	else
	{
		// User selected an empty emitter so set the emitter to copy to null, and disable creationg of default modules.
		EmitterToCopy = nullptr;
		bAddDefaultModulesAndRenderersToEmptyEmitter = false;
	}

	return true;
}

UNiagaraNodeFunctionCall* AddModuleFromAssetPath(FString AssetPath, UNiagaraNodeOutput& TargetOutputNode)
{
	FSoftObjectPath AssetRef(AssetPath);
	UNiagaraScript* AssetScript = Cast<UNiagaraScript>(AssetRef.TryLoad());
	FAssetData ScriptAssetData(AssetScript);
	if (ScriptAssetData.IsValid())
	{
		return FNiagaraStackGraphUtilities::AddScriptModuleToStack(ScriptAssetData, TargetOutputNode);
	}
	else
	{
		UE_LOG(LogNiagaraEditor, Error, TEXT("Failed to create default modules for emitter.  Missing %s"), *AssetRef.ToString());
		return nullptr;
	}
}

template<typename ValueType>
void SetRapidIterationParameter(FString UniqueEmitterName, UNiagaraScript& TargetScript, UNiagaraNodeFunctionCall& TargetFunctionCallNode,
	FName InputName, FNiagaraTypeDefinition InputType, ValueType Value)
{
	FNiagaraParameterHandle InputHandle = FNiagaraParameterHandle::CreateModuleParameterHandle(InputName);
	FNiagaraParameterHandle AliasedInputHandle = FNiagaraParameterHandle::CreateAliasedModuleParameterHandle(InputHandle, &TargetFunctionCallNode);
	FNiagaraVariable RapidIterationParameter = FNiagaraStackGraphUtilities::CreateRapidIterationParameter(UniqueEmitterName, TargetScript.GetUsage(),
		AliasedInputHandle.GetParameterHandleString(), InputType);
	RapidIterationParameter.SetValue(Value);
	bool bAddParameterIfMissing = true;
	TargetScript.RapidIterationParameters.SetParameterData(RapidIterationParameter.GetData(), RapidIterationParameter, bAddParameterIfMissing);
}

UObject* UNiagaraEmitterFactoryNew::FactoryCreateNew(UClass* Class, UObject* InParent, FName Name, EObjectFlags Flags, UObject* Context, FFeedbackContext* Warn)
{
	check(Class->IsChildOf(UNiagaraEmitter::StaticClass()));

	const UNiagaraEditorSettings* Settings = GetDefault<UNiagaraEditorSettings>();
	check(Settings);

	UNiagaraEmitter* NewEmitter;

	if (EmitterToCopy != nullptr)
	{
		NewEmitter = Cast<UNiagaraEmitter>(StaticDuplicateObject(EmitterToCopy, InParent, Name, Flags, Class));
		NewEmitter->bIsTemplateAsset = false;
		NewEmitter->TemplateAssetDescription = FText();
	}
	else
	{
		// Create an empty emitter, source, and graph.
		NewEmitter = NewObject<UNiagaraEmitter>(InParent, Class, Name, Flags | RF_Transactional);
		NewEmitter->SimTarget = ENiagaraSimTarget::CPUSim;

		UNiagaraScriptSource* Source = NewObject<UNiagaraScriptSource>(NewEmitter, NAME_None, RF_Transactional);
		UNiagaraGraph* CreatedGraph = NewObject<UNiagaraGraph>(Source, NAME_None, RF_Transactional);
		Source->NodeGraph = CreatedGraph;

		// Fix up source pointers.
		NewEmitter->GraphSource = Source;
		NewEmitter->SpawnScriptProps.Script->SetSource(Source);
		NewEmitter->UpdateScriptProps.Script->SetSource(Source);
		NewEmitter->EmitterSpawnScriptProps.Script->SetSource(Source);
		NewEmitter->EmitterUpdateScriptProps.Script->SetSource(Source);
		NewEmitter->GetGPUComputeScript()->SetSource(Source);

		// Initialize the scripts for output.
		UNiagaraNodeOutput* EmitterSpawnOutputNode = FNiagaraStackGraphUtilities::ResetGraphForOutput(*Source->NodeGraph, ENiagaraScriptUsage::EmitterSpawnScript, NewEmitter->EmitterSpawnScriptProps.Script->GetUsageId());
		UNiagaraNodeOutput* EmitterUpdateOutputNode = FNiagaraStackGraphUtilities::ResetGraphForOutput(*Source->NodeGraph, ENiagaraScriptUsage::EmitterUpdateScript, NewEmitter->EmitterUpdateScriptProps.Script->GetUsageId());
		UNiagaraNodeOutput* ParticleSpawnOutputNode = FNiagaraStackGraphUtilities::ResetGraphForOutput(*Source->NodeGraph, ENiagaraScriptUsage::ParticleSpawnScript, NewEmitter->SpawnScriptProps.Script->GetUsageId());
		UNiagaraNodeOutput* ParticleUpdateOutputNode = FNiagaraStackGraphUtilities::ResetGraphForOutput(*Source->NodeGraph, ENiagaraScriptUsage::ParticleUpdateScript, NewEmitter->UpdateScriptProps.Script->GetUsageId());

		checkf(EmitterSpawnOutputNode != nullptr && EmitterUpdateOutputNode != nullptr && ParticleSpawnOutputNode != nullptr && ParticleUpdateOutputNode != nullptr,
			TEXT("Failed to create output nodes for emitter scripts."));

		if (bAddDefaultModulesAndRenderersToEmptyEmitter)
		{
			NewEmitter->AddRenderer(NewObject<UNiagaraSpriteRendererProperties>(NewEmitter, "Renderer"));

			AddModuleFromAssetPath(TEXT("/Niagara/Modules/Emitter/EmitterLifeCycle.EmitterLifeCycle"), *EmitterUpdateOutputNode);
			
			UNiagaraNodeFunctionCall* SpawnRateNode = AddModuleFromAssetPath(TEXT("/Niagara/Modules/Emitter/SpawnRate.SpawnRate"), *EmitterUpdateOutputNode);
			if (SpawnRateNode != nullptr)
			{
				SetRapidIterationParameter(NewEmitter->GetUniqueEmitterName(), *NewEmitter->EmitterUpdateScriptProps.Script, *SpawnRateNode,
					"SpawnRate", FNiagaraTypeDefinition::GetFloatDef(), 10.0f);
			}

			AddModuleFromAssetPath(TEXT("/Niagara/Modules/Spawn/Location/SystemLocation.SystemLocation"), *ParticleSpawnOutputNode);

			UNiagaraNodeFunctionCall* AddVelocityNode = AddModuleFromAssetPath(TEXT("/Niagara/Modules/Spawn/Velocity/AddVelocity.AddVelocity"), *ParticleSpawnOutputNode);
			if (AddVelocityNode != nullptr)
			{
				SetRapidIterationParameter(NewEmitter->GetUniqueEmitterName(), *NewEmitter->SpawnScriptProps.Script, *AddVelocityNode,
					"Velocity", FNiagaraTypeDefinition::GetVec3Def(), FVector(0.0f, 0.0f, 100.0f));
			}

			TArray<FNiagaraVariable> Vars =
			{
				SYS_PARAM_PARTICLES_SPRITE_SIZE,
				SYS_PARAM_PARTICLES_SPRITE_ROTATION,
				SYS_PARAM_PARTICLES_LIFETIME
			};

			TArray<FString> Defaults = 
			{
				FNiagaraConstants::GetAttributeDefaultValue(SYS_PARAM_PARTICLES_SPRITE_SIZE),
				FNiagaraConstants::GetAttributeDefaultValue(SYS_PARAM_PARTICLES_SPRITE_ROTATION),
				FNiagaraConstants::GetAttributeDefaultValue(SYS_PARAM_PARTICLES_LIFETIME)
			};

			FNiagaraStackGraphUtilities::AddParameterModuleToStack(Vars, *ParticleSpawnOutputNode, INDEX_NONE, Defaults);

			AddModuleFromAssetPath(TEXT("/Niagara/Modules/Update/Lifetime/UpdateAge.UpdateAge"), *ParticleUpdateOutputNode);
			AddModuleFromAssetPath(TEXT("/Niagara/Modules/Update/Color/Color.Color"), *ParticleUpdateOutputNode);
			AddModuleFromAssetPath(TEXT("/Niagara/Modules/Solvers/SolveForcesAndVelocity.SolveForcesAndVelocity"), *ParticleUpdateOutputNode);
		}

		FNiagaraStackGraphUtilities::RelayoutGraph(*Source->NodeGraph);
		NewEmitter->bInterpolatedSpawning = true;
		NewEmitter->SpawnScriptProps.Script->SetUsage(ENiagaraScriptUsage::ParticleSpawnScriptInterpolated);
	}
	
	return NewEmitter;
}

#undef LOCTEXT_NAMESPACE
