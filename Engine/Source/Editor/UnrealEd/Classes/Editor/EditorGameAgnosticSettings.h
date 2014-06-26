// Copyright 1998-2014 Epic Games, Inc. All Rights Reserved.

#pragma once
#include "EditorGameAgnosticSettings.generated.h"

UCLASS(config=EditorGameAgnostic)
class UEditorGameAgnosticSettings : public UObject
{
	GENERATED_UCLASS_BODY()

	/** When checked, the most recently loaded project will be auto-loaded at editor startup if no other project was specified on the command line */
	UPROPERTY(EditAnywhere, Category=Startup)
	bool bLoadTheMostRecentlyLoadedProjectAtStartup; // Note that this property is NOT config since it is not necessary to save the value to ini. It is determined at startup in UEditorEngine::InitEditor().


	// =====================================================================
	// The following options are NOT exposed in the preferences Editor
	// (usually because there is a different way to set them interactively!)

	/** Game project files that were recently opened in the editor */
	UPROPERTY(config)
	TArray<FString> RecentlyOpenedProjectFiles;

	/** The paths of projects created with the new project wizard. This is used to populate the "Path" field of the new project dialog. */
	UPROPERTY(config)
	TArray<FString> CreatedProjectPaths;

	UPROPERTY(config)
	bool bCopyStarterContentPreference;

	UPROPERTY(config)
	bool bShowPerformanceWarningPreference;

	/** The id's of the surveys completed */
	UPROPERTY(config)
	TArray<FGuid> CompletedSurveys;

	/** The id's of the surveys currently in-progress */
	UPROPERTY(config)
	TArray<FGuid> InProgressSurveys;

	// Begin UObject Interface
	virtual void PostEditChangeProperty( struct FPropertyChangedEvent& PropertyChangedEvent) override;
	// End UObject Interface
};