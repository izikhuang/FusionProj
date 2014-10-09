// Copyright 1998-2014 Epic Games, Inc. All Rights Reserved.

#pragma once

#include "EdGraph/EdGraphPin.h" // for EBlueprintPinStyleType
#include "BlueprintEditorSettings.generated.h"


UCLASS(config=EditorUserSettings)
class BLUEPRINTGRAPH_API UBlueprintEditorSettings
	:	public UObject
{
	GENERATED_UCLASS_BODY()

// General Settings
public:
	/** Determines if Blueprints are saved whenever you (successfully) compile them */
	UPROPERTY(EditAnywhere, config, Category=General)
	bool bSaveOnCompile;

// Style Settings
public:
	/** Should arrows indicating data/execution flow be drawn halfway along wires? */
	UPROPERTY(EditAnywhere, config, Category=VisualStyle, meta=(DisplayName="Draw midpoint arrows in Blueprints"))
	bool bDrawMidpointArrowsInBlueprints;

// UX Settings
public:
	/** Determines if lightweight tutorial text shows up at the top of empty blueprint graphs */
	UPROPERTY(EditAnywhere, config, Category=UserExperience)
	bool bShowGraphInstructionText;

	/** If enabled, will use the blueprint's (or output pin's) class to narrow down context menu results. */
	UPROPERTY(EditAnywhere, config, Category=UserExperience)
	bool bUseTargetContextForNodeMenu;

	/** If enabled, then ALL component functions are exposed to the context menu (when the contextual target is a component owner). Ignores "ExposeFunctionCategories" metadata for components. */
	UPROPERTY(EditAnywhere, config, Category=UserExperience, meta=(DisplayName="Context Menu: Expose All Sub-Component Functions"))
	bool bExposeAllMemberComponentFunctions;

	/** If enabled, then a separate section with your Blueprint favorites will be pined to the top of the context menu. */
	UPROPERTY(EditAnywhere, config, Category=UserExperience, meta=(DisplayName="Context Menu: Show Favorites Section"))
	bool bShowContextualFavorites;

	/** If enabled, then your Blueprint favorites will be uncategorized, leaving you with less nested categories to sort through. */
	UPROPERTY(EditAnywhere, config, Category=UserExperience)
	bool bFlattenFavoritesMenus;

	/** If set, then the new refactored menu system will be replaced with the old (legacy) system (as a fallback, in case the new system has unforeseen problems)*/
	UPROPERTY(EditAnywhere, AdvancedDisplay, config, Category=UserExperience)
	bool bUseLegacyMenuingSystem;

// Developer Settings
public:
	/** If enabled, tooltips on action menu items will show the associated action's signature id (can be used to setup custom favorites menus).*/
	UPROPERTY(EditAnywhere, config, Category=DeveloperTools)
	bool bShowActionMenuItemSignatures;

// Perf Settings
public:
	/** If enabled, additional details will be displayed in the Compiler Results tab after compiling a blueprint. */
	UPROPERTY(EditAnywhere, config, Category=Performance)
	bool bShowDetailedCompileResults;

	/** Minimum event time threshold used as a filter when additional details are enabled for display in the Compiler Results tab. A value of zero means that all events will be included in the final summary. */
	UPROPERTY(EditAnywhere, AdvancedDisplay, config, Category=Performance, DisplayName="Compile Event Results Threshold (ms)", meta=(ClampMin="0", UIMin="0"))
	int32 CompileEventDisplayThresholdMs;

	/** The node template cache is used to speed up blueprint menuing. This determines the peak data size for that cache. */
	UPROPERTY(EditAnywhere, AdvancedDisplay, config, Category=Performance, DisplayName="Node-Template Cache Cap (MB)", meta=(ClampMin="0", UIMin="0"))
	float NodeTemplateCacheCapMB;
};
