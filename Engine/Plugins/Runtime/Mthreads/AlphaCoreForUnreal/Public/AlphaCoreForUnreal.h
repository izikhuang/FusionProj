// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "AlphaCore.h"

class FAlphaCoreForUnrealModule : public IModuleInterface
{
public:

	static FString GetModularFeatureName()
	{
		static FString FeatureName = FString(TEXT("AlphaCoreForUnreal"));
		return FeatureName;
	}

	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

private:
	void StartupAlphaCore();
	void ShutdownAlphaCore();
};
