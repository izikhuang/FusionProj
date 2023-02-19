// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class FAlphaCoreRuntimeModule : public IModuleInterface
{
public:

	static FString GetModularFeatureName()
	{
		static FString FeatureName = FString(TEXT("AlphaCoreRuntime"));
		return FeatureName;
	}

	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

private:
	void StartupAlphaCore();
	void ShutdownAlphaCore();
};
