// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

/*==============================================================================
	MaterialInstanceDynamic.cpp: MaterialInstanceDynamic implementation.
==============================================================================*/

#include "Materials/MaterialInstanceDynamic.h"
#include "UObject/Package.h"
#include "Materials/MaterialInstanceSupport.h"
#include "Engine/Texture.h"
#include "Misc/RuntimeErrors.h"
#include "UnrealEngine.h"
#include "Materials/MaterialUniformExpressions.h"
#include "Stats/StatsMisc.h"

UMaterialInstanceDynamic::UMaterialInstanceDynamic(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
}

UMaterialInstanceDynamic* UMaterialInstanceDynamic::Create(UMaterialInterface* ParentMaterial, UObject* InOuter)
{
	UObject* Outer = InOuter ? InOuter : GetTransientPackage();
	UMaterialInstanceDynamic* MID = NewObject<UMaterialInstanceDynamic>(Outer);
	MID->SetParentInternal(ParentMaterial, false);
	return MID;
}

UMaterialInstanceDynamic* UMaterialInstanceDynamic::Create(UMaterialInterface* ParentMaterial, UObject* InOuter, FName Name)
{
	UObject* Outer = InOuter ? InOuter : GetTransientPackage();
	UMaterialInstanceDynamic* MID = NewObject<UMaterialInstanceDynamic>(Outer, Name);
	MID->SetParentInternal(ParentMaterial, false);
	return MID;
}

void UMaterialInstanceDynamic::SetVectorParameterValue(FName ParameterName, FLinearColor Value)
{
	FMaterialParameterInfo ParameterInfo(ParameterName); // @TODO: This will only work for non-layered parameters
	SetVectorParameterValueInternal(ParameterInfo,Value);
}

FLinearColor UMaterialInstanceDynamic::K2_GetVectorParameterValue(FName ParameterName)
{
	FLinearColor Result(0,0,0);
	FMaterialParameterInfo ParameterInfo(ParameterName); // @TODO: This will only work for non-layered parameters
	Super::GetVectorParameterValue(ParameterInfo, Result);
	return Result;
}

void UMaterialInstanceDynamic::SetScalarParameterValue(FName ParameterName, float Value)
{
	FMaterialParameterInfo ParameterInfo(ParameterName); // @TODO: This will only work for non-layered parameters
	SetScalarParameterValueInternal(ParameterInfo,Value);
}

bool UMaterialInstanceDynamic::InitializeScalarParameterAndGetIndex(const FName& ParameterName, float Value, int32& OutParameterIndex)
{
	OutParameterIndex = INDEX_NONE;

	FMaterialParameterInfo ParameterInfo(ParameterName); // @TODO: This will only work for non-layered parameters
	SetScalarParameterValueInternal(ParameterInfo, Value);

	OutParameterIndex = GameThread_FindParameterIndexByName(ScalarParameterValues, ParameterInfo);

	return (OutParameterIndex != INDEX_NONE);
}

bool UMaterialInstanceDynamic::SetScalarParameterByIndex(int32 ParameterIndex, float Value)
{
	return SetScalarParameterByIndexInternal(ParameterIndex, Value);
}

bool UMaterialInstanceDynamic::InitializeVectorParameterAndGetIndex(const FName& ParameterName, const FLinearColor& Value, int32& OutParameterIndex)
{
	OutParameterIndex = INDEX_NONE;

	FMaterialParameterInfo ParameterInfo(ParameterName); // @TODO: This will only work for non-layered parameters
	SetVectorParameterValueInternal(ParameterInfo, Value);

	OutParameterIndex = GameThread_FindParameterIndexByName(VectorParameterValues, ParameterInfo);

	return (OutParameterIndex != INDEX_NONE);
}

bool UMaterialInstanceDynamic::SetVectorParameterByIndex(int32 ParameterIndex, const FLinearColor& Value)
{
	return SetVectorParameterByIndexInternal(ParameterIndex, Value);
}

float UMaterialInstanceDynamic::K2_GetScalarParameterValue(FName ParameterName)
{
	float Result = 0.f;
	FMaterialParameterInfo ParameterInfo(ParameterName); // @TODO: This will only work for non-layered parameters
	Super::GetScalarParameterValue(ParameterInfo, Result);
	return Result;
}

void UMaterialInstanceDynamic::SetTextureParameterValue(FName ParameterName, UTexture* Value)
{
	// Save the texture renaming as it will be useful to remap the texture streaming data.
	UTexture* RenamedTexture = NULL;

	FMaterialParameterInfo ParameterInfo(ParameterName); // @TODO: This will only work for non-layered parameters
	Super::GetTextureParameterValue(ParameterInfo, RenamedTexture);

	if (Value && RenamedTexture && Value->GetFName() != RenamedTexture->GetFName())
	{
		RenamedTextures.FindOrAdd(Value->GetFName()).AddUnique(RenamedTexture->GetFName());
	}

	SetTextureParameterValueInternal(ParameterInfo,Value);
}

UTexture* UMaterialInstanceDynamic::K2_GetTextureParameterValue(FName ParameterName)
{
	UTexture* Result = NULL;
	FMaterialParameterInfo ParameterInfo(ParameterName); // @TODO: This will only work for non-layered parameters
	Super::GetTextureParameterValue(ParameterInfo, Result);
	return Result;
}

void UMaterialInstanceDynamic::SetFontParameterValue(const FMaterialParameterInfo& ParameterInfo,class UFont* FontValue,int32 FontPage)
{
	SetFontParameterValueInternal(ParameterInfo,FontValue,FontPage);
}

void UMaterialInstanceDynamic::ClearParameterValues()
{
	ClearParameterValuesInternal();
}


// could be optimized but surely faster than GetAllVectorParameterNames()
void GameThread_FindAllScalarParameterNames(UMaterialInstance* MaterialInstance, TArray<FName>& InOutNames)
{
	while(MaterialInstance)
	{
		for(int32 i = 0, Num = MaterialInstance->ScalarParameterValues.Num(); i < Num; ++i)
		{
			InOutNames.AddUnique(MaterialInstance->ScalarParameterValues[i].ParameterInfo.Name);
		}

		MaterialInstance = Cast<UMaterialInstance>(MaterialInstance->Parent);
	}
}

// could be optimized but surely faster than GetAllVectorParameterNames()
void GameThread_FindAllVectorParameterNames(UMaterialInstance* MaterialInstance, TArray<FName>& InOutNames)
{
	while(MaterialInstance)
	{
		for(int32 i = 0, Num = MaterialInstance->VectorParameterValues.Num(); i < Num; ++i)
		{
			InOutNames.AddUnique(MaterialInstance->VectorParameterValues[i].ParameterInfo.Name);
		}

		MaterialInstance = Cast<UMaterialInstance>(MaterialInstance->Parent);
	}
}

// Finds a parameter by name from the game thread, traversing the chain up to the BaseMaterial.
FScalarParameterValue* GameThread_GetScalarParameterValue(UMaterialInstance* MaterialInstance, FName Name)
{
	UMaterialInterface* It = 0;
	FMaterialParameterInfo ParameterInfo(Name); // @TODO: This will only work for non-layered parameters

	while(MaterialInstance)
	{
		if(FScalarParameterValue* Ret = GameThread_FindParameterByName(MaterialInstance->ScalarParameterValues, ParameterInfo))
		{
			return Ret;
		}

		It = MaterialInstance->Parent;
		MaterialInstance = Cast<UMaterialInstance>(It);
	}

	return 0;
}

// Finds a parameter by name from the game thread, traversing the chain up to the BaseMaterial.
FVectorParameterValue* GameThread_GetVectorParameterValue(UMaterialInstance* MaterialInstance, FName Name)
{
	UMaterialInterface* It = 0;
	FMaterialParameterInfo ParameterInfo(Name); // @TODO: This will only work for non-layered parameters

	while(MaterialInstance)
	{
		if(FVectorParameterValue* Ret = GameThread_FindParameterByName(MaterialInstance->VectorParameterValues, ParameterInfo))
		{
			return Ret;
		}

		It = MaterialInstance->Parent;
		MaterialInstance = Cast<UMaterialInstance>(It);
	}

	return 0;
}

void UMaterialInstanceDynamic::K2_InterpolateMaterialInstanceParams(UMaterialInstance* SourceA, UMaterialInstance* SourceB, float Alpha)
{
	if(SourceA && SourceB)
	{
		UMaterial* BaseA = SourceA->GetBaseMaterial();
		UMaterial* BaseB = SourceB->GetBaseMaterial();

		if(BaseA == BaseB)
		{
			// todo: can be optimized, at least we can reserve
			TArray<FName> Names;

			GameThread_FindAllScalarParameterNames(SourceA, Names);
			GameThread_FindAllScalarParameterNames(SourceB, Names);

			// Interpolate the scalar parameters common to both materials
			for(int32 Idx = 0, Count = Names.Num(); Idx < Count; ++Idx)
			{
				FName Name = Names[Idx];

				auto ParamValueA = GameThread_GetScalarParameterValue(SourceA, Name);
				auto ParamValueB = GameThread_GetScalarParameterValue(SourceB, Name);

				if(ParamValueA || ParamValueB)
				{
					auto Default = 0.0f;

					if(!ParamValueA || !ParamValueB)
					{
						BaseA->GetScalarParameterValue(Name, Default);
					}

					auto ValueA = ParamValueA ? ParamValueA->ParameterValue : Default;
					auto ValueB = ParamValueB ? ParamValueB->ParameterValue : Default;

					SetScalarParameterValue(Name, FMath::Lerp(ValueA, ValueB, Alpha));
				}
			}

			// reused array to minimize further allocations
			Names.Empty();
			GameThread_FindAllVectorParameterNames(SourceA, Names);
			GameThread_FindAllVectorParameterNames(SourceB, Names);

			// Interpolate the vector parameters common to both
			for(int32 Idx = 0, Count = Names.Num(); Idx < Count; ++Idx)
			{
				FName Name = Names[Idx];

				auto ParamValueA = GameThread_GetVectorParameterValue(SourceA, Name);
				auto ParamValueB = GameThread_GetVectorParameterValue(SourceB, Name);

				if(ParamValueA || ParamValueB)
				{
					auto Default = FLinearColor(EForceInit::ForceInit);

					if(!ParamValueA || !ParamValueB)
					{
						BaseA->GetVectorParameterValue(Name, Default);
					}

					auto ValueA = ParamValueA ? ParamValueA->ParameterValue : Default;
					auto ValueB = ParamValueB ? ParamValueB->ParameterValue : Default;

					SetVectorParameterValue(Name, FMath::Lerp(ValueA, ValueB, Alpha));
				}
			}
		}
		else
		{
			// to find bad usage of this method
			// Maybe we can log a content error instead
			// ensure(BaseA == BaseB);
		}
	}
}

void UMaterialInstanceDynamic::K2_CopyMaterialInstanceParameters(UMaterialInterface* Source, bool bQuickParametersOnly /*= false*/)
{
	if (bQuickParametersOnly)
	{
		CopyMaterialUniformParameters(Source);	
	}
	else
	{
		CopyMaterialInstanceParameters(Source);
	}
}

void UMaterialInstanceDynamic::CopyMaterialUniformParameters(UMaterialInterface* Source)
{
	CopyMaterialUniformParametersInternal(Source);
}

void UMaterialInstanceDynamic::CopyInterpParameters(UMaterialInstance* Source)
{
	// we might expose as blueprint function so we have the input a pointer instead of a reference
	if(Source)
	{
		// copy the array and update the renderer data structures

		for (auto& it : Source->ScalarParameterValues)
		{
			SetScalarParameterValue(it.ParameterInfo.Name, it.ParameterValue);
		}

		for (auto& it : Source->VectorParameterValues)
		{
			SetVectorParameterValue(it.ParameterInfo.Name, it.ParameterValue);
		}

		for (auto& it : Source->TextureParameterValues)
		{
			SetTextureParameterValue(it.ParameterInfo.Name, it.ParameterValue);
		}

		for (auto& it : Source->FontParameterValues)
		{
			SetFontParameterValue(it.ParameterInfo.Name, it.FontValue, it.FontPage);
		}
	}
}

void UMaterialInstanceDynamic::CopyParameterOverrides(UMaterialInstance* MaterialInstance)
{
	ClearParameterValues();
	if (ensureAsRuntimeWarning(MaterialInstance != nullptr))
	{
		VectorParameterValues = MaterialInstance->VectorParameterValues;
		ScalarParameterValues = MaterialInstance->ScalarParameterValues;
		TextureParameterValues = MaterialInstance->TextureParameterValues;
		FontParameterValues = MaterialInstance->FontParameterValues;
	}
	InitResources();
}

float UMaterialInstanceDynamic::GetTextureDensity(FName TextureName, const struct FMeshUVChannelInfo& UVChannelData) const
{
	float Density = Super::GetTextureDensity(TextureName, UVChannelData);

	// Also try any renames. Note that even though it could be renamed, the texture could still be used by the parent.
	const TArray<FName>* Renames = RenamedTextures.Find(TextureName);
	if (Renames)
	{
		for (FName Rename : *Renames)
		{
			Density = FMath::Max<float>(Density, Super::GetTextureDensity(Rename, UVChannelData));
		}
	}
	return Density;
}
