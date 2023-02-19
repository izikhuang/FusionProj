// Copyright Epic Games, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "Components/StaticMeshComponent.h"
#include "Components/PrimitiveComponent.h"
#include "DrawDebugHelpers.h"
#include "StormOperations/StormOperationComponent.h"
#include "OPFieldSourceComponent.generated.h"


UCLASS(Blueprintable, ClassGroup = Rendering, hideCategories = (Activation, Collision, Cooking, HLOD, Navigation), meta = (BlueprintSpawnableComponent))
class ALPHACOREFORUNREAL_API UOPFieldSourceComponent : public UAlphaCoreStormOperationComponent
{
	GENERATED_UCLASS_BODY()
public:

	virtual ~UOPFieldSourceComponent();

	//////////////////////////////////
	//// FieldSource Propertyies
	//////////////////////////////////
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|FieldSource")
		FVector2D Frequency;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|FieldSource")
		float NoiseAmp;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|FieldSource")
		float NoiseSize;

	virtual FString GetOPType() override { return FString("FieldSource"); }
};
