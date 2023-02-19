// Copyright Epic Games, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "Components/StaticMeshComponent.h"
#include "Components/PrimitiveComponent.h"
#include "DrawDebugHelpers.h"
#include "StormOperations/StormOperationComponent.h"
#include "OPVerticityConfinementComponent.generated.h"


UCLASS(Blueprintable, ClassGroup = Rendering, hideCategories = (Activation, Collision, Cooking, HLOD, Navigation), meta = (BlueprintSpawnableComponent))
class ALPHACOREFORUNREAL_API UOPVerticityConfinementComponent : public UAlphaCoreStormOperationComponent
{
	GENERATED_UCLASS_BODY()
public:

	virtual ~UOPVerticityConfinementComponent();

	////////////////////////////////////////
	//// VerticityConfinement Propertyies
	////////////////////////////////////////
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|VerticityConfinement")
	float ConfinementScale = 0.0;

	virtual FString GetOPType() override { return FString("VerticityConfinement"); }
};
