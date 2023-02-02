// Copyright Epic Games, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "Components/StaticMeshComponent.h"
#include "DrawDebugHelpers.h"
#include "StormEmitterComponent.generated.h"


UCLASS(Blueprintable, ClassGroup = Rendering, hideCategories = (Activation, Collision, Cooking, HLOD, Navigation), meta = (BlueprintSpawnableComponent))
class ALPHACOREFORUNREAL_API UStormEmitterComponent : public UStaticMeshComponent
{
	GENERATED_UCLASS_BODY()
public:

	virtual ~UStormEmitterComponent();


};
