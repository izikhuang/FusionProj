// Copyright Epic Games, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "Components/StaticMeshComponent.h"
#include "Components/PrimitiveComponent.h"
#include "DrawDebugHelpers.h"
#include "StormOperations/StormOperationComponent.h"
#include "OPVerticityConfinementComponent.generated.h"


UCLASS(HideCategories = (StaticMesh, Rendering, Mobile, Transform, Activation, Collision, Cooking, HLOD, Navigation, LOD, RayTracing, TextureStreaming, Lighting, MaterialParameters, Physics, VirtualTexture, Mobility, Tags, AssetUserData))
class ALPHACORESTORMSYSTEM_API UOPVerticityConfinementComponent : public UAlphaCoreStormOperationComponent
{
	GENERATED_UCLASS_BODY()
public:

	virtual ~UOPVerticityConfinementComponent();

	////////////////////////////////////////
	//// VerticityConfinement Propertyies
	////////////////////////////////////////
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|VerticityConfinement")
	float ConfinementScale = 0.1f;

	virtual FString GetOPType() override { return FString("VerticityConfinement"); }
};
