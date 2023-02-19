// Copyright Epic Games, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "Components/StaticMeshComponent.h"
#include "Components/PrimitiveComponent.h"
#include "DrawDebugHelpers.h"
#include "StormOperations/StormOperationComponent.h"
#include "OPFieldSourceComponent.generated.h"


UCLASS(HideCategories = (Activation, Collision, Cooking, HLOD, Navigation, LOD, RayTracing, TextureStreaming, Lighting, MaterialParameters,Physics, VirtualTexture,Mobility, Mobile, Tags,AssetUserData))
class ALPHACORESTORMSYSTEM_API UOPFieldSourceComponent : public UAlphaCoreStormOperationComponent
{
	GENERATED_UCLASS_BODY()
public:

	virtual ~UOPFieldSourceComponent();

	//////////////////////////////////
	//// FieldSource Propertyies
	//////////////////////////////////
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Basic", meta = (DisplayName = "ProjectToTerrain"))
		bool Projection = false;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Basic", meta = (ClampMin = "2", UIMin = "2", ClampMax = "4", UIMax = "4", DisplayName = "Thickness"))
		int Prolong = 2;


	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Noise|FirstShape", meta = (ClampMin = "0", UIMin = "0", ClampMax = "2", UIMax = "2"))
		float NoiseAmp = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Noise|FirstShape", meta = (ClampMin = "0", UIMin = "0", ClampMax = "1", UIMax = "1"))
		float NoiseRoughness = 0.5f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Noise|FirstShape", meta = (ClampMin = "0", UIMin = "0", ClampMax = "10", UIMax = "10"))
		int NoiseTurbulence = 3;

	//UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Noise|FirstShape", meta = (ClampMin = "0", UIMin = "0", ClampMax = "1", UIMax = "1"))
		float NoiseAttenuation = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Noise|FirstShape", meta = (ClampMin = "0", UIMin = "0", ClampMax = "1", UIMax = "1"))
		FVector2D Frequency = { 0.05,0.05 };

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Noise|FirstShape", meta = (ClampMin = "-10", UIMin = "-10", ClampMax = "10", UIMax = "10"))
		FVector4 NoiseOffset = FVector4(0.0, 0.0, 0.0, 1.0);
	
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Noise|SecondShape", meta = (ClampMin = "-5", UIMin = "-5", ClampMax = "5", UIMax = "5"))
		bool UseSecondShape = false;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Noise|SecondShape", meta = (ClampMin = "0", UIMin = "0", ClampMax = "2", UIMax = "2"))
		float NoiseAmp2 = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Noise|SecondShape", meta = (ClampMin = "0", UIMin = "0", ClampMax = "1", UIMax = "1"))
		float NoiseRoughness2 = 0.5f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Noise|SecondShape", meta = (ClampMin = "0", UIMin = "0", ClampMax = "10", UIMax = "10"))
		int NoiseTurbulence2 = 3;

	//UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Noise|SecondShape", meta = (ClampMin = "0", UIMin = "0", ClampMax = "1", UIMax = "1"))
		float NoiseAttenuation2 = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Noise|SecondShape", meta = (ClampMin = "0", UIMin = "0", ClampMax = "1", UIMax = "1"))
		FVector2D Frequency2 = { 0.05,0.05 };

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Noise|SecondShape", meta = (ClampMin = "-10", UIMin = "-10", ClampMax = "10", UIMax = "10"))
		FVector4 NoiseOffset2 = FVector4(0.0,0.0,0.0,1.0);



	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Atomosphere", meta = (ClampMin = "0", UIMin = "0", ClampMax = "2", UIMax = "2"))
		float HeatNoiseScale = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Atomosphere", meta = (ClampMin = "0", UIMin = "0", ClampMax = "1", UIMax = "1"))
		float RelHumidityGround = 0.7f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Atomosphere", meta = (ClampMin = "0", UIMin = "0", ClampMax = "1", UIMax = "1"))
		float DensityNoiseScale = 0.1f;



	virtual FString GetOPType() override { return FString("FieldSource"); }
};
