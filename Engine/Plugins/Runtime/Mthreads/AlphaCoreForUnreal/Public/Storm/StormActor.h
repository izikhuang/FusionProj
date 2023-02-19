#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "UObject/ObjectPtr.h"
#include "Engine/StaticMeshActor.h"
#include "Components/StaticMeshComponent.h"
#include "Components/SceneComponent.h"
#include "Components/BoxComponent.h"
#include "StormOperations/StormOperationComponent.h"
#include "AxUESceneManager.h"
#include "StormActor.generated.h"

UCLASS()
class ALPHACOREFORUNREAL_API AStormActor : public AActor
{
	GENERATED_BODY()
public:
	AStormActor();
	~AStormActor() {};
	USceneComponent* SceneComponent;
	UBoxComponent* StormBoundingBox;

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;
private:
	TArray<FVector> GetMountainMeshData(int lod);
public:
	///////////////////////////
	//// Test Propertyies
	///////////////////////////
	UPROPERTY(EditAnywhere, Category = "AlphaCore")
	TArray<TObjectPtr<class UStaticMeshComponent>> StaticMeshs;

	UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Test")
	FString AlphaCoreJson;

	UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Test")
	TMap<TObjectPtr<class AStaticMeshActor>, uint32> StormCollisionActorsAndLOD;

	UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Test")
	TObjectPtr<class UCurveLinearColor> VolumeDensity = nullptr;

	UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Test")
	TObjectPtr<class UCurveLinearColor> VolumeColor = nullptr;

	UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Test")
	FLinearColor Color = FLinearColor(1.0, 1.0, 1.0, 1.0);
	
	
	///////////////////////////
	//// Storm Base Properties
	///////////////////////////
	
	// StaticMountainComponent used to cast height field.
	UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Base")
	TObjectPtr<class AStaticMeshActor> StaticMountain;
	
	UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Base", meta = (ClampMin = "0", UIMin = "0", ClampMax = "10", UIMax = "10"))
	int StaticMountainLOD;
	
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Base", meta = (ClampMin = "10", UIMin = "10", ClampMax = "512", UIMax = "512"))
	FIntVector StormSimulationFieldResolution = { 128,128,128 };

	///////////////////////////
	//// Storm Emitter
	///////////////////////////
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Operations")
	TArray<TObjectPtr<class UAlphaCoreStormOperationComponent>> StormOperation;


	///////////////////////////
	//// Storm Propertyies
	///////////////////////////
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Storm")
	float TemLapseRateLow;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Storm")
	float TemLapseRateHigh;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Storm")
	float TemInversionHeight;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Storm")
	float AuthenticDomainHeight;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Storm")
	float DiffusionCoeff;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Storm")
	float HeatEmitterAmp;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Storm")
	float HeatNoiseScale;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Storm")
	float RelHumidityGround;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Storm")
	float DensityNoiseScale;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Storm")
	float WindSpeed;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Storm")
	float WindIntensity;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Storm", meta = (ClampMin = "0", UIMin = "0", ClampMax = "5", UIMax = "5"))
	FVector WindDirection;




public:

	void OnAlphaCoreJsonChanged();
	void OnSamplesPerPixelChanged();
	void OnStormCollisionActorsAndLODChanged();
	void OnVolumeDensityChanged();
	void OnVolumeColorChanged();
	void OnColorChanged();

	///////////////////////////
	////Property Changed Event
	///////////////////////////
	void OnTemLapseRateLowChanged();
	void OnTemLapseRateHighChanged();
	void OnTemInversionHeightChanged();
	void OnAuthenticDomainHeightChanged();
	void OnDiffusionCoeffChanged();
	void OnHeatEmitterAmpChanged();
	void OnHeatNoiseScaleChanged();
	void OnRelHumidityGroundChanged();
	void OnDensityNoiseScaleChanged();
	void OnWindSpeedChanged();
	void OnWindIntensityChanged();
	void OnWindDirectionChanged();
	void OnWindFieldTypeChanged();
	void OnFrequencyChanged();
	void OnNoiseAmpChanged();
	void OnNoiseSizeChanged();
};