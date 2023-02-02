#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "UObject/ObjectPtr.h"
#include "Engine/StaticMeshActor.h"
#include "Components/StaticMeshComponent.h"
#include "Components/SceneComponent.h"
#include "Components/BoxComponent.h"
#include "StormEmitterComponent.h"
#include "AxUESceneManager.h"
#include "StormActor.generated.h"

UCLASS()
class ALPHACOREFORUNREAL_API AStormActor : public AActor
{
	GENERATED_BODY()
public:
	// Sets default values for this actor's properties
	AStormActor();
	~AStormActor() {};
	TArray<FVector> GetLODMeshData(const TObjectPtr<class AStaticMeshActor> staticMeshActor, int lod);

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
private:
	//UStaticMesh* CreateMeshData();
public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	UPROPERTY(EditAnywhere, Category = "AlphaCoreStorm")
	USceneComponent* SceneComponent;

	// StormBoundingBox used to cast SimField Size
	UBoxComponent* StormBoundingBox;

	// StaticMountainComponent used to cast height field.
	UPROPERTY(EditAnywhere, Category = "AlphaCoreStorm")
	UStaticMeshComponent* StaticMountainComponent;

	UPROPERTY(EditAnywhere, Category = "AlphaCoreStorm")
	UStormEmitterComponent* StormEmitter;
public:
	///////////////////////////
	//// Test Propertyies
	///////////////////////////
	UPROPERTY(EditAnywhere, Category = "AlphaCoreStorm|Test")
		TArray<TObjectPtr<class UStaticMeshComponent>> StaticMeshs;

	UPROPERTY(EditAnywhere, Category = "AlphaCoreStorm|Test")
		FString AlphaCoreJson;

	UPROPERTY(EditAnywhere, Category = "AlphaCoreStorm|Test")
		TMap<TObjectPtr<class AStaticMeshActor>, uint32> StormCollisionActorsAndLOD;

	UPROPERTY(EditAnywhere, Category = "AlphaCoreStorm|Test")
		TObjectPtr<class UCurveLinearColor> VolumeDensity = nullptr;

	UPROPERTY(EditAnywhere, Category = "AlphaCoreStorm|Test")
		TObjectPtr<class UCurveLinearColor> VolumeColor = nullptr;

	UPROPERTY(EditAnywhere, Category = "AlphaCoreStorm|Test")
		FLinearColor Color = FLinearColor(1.0, 1.0, 1.0, 1.0);

	///////////////////////////
	//// Storm Emitter
	///////////////////////////




	///////////////////////////
	//// Storm Field Content
	///////////////////////////
	// StormSimFieldResolution used to set Sim field resolution.
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|StormField", meta = (ClampMin = "10", UIMin = "10", ClampMax = "512", UIMax = "512"))
	FIntVector StormSimFieldResolution = {128,128,128};


	///////////////////////////
	//// Storm Propertyies
	///////////////////////////
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|Storm")
	float TemLapseRateLow;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|Storm")
	float TemLapseRateHigh;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|Storm")
	float TemInversionHeight;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|Storm")
	float AuthenticDomainHeight;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|Storm")
	float DiffusionCoeff;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|Storm")
	float HeatEmitterAmp;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|Storm")
	float HeatNoiseScale;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|Storm")
	float RelHumidityGround;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|Storm")
	float DensityNoiseScale;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|Storm")
	float WindSpeed;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|Storm")
	float WindIntensity;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|Storm", meta = (ClampMin = "0", UIMin = "0", ClampMax = "5", UIMax = "5"))
	FVector WindDirection;


	//////////////////////////////////
	//// FieldSource Propertyies
	//////////////////////////////////
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|FieldSource")
	FVector2D Frequency;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|FieldSource")
	float NoiseAmp;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCoreStorm|FieldSource")
	float NoiseSize;


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