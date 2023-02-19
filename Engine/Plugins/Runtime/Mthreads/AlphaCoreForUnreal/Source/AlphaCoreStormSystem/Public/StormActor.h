#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "UObject/ObjectPtr.h"
#include "Engine/StaticMeshActor.h"
#include "Components/StaticMeshComponent.h"
#include "Components/SceneComponent.h"
#include "Components/BoxComponent.h"
#include "StormOperations/StormOperationComponent.h"
#include "StormOperations/OPVerticityConfinementComponent.h"
#include "StormOperations/OPFieldSourceComponent.h"
#include "AlphaCoreSceneManager.h"
#include "AlphaCoreUtils.h"

#include <AlphaCore.h>

#include "StormActor.generated.h"



UCLASS(HideCategories = (Rendering, Collision, Replication, HLOD, WorldPartition, Input, Actor, Cooking, DataLayers))
class ALPHACORESTORMSYSTEM_API AStormActor : public AActor
{
	GENERATED_BODY()
public:
	AStormActor();
	~AStormActor();
	virtual void Destroyed() override;

	TObjectPtr<class USceneComponent> SceneComponent;
	TObjectPtr<class UBoxComponent> StormBoundingBox;
	//FString TestString;
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;
private:
	AxGeometry* GetMountainMeshData();

private:
	std::vector<AxMicroSolverBase*> CreateAxMicroSolvers();
	AxMicroSolverBase* CreateFieldSourceSolver(TObjectPtr<class UAlphaCoreStormOperationComponent> operation, const std::string name);
	AxMicroSolverBase* CreateVerticityConfinementSolver(TObjectPtr<class UAlphaCoreStormOperationComponent> operation, const std::string name);
	AxMicroSolverBase* CreateWindSolver(const std::string name);

public:
	
	AxStormSysObject StormObject;
	
	///////////////////////////
	//// Storm Base Properties
	///////////////////////////
	
	// StaticMountainComponent used to cast height field.
	//UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Basic")
	//bool UseHeightField = false;

	UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Basic")
	TArray<TObjectPtr<class AStaticMeshActor>> StaticMountains;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Basic", meta = (ClampMin = "0.1", UIMin = "0.1", ClampMax = "100", UIMax = "100"))
		float SimulationVoxleSize = 1.0f;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Basic")
		FIntVector SimulationResolution;
	//UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Basic", meta = (ClampMin = "0", UIMin = "0", ClampMax = "10", UIMax = "10"))
	//int StaticMountainLOD;


	///////////////////////////
	//// Storm Emitter
	///////////////////////////
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Operations")
	TArray<TObjectPtr<class UAlphaCoreStormOperationComponent>> StormOperations;


	///////////////////////////
	//// Storm Propertyies
	///////////////////////////

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Atmosphere", meta = (ClampMin = "0", UIMin = "0", ClampMax = "5", UIMax = "5"))
		float HeatEmitterAmp = 2.0f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Atmosphere", meta = (ClampMin = "3000", UIMin = "3000", ClampMax = "8000", UIMax = "8000"))
		float AuthenticDomainHeight = 6400.0f;

	//UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Atmosphere", meta = (ClampMin = "0", UIMin = "0", ClampMax = "0.1", UIMax = "0.1"))
		float DiffusionCoeff = 0.01f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Atmosphere", meta = (ClampMin = "0", UIMin = "0", ClampMax = "100", UIMax = "100"))
		float BuoyancyScale = 50.0f;


	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Atmosphere")
		float CloudPositionOffsetZ = 0.0f;




	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Wind", meta = (ClampMin = "0", UIMin = "0", ClampMax = "2", UIMax = "2"))
		float WindSpeed = 0.3f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Wind", meta = (ClampMin = "0", UIMin = "0", ClampMax = "1", UIMax = "1"))
		float WindIntensity = 0.2f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AlphaCore|Storm|Wind", meta = (ClampMin = "-1", UIMin = "-1", ClampMax = "1", UIMax = "1"))
		FVector WindDirection;

	///////////////////////////
	//// Render Propertyies
	///////////////////////////
	UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Render", meta = (ClampMin = "0", UIMin = "0", ClampMax = "1000", UIMax = "1000"))
	float DensityScale= 1.0f;

	UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Render", meta = (ClampMin = "0", UIMin = "0", ClampMax = "1000", UIMax = "1000"))
	float StepSize = 1.0f;

	UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Render", meta = (ClampMin = "0", UIMin = "0", ClampMax = "10", UIMax = "10"))
	float ShadowScale = 1.0f;

	UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Render")
	bool UsePhase = true;

	UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Render", meta = (ClampMin = "-1", UIMin = "-1", ClampMax = "1", UIMax = "1"))
	float Phase = 0.3f;

	UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Render", meta = (ClampMin = "0", UIMin = "0", ClampMax = "100", UIMax = "100"))
	FVector2f InputDensityMinMax = FVector2f(0.0f,1.0f);


	//UPROPERTY(EditAnywhere, Category = "AlphaCore|Storm|Render")
	FLinearColor VolumeColor = FLinearColor(1.0, 1.0, 1.0, 1.0);



public:
	void SetRenderData();


	///////////////////////////
	//// Simulation Button
	///////////////////////////
	void OnClearButtonClicked();
	void OnInitButtonClicked();

	///////////////////////////
	////Property Changed Event
	///////////////////////////
	void OnVoxelSizeChanged();

	void OnHeatEmitterAmpChanged();
	void OnAuthenticDomainHeightChanged();
	void OnDiffusionCoeffChanged();
	void OnBuoyancyScaleChanged();
	void OnCloudOffsetChanged();

	void OnWindSpeedChanged();
	void OnWindIntensityChanged();
	void OnWindDirectionChanged();

	void OnRenderDataChanged();

	void OnOperationsValueChanged();
};