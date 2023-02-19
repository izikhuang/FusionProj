// Fill out your copyright notice in the Description page of Project Settings.


#include "StormActor.h"
#include "Rendering/PositionVertexBuffer.h"
#include "StaticMeshResources.h"
#include "Engine/StaticMesh.h"
#include "DrawDebugHelpers.h"

// AlphaCore Native
#include "Op/Grid/AxOP_FieldSource.h"
#include "Op/Grid/AxOP_FieldWindForce.h"
#include "Op/Grid/AxOP_FieldVorticityConfinement.h"

//#include "MeshDescription.h"
//#include "MeshDescriptionBuilder.h"
//#include "StaticMeshAttributes.h"

DEFINE_LOG_CATEGORY_STATIC(StormActor, Log, All);


// Sets default values
AStormActor::AStormActor()
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	SceneComponent = CreateDefaultSubobject<USceneComponent>("SceneComponent");
	RootComponent = SceneComponent;

	StormBoundingBox = CreateDefaultSubobject<UBoxComponent>("StormBoundingBox");
	FAttachmentTransformRules keepRelative(EAttachmentRule::KeepRelative, false);
	StormBoundingBox->AttachToComponent(SceneComponent, keepRelative);

	//if (!StormObject) StormObject = new AxStormSysObject();
	FString name = this->GetName();
	StormObject.SetName(TCHAR_TO_UTF8(*name));
	this->SetActorEnableCollision(false);
	//TestString = FString("TestttString");
}

AStormActor::~AStormActor() 
{
	OnClearButtonClicked();
}

void AStormActor::Destroyed()
{
	OnClearButtonClicked();
}
// Called when the game starts or when spawned
void AStormActor::BeginPlay()
{
	Super::BeginPlay();
	OnClearButtonClicked();
	auto SceneManager = AlphaCoreSceneManager::GetInstance();

	AxSimWorld* SimWorld = SceneManager->GetWorld();
	SimWorld->SetFrame(0);
	SimWorld->SetRenderImageEmpty();
	SimWorld->SetDepthImageEmpty();
	OnInitButtonClicked();

	for (auto operation : StormOperations)
	{
		
		if (!operation) continue;
		operation->SetVisibility(false);

		//if (operation->GetOPType() == "") continue;

		//auto OPType = operation->GetOPType();
		//if (OPType == "FieldSource")
		//{
		//	UAlphaCoreStormOperationComponent* operationHandle = operation.Get();
		//	UOPFieldSourceComponent* OPHandle = (UOPFieldSourceComponent*)operationHandle;
		//	OPHandle->SetVisibility(fasle);
		//	operationHandle->SetVisi
		//	auto mesh = OPHandle->GetStaticMesh();
		//}
	}

}

// Called every frame
void AStormActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}


// Private Functions 
AxGeometry* AStormActor::GetMountainMeshData()
{
	AxGeometry* geo = new AxGeometry();
	TArray<uint32> indices;
	int IndicesNum = 0;
	int VertexNum = 0;
	for (int i = 0; i < StaticMountains.Num(); i++)
	{
		TObjectPtr<class AStaticMeshActor> StaticMountain = StaticMountains[i];
		if (!StaticMountain.Get()) continue;
		UStaticMeshComponent* StaticMeshComponent = StaticMountains[i]->GetStaticMeshComponent();
		if (!IsValidLowLevel()) return nullptr;
		if (!StaticMeshComponent) continue;
		if (!StaticMeshComponent->GetStaticMesh()) continue;
		if (!StaticMeshComponent->GetStaticMesh()->GetRenderData()) continue;

		if (StaticMeshComponent->GetStaticMesh()->GetRenderData()->LODResources.Num() > 0)
		{
			FStaticMeshRenderData* renderData = StaticMeshComponent->GetStaticMesh()->GetRenderData();
			//int32 LODNum = renderData->LODResources.Num();
			//lod = lod < 0 ? 0 : lod;
			//lod = lod > LODNum - 1 ? LODNum - 1 : lod;
			FPositionVertexBuffer* VertexBuffer = &renderData->LODResources[0].VertexBuffers.PositionVertexBuffer;
			if (VertexBuffer)
			{
				const int32 VertexCount = VertexBuffer->GetNumVertices();
				geo->ResizePoints(VertexNum + VertexCount);

				for (int32 Index = 0; Index < VertexCount; Index++)
				{
					FVector3f point = VertexBuffer->VertexPosition(Index);

					FVector p = static_cast<FVector>(point);
					FVector WorldSpaceVertexLocation = StaticMountain->GetActorLocation() + StaticMountain->GetTransform().TransformVector(p);
					AxVector3 axPos = AlphaCoreUtils::ConvertUnrealPositionToAlphaCore(WorldSpaceVertexLocation);
					geo->SetPointPosition((AxUInt32)(Index+ VertexNum), axPos.x, axPos.y, axPos.z);
				}

				// Set Indicies
				renderData->LODResources[0].IndexBuffer.GetCopy(indices);
				std::vector<int> axIndices;
				for (int j = 0; j < indices.Num(); j++)
					axIndices.push_back(indices[j] + VertexNum);
				for (int j = 0; j < indices.Num() / 3; j++)
					geo->AddPrimitive(3, AlphaCore::AxPrimitiveType::kPrimPolyon, axIndices.data() + j * 3);
				
				IndicesNum += indices.Num();
				VertexNum += VertexCount;
			}
		}
	}

	if (VertexNum == 0)
	{
		UE_LOG(StormActor, Warning, TEXT("No Mountain find in this actor, Maybe you would Attach One!"));
		return nullptr;
	}
	return geo;
}

std::vector<AxMicroSolverBase*> AStormActor::CreateAxMicroSolvers()
{
	std::vector<AxMicroSolverBase*> MicroSolvers;
	int i = 0;
	for (auto operation : StormOperations)
	{

		if (!operation) continue;
		if (operation->GetOPType() == "") continue;

		auto OPType = operation->GetOPType();
		if (OPType == "FieldSource") 
		{
			std::string solverType = "AxField_Source";
			std::string solverName = solverType + std::to_string(i);;
			auto solver = CreateFieldSourceSolver(operation, solverName);
			if (!solver) continue;
			MicroSolvers.push_back(solver);
		}
		if (OPType == "VerticityConfinement")
		{
			std::string solverType = "AxField_VorticityConfinement";
			std::string solverName = solverType + std::to_string(i);;
			auto solver = CreateVerticityConfinementSolver(operation, solverName);
			if (!solver) continue;
			MicroSolvers.push_back(solver);
		}
		i++;
	}

	auto solver = CreateWindSolver("StormWindSolver");
	if (solver) 
		MicroSolvers.push_back(solver);

	return MicroSolvers;
}

AxMicroSolverBase* AStormActor::CreateFieldSourceSolver(TObjectPtr<class UAlphaCoreStormOperationComponent> operation, const std::string name)
{
	UAlphaCoreStormOperationComponent* operationHandle = operation.Get();
	if (!operationHandle) return nullptr;
	if (operationHandle->GetOPType() != "FieldSource") return nullptr;
	UOPFieldSourceComponent* OPHandle = (UOPFieldSourceComponent*)operationHandle;

	std::string solverType = "AxField_Source";
	AxMicroSolverBase* solver = AxMicroSolverFactory::GetInstance()->CreateSolver(solverType);
	if (!solver) return nullptr;
	solver->SetName(name);
	AxOP_FieldSource* realSolver = (AxOP_FieldSource*)solver;

	// Get parms from UOPFieldSourceComponent.
	bool projection = OPHandle->Projection;
	int prolong = OPHandle->Prolong;


	float noiseAmp = OPHandle->NoiseAmp;
	float noiseRoughness = OPHandle->NoiseRoughness;
	float noiseTurbulence = OPHandle->NoiseTurbulence;
	float noiseAttenuation = OPHandle->NoiseAttenuation;
	AxVector2 noiseFrequency;
	noiseFrequency.x = OPHandle->Frequency.X;
	noiseFrequency.y = OPHandle->Frequency.Y;
	AxVector4 noiseOffset;
	noiseOffset.x = OPHandle->NoiseOffset.X;
	noiseOffset.y = OPHandle->NoiseOffset.Y;
	noiseOffset.z = OPHandle->NoiseOffset.Z;
	noiseOffset.w = OPHandle->NoiseOffset.W;

	bool useSecondShape = OPHandle->UseSecondShape;
	float noiseAmp2 = OPHandle->NoiseAmp2;
	float noiseRoughness2 = OPHandle->NoiseRoughness2;
	float noiseTurbulence2 = OPHandle->NoiseTurbulence2;
	float noiseAttenuation2 = OPHandle->NoiseAttenuation2;
	AxVector2 noiseFrequency2;
	noiseFrequency2.x = OPHandle->Frequency2.X;
	noiseFrequency2.y = OPHandle->Frequency2.Y;
	AxVector4 noiseOffset2;
	noiseOffset2.x = OPHandle->NoiseOffset2.X;
	noiseOffset2.y = OPHandle->NoiseOffset2.Y;
	noiseOffset2.z = OPHandle->NoiseOffset2.Z;
	noiseOffset2.w = OPHandle->NoiseOffset2.W;



	float heatNoiseScale = OPHandle->HeatNoiseScale;
	float relHumidityGround = OPHandle->RelHumidityGround;
	float densityNoiseScale = OPHandle->DensityNoiseScale;

	// Set Data
	realSolver->simParam.ProjectToTerrain.Set(projection?1:0);
	realSolver->simParam.Prolong.Set(prolong);

	realSolver->simParam.NoiseAmp.Set(noiseAmp);
	realSolver->simParam.NoiseRoughness.Set(noiseRoughness);
	realSolver->simParam.NoiseTurbulence.Set(noiseTurbulence);
	realSolver->simParam.NoiseAttenuation.Set(noiseAttenuation);
	realSolver->simParam.Frequency.Set(noiseFrequency);
	realSolver->simParam.Offset.Set(noiseOffset);

	realSolver->simParam.SecondaryShape.Set(useSecondShape);
	realSolver->simParam.NoiseAmp2.Set(noiseAmp2);
	realSolver->simParam.NoiseRoughness2.Set(noiseRoughness2);
	realSolver->simParam.NoiseTurbulence2.Set(noiseTurbulence2);
	realSolver->simParam.NoiseAttenuation2.Set(noiseAttenuation2);
	realSolver->simParam.Frequency2.Set(noiseFrequency2);
	realSolver->simParam.Offset2.Set(noiseOffset2);

	realSolver->simParam.HeatNoiseScale.Set(heatNoiseScale);
	realSolver->simParam.RelHumidityGround.Set(relHumidityGround);
	realSolver->simParam.DensityNoiseScale.Set(densityNoiseScale);


	// Get Data From Static Mesh
	auto mesh = OPHandle->GetStaticMesh();
	if (mesh->GetFName() == "FieldSourceNoiseAsset")
	{
		// Center
		FTransform transform = OPHandle->GetComponentTransform();
		FVector3d translation = transform.GetTranslation();
		AxVector3 center = AlphaCoreUtils::ConvertUnrealPositionToAlphaCore(translation);
		realSolver->simParam.Center.Set(center);
		// RectSize
		AxVector2 rectSize;
		FVector3d scale = transform.GetScale3D();
		rectSize.x = scale.X;	 // Check
		rectSize.y = scale.Y;	 // Check
		realSolver->simParam.RectSize.Set(rectSize);
	}
	else
	{
		UE_LOG(StormActor, Warning, TEXT("OPFieldSourceComponent Must attach a FieldSourceNoiseAsset, %s Attached"), *(mesh->GetFName().ToString()));
		realSolver->simParam.Center.Set(MakeVector3(1.0f, 1.0f,1.0f));
		realSolver->simParam.RectSize.Set(MakeVector2(1.0f, 1.0f));

	}
	
	realSolver->simParam.SourcingEmitterType.Set(2); // TODO: Check Procesual GEO
	realSolver->simParam.EmitterForAtmosphere.Set(1);

	return solver;
}

AxMicroSolverBase* AStormActor::CreateVerticityConfinementSolver(TObjectPtr<class UAlphaCoreStormOperationComponent> operation, const std::string name)
{
	UAlphaCoreStormOperationComponent* operationHandle = operation.Get();
	if (!operationHandle) return nullptr;
	if (operationHandle->GetOPType() != "VerticityConfinement") return nullptr;

	UOPVerticityConfinementComponent* OPHandle = (UOPVerticityConfinementComponent*)operationHandle;

	std::string solverType = "AxField_VorticityConfinement";
	AxMicroSolverBase* solver = AxMicroSolverFactory::GetInstance()->CreateSolver(solverType);
	if (!solver) return nullptr;
	solver->SetName(name);

	AxOP_FieldVorticityConfinement* realSolver = (AxOP_FieldVorticityConfinement*)solver;

	// Get parms from UOPVerticityConfinementComponent.
	float confinementScale = OPHandle->ConfinementScale;
	realSolver->simParam.Confinementscale.Set(confinementScale);

	return solver;
}

AxMicroSolverBase* AStormActor::CreateWindSolver(const std::string name)
{
	std::string solverType = "AxField_WindForce";
	AxMicroSolverBase* solver = AxMicroSolverFactory::GetInstance()->CreateSolver(solverType);
	if (!solver) return nullptr;
	solver->SetName(name);

	AxOP_FieldWindForce* realSolver = (AxOP_FieldWindForce*)solver;

	// Set Parms from StormActor
	realSolver->simParam.WindIntensity.Set(WindIntensity);
	realSolver->simParam.WindSpeed.Set(WindSpeed);
	AxVector3 windDirection = AlphaCoreUtils::ConvertUnrealPositionToAlphaCore(WindDirection);
	realSolver->simParam.WindDirection.Set(windDirection);

	return solver;
}


// Property Changed Event
///////////////////////////
//// Simulation Fields
///////////////////////////


///////////////////////////
//// Simulation Button
///////////////////////////

void AStormActor::OnClearButtonClicked() 
{
	auto SceneManager = AlphaCoreSceneManager::GetInstance();
	while (true)
	{
		if (SceneManager->GetSimWorldStatus() != 2)
		{
			SceneManager->SetSimWorldNotInit();
			break;
		}
	}
	AxSimWorld* AxSimWorld = SceneManager->GetWorld();
	std::string name = StormObject.GetName();
	if (AxSimWorld->IsObjectInWorld(name))
	{
		AxSimWorld->RemoveObjectByName(name);
	}

	StormObject.ClearAndDestory();
	StormObject.SetHeightFieldGeometry(nullptr);
	StormObject.SetRenderDensityEmpty();
	StormObject.m_PostSimCallstack.clear();

	SceneManager->SetSimWorldInited();
	UE_LOG(StormActor, Display, TEXT("Object %s Removed!"), *FString(name.c_str()));
}

void AStormActor::OnInitButtonClicked()
{
	auto SceneManager = AlphaCoreSceneManager::GetInstance();

	while (true)
	{
		if (SceneManager->GetSimWorldStatus() != 2) 
		{
			SceneManager->SetSimWorldNotInit();
			break;
		}
	}
	
	

	// AlphaCore Simulation Init
	AxSimWorld* AxSimWorld = AlphaCoreSceneManager::GetInstance()->GetWorld();
	FString ActorName = this->GetName();

	std::string name(TCHAR_TO_UTF8(*ActorName));
	if (AxSimWorld->IsObjectInWorld(name)) 
	{
		AxSimWorld->RemoveObjectByName(name);
	}



	StormObject.ClearAndDestory();
	//StormObject.RegistSimObjectRenderImage();
	StormObject.SetHeightFieldGeometry(nullptr);
	StormObject.SetRenderDensityEmpty();
	StormObject.m_PostSimCallstack.clear();
	StormObject.SetName(name);

	//auto worldRenderImage = AxSimWorld->GetRenderImage();
	//if (worldRenderImage)
	//{
	//	auto width = worldRenderImage->GetResolution().x;
	//	auto height = worldRenderImage->GetResolution().y;
	//	StormObject.RegisterImage(width, height);
	//}

	// Beigin to Init StormObject

	// AxStormSysSimData
	FVector3d boundMin = StormBoundingBox->Bounds.GetBox().Min;
	FVector3d boundMax = StormBoundingBox->Bounds.GetBox().Max;
	FVector3d UnrealSize = boundMax - boundMin;
	FVector3d UnrealPivot = (boundMax + boundMin) / 2.0;

	AxVector3 fieldSize = AlphaCoreUtils::ConvertUnrealPositionToAlphaCore(UnrealSize);
	AxVector3 fieldPivot = AlphaCoreUtils::ConvertUnrealPositionToAlphaCore(UnrealPivot);

	StormObject.simParam.ComputeArch.Set(2);
	StormObject.simParam.VoxelSize.Set(SimulationVoxleSize);
	StormObject.simParam.Pivot.Set(fieldPivot);
	StormObject.simParam.Size.Set(fieldSize);

	// HeightFiled
	auto mountainGeo = GetMountainMeshData();
	StormObject.SetHeightFieldGeometry(mountainGeo);
	StormObject.simParam.Substeps.Set(1);

	///////////////////
	// Property Init
	///////////////////
	{
		// Render Data
		SetRenderData();
		// Atmosphere
		StormObject.simParam.HeatEmitterAmp.Set(HeatEmitterAmp);
		StormObject.simParam.AuthenticDomainHeight.Set(AuthenticDomainHeight);
		StormObject.simParam.BuoyScale.Set(BuoyancyScale);
		StormObject.simParam.BuoyancyScale.Set(BuoyancyScale);
		StormObject.simParam.DiffusionCoeff.Set(DiffusionCoeff);
		StormObject.simParam.CloudPosOffset.Set(CloudPositionOffsetZ);
		// Wind
		StormObject.simParam.WindSpeed.Set(WindSpeed);
		StormObject.simParam.WindIntensity.Set(WindIntensity);
		StormObject.simParam.WindDirection.Set(AlphaCoreUtils::ConvertFromUnrealToAlphaCoreWithoutScale(WindDirection));
		StormObject.simParam.WindFieldType.Set(1);

		// Operations
		StormObject.m_PostSimCallstack = CreateAxMicroSolvers();
	}

	for (int i = 0; i < StormObject.m_PostSimCallstack.size(); i++)
	{
		StormObject.m_PostSimCallstack[i]->SetSimWorld(AxSimWorld);
	}

	StormObject.SetSimCacheOutputMark(false);
	StormObject.SetSPMDBackendAPI(AlphaCore::CUDA);
	StormObject.SetCookTimes(0);

	// StormObject Inited

	AxSimWorld->AddObject(&StormObject);

	SceneManager->SetSimWorldInited();
}


///////////////////////////
////Property Changed Event
///////////////////////////
void AStormActor::SetRenderData()
{
	AxGasVolumeMaterial material;
	material.stepSize = StepSize;
	material.densityScale = DensityScale;
	material.shadowScale = ShadowScale;
	material.usePhase = UsePhase;
	material.phase = Phase;
	material.minMaxInputDensity = { InputDensityMinMax.X, InputDensityMinMax.Y/1000.f };
	material.needUpdate = false;
	StormObject.SetRenderMaterial(&material);
}
void AStormActor::OnRenderDataChanged()
{
	SetRenderData();
	//UE_LOG(StormActor, Warning, TEXT("AStormActor::OnRenderDataChanged()"));
}


void AStormActor::OnVoxelSizeChanged()
{
	FVector3d boundMin = StormBoundingBox->Bounds.GetBox().Min;
	FVector3d boundMax = StormBoundingBox->Bounds.GetBox().Max;
	FVector3d UnrealSize = boundMax - boundMin;
	AxVector3 fieldSize = AlphaCoreUtils::ConvertUnrealPositionToAlphaCore(UnrealSize);

	SimulationResolution.X = int(fieldSize.x / SimulationVoxleSize);
	SimulationResolution.Z = int(fieldSize.y / SimulationVoxleSize);
	SimulationResolution.Y = int(fieldSize.z / SimulationVoxleSize);
}

void AStormActor::OnHeatEmitterAmpChanged()
{
	//UE_LOG(StormActor, Warning, TEXT("AStormActor::OnHeatEmitterAmpChanged()"));
	StormObject.simParam.HeatEmitterAmp.Set(HeatEmitterAmp);
}
void AStormActor::OnAuthenticDomainHeightChanged()
{
	//UE_LOG(StormActor, Warning, TEXT("AStormActor::OnAuthenticDomainHeightChanged()"));
	StormObject.simParam.AuthenticDomainHeight.Set(AuthenticDomainHeight);
}
void AStormActor::OnDiffusionCoeffChanged()
{
	//UE_LOG(StormActor, Warning, TEXT("AStormActor::OnDiffusionCoeffChanged()"));
	StormObject.simParam.DiffusionCoeff.Set(DiffusionCoeff);
}
void AStormActor::OnBuoyancyScaleChanged()
{
	//UE_LOG(StormActor, Warning, TEXT("AStormActor::OnBuoyancyScaleChanged()"));
	StormObject.simParam.BuoyScale.Set(BuoyancyScale);
	StormObject.simParam.BuoyancyScale.Set(BuoyancyScale);
}
void AStormActor::OnCloudOffsetChanged()
{
	//UE_LOG(StormActor, Warning, TEXT("AStormActor::OnCloudOffsetChanged()"));
	StormObject.simParam.CloudPosOffset.Set(CloudPositionOffsetZ);
}


void AStormActor::OnWindSpeedChanged()
{
	//UE_LOG(StormActor, Display, TEXT("AStormActor::OnWindSpeedChanged()"));
	for (int i = 0; i < StormObject.m_PostSimCallstack.size(); ++i)
	{
		auto solver = StormObject.m_PostSimCallstack[i];
		if (solver->GetSolverName() == "StormWindSolver")
		{
			AxOP_FieldWindForce* realSolver = (AxOP_FieldWindForce*)solver;
			realSolver->simParam.WindSpeed.Set(WindSpeed);
		}
	}
	StormObject.simParam.WindSpeed.Set(WindSpeed);
}
void AStormActor::OnWindIntensityChanged()
{
	//UE_LOG(StormActor, Display, TEXT("AStormActor::OnWindIntensityChanged()"));
	for (int i = 0; i < StormObject.m_PostSimCallstack.size(); ++i)
	{
		auto solver = StormObject.m_PostSimCallstack[i];
		if (solver->GetSolverName() == "StormWindSolver")
		{
			AxOP_FieldWindForce* realSolver = (AxOP_FieldWindForce*)solver;
			realSolver->simParam.WindIntensity.Set(WindIntensity);
		}
	}
	StormObject.simParam.WindIntensity.Set(WindIntensity);
}
void AStormActor::OnWindDirectionChanged()
{
	//UE_LOG(StormActor, Display, TEXT("AStormActor::OnWindDirectionChanged()"));
	AxVector3 windDirection = AlphaCoreUtils::ConvertUnrealPositionToAlphaCore(WindDirection);
	for (int i = 0; i < StormObject.m_PostSimCallstack.size(); ++i)
	{
		auto solver = StormObject.m_PostSimCallstack[i];
		if (solver->GetSolverName() == "StormWindSolver")
		{
			AxOP_FieldWindForce* realSolver = (AxOP_FieldWindForce*)solver;
			
			realSolver->simParam.WindDirection.Set(windDirection);
		}
	}

	StormObject.simParam.WindDirection.Set(windDirection);
}

void AStormActor::OnOperationsValueChanged()
{
	UE_LOG(StormActor, Display, TEXT("Operation Data Changed, Storm Simulation need to Init"));
}
