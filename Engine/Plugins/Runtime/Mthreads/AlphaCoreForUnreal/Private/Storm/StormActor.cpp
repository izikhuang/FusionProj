// Fill out your copyright notice in the Description page of Project Settings.


#include "Storm/StormActor.h"
#include "Rendering/PositionVertexBuffer.h"
#include "StaticMeshResources.h"
#include "Engine/StaticMesh.h"
#include "DrawDebugHelpers.h"

//#include "MeshDescription.h"
//#include "MeshDescriptionBuilder.h"
//#include "StaticMeshAttributes.h"


// Sets default values
AStormActor::AStormActor()
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	SceneComponent = CreateDefaultSubobject<USceneComponent>("SceneComponent");
	StormBoundingBox = CreateDefaultSubobject<UBoxComponent>("StormBoundingBox");

	FAttachmentTransformRules keepRelative(EAttachmentRule::KeepRelative, false);
	StormBoundingBox->AttachToComponent(SceneComponent, keepRelative);

	RootComponent = SceneComponent;
}

// Called when the game starts or when spawned
void AStormActor::BeginPlay()
{
	Super::BeginPlay();
	//if (StaticMountainComponent) StaticMountainComponent->SetHiddenInGame(true);
	
	for (auto& mesh : StaticMeshs)
	{
		if (mesh) mesh->SetHiddenInGame(true);
	}
}

TArray<FVector> AStormActor::GetMountainMeshData(int lod)
{
	TArray<FVector> vertices = TArray<FVector>();
	UStaticMeshComponent* StaticMeshComponent = StaticMountain->GetStaticMeshComponent();

	// Vertex Buffer
	if (!IsValidLowLevel()) return vertices;
	if (!StaticMeshComponent) return vertices;
	if (!StaticMeshComponent->GetStaticMesh()) return vertices;
	if (!StaticMeshComponent->GetStaticMesh()->GetRenderData()) return vertices;
	if (StaticMeshComponent->GetStaticMesh()->GetRenderData()->LODResources.Num() > 0)
	{
		FStaticMeshRenderData* renderData = StaticMeshComponent->GetStaticMesh()->GetRenderData();
		int32 LODNum = renderData->LODResources.Num();

		lod = lod < 0 ? 0 : lod;
		lod = lod > LODNum - 1 ? LODNum - 1 : lod;

		// LOD.IndexBuffer
		FPositionVertexBuffer* VertexBuffer = &renderData->LODResources[lod].VertexBuffers.PositionVertexBuffer;
		if (VertexBuffer)
		{
			const int32 VertexCount = VertexBuffer->GetNumVertices();
			for (int32 Index = 0; Index < VertexCount; Index++)
			{
				FVector3f point = VertexBuffer->VertexPosition(Index);

				FVector p = static_cast<FVector>(point);
				FVector WorldSpaceVertexLocation = StaticMountain->GetActorLocation() + StaticMountain->GetTransform().TransformVector(p);

				////This is in the Static Mesh Actor Class, so it is location and tranform of the SMActor
				vertices.Add(WorldSpaceVertexLocation);
			}
		}
	}

	return vertices;
}

// Called every frame
void AStormActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	//StaticMountainComponent->GetLod


	FBoxSphereBounds bound = RootComponent->Bounds;
	auto box = bound.GetBox();
	auto min = box.Min;
	auto max = box.Max;
	UE_LOG(LogTemp, Warning, TEXT("RootComponent Bounds--- Min: %f,%f,%f  Max:%f,%f,%f"), min.X, min.Y, min.Z, max.X, max.Y, max.Z);

	TArray<FVector> Points = GetMountainMeshData(StaticMountainLOD);
	for (FVector point : Points)
	{
		DrawDebugPoint(GetWorld(), point, 20, FColor::Orange);
	}

}


///////////////////////////
////Property Changed Event
///////////////////////////

void AStormActor::OnAlphaCoreJsonChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnAlphaCoreJsonChanged"));
}

void AStormActor::OnSamplesPerPixelChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnSamplesPerPixelChanged()"));
}

void AStormActor::OnStormCollisionActorsAndLODChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnStormCollisionActorsAndLODChanged()"));
}
void AStormActor::OnVolumeDensityChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnVolumeDensityChanged()"));
}
void AStormActor::OnVolumeColorChanged() 
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnVolumeColorChanged()"));
}
void AStormActor::OnColorChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnColorChanged()"));
}




///////////////////////////
////Property Changed Event
///////////////////////////
void AStormActor::OnTemLapseRateLowChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnTemLapseRateLowChanged()"));
}

void AStormActor::OnTemLapseRateHighChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnTemLapseRateHighChanged()"));
}

void AStormActor::OnTemInversionHeightChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnTemInversionHeightChanged()"));
}

void AStormActor::OnAuthenticDomainHeightChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnAuthenticDomainHeightChanged()"));
}

void AStormActor::OnDiffusionCoeffChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnDiffusionCoeffChanged()"));
}

void AStormActor::OnHeatEmitterAmpChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnHeatEmitterAmpChanged()"));
}

void AStormActor::OnHeatNoiseScaleChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnHeatNoiseScaleChanged()"));
}

void AStormActor::OnRelHumidityGroundChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnRelHumidityGroundChanged()"));
}

void AStormActor::OnDensityNoiseScaleChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnDensityNoiseScaleChanged()"));
}

void AStormActor::OnWindSpeedChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnWindSpeedChanged()"));
}

void AStormActor::OnWindIntensityChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnWindIntensityChanged()"));
}

void AStormActor::OnWindDirectionChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnWindDirectionChanged()"));
}

void AStormActor::OnWindFieldTypeChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnWindFieldTypeChanged()"));
}

void AStormActor::OnFrequencyChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnFrequencyChanged()"));
}

void AStormActor::OnNoiseAmpChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnNoiseAmpChanged()"));
}

void AStormActor::OnNoiseSizeChanged()
{
	UE_LOG(LogTemp, Warning, TEXT("AStormActor::OnNoiseSizeChanged"));
}

