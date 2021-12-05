/*
* Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "DDGIVolume.h"
#include "DDGIVolumeComponent.h"
#include "Components/BoxComponent.h"
#include "Components/BillboardComponent.h"
#include "Engine/CollisionProfile.h"
#include "UObject/ConstructorHelpers.h"

ADDGIVolume::ADDGIVolume(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer)
{
    DDGIVolumeComponent = CreateDefaultSubobject<UDDGIVolumeComponent>(TEXT("DDGI"));
    DDGIVolumeComponent->bHiddenInGame = false;
    RootComponent = DDGIVolumeComponent;

#if WITH_EDITORONLY_DATA
    BoxComponent = CreateDefaultSubobject<UBoxComponent>(TEXT("Volume"));
    SpriteComponent = CreateEditorOnlyDefaultSubobject<UBillboardComponent>(TEXT("Sprite"));
    if (!IsRunningCommandlet())
    {
        if(BoxComponent != nullptr)
        {
            BoxComponent->SetBoxExtent(FVector{ 100.0f, 100.0f, 100.0f });
            BoxComponent->bHiddenInGame = true;
            BoxComponent->SetCollisionProfileName(UCollisionProfile::NoCollision_ProfileName);
            BoxComponent->SetupAttachment(RootComponent);
        }

        if(SpriteComponent != nullptr)
        {
            static ConstructorHelpers::FObjectFinderOptional<UTexture2D> DecalTexture(TEXT("/Engine/EditorResources/EmptyActor"));

            SpriteComponent->Sprite = DecalTexture.Get();
            SpriteComponent->SetRelativeScale3D_Direct(FVector(0.5f, 0.5f, 0.5f));
            SpriteComponent->bHiddenInGame = true;
            SpriteComponent->SetUsingAbsoluteScale(true);
            SpriteComponent->SetCollisionProfileName(UCollisionProfile::NoCollision_ProfileName);
            SpriteComponent->bIsScreenSizeScaled = true;

            SpriteComponent->SetupAttachment(RootComponent);
        }
    }
#endif // WITH_EDITORONLY_DATA

#if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.bStartWithTickEnabled = true;
#endif
}

#if WITH_EDITOR
void ADDGIVolume::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
    Super::PostEditChangeProperty(PropertyChangedEvent);
    DDGIVolumeComponent->UpdateRenderThreadData();
}

void ADDGIVolume::PostEditMove(bool bFinished)
{
    Super::PostEditMove(bFinished);
    DDGIVolumeComponent->UpdateRenderThreadData();
}
#endif
