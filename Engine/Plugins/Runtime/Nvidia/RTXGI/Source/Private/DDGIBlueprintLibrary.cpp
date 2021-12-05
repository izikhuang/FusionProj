/*
* Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "DDGIBlueprintLibrary.h"
#include "DDGIVolumeComponent.h"
#include "RenderingThread.h"

UDDGIBlueprintLibrary::UDDGIBlueprintLibrary(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer){}

void UDDGIBlueprintLibrary::ClearProbeData(const UDDGIVolumeComponent* DDGIVolumeComponent)
{
    FDDGIVolumeSceneProxy* DDGIProxy = DDGIVolumeComponent->SceneProxy;

    ENQUEUE_RENDER_COMMAND(DDGIClearProbeData)(
        [DDGIProxy, DDGIVolumeComponent](FRHICommandListImmediate& RHICmdList)
        {
            DDGIVolumeComponent->SceneProxy->ResetTextures_RenderThread(RHICmdList);
        }
    );
}

void UDDGIBlueprintLibrary::DisableVolume(
    UDDGIVolumeComponent* DDGIVolumeComponent
)
{
    DDGIVolumeComponent->EnableVolumeComponent(false);
}

void UDDGIBlueprintLibrary::EnableVolume(
    UDDGIVolumeComponent* DDGIVolumeComponent
)
{
    DDGIVolumeComponent->EnableVolumeComponent(true);
}
