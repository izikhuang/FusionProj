/*
* Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "Kismet/BlueprintFunctionLibrary.h"

#include "DDGIBlueprintLibrary.generated.h"

UCLASS(MinimalAPI, meta = (ScriptName = "DDGILibrary"))
class UDDGIBlueprintLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_UCLASS_BODY()

    UFUNCTION(BlueprintCallable, Category = "DDGI")
    static void ClearProbeData(const UDDGIVolumeComponent* DDGIVolumeComponent);

    UFUNCTION(BlueprintCallable, Category = "DDGI")
    static void DisableVolume(UDDGIVolumeComponent* DDGIVolumeComponent);

    UFUNCTION(BlueprintCallable, Category = "DDGI")
    static void EnableVolume(UDDGIVolumeComponent* DDGIVolumeComponent);

};
