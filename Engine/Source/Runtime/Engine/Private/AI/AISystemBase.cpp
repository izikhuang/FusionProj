// Copyright 1998-2018 Epic Games, Inc. All Rights Reserved.

#include "AI/AISystemBase.h"
#include "Templates/Casts.h"
#include "GameFramework/GameModeBase.h"


UAISystemBase::UAISystemBase(const FObjectInitializer& ObjectInitializer)
: Super(ObjectInitializer)
{
}

FName UAISystemBase::GetAISystemModuleName()
{
	UAISystemBase* AISystemDefaultObject = Cast<UAISystemBase>(StaticClass()->GetDefaultObject());
	return AISystemDefaultObject != NULL ? AISystemDefaultObject->AISystemModuleName : TEXT("");
}

FSoftClassPath UAISystemBase::GetAISystemClassName()
{
	UAISystemBase* AISystemDefaultObject = Cast<UAISystemBase>(StaticClass()->GetDefaultObject());
	return AISystemDefaultObject != NULL ? AISystemDefaultObject->AISystemClassName : FSoftClassPath();
}

void UAISystemBase::CleanupWorld(bool bSessionEnded, bool bCleanupResources, UWorld* NewWorld)
{
	FGameModeEvents::OnGameModeMatchStateSetEvent().Remove(OnMatchStateSetHandle);
}

void UAISystemBase::StartPlay()
{
	OnMatchStateSetHandle = FGameModeEvents::OnGameModeMatchStateSetEvent().AddUObject(this, &UAISystemBase::OnMatchStateSet);
}

void UAISystemBase::OnMatchStateSet(FName NewMatchState)
{

}

bool UAISystemBase::ShouldInstantiateInNetMode(ENetMode NetMode)
{
	UAISystemBase* AISystemDefaultObject = Cast<UAISystemBase>(StaticClass()->GetDefaultObject());
	return AISystemDefaultObject && (AISystemDefaultObject->bInstantiateAISystemOnClient == true || NetMode != NM_Client);
}
