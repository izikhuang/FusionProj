
#include "StormOperations/OPFieldSourceComponent.h"

UOPFieldSourceComponent::UOPFieldSourceComponent(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	//bUseDefaultCollision = 0;
	Super::SetCollisionProfileName(FName("NoCollision"), true);
	this->GetCollisionEnabled();
	Super::SetCollisionEnabled(ECollisionEnabled::NoCollision);
}

UOPFieldSourceComponent::~UOPFieldSourceComponent()
{
}


