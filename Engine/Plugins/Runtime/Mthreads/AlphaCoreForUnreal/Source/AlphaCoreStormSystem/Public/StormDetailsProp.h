#pragma once

#include "IDetailCustomization.h"
#include "StormActor.h"

class ALPHACORESTORMSYSTEM_API FStormDetailsProp : public IDetailCustomization
{
public:

	static TSharedRef<IDetailCustomization> MakeInstance();

	virtual void CustomizeDetails(IDetailLayoutBuilder& DetailBuilder) override;
protected:

	friend class AStormActor;
	friend class AActor;
};