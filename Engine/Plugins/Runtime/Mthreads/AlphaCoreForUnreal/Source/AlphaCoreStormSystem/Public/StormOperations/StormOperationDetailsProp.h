#pragma once

#include "IDetailCustomization.h"

class ALPHACORESTORMSYSTEM_API FFieldSourceDetailsProp : public IDetailCustomization
{
public:

	static TSharedRef<IDetailCustomization> MakeInstance();

	virtual void CustomizeDetails(IDetailLayoutBuilder& DetailBuilder) override;

};

class ALPHACORESTORMSYSTEM_API FVerticityConfinementDetailsProp : public IDetailCustomization
{
public:

	static TSharedRef<IDetailCustomization> MakeInstance();

	virtual void CustomizeDetails(IDetailLayoutBuilder& DetailBuilder) override;

};