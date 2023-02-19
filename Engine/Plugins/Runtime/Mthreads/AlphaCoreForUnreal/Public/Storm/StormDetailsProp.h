#pragma once

#include "IDetailCustomization.h"

class ALPHACOREFORUNREAL_API FStormDetailsProp : public IDetailCustomization
{
public:

	static TSharedRef<IDetailCustomization> MakeInstance();

	virtual void CustomizeDetails(IDetailLayoutBuilder& DetailBuilder) override;

};