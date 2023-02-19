#pragma once

#include "StormOperations/StormOperationDetailsProp.h"

//#include 

#include "DetailWidgetRow.h"
#include "DetailCategoryBuilder.h"
#include "DetailLayoutBuilder.h"
//#include "Input/Reply.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Input/SButton.h"


#define LOCTEXT_NAMESPACE "FieldSourceDetails"

TSharedRef<IDetailCustomization> FFieldSourceDetailsProp::MakeInstance()
{
	return MakeShareable(new FFieldSourceDetailsProp());
}

void FFieldSourceDetailsProp::CustomizeDetails(IDetailLayoutBuilder& DetailBuilder)
{
	UE_LOG(LogTemp, Warning, TEXT("FFieldSourceDetailsProp::CustomizeDetails"));
}

#undef LOCTEXT_NAMESPACE



#define LOCTEXT_NAMESPACE "VerticityConfinementDetails"


TSharedRef<IDetailCustomization> FVerticityConfinementDetailsProp::MakeInstance()
{
	return MakeShareable(new FVerticityConfinementDetailsProp());
}

void FVerticityConfinementDetailsProp::CustomizeDetails(IDetailLayoutBuilder& DetailBuilder)
{
	UE_LOG(LogTemp, Warning, TEXT("FVerticityConfinementDetailsProp::CustomizeDetails"));
}

#undef LOCTEXT_NAMESPACE

