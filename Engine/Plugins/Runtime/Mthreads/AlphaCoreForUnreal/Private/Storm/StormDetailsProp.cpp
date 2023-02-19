#pragma once

#include "Storm/StormDetailsProp.h"
#include "Storm/StormActor.h"

#include "DetailWidgetRow.h"
#include "DetailCategoryBuilder.h"
#include "DetailLayoutBuilder.h"
//#include "Input/Reply.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Input/SButton.h"


#define LOCTEXT_NAMESPACE "StormActorDetails"


TSharedRef<IDetailCustomization> FStormDetailsProp::MakeInstance()
{
	return MakeShareable(new FStormDetailsProp());
}

void FStormDetailsProp::CustomizeDetails(IDetailLayoutBuilder& DetailBuilder)
{
	TArray<TWeakObjectPtr<UObject>> Objects;
	DetailBuilder.GetObjectsBeingCustomized(Objects);
	if (Objects.Num() != 1)	return;
	
	UE_LOG(LogTemp, Warning, TEXT("RootComponent Bounds %s"), *(Objects[0]->GetName()));
	

	IDetailCategoryBuilder& Category = DetailBuilder.EditCategory("AlphaCore", LOCTEXT("RowNameText", "Task Json"), ECategoryPriority::Important);

	Category.AddCustomRow(LOCTEXT("Keyword", "Vera Task Json"))
		.NameContent()
		[
			SNew(STextBlock)
			.Text(LOCTEXT("TitleNameText", "Update Vera Param"))
			.Font(IDetailLayoutBuilder::GetDetailFont())
		]
		.ValueContent().HAlign(HAlign_Fill)
		[
			SNew(SButton)
			.Text(LOCTEXT("ButtonText", "Update"))
			.OnClicked_Lambda(
				[Objects]()
				{
					for (auto Object : Objects)
					{
						if (Object.IsValid() && Object->IsA(AStormActor::StaticClass()))
						{
							Cast<AStormActor>(Object)->OnAlphaCoreJsonChanged();
						}
					}

					return FReply::Handled();
				}
			)
		];



	// Actor Under Control
	TWeakObjectPtr<AStormActor> StormActor = Cast<AStormActor>(Objects[0].Get());

	// Property Handles 
	TSharedPtr<IPropertyHandle> AlphaCoreJsonHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, AlphaCoreJson));
	TSharedPtr<IPropertyHandle> StormCollisionActorsAndLODHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, StormCollisionActorsAndLOD));
	TSharedPtr<IPropertyHandle> VolumeDensityHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, VolumeDensity));
	TSharedPtr<IPropertyHandle> VolumeColorHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, VolumeColor));
	TSharedPtr<IPropertyHandle> ColorPropertyHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, Color));
	//TSharedPtr<IPropertyHandle> StormBoundingBoxHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, StormBoundingBoxComponent));
	
	TSharedPtr<IPropertyHandle> TemLapseRateLowHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, TemLapseRateLow));
	TSharedPtr<IPropertyHandle> TemLapseRateHighHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, TemLapseRateHigh));
	TSharedPtr<IPropertyHandle> TemInversionHeightHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, TemInversionHeight));
	TSharedPtr<IPropertyHandle> AuthenticDomainHeightHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, AuthenticDomainHeight));
	TSharedPtr<IPropertyHandle> DiffusionCoeffHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, DiffusionCoeff));
	TSharedPtr<IPropertyHandle> HeatEmitterAmpHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, HeatEmitterAmp));
	TSharedPtr<IPropertyHandle> HeatNoiseScaleHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, HeatNoiseScale));
	TSharedPtr<IPropertyHandle> RelHumidityGroundHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, RelHumidityGround));
	TSharedPtr<IPropertyHandle> DensityNoiseScaleHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, DensityNoiseScale));
	TSharedPtr<IPropertyHandle> WindSpeedHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, WindSpeed));
	TSharedPtr<IPropertyHandle> WindIntensityHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, WindIntensity));
	TSharedPtr<IPropertyHandle> WindDirectionHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, WindDirection));
	//TSharedPtr<IPropertyHandle> FrequencyHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, Frequency));
	//TSharedPtr<IPropertyHandle> NoiseAmpHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, NoiseAmp));
	//TSharedPtr<IPropertyHandle> NoiseSizeHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, NoiseSize));


	// Property Changed Lambda
	auto OnPropertyAlphaCoreJsonChanged = [=] { StormActor->OnAlphaCoreJsonChanged(); };
	auto OnPropertyStormCollisionActorsAndLODChanged = [=] { StormActor->OnStormCollisionActorsAndLODChanged(); };
	auto OnPropertyVolumeDensityChanged = [=] { StormActor->OnVolumeDensityChanged(); };
	auto OnPropertyVolumeColorChanged = [=] { StormActor->OnVolumeColorChanged(); };
	auto OnPropertyColorChanged = [=] { StormActor->OnColorChanged(); };
	auto OnStormBoundingBoxChanged = [=] { UE_LOG(LogTemp, Warning, TEXT("RootComponent Bounds")); };


	auto OnPropertyTemLapseRateLowChanged = [=] { StormActor->OnTemLapseRateLowChanged(); };
	auto OnPropertyTemLapseRateHighChanged = [=] { StormActor->OnTemLapseRateHighChanged(); };
	auto OnPropertyTemInversionHeightChanged = [=] { StormActor->OnTemInversionHeightChanged(); };
	auto OnPropertyAuthenticDomainHeightChanged = [=] { StormActor->OnAuthenticDomainHeightChanged(); };
	auto OnPropertyDiffusionCoeffChanged = [=] { StormActor->OnDiffusionCoeffChanged(); };
	auto OnPropertyHeatEmitterAmpChanged = [=] { StormActor->OnHeatEmitterAmpChanged(); };
	auto OnPropertyHeatNoiseScaleChanged = [=] { StormActor->OnHeatNoiseScaleChanged(); };
	auto OnPropertyRelHumidityGroundChanged = [=] { StormActor->OnRelHumidityGroundChanged(); };
	auto OnPropertyDensityNoiseScaleChanged = [=] { StormActor->OnDensityNoiseScaleChanged(); };
	auto OnPropertyWindSpeedChanged = [=] { StormActor->OnWindSpeedChanged(); };
	auto OnPropertyWindIntensityChanged = [=] { StormActor->OnWindIntensityChanged(); };
	auto OnPropertyWindDirectionChanged = [=] { StormActor->OnWindDirectionChanged(); };
	//auto OnPropertyFrequencyChanged = [=] { StormActor->OnFrequencyChanged(); };
	//auto OnPropertyNoiseAmpChanged = [=] { StormActor->OnNoiseAmpChanged(); };
	//auto OnPropertyNoiseSizeChanged = [=] { StormActor->OnNoiseSizeChanged(); };

	// Bind Property Changed Function
	//StormBoundingBoxHandle->SetOnChildPropertyValueChanged(FSimpleDelegate::CreateLambda(OnStormBoundingBoxChanged));

	AlphaCoreJsonHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyAlphaCoreJsonChanged));
	StormCollisionActorsAndLODHandle->SetOnChildPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyStormCollisionActorsAndLODChanged));
	StormCollisionActorsAndLODHandle->SetOnPropertyResetToDefault(FSimpleDelegate::CreateLambda(OnPropertyStormCollisionActorsAndLODChanged));
	VolumeDensityHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyVolumeDensityChanged));
	VolumeColorHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyVolumeColorChanged));
	ColorPropertyHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyColorChanged));

	TemLapseRateLowHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyTemLapseRateLowChanged));
	TemLapseRateHighHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyTemLapseRateHighChanged));
	TemInversionHeightHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyTemInversionHeightChanged));
	AuthenticDomainHeightHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyAuthenticDomainHeightChanged));
	DiffusionCoeffHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyDiffusionCoeffChanged));
	HeatEmitterAmpHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyHeatEmitterAmpChanged));
	HeatNoiseScaleHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyHeatNoiseScaleChanged));
	RelHumidityGroundHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyRelHumidityGroundChanged));
	DensityNoiseScaleHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyDensityNoiseScaleChanged));
	WindSpeedHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyWindSpeedChanged));
	WindIntensityHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyWindIntensityChanged));
	WindDirectionHandle->SetOnChildPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyWindDirectionChanged));
	//FrequencyHandle->SetOnChildPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyFrequencyChanged));
	//NoiseAmpHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyNoiseAmpChanged));
	//NoiseSizeHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyNoiseSizeChanged));

}


