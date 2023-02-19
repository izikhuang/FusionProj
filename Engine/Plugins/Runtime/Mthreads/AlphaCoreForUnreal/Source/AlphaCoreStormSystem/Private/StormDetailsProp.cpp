#pragma once

#include "StormDetailsProp.h"


#include "DetailWidgetRow.h"
#include "DetailCategoryBuilder.h"
#include "DetailLayoutBuilder.h"
//#include "Input/Reply.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Input/SButton.h"


#define LOCTEXT_NAMESPACE "StormActorDetails"

DEFINE_LOG_CATEGORY_STATIC(StormDetailsProp, Log, All);

TSharedRef<IDetailCustomization> FStormDetailsProp::MakeInstance()
{
	return MakeShareable(new FStormDetailsProp());
}

void FStormDetailsProp::CustomizeDetails(IDetailLayoutBuilder& DetailBuilder)
{
	TArray<TWeakObjectPtr<UObject>> Objects;
	DetailBuilder.GetObjectsBeingCustomized(Objects);

	// Actor Under Control
	TWeakObjectPtr<AStormActor> CurrentStormActor;
	for (auto Object : Objects)
	{
		if (Object.IsValid() && Object->IsA(AStormActor::StaticClass()))
		{
			CurrentStormActor = Cast<AStormActor>(Object);
		}
	}

	if (!CurrentStormActor.IsValid()) return;




	// Custom Widget
	IDetailCategoryBuilder& Category = DetailBuilder.EditCategory("AlphaCore", LOCTEXT("RowNameText", "AlphaCore"), ECategoryPriority::Important);

	/*
	{
		//auto GetCurveTypeText = []()
		//{
		//	return FText::FromString("3333333");
		//};

		FDetailWidgetRow& CheckBoxRow = Category.AddCustomRow(LOCTEXT("Keyword", "Emitter Type"));
		//TSharedPtr< SCheckBox > CheckBoxExportLODs;

		//TSharedPtr<SComboBox<TSharedPtr<FString>>> TypeComboBox;
		TSharedPtr<SComboBox<TSharedPtr<FString>>> StaticMesh;

		TArray<TSharedPtr<FString>> ComboTEXT;
		ComboTEXT.Reset();
		ComboTEXT.Add(MakeShareable(new FString("1")));
		ComboTEXT.Add(MakeShareable(new FString("2")));
		ComboTEXT.Add(MakeShareable(new FString("3")));

		TSharedPtr<FString> IntialSelec;
		IntialSelec = ComboTEXT[0];

		SMyCombo myCombo;
		TSharedRef<SWidget> myCombo = MakeShared<SMyCombo>();
		TSharedRef<SHorizontalBox> WidgetBox = SNew(SHorizontalBox);
		WidgetBox->AddSlot().AttachWidget(myCombo);
		//[
			//myCombo
			//SAssignNew(StaticMesh, SComboBox<TSharedPtr<FString>>)
			//.OptionsSource(&ComboTEXT)
			//.InitiallySelectedItem(IntialSelec)
			//.OnGenerateWidget_Lambda
			//(
			//	[](TSharedPtr<FString> InItem) 
			//	{
			//		FText ChoiceEntryText = FText::FromString(*InItem);
			//		return SNew(STextBlock)
			//			.Text(ChoiceEntryText);
			//	}
			//)
			//.OnSelectionChanged_Lambda([=](TSharedPtr<FString> NewChoice, ESelectInfo::Type SelectType)
			//{
			//		FText ret = FText::FromString(*NewChoice);
			//	return SNew(STextBlock)
			//		.Text(ret);
			//})
			//.OnSelectionChanged_Lambda
			//(
			//	[]() {return FText::FromString("3333"); }
			//)

		//];
		//WidgetBox->AddSlot()
		//.VAlign(VAlign_Fill)
		//[
		//	SAssignNew(StaticMesh, SComboBox<TSharedPtr<FString>>)
		//	.Content()
		//[
		//	SNew(STextBlock)
		//	.Text(LOCTEXT("TitleNameText", "Static Mesh222"))
		//]
		//];

		CheckBoxRow
			.NameContent()
			[
				SNew(STextBlock)
				.Text(LOCTEXT("TitleNameText", "Emitter Type"))
				.Font(IDetailLayoutBuilder::GetDetailFont())
			];
		//.ValueContent()
		//	[
		//		SNew(SHorizontalBox)
		//		+ SHorizontalBox::Slot()
		//	.Padding(0)
		//	.AutoWidth()
		//	[
		//		SAssignNew(StaticMesh, SComboBox<TSharedPtr<FString>>)
		//		.OptionsSource(&ComboTEXT)
		//		.InitiallySelectedItem(ComboTEXT[0])
		//		


		//		//SNew(SButton)
		//		//.Text(LOCTEXT("Import", "Import"))
		//		//.ToolTipText(LOCTEXT("ImportRulesTable_Tooltip", "Import Rules Table to the Config"))
		//		//.IsEnabled_Lambda([this, ScannerSettingsIns]()->bool
		//		//{
		//		//	return ScannerSettingsIns->bUseRulesTable;
		//		//})
		//		//.OnClicked_Lambda([this, ScannerSettingsIns]()
		//		//{
		//		//	if (ScannerSettingsIns)
		//		//	{
		//		//		ScannerSettingsIns->HandleImportRulesTable();
		//		//	}
		//		//	return(FReply::Handled());
		//		//})
		//	]
		//];

		CheckBoxRow.ValueWidget.Widget = myCombo;
	}
	*/
	// Control Buttons

	{
		Category.InitiallyCollapsed(false);

		Category.AddCustomRow(LOCTEXT("Keyword", "Storm Init"))
			.NameContent()
			[
				SNew(STextBlock)
				.Text(LOCTEXT("TitleNameText", "Storm Init"))
				.Font(IDetailLayoutBuilder::GetDetailFont())
			]
			.ValueContent().HAlign(HAlign_Fill)
			[
				SNew(SButton)
				.Text(LOCTEXT("ButtonText", "Init"))
				.OnClicked_Lambda(
					[Objects]()
					{
						for (auto Object : Objects)
						{
							if (Object.IsValid() && Object->IsA(AStormActor::StaticClass()))
							{
								Cast<AStormActor>(Object)->OnInitButtonClicked();
							}
						}
						return FReply::Handled();
					}
				)
			];

		Category.AddCustomRow(LOCTEXT("Keyword", "Storm Clear"))
			.NameContent()
			[
				SNew(STextBlock)
				.Text(LOCTEXT("TitleNameText", "Storm Clear"))
				.Font(IDetailLayoutBuilder::GetDetailFont())
			]
			.ValueContent().HAlign(HAlign_Fill)
			[
				SNew(SButton)
				.Text(LOCTEXT("ButtonText", "Clear"))
				.OnClicked_Lambda(
				[Objects]()
				{
					for (auto Object : Objects)
					{
						if (Object.IsValid() && Object->IsA(AStormActor::StaticClass()))
						{
							Cast<AStormActor>(Object)->OnClearButtonClicked();
						}
					}
				return FReply::Handled();
			})];

	}



	///////////////////////////////////
	// Property Changed Lambda
	///////////////////////////////////
	// 
	auto OnPropertyVoxelSizeChanged = [=] { CurrentStormActor->OnVoxelSizeChanged(); };
	// Atmosphere
	auto OnPropertyHeatEmitterAmpChanged = [=] { CurrentStormActor->OnHeatEmitterAmpChanged(); };
	auto OnPropertyAuthenticDomainHeightChanged = [=] { CurrentStormActor->OnAuthenticDomainHeightChanged(); };
	auto OnPropertyDiffusionCoeffChanged = [=] { CurrentStormActor->OnDiffusionCoeffChanged(); };
	auto OnPropertyBuoyancyScaleChanged = [=] { CurrentStormActor->OnBuoyancyScaleChanged(); };
	auto OnPropertyCloudPositionOffsetChanged = [=] { CurrentStormActor->OnCloudOffsetChanged(); };

	// Wind
	auto OnPropertyWindSpeedChanged = [=] { CurrentStormActor->OnWindSpeedChanged(); };
	auto OnPropertyWindIntensityChanged = [=] { CurrentStormActor->OnWindIntensityChanged(); };
	auto OnPropertyWindDirectionChanged = [=] { CurrentStormActor->OnWindDirectionChanged(); };

	// Operations
	auto OnPropertyStormOperationChanged = [=] { CurrentStormActor->OnOperationsValueChanged(); };

	// Render Data
	auto OnOnRenderDataChangedHandleChanged = [=] { CurrentStormActor->OnRenderDataChanged(); };





	///////////////////////////
	// Property Handles 
	///////////////////////////
	TSharedPtr<IPropertyHandle> SimulationVoxleSizeHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, SimulationVoxleSize));

	// Render Data
	TSharedPtr<IPropertyHandle> DensityScaleHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, DensityScale));
	TSharedPtr<IPropertyHandle> StepSizeHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, StepSize));
	TSharedPtr<IPropertyHandle> ShadowScaleHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, ShadowScale));
	TSharedPtr<IPropertyHandle> UsePhaseHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, UsePhase));
	TSharedPtr<IPropertyHandle> PhaseHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, Phase));
	TSharedPtr<IPropertyHandle> InputDensityMinMaxHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, InputDensityMinMax));
	TSharedPtr<IPropertyHandle> VolumeColorHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, VolumeColor));

	// Atmosphere
	TSharedPtr<IPropertyHandle> HeatEmitterAmpHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, HeatEmitterAmp));
	TSharedPtr<IPropertyHandle> AuthenticDomainHeightHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, AuthenticDomainHeight));
	TSharedPtr<IPropertyHandle> DiffusionCoeffHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, DiffusionCoeff));
	TSharedPtr<IPropertyHandle> BuoyancyScaleHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, BuoyancyScale));
	TSharedPtr<IPropertyHandle> CloudPositionOffsetZHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, CloudPositionOffsetZ));

	// Wind
	TSharedPtr<IPropertyHandle> WindSpeedHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, WindSpeed));
	TSharedPtr<IPropertyHandle> WindIntensityHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, WindIntensity));
	TSharedPtr<IPropertyHandle> WindDirectionHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, WindDirection));

	// Operations
	TSharedPtr<IPropertyHandle> StormOperationHandle = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, StormOperations));



	// Bind Property Changed Function
	
	SimulationVoxleSizeHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyVoxelSizeChanged));
	// RenderData
	DensityScaleHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnOnRenderDataChangedHandleChanged));
	StepSizeHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnOnRenderDataChangedHandleChanged));
	ShadowScaleHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnOnRenderDataChangedHandleChanged));
	UsePhaseHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnOnRenderDataChangedHandleChanged));
	PhaseHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnOnRenderDataChangedHandleChanged));
	InputDensityMinMaxHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnOnRenderDataChangedHandleChanged));
	InputDensityMinMaxHandle->SetOnChildPropertyValueChanged(FSimpleDelegate::CreateLambda(OnOnRenderDataChangedHandleChanged));
	VolumeColorHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnOnRenderDataChangedHandleChanged));

	// Atmosphere
	HeatEmitterAmpHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyHeatEmitterAmpChanged));
	AuthenticDomainHeightHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyAuthenticDomainHeightChanged));
	DiffusionCoeffHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyDiffusionCoeffChanged));
	BuoyancyScaleHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyBuoyancyScaleChanged));
	CloudPositionOffsetZHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyCloudPositionOffsetChanged));

	// Wind
	WindSpeedHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyWindSpeedChanged));
	WindIntensityHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyWindIntensityChanged));
	WindDirectionHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyWindDirectionChanged));
	WindDirectionHandle->SetOnChildPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyWindDirectionChanged));

	// Operations
	StormOperationHandle->SetOnPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyStormOperationChanged));
	StormOperationHandle->SetOnChildPropertyValueChanged(FSimpleDelegate::CreateLambda(OnPropertyStormOperationChanged));




	// Transform Propery
	//TSharedPtr<IPropertyHandle> transform = DetailBuilder.GetProperty(GET_MEMBER_NAME_CHECKED(AStormActor, SceneComponent));
	//transform->SetOnChildPropertyValueChanged(FSimpleDelegate::CreateLambda(OnSimulationFieldHandleChanged));

}


#undef LOCTEXT_NAMESPACE