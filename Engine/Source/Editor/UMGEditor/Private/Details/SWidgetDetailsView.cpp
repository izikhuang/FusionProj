// Copyright 1998-2014 Epic Games, Inc. All Rights Reserved.

#include "UMGEditorPrivatePCH.h"
#include "SWidgetDetailsView.h"

#include "BlueprintEditor.h"
#include "IDetailsView.h"
#include "Kismet2NameValidators.h"

#define LOCTEXT_NAMESPACE "UMG"

void SWidgetDetailsView::Construct(const FArguments& InArgs, TSharedPtr<FWidgetBlueprintEditor> InBlueprintEditor)
{
	BlueprintEditor = InBlueprintEditor;

	// Create a property view
	FPropertyEditorModule& EditModule = FModuleManager::Get().GetModuleChecked<FPropertyEditorModule>("PropertyEditor");

	FNotifyHook* NotifyHook = InBlueprintEditor.Get();
	FDetailsViewArgs DetailsViewArgs( /*bUpdateFromSelection=*/ false, /*bLockable=*/ false, /*bAllowSearch=*/ true, /*bObjectsUseNameArea=*/ true, /*bHideSelectionTip=*/ true, /*InNotifyHook=*/ NotifyHook, /*InSearchInitialKeyFocus=*/ false, /*InViewIdentifier=*/ NAME_None);
	DetailsViewArgs.bHideActorNameArea = true;

	PropertyView = EditModule.CreateDetailView(DetailsViewArgs);

	ChildSlot
	[
		SNew(SVerticalBox)

		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 0, 0, 6)
		[
			SNew(SHorizontalBox)
			.Visibility(this, &SWidgetDetailsView::GetNameAreaVisibility)

			+ SHorizontalBox::Slot()
			.AutoWidth()
			.Padding(0, 0, 3, 0)
			.VAlign(VAlign_Center)
			[
				SNew(SImage)
				.Image(this, &SWidgetDetailsView::GetNameIcon)
			]

			+ SHorizontalBox::Slot()
			.AutoWidth()
			.Padding(0, 0, 6, 0)
			[
				SNew(SBox)
				.WidthOverride(200.0f)
				.VAlign(VAlign_Center)
				[
					SAssignNew(NameTextBox, SEditableTextBox)
					.SelectAllTextWhenFocused(true)
					.HintText(LOCTEXT("Name", "Name"))
					.Text(this, &SWidgetDetailsView::GetNameText)
					.OnTextChanged(this, &SWidgetDetailsView::HandleNameTextChanged)
					.OnTextCommitted(this, &SWidgetDetailsView::HandleNameTextCommitted)
				]
			]

			+ SHorizontalBox::Slot()
			.AutoWidth()
			[
				SNew(SCheckBox)
				.IsChecked(this, &SWidgetDetailsView::GetIsVariable)
				.OnCheckStateChanged(this, &SWidgetDetailsView::HandleIsVariableChanged)
				[
					SNew(STextBlock)
					.Text(LOCTEXT("IsVariable", "Is Variable"))
				]
			]
		]

		+ SVerticalBox::Slot()
		.FillHeight(1.0f)
		[
			PropertyView.ToSharedRef()
		]
	];

	BlueprintEditor.Pin()->OnSelectedWidgetsChanging.AddRaw(this, &SWidgetDetailsView::OnEditorSelectionChanging);
	BlueprintEditor.Pin()->OnSelectedWidgetsChanged.AddRaw(this, &SWidgetDetailsView::OnEditorSelectionChanged);

	RegisterCustomizations();
}

SWidgetDetailsView::~SWidgetDetailsView()
{
	if ( BlueprintEditor.IsValid() )
	{
		BlueprintEditor.Pin()->OnSelectedWidgetsChanging.RemoveAll(this);
		BlueprintEditor.Pin()->OnSelectedWidgetsChanged.RemoveAll(this);
	}
}

void SWidgetDetailsView::RegisterCustomizations()
{
	FOnGetDetailCustomizationInstance LayoutDelegateDetails = FOnGetDetailCustomizationInstance::CreateStatic(&FBlueprintWidgetCustomization::MakeInstance, BlueprintEditor.Pin()->GetBlueprintObj());
	PropertyView->RegisterInstancedCustomPropertyLayout(UWidget::StaticClass(), LayoutDelegateDetails);

	FOnGetPropertyTypeCustomizationInstance CanvasSlotCustomization = FOnGetPropertyTypeCustomizationInstance::CreateStatic(&FCanvasSlotCustomization::MakeInstance, BlueprintEditor.Pin()->GetBlueprintObj());

	static FName PropertyEditor("PropertyEditor");
	FPropertyEditorModule& PropertyModule = FModuleManager::GetModuleChecked<FPropertyEditorModule>(PropertyEditor);
	PropertyModule.RegisterCustomPropertyTypeLayout(TEXT("PanelSlot"), CanvasSlotCustomization);
}

void SWidgetDetailsView::OnEditorSelectionChanging()
{
	ClearFocusIfOwned();
}

void SWidgetDetailsView::OnEditorSelectionChanged()
{
	TSet< FWidgetReference > SelectedWidgets = BlueprintEditor.Pin()->GetSelectedWidgets();

	SelectedObjects.Empty();
	for ( FWidgetReference& WidgetRef : SelectedWidgets )
	{
		SelectedObjects.Add(WidgetRef.GetPreview());
	}

	const bool bForceRefresh = false;
	PropertyView->SetObjects(SelectedObjects, bForceRefresh);
}

void SWidgetDetailsView::ClearFocusIfOwned()
{
	static bool bIsReentrant = false;
	if ( !bIsReentrant )
	{
		bIsReentrant = true;
		// When the selection is changed, we may be potentially actively editing a property,
		// if this occurs we need need to immediately clear keyboard focus
		if ( FSlateApplication::Get().HasFocusedDescendants(AsShared()) )
		{
			FSlateApplication::Get().ClearKeyboardFocus(EKeyboardFocusCause::Mouse);
		}
		bIsReentrant = false;
	}
}

EVisibility SWidgetDetailsView::GetNameAreaVisibility() const
{
	if ( SelectedObjects.Num() == 1 )
	{
		return EVisibility::Visible;
	}
	
	return EVisibility::Collapsed;
}

const FSlateBrush* SWidgetDetailsView::GetNameIcon() const
{
	if ( SelectedObjects.Num() == 1 )
	{
		UWidget* Widget = Cast<UWidget>(SelectedObjects[0].Get());
		if ( Widget )
		{
			return Widget->GetEditorIcon();
		}
	}

	return NULL;
}

FText SWidgetDetailsView::GetNameText() const
{
	if ( SelectedObjects.Num() == 1 )
	{
		UWidget* Widget = Cast<UWidget>(SelectedObjects[0].Get());
		if ( Widget )
		{
			return FText::FromString(Widget->GetLabel());
		}
	}
	
	return FText::GetEmpty();
}


void SWidgetDetailsView::HandleNameTextChanged(const FText& Text)
{
	FText OutErrorMessage;
	if ( !HandleVerifyNameTextChanged(Text, OutErrorMessage) )
	{
		NameTextBox->SetError(OutErrorMessage);
	}
	else
	{
		NameTextBox->SetError(FText::GetEmpty());
	}
}

bool SWidgetDetailsView::HandleVerifyNameTextChanged(const FText& InText, FText& OutErrorMessage)
{
	if ( SelectedObjects.Num() == 1 )
	{
		UWidget* Widget = Cast<UWidget>(SelectedObjects[0].Get());

		FString NewName = InText.ToString();

		UWidgetBlueprint* Blueprint = BlueprintEditor.Pin()->GetWidgetBlueprintObj();
		UWidget* ExistingWidget = Blueprint->WidgetTree->FindWidget(NewName);

		FKismetNameValidator Validator(Blueprint);

		const bool bUniqueNameForVariable = ( EValidatorResult::Ok == Validator.IsValid(NewName) );

		if ( ( ExistingWidget != NULL && ExistingWidget != Widget ) || !bUniqueNameForVariable )
		{
			OutErrorMessage = LOCTEXT("NameConflict", "NameConflict");
			return false;
		}

		return true;
	}
	else
	{
		return false;
	}
}

void SWidgetDetailsView::HandleNameTextCommitted(const FText& Text, ETextCommit::Type CommitType)
{
	static bool IsReentrant = false;

	if ( !IsReentrant )
	{
		IsReentrant = true;
		if ( SelectedObjects.Num() == 1 )
		{
			UWidget* Widget = Cast<UWidget>(SelectedObjects[0].Get());
			FWidgetBlueprintEditorUtils::RenameWidget(BlueprintEditor.Pin().ToSharedRef(), Widget->GetFName(), FName(*Text.ToString()));
		}
		IsReentrant = false;
	}
}

ESlateCheckBoxState::Type SWidgetDetailsView::GetIsVariable() const
{
	if ( SelectedObjects.Num() == 1 )
	{
		UWidget* Widget = Cast<UWidget>(SelectedObjects[0].Get());
		if ( Widget )
		{
			return Widget->bIsVariable ? ESlateCheckBoxState::Checked : ESlateCheckBoxState::Unchecked;
		}
	}

	return ESlateCheckBoxState::Unchecked;
}

void SWidgetDetailsView::HandleIsVariableChanged(ESlateCheckBoxState::Type CheckState)
{
	if ( SelectedObjects.Num() == 1 )
	{
		TSharedPtr<FWidgetBlueprintEditor> BPEditor = BlueprintEditor.Pin();

		UWidget* Widget = Cast<UWidget>(SelectedObjects[0].Get());
		UWidgetBlueprint* Blueprint = BlueprintEditor.Pin()->GetWidgetBlueprintObj();
		
		FWidgetReference WidgetRef = FWidgetReference::FromTemplate(BPEditor, Blueprint->WidgetTree->FindWidget(Widget->GetName()));
		if ( WidgetRef.IsValid() )
		{
			UWidget* Template = WidgetRef.GetTemplate();
			UWidget* Preview = WidgetRef.GetPreview();

			const FScopedTransaction Transaction(LOCTEXT("VariableToggle", "Variable Toggle"));
			Template->Modify();
			Preview->Modify();

			Template->bIsVariable = Preview->bIsVariable = CheckState == ESlateCheckBoxState::Checked ? true : false;

			// Refresh references and flush editors
			FBlueprintEditorUtils::MarkBlueprintAsStructurallyModified(Blueprint);
		}
	}
}

#undef LOCTEXT_NAMESPACE
