// Copyright 1998-2015 Epic Games, Inc. All Rights Reserved.

#include "SlatePrivatePCH.h"
#include "Menu.h"
#include "SPopup.h"
#include "SMenuAnchor.h"

#define LOCTEXT_NAMESPACE "MenuStack"

namespace FMenuStackDefs
{
	/** Maximum size of menus as a fraction of the work area height */
	const float MaxMenuScreenHeightFraction = 0.8f;
	const float AnimationDuration = 0.15f;
}

/** Overlay widget class used to hold menu contents and display them on top of the current window */
class SMenuPanel : public SOverlay
{
public:
	SLATE_BEGIN_ARGS(SMenuPanel)
	{
		_Visibility = EVisibility::SelfHitTestInvisible;
	}

	SLATE_END_ARGS()

	void Construct(const FArguments& InArgs)
	{
		SOverlay::Construct(SOverlay::FArguments());
	}

	void PushMenu(TSharedRef<FMenuBase> InMenu, const FVector2D& InLocation)
	{
		check(InMenu->GetContent().IsValid());

		TSharedPtr<SWindow> ParentWindow = InMenu->GetParentWindow();
		check(ParentWindow.IsValid());

		// Transform InLocation into a position local to this panel (assumes the panel is in an overlay that covers the whole of the panel window) 
		FVector2D PanelInScreen = ParentWindow->GetRectInScreen().GetTopLeft();
		FVector2D PanelInWindow = ParentWindow->GetLocalToScreenTransform().Inverse().TransformPoint(PanelInScreen);
		FVector2D LocationInWindow = ParentWindow->GetLocalToScreenTransform().Inverse().TransformPoint(InLocation);
		FVector2D LocationInPanel = LocationInWindow - PanelInWindow;

		// Add the new menu into a slot on this panel and set the padding so that its position is correct
		AddSlot()
		.HAlign(HAlign_Left)
		.VAlign(VAlign_Top)
		.Padding(LocationInPanel.X, LocationInPanel.Y, 0, 0)
		[
			InMenu->GetContent().ToSharedRef()
		];

		// Make sure that the menu will remove itself from the panel when dismissed
		InMenu->GetOnMenuDismissed().AddSP(this, &SMenuPanel::OnMenuClosed);
	}

	void OnMenuClosed(TSharedRef<IMenu> InMenu)
	{
		RemoveSlot(InMenu->GetContent().ToSharedRef());
	}
};


namespace
{
	/** Widget that wraps any menu created in FMenuStack to provide default key handling, focus tracking and helps us spot menus in widget paths */
	DECLARE_DELEGATE_RetVal_OneParam(FReply, FOnKeyDown, FKey)
	DECLARE_DELEGATE_OneParam(FOnMenuLostFocus, const FWidgetPath&)
	class SMenuContentWrapper : public SCompoundWidget
	{
	public:
		SLATE_BEGIN_ARGS(SMenuContentWrapper) 
				: _MenuContent()
				, _OnKeyDown()
			{}

			SLATE_DEFAULT_SLOT(FArguments, MenuContent)
			SLATE_EVENT(FOnKeyDown, OnKeyDown)
			SLATE_EVENT(FOnMenuLostFocus, OnMenuLostFocus)
		SLATE_END_ARGS()

		/** Construct this widget */
		void Construct(const FArguments& InArgs)
		{
			OnKeyDownDelegate = InArgs._OnKeyDown;
			OnMenuLostFocus = InArgs._OnMenuLostFocus;
			ChildSlot
			[
				InArgs._MenuContent.Widget
			];
		}

		virtual void OnFocusChanging(const FWeakWidgetPath& PreviousFocusPath, const FWidgetPath& NewWidgetPath) override
		{
			// if focus changed and this menu content had focus (or one of its children did) then inform the stack via the OnMenuLostFocus event
			if (OnMenuLostFocus.IsBound() && PreviousFocusPath.ContainsWidget(AsShared()))
			{
				return OnMenuLostFocus.Execute(NewWidgetPath);
			}
		}

	private:

		/** This widget must support keyboard focus */
		virtual bool SupportsKeyboardFocus() const override
		{
			return true;
		}

		virtual FReply OnKeyDown( const FGeometry& MyGeometry, const FKeyEvent& InKeyEvent ) override
		{
			if (OnKeyDownDelegate.IsBound())
			{
				return OnKeyDownDelegate.Execute(InKeyEvent.GetKey());
			}

			return FReply::Unhandled();
		}

		/** Delegate to forward keys down events on the menu */
		FOnKeyDown OnKeyDownDelegate;

		/** Delegate to inform the stack that a menu has lost focus and might need to be closed */
		FOnMenuLostFocus OnMenuLostFocus;
	};


	/** Global handler used to handle key presses on popup menus */
	FReply OnMenuKeyDown(const FKey Key)
	{
		if (Key == EKeys::Escape)
		{
			FSlateApplication::Get().DismissAllMenus();
			return FReply::Handled();
		}

		return FReply::Unhandled();
	}

}	// anon namespace


TSharedRef<SWindow> FMenuStack::PushMenu( const TSharedRef<SWindow>& ParentWindow, const TSharedRef<SWidget>& InContent, const FVector2D& SummonLocation, const FPopupTransitionEffect& TransitionEffect, const bool bFocusImmediately, const bool bShouldAutoSize, const FVector2D& WindowSize, const FVector2D& SummonLocationSize)
{
	// Deprecated method that is no longer called by its deprecated counterpart in FSlateApplication so it will only create a dummy message warning the caller not to use this method
	// Backwards compatibility of the API is taken care of by FSlateApplication.
	ensureMsgf(false, TEXT("PushMenu() returning an SWindow is deprecated. Use Push() that returns an IMenu instead."));

	FSlateRect Anchor(SummonLocation, SummonLocation + SummonLocationSize);
	FVector2D ExpectedSize(300, 200);
	FVector2D AdjustedSummonLocation = FSlateApplication::Get().CalculatePopupWindowPosition(Anchor, ExpectedSize, Orient_Vertical);

	TSharedRef<SWindow> NewMenuWindow =
		SNew(SWindow)
		.IsPopupWindow(true)
		.SizingRule(bShouldAutoSize ? ESizingRule::Autosized : ESizingRule::UserSized)
		.ScreenPosition(AdjustedSummonLocation)
		.AutoCenter(EAutoCenter::None)
		.ClientSize(ExpectedSize)
		.FocusWhenFirstShown(true)
		.ActivateWhenFirstShown(true)
		[
			SNew(SVerticalBox)
			+ SVerticalBox::Slot()
			.VAlign(VAlign_Center)
			.HAlign(HAlign_Center)
			[
				SNew(STextBlock)
				.Text(LOCTEXT("PushMenuDeprecatedWarning", "WARNING: PushMenu() returning an SWindow is deprecated. Use Push() that returns an IMenu instead."))
			]
		];

	FSlateApplication::Get().AddWindowAsNativeChild(NewMenuWindow, ParentWindow, true);

	return NewMenuWindow;
}

TSharedRef<IMenu> FMenuStack::Push(const FWidgetPath& InOwnerPath, const TSharedRef<SWidget>& InContent, const FVector2D& SummonLocation, const FPopupTransitionEffect& TransitionEffect, const bool bFocusImmediately, const FVector2D& SummonLocationSize, TOptional<EPopupMethod> InMethod)
{
	FSlateRect Anchor(SummonLocation, SummonLocation + SummonLocationSize);
	TSharedPtr<IMenu> ParentMenu;

	if (HasMenus())
	{
		// Find a menu in the stack in InOwnerPath to determine the level to insert this 
		ParentMenu = FindMenuInWidgetPath(InOwnerPath);
		check(HostWindow.IsValid());
	}

	if (!ParentMenu.IsValid())
	{
		// pushing a new root menu (leave ParentMenu invalid)

		// The active method is determined when a new root menu is pushed
		ActiveMethod = InMethod.IsSet() ? InMethod : QueryPopupMethod(InOwnerPath);

		// The host window is determined when a new root menu is pushed
		SetHostWindow(InOwnerPath.GetWindow());
	}

	return PushInternal(ParentMenu, InContent, Anchor, TransitionEffect, bFocusImmediately);
}

TSharedRef<IMenu> FMenuStack::Push(const TSharedPtr<IMenu>& InParentMenu, const TSharedRef<SWidget>& InContent, const FVector2D& SummonLocation, const FPopupTransitionEffect& TransitionEffect, const bool bFocusImmediately, const FVector2D& SummonLocationSize)
{
	check(Stack.Contains(InParentMenu));
	check(HostWindow.IsValid());

	FSlateRect Anchor(SummonLocation, SummonLocation + SummonLocationSize);

	return PushInternal(InParentMenu, InContent, Anchor, TransitionEffect, bFocusImmediately);
}

TSharedRef<IMenu> FMenuStack::PushHosted(const FWidgetPath& InOwnerPath, const TSharedRef<IMenuHost>& InMenuHost, const TSharedRef<SWidget>& InContent, TSharedPtr<SWidget>& OutWrappedContent, const FPopupTransitionEffect& TransitionEffect)
{
	TSharedPtr<IMenu> ParentMenu;

	if (HasMenus())
	{
		// Find a menu in the stack in InOwnerPath to determine the level to insert this 
		ParentMenu = FindMenuInWidgetPath(InOwnerPath);
		check(HostWindow.IsValid());
	}

	if (!ParentMenu.IsValid())
	{
		// pushing a new root menu (leave ParentMenu invalid)

		// The active method is determined when a new root menu is pushed and hosted menus are always UseCurrentWindow
		ActiveMethod = EPopupMethod::UseCurrentWindow;

		// The host window is determined when a new root menu is pushed
		SetHostWindow(InOwnerPath.GetWindow());
	}

	return PushHosted(ParentMenu, InMenuHost, InContent, OutWrappedContent, TransitionEffect);
}

TSharedRef<IMenu> FMenuStack::PushHosted(const TSharedPtr<IMenu>& InParentMenu, const TSharedRef<IMenuHost>& InMenuHost, const TSharedRef<SWidget>& InContent, TSharedPtr<SWidget>& OutWrappedContent, const FPopupTransitionEffect& TransitionEffect)
{
	check(HostWindow.IsValid());

	// Create a FMenuInHostWidget
	TSharedRef<SWidget> WrappedContent = WrapContent(InContent);
	TSharedRef<FMenuInHostWidget> OutMenu = MakeShareable(new FMenuInHostWidget(InMenuHost, WrappedContent));
	PendingNewMenu = OutMenu;

	// Set the returned content - this must be drawn by the hosting widget until the menu gets dismissed and calls IMenuHost::OnMenuDismissed on its host
	OutWrappedContent = WrappedContent;

	// Register to get a callback when it's dismissed - to fixup stack
	OutMenu->GetOnMenuDismissed().AddRaw(this, &FMenuStack::OnMenuDestroyed);

	PostPush(InParentMenu, OutMenu);

	PendingNewMenu.Reset();

	return OutMenu;
}

TSharedRef<IMenu> FMenuStack::PushInternal(const TSharedPtr<IMenu>& InParentMenu, const TSharedRef<SWidget>& InContent, FSlateRect Anchor, const FPopupTransitionEffect& TransitionEffect, const bool bFocusImmediately)
{
	FPrePushArgs PrePushArgs;
	PrePushArgs.Content = InContent;
	PrePushArgs.Anchor = Anchor;
	PrePushArgs.TransitionEffect = TransitionEffect;
	PrePushArgs.bFocusImmediately = bFocusImmediately;

	// Pre-push stage
	//   Determines correct layout
	//   Wraps content
	//   Other common setup steps needed by all (non-hosted) menus
	const FPrePushResults PrePushResults = PrePush(PrePushArgs);

	// Menu object creation stage
	TSharedRef<FMenuBase> OutMenu = ActiveMethod.Get(EPopupMethod::CreateNewWindow) == EPopupMethod::CreateNewWindow
		? PushNewWindow(InParentMenu, PrePushResults)
		: PushPopup(InParentMenu, PrePushResults);

	// Post-push stage
	//   Updates the stack and content map member variables
	PostPush(InParentMenu, OutMenu);

	PendingNewMenu.Reset();

	return OutMenu;
}

FMenuStack::FPrePushResults FMenuStack::PrePush(const FPrePushArgs& InArgs)
{
	FPrePushResults OutResults;
	OutResults.bFocusImmediately = InArgs.bFocusImmediately;
	if (InArgs.bFocusImmediately)
	{
		OutResults.WidgetToFocus = InArgs.Content;
	}

	// Only enable window position/size transitions if we're running at a decent frame rate
	OutResults.bAllowAnimations = FSlateApplication::Get().AreMenuAnimationsEnabled() && FSlateApplication::Get().IsRunningAtTargetFrameRate();

	// Calc the max height available on screen for the menu
	float MaxHeight;
	if (ActiveMethod.Get(EPopupMethod::CreateNewWindow) == EPopupMethod::CreateNewWindow)
	{
		FSlateRect WorkArea = FSlateApplication::Get().GetWorkArea(InArgs.Anchor);
		MaxHeight = FMenuStackDefs::MaxMenuScreenHeightFraction * WorkArea.GetSize().Y;
	}
	else
	{
		MaxHeight = FMenuStackDefs::MaxMenuScreenHeightFraction * HostWindow->GetClientSizeInScreen().Y;
	}

	bool bAnchorSetsMinWidth = InArgs.TransitionEffect.SlideDirection == FPopupTransitionEffect::ComboButton;

	// Wrap menu content in a box needed for various sizing and tracking purposes
	FOptionalSize OptionalMinWidth = bAnchorSetsMinWidth ? InArgs.Anchor.GetSize().X : FOptionalSize();
	FOptionalSize OptionalMinHeight = MaxHeight;

	// Wrap content in an SPopup before the rest of the wrapping process - should make menus appear on top using deferred presentation
	TSharedRef<SWidget> TempContent = SNew(SPopup)[InArgs.Content.ToSharedRef()];

	OutResults.WrappedContent = WrapContent(TempContent, OptionalMinWidth, OptionalMinHeight);

	// @todo slate: Assumes that popup is not Scaled up or down from application scale.
	OutResults.WrappedContent->SlatePrepass(FSlateApplication::Get().GetApplicationScale());
	// @todo slate: Doesn't take into account potential window border size
	OutResults.ExpectedSize = OutResults.WrappedContent->GetDesiredSize();

	EOrientation Orientation = (InArgs.TransitionEffect.SlideDirection == FPopupTransitionEffect::SubMenu) ? Orient_Horizontal : Orient_Vertical;

	// Calculate the correct position for the menu based on the popup method
	if (ActiveMethod.Get(EPopupMethod::CreateNewWindow) == EPopupMethod::CreateNewWindow)
	{
		// Places the menu's window in the work area
		OutResults.AnimStartLocation = OutResults.AnimFinalLocation = FSlateApplication::Get().CalculatePopupWindowPosition(InArgs.Anchor, OutResults.ExpectedSize, Orientation);
	}
	else
	{
		// Places the menu's content in the host window
		const FVector2D ProposedPlacement(
			Orientation == Orient_Horizontal ? InArgs.Anchor.Right : InArgs.Anchor.Left,
			Orientation == Orient_Horizontal ? InArgs.Anchor.Top : InArgs.Anchor.Bottom);

		OutResults.AnimStartLocation = OutResults.AnimFinalLocation =
			ComputePopupFitInRect(InArgs.Anchor, FSlateRect(ProposedPlacement, ProposedPlacement + OutResults.ExpectedSize), Orientation, HostWindow->GetClientRectInScreen());
	}

	// Start the pop-up menu at an offset location, then animate it to its target location over time
	// @todo: Anims aren't supported or attempted - this is legacy code left in in case we reinstate menu anims
	if (OutResults.bAllowAnimations)
	{
		const bool bSummonRight = OutResults.AnimFinalLocation.X >= OutResults.AnimStartLocation.X;
		const bool bSummonBelow = OutResults.AnimFinalLocation.Y >= OutResults.AnimStartLocation.Y;
		const int32 SummonDirectionX = bSummonRight ? 1 : -1;
		const int32 SummonDirectionY = bSummonBelow ? 1 : -1;

		switch (InArgs.TransitionEffect.SlideDirection)
		{
		case FPopupTransitionEffect::None:
			// No sliding
			break;

		case FPopupTransitionEffect::ComboButton:
			OutResults.AnimStartLocation.Y = FMath::Max(OutResults.AnimStartLocation.Y + 30.0f * SummonDirectionY, 0.0f);
			break;

		case FPopupTransitionEffect::TopMenu:
			OutResults.AnimStartLocation.Y = FMath::Max(OutResults.AnimStartLocation.Y + 60.0f * SummonDirectionY, 0.0f);
			break;

		case FPopupTransitionEffect::SubMenu:
			OutResults.AnimStartLocation.X += 60.0f * SummonDirectionX;
			break;

		case FPopupTransitionEffect::TypeInPopup:
			OutResults.AnimStartLocation.Y = FMath::Max(OutResults.AnimStartLocation.Y + 30.0f * SummonDirectionY, 0.0f);
			break;

		case FPopupTransitionEffect::ContextMenu:
			OutResults.AnimStartLocation.X += 30.0f * SummonDirectionX;
			OutResults.AnimStartLocation.Y += 50.0f * SummonDirectionY;
			break;
		}
	}

	// Release the mouse so that context can be properly restored upon closing menus.  See CL 1411833 before changing this.
	if (InArgs.bFocusImmediately)
	{
		FSlateApplication::Get().ReleaseMouseCapture();
	}

	return OutResults;
}

TSharedRef<FMenuBase> FMenuStack::PushNewWindow(TSharedPtr<IMenu> InParentMenu, const FPrePushResults& InPrePushResults)
{
	check(ActiveMethod.GetValue() == EPopupMethod::CreateNewWindow);

	// Start pop-up windows out transparent, then fade them in over time
	const EWindowTransparency Transparency(InPrePushResults.bAllowAnimations ? EWindowTransparency::PerWindow : EWindowTransparency::None);
	const float InitialWindowOpacity = InPrePushResults.bAllowAnimations ? 0.0f : 1.0f;
	const float TargetWindowOpacity = 1.0f;

	// Create a new window for the menu
	TSharedRef<SWindow> NewMenuWindow = SNew(SWindow)
		.IsPopupWindow(true)
		.SizingRule(ESizingRule::Autosized)
		.ScreenPosition(InPrePushResults.AnimStartLocation)
		.AutoCenter(EAutoCenter::None)
		.ClientSize(InPrePushResults.ExpectedSize)
		.InitialOpacity(InitialWindowOpacity)
		.SupportsTransparency(Transparency)
		.FocusWhenFirstShown(InPrePushResults.bFocusImmediately)
		.ActivateWhenFirstShown(InPrePushResults.bFocusImmediately)
		[
			InPrePushResults.WrappedContent.ToSharedRef()
		];

	PendingNewWindow = NewMenuWindow;

	if (InPrePushResults.bFocusImmediately && InPrePushResults.WidgetToFocus.IsValid())
	{
		// Focus the unwrapped content rather than just the window
		NewMenuWindow->SetWidgetToFocusOnActivate(InPrePushResults.WidgetToFocus);
	}

	TSharedRef<FMenuInWindow> Menu = MakeShareable(new FMenuInWindow(NewMenuWindow, InPrePushResults.WrappedContent.ToSharedRef()));
	PendingNewMenu = Menu;

	TSharedPtr<SWindow> ParentWindow;
	if (InParentMenu.IsValid())
	{
		ParentWindow = InParentMenu->GetParentWindow();
	}
	else
	{
		ParentWindow = HostWindow;
	}

	FSlateApplication::Get().AddWindowAsNativeChild(NewMenuWindow, ParentWindow.ToSharedRef(), true);

	// Kick off the intro animation!
	// @todo: Anims aren't supported or attempted - this is legacy code left in in case we reinstate menu anims
	if (InPrePushResults.bAllowAnimations)
	{
		FCurveSequence Sequence;
		Sequence.AddCurve(0, FMenuStackDefs::AnimationDuration, ECurveEaseFunction::CubicOut);
		NewMenuWindow->MorphToPosition(Sequence, TargetWindowOpacity, InPrePushResults.AnimFinalLocation);
	}

	PendingNewWindow.Reset();

	return Menu;
}

TSharedRef<FMenuBase> FMenuStack::PushPopup(TSharedPtr<IMenu> InParentMenu, const FPrePushResults& InPrePushResults)
{
	check(ActiveMethod.GetValue() == EPopupMethod::UseCurrentWindow);

	// Create a FMenuInPopup
	check(InPrePushResults.WrappedContent.IsValid());
	TSharedRef<FMenuInPopup> Menu = MakeShareable(new FMenuInPopup(InPrePushResults.WrappedContent.ToSharedRef()));
	PendingNewMenu = Menu;

	// Register to get callback when it's dismissed - to fixup stack
	Menu->GetOnMenuDismissed().AddRaw(this, &FMenuStack::OnMenuDestroyed);

	// Add it to a slot on the menus panel widget
	HostWindowPopupPanel->PushMenu(Menu, InPrePushResults.AnimFinalLocation);

	if (InPrePushResults.bFocusImmediately && InPrePushResults.WidgetToFocus.IsValid())
	{
		FSlateApplication::Get().SetKeyboardFocus(InPrePushResults.WidgetToFocus, EFocusCause::SetDirectly);
	}

	return Menu;
}

void FMenuStack::PostPush(TSharedPtr<IMenu> InParentMenu, TSharedRef<FMenuBase> InMenu)
{
	// Determine at which index we insert this new menu in the stack
	int32 InsertIndex = 0;
	if (InParentMenu.IsValid())
	{
		int32 ParentIndex = Stack.IndexOfByKey(InParentMenu);
		check(ParentIndex != INDEX_NONE);

		InsertIndex = ParentIndex + 1;
	}

	// Insert before dismissing others to stop the stack accidentally emptying briefly mid-push and reseting some state
	Stack.Insert(InMenu, InsertIndex);
	CachedContentMap.Add(InMenu->GetContent(), InMenu);

	// Dismiss menus after the insert point
	if (Stack.Num() > InsertIndex + 1)
	{
		DismissFrom(Stack[InsertIndex + 1]);

		// tidy the stack data now (it will happen via callbacks from the dismissed menus but that might be delayed)
		for (int32 StackIndex = Stack.Num() - 1; StackIndex > InsertIndex; --StackIndex)
		{
			CachedContentMap.Remove(Stack[StackIndex]->GetContent());
			Stack.RemoveAt(StackIndex);
		}
	}

	// When a new menu is pushed, if we are not already in responsive mode for Slate UI, enter it now
	// to ensure the menu is responsive in low FPS situations
	if (!ThrottleHandle.IsValid())
	{
		ThrottleHandle = FSlateThrottleManager::Get().EnterResponsiveMode();
	}
}

EPopupMethod FMenuStack::QueryPopupMethod(const FWidgetPath& PathToQuery)
{
	for (int32 WidgetIndex = PathToQuery.Widgets.Num() - 1; WidgetIndex >= 0; --WidgetIndex)
	{
		TOptional<EPopupMethod> PopupMethod = PathToQuery.Widgets[WidgetIndex].Widget->OnQueryPopupMethod();
		if (PopupMethod.IsSet())
		{
			return PopupMethod.GetValue();
		}
	}

	return EPopupMethod::CreateNewWindow;
}

void FMenuStack::Dismiss(int32 FirstStackIndexToRemove)
{
	// deprecated
	DismissInternal(FirstStackIndexToRemove);
}

void FMenuStack::DismissFrom(const TSharedPtr<IMenu>& InFromMenu)
{
	int32 Index = Stack.IndexOfByKey(InFromMenu);
	if (Index != INDEX_NONE)
	{
		DismissInternal(Index);
	}
}

void FMenuStack::DismissAll()
{
	const int32 TopLevel = 0;
	DismissInternal(TopLevel);
}

void FMenuStack::DismissInternal(int32 FirstStackIndexToRemove)
{
	// Dismiss the stack in reverse order so we destroy children before parents (causes focusing issues if done the other way around)
	for (int32 StackIndex = Stack.Num() - 1; StackIndex >= FirstStackIndexToRemove; --StackIndex)
	{
		Stack[StackIndex]->Dismiss();
	}
}

void FMenuStack::SetHostWindow(TSharedPtr<SWindow> InWindow)
{
	if (HostWindow != InWindow)
	{
		// If the host window is changing, remove the popup panel from the previous host
		if (HostWindow.IsValid() && HostWindowPopupPanel.IsValid())
		{
			HostWindow->RemoveOverlaySlot(HostWindowPopupPanel.ToSharedRef());
			HostWindowPopupPanel.Reset();
		}

		HostWindow = InWindow;

		// If the host window is changing, add a popup panel to the new host
		if (HostWindow.IsValid())
		{
			// setup a popup layer on the new window
			HostWindow->AddOverlaySlot()
			[
				SAssignNew(HostWindowPopupPanel, SMenuPanel)
			];
		}
	}
}

void FMenuStack::OnMenuDestroyed(TSharedRef<IMenu> InMenu)
{
	int32 Index = Stack.IndexOfByKey(InMenu);
	if (Index != INDEX_NONE)
	{
		// Dismiss menus below this one
		for (int32 StackIndex = Stack.Num() - 1; StackIndex > Index; --StackIndex)
		{
			Stack[StackIndex]->Dismiss();	// this will cause OnMenuDestroyed() to re-enter
		}

		// Clean up the stack and content map arrays
		for (int32 StackIndex = Stack.Num() - 1; StackIndex >= Index; --StackIndex)
		{
			CachedContentMap.Remove(Stack[StackIndex]->GetContent());
			Stack.RemoveAt(StackIndex);
		}

		// Leave responsive mode once the last menu closes
		if (Stack.Num() == 0)
		{
			if (ThrottleHandle.IsValid())
			{
				FSlateThrottleManager::Get().LeaveResponsiveMode(ThrottleHandle);
			}

			SetHostWindow(TSharedPtr<SWindow>());
		}
	}
}

void FMenuStack::OnMenuContentLostFocus(const FWidgetPath& InFocussedPath)
{
	// Window activation messages will make menus collapse when in CreateNewWindow mode
	// In UseCurrentWindow mode we must look for focus moving from menus
	if (ActiveMethod.Get(EPopupMethod::CreateNewWindow) == EPopupMethod::UseCurrentWindow && HasMenus() && !PendingNewMenu.IsValid())
	{
		// If focus is switching determine which of our menus it is in, if any
		TSharedPtr<IMenu> FocussedMenu = FindMenuInWidgetPath(InFocussedPath);

		if (FocussedMenu.IsValid())
		{
			// dismiss menus below FocussedMenu
			int32 FocussedIndex = Stack.IndexOfByKey(FocussedMenu);
			check(FocussedIndex != INDEX_NONE);

			if (Stack.Num() > FocussedIndex + 1)
			{
				DismissFrom(Stack[FocussedIndex + 1]);
			}
		}
		else
		{
			// Focus has moved outside all menus - collapse the stack
			DismissAll();
		}
	}
}

TSharedRef<SWidget> FMenuStack::WrapContent(TSharedRef<SWidget> InContent, FOptionalSize OptionalMinWidth, FOptionalSize OptionalMinHeight)
{
	// Wrap menu content in a box that limits its maximum height
	// and in a SMenuContentWrapper that handles key presses and focus changes.
	return SNew(SMenuContentWrapper)
		.OnKeyDown_Static(&OnMenuKeyDown)
		.OnMenuLostFocus_Raw(this, &FMenuStack::OnMenuContentLostFocus)
		.MenuContent()
		[
			SNew(SBox)
			.MinDesiredWidth(OptionalMinWidth)
			.MaxDesiredHeight(OptionalMinHeight)
			[
				InContent
			]
		];
}

TSharedPtr<IMenu> FMenuStack::FindMenuInWidgetPath(const FWidgetPath& PathToQuery) const
{
	for (int32 PathIndex = PathToQuery.Widgets.Num() - 1; PathIndex >= 0; --PathIndex)
	{
		TSharedPtr<const SWidget> Widget = PathToQuery.Widgets[PathIndex].Widget;
		const TSharedPtr<FMenuBase>* FoundMenu = CachedContentMap.Find(Widget);
		if (FoundMenu != nullptr)
		{
			return *FoundMenu;
		}
	}
	return TSharedPtr<IMenu>();
}

void FMenuStack::RemoveWindow( TSharedRef<SWindow> WindowToRemove )
{
	// deprecated
	OnWindowDestroyed(WindowToRemove);
}

void FMenuStack::OnWindowDestroyed(TSharedRef<SWindow> InWindow)
{
	if (HostWindow == InWindow)
	{
		// If the host window is destroyed, collapse the whole stack and reset all state
		Stack.Empty();
		CachedContentMap.Empty();

		SetHostWindow(TSharedPtr<SWindow>());
	}
	else
	{
		// A window was requested to be destroyed, so make sure it's not in the menu stack to avoid it
		// becoming a parent to a freshly-created window!
		TSharedPtr<IMenu> Menu = FindMenuFromWindow(InWindow);

		if (Menu.IsValid())
		{
			OnMenuDestroyed(Menu.ToSharedRef());
		}
	}
}

void FMenuStack::OnWindowActivated( TSharedRef<SWindow> ActivatedWindow )
{
	if (ActivatedWindow != PendingNewWindow && HasMenus())
	{
		TWeakPtr<IMenu> ActivatedMenu = FindMenuFromWindow(ActivatedWindow);

		if (ActivatedMenu.IsValid())
		{
			// Dismiss menus below ActivatedMenu
			int32 ActivatedIndex = Stack.IndexOfByKey(ActivatedMenu);
			check(ActivatedIndex != INDEX_NONE);

			if (Stack.Num() > ActivatedIndex + 1)
			{
				DismissFrom(Stack[ActivatedIndex + 1]);
			}
		}
		else
		{
			// Activated a window that isn't a menu - collapse the stack
			DismissAll();
		}
	}
}


int32 FMenuStack::FindLocationInStack( TSharedPtr<SWindow> WindowToFind ) const
{
	// deprecated
	if (WindowToFind.IsValid())
	{
		TWeakPtr<IMenu> Menu = FindMenuFromWindow(WindowToFind.ToSharedRef());

		return Stack.IndexOfByKey(Menu);
	}

	// The window was not found
	return INDEX_NONE;
}

TSharedPtr<IMenu> FMenuStack::FindMenuFromWindow(TSharedRef<SWindow> InWindow) const
{
	const TSharedPtr<FMenuBase>* FoundMenu = Stack.FindByPredicate([InWindow](TSharedPtr<FMenuBase> Menu) { return Menu->GetOwnedWindow() == InWindow; });
	if (FoundMenu != nullptr)
	{
		return *FoundMenu;
	}
	return TSharedPtr<IMenu>();
}

FSlateRect FMenuStack::GetToolTipForceFieldRect(TSharedRef<IMenu> InMenu, const FWidgetPath& PathContainingMenu) const
{
	FSlateRect ForceFieldRect(0, 0, 0, 0);

	int32 StackLevel = Stack.IndexOfByKey(InMenu);

	if (StackLevel != INDEX_NONE)
	{
		for (int32 StackLevelIndex = StackLevel + 1; StackLevelIndex < Stack.Num(); ++StackLevelIndex)
		{
			TSharedPtr<SWidget> MenuContent = Stack[StackLevelIndex]->GetContent();
			if (MenuContent.IsValid())
			{
				FWidgetPath WidgetPath = PathContainingMenu.GetPathDownTo(MenuContent.ToSharedRef());
				if (!WidgetPath.IsValid())
				{
					FSlateApplication::Get().GeneratePathToWidgetChecked(MenuContent.ToSharedRef(), WidgetPath);
				}
				if (WidgetPath.IsValid())
				{
					const FGeometry& ContentGeometry = WidgetPath.Widgets.Last().Geometry;
					ForceFieldRect = ForceFieldRect.Expand(ContentGeometry.GetClippingRect());
				}
			}
		}
	}

	return ForceFieldRect;
}

bool FMenuStack::HasMenus() const
{
	return Stack.Num() > 0;
}

bool FMenuStack::HasOpenSubMenus( const TSharedRef<SWindow>& Window ) const
{
	// deprecated
	TWeakPtr<IMenu> Menu = FindMenuFromWindow(Window);
	if (Menu.IsValid())
	{
		return Menu != Stack.Last();
	}
	return false;
}

bool FMenuStack::HasOpenSubMenus(TSharedPtr<IMenu> InMenu) const
{
	int32 StackIndex = Stack.IndexOfByKey(InMenu);
	return StackIndex != INDEX_NONE && StackIndex < Stack.Num() - 1;
}

int32 FMenuStack::GetNumStackLevels() const
{
	return Stack.Num();
}

TSharedPtr<SWindow> FMenuStack::GetHostWindow() const
{
	return HostWindow;
}


FMenuWindowList& FMenuStack::GetWindowsAtStackLevel( const int32 StackLevelIndex )
{
	// deprecated
	// can't support this so return an empty list
	ensure(false);
	static FMenuWindowList EmptyList;
	return EmptyList;
}

#undef LOCTEXT_NAMESPACE
