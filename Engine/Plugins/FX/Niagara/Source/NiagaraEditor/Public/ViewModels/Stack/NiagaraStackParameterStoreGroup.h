// Copyright 1998-2018 Epic Games, Inc. All Rights Reserved.

#pragma once

#include "ViewModels/Stack/NiagaraStackItemGroup.h"
#include "NiagaraCommon.h"
#include "NiagaraParameterStore.h"
#include "ViewModels/Stack/NiagaraStackParameterStoreEntry.h"
#include "NiagaraStackParameterStoreGroup.generated.h"

class FNiagaraScriptViewModel;

UCLASS()
class NIAGARAEDITOR_API UNiagaraStackParameterStoreGroup : public UNiagaraStackItemGroup
{
	GENERATED_BODY()
		
public:
	void Initialize(FRequiredEntryData InRequiredEntryData,	UObject* InOwner, FNiagaraParameterStore* InParameterStore);

	void AddUserParameter(FNiagaraVariable ParameterVariable);

protected:
	virtual void RefreshChildrenInternal(const TArray<UNiagaraStackEntry*>& CurrentChildren, TArray<UNiagaraStackEntry*>& NewChildren, TArray<FStackIssue>& NewIssues) override;

private:
	TWeakObjectPtr<UObject> Owner;
	FNiagaraParameterStore* ParameterStore;
	FDelegateHandle ParameterStoreChangedHandle;
	TSharedPtr<INiagaraStackItemGroupAddUtilities> AddUtilities;
};

UCLASS()
class UNiagaraStackParameterStoreItem : public UNiagaraStackItem
{
	GENERATED_BODY()

public:
	void Initialize(FRequiredEntryData InRequiredEntryData, UObject* InOwner, FNiagaraParameterStore* InParameterStore);

	virtual FText GetDisplayName() const override;

protected:
	virtual void FinalizeInternal() override;

	virtual void RefreshChildrenInternal(const TArray<UNiagaraStackEntry*>& CurrentChildren, TArray<UNiagaraStackEntry*>& NewChildren, TArray<FStackIssue>& NewIssues) override;

private:
	void ParameterStoreChanged();

private:
	TWeakObjectPtr<UObject> Owner;
	FNiagaraParameterStore* ParameterStore;
	FDelegateHandle ParameterStoreChangedHandle;
};