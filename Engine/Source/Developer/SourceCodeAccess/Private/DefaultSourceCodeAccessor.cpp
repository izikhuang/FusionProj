// Copyright 1998-2014 Epic Games, Inc. All Rights Reserved.

#include "SourceCodeAccessPrivatePCH.h"
#include "DefaultSourceCodeAccessor.h"

#define LOCTEXT_NAMESPACE "DefaultSourceCodeAccessor"

bool FDefaultSourceCodeAccessor::CanAccessSourceCode() const
{
	return false;
}

FName FDefaultSourceCodeAccessor::GetFName() const
{
	return FName("None");
}

FText FDefaultSourceCodeAccessor::GetNameText() const
{
	return LOCTEXT("DefaultDisplayName", "None");
}

FText FDefaultSourceCodeAccessor::GetDescriptionText() const
{
	return LOCTEXT("DefaultDisplayDesc", "Do not open source code files.");
}

bool FDefaultSourceCodeAccessor::OpenSolution()
{
	return false;
}

bool FDefaultSourceCodeAccessor::OpenFileAtLine(const FString& FullPath, int32 LineNumber, int32 ColumnNumber)
{
	return false;
}

bool FDefaultSourceCodeAccessor::OpenSourceFiles(const TArray<FString>& AbsoluteSourcePaths)
{
	return false;
}

bool FDefaultSourceCodeAccessor::SaveAllOpenDocuments() const
{
	return false;
}

#undef LOCTEXT_NAMESPACE