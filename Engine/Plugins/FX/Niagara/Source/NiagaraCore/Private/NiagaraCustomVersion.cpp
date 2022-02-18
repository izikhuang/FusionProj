// Copyright Epic Games, Inc. All Rights Reserved.

#include "NiagaraCustomVersion.h"
#include "Serialization/CustomVersion.h"

const FGuid FNiagaraCustomVersion::GUID(0xFCF57AFA, 0x50764283, 0xB9A9E658, 0xFFA02D32);

// Register the custom version with core
FCustomVersionRegistration GRegisterNiagaraCustomVersion(FNiagaraCustomVersion::GUID, FNiagaraCustomVersion::LatestVersion, TEXT("NiagaraVer"));

// Note: When encountering a conflict on this file please generate a new GUID
const FGuid FNiagaraCustomVersion::LatestScriptCompileVersion(0x2D0D660A, 0x6E4B44C3, 0x8AA99B65, 0xC58EC4F1);
