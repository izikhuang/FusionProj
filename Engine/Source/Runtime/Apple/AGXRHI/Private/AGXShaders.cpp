// Copyright Epic Games, Inc. All Rights Reserved.

/*=============================================================================
	AGXShaders.cpp: AGX RHI shader implementation.
=============================================================================*/

#include "AGXRHIPrivate.h"

#include "Shaders/Debugging/AGXShaderDebugCache.h"
#include "Shaders/AGXCompiledShaderKey.h"
#include "Shaders/AGXCompiledShaderCache.h"
#include "Shaders/AGXShaderLibrary.h"

#include "HAL/FileManager.h"
#include "HAL/PlatformFileManager.h"
#include "Misc/Paths.h"
#include "MetalShaderResources.h"
#include "AGXResources.h"
#include "AGXProfiler.h"
#include "AGXCommandBuffer.h"
#include "Serialization/MemoryReader.h"
#include "Misc/FileHelper.h"
#include "Misc/ScopeRWLock.h"
#include "Misc/Compression.h"
#include "Misc/MessageDialog.h"

#define SHADERCOMPILERCOMMON_API
#	include "Developer/ShaderCompilerCommon/Public/ShaderCompilerCommon.h"
#undef SHADERCOMPILERCOMMON_API


NSString* AGXDecodeMetalSourceCode(uint32 CodeSize, TArray<uint8> const& CompressedSource)
{
	NSString* GlslCodeNSString = nil;
	if (CodeSize && CompressedSource.Num())
	{
		TArray<ANSICHAR> UncompressedCode;
		UncompressedCode.AddZeroed(CodeSize+1);
		bool bSucceed = FCompression::UncompressMemory(NAME_Zlib, UncompressedCode.GetData(), CodeSize, CompressedSource.GetData(), CompressedSource.Num());
		if (bSucceed)
		{
			GlslCodeNSString = [[NSString stringWithUTF8String:UncompressedCode.GetData()] retain];
		}
	}
	return GlslCodeNSString;
}

mtlpp::LanguageVersion AGXValidateVersion(uint32 Version)
{
	static uint32 MetalMacOSVersions[][3] = {
		{10,15,0},
		{11,0,0},
        {12,0,0},
	};
	static uint32 MetaliOSVersions[][3] = {
		{13,0,0},
		{14,0,0},
		{15,0,0},
	};
	static TCHAR const* StandardNames[] =
	{
		TEXT("Metal 2.2"),
		TEXT("Metal 2.3"),
		TEXT("Metal 2.4"),
	};
	
	mtlpp::LanguageVersion Result = mtlpp::LanguageVersion::Version2_2;
	switch(Version)
	{
		case 7:
			Result = mtlpp::LanguageVersion::Version2_4;
			break;
		case 6:
			Result = mtlpp::LanguageVersion::Version2_3;
			break;
		case 5:
			Result = mtlpp::LanguageVersion::Version2_2;
			break;
		default:
			Result = mtlpp::LanguageVersion::Version2_2;
			break;
	}
	
	if (!FApplePlatformMisc::IsOSAtLeastVersion(MetalMacOSVersions[Version], MetaliOSVersions[Version], MetaliOSVersions[Version]))
	{
		FFormatNamedArguments Args;
		Args.Add(TEXT("ShaderVersion"), FText::FromString(FString(StandardNames[Version])));
#if PLATFORM_MAC
		Args.Add(TEXT("RequiredOS"), FText::FromString(FString::Printf(TEXT("macOS %d.%d.%d"), MetalMacOSVersions[Version][0], MetalMacOSVersions[Version][1], MetalMacOSVersions[Version][2])));
#else
		Args.Add(TEXT("RequiredOS"), FText::FromString(FString::Printf(TEXT("%d.%d.%d"), MetaliOSVersions[Version][0], MetaliOSVersions[Version][1], MetaliOSVersions[Version][2])));
#endif
		FText LocalizedMsg = FText::Format(NSLOCTEXT("AGXRHI", "ShaderVersionUnsupported", "The current OS version does not support {ShaderVersion} required by the project. You must upgrade to {RequiredOS} to run this project."),Args);
		
		FText Title = NSLOCTEXT("AGXRHI", "ShaderVersionUnsupportedTitle", "Shader Version Unsupported");
		FMessageDialog::Open(EAppMsgType::Ok, LocalizedMsg, &Title);
		
		FPlatformMisc::RequestExit(true);
	}
	
	return Result;
}
