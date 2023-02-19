#pragma once

#include "CoreMinimal.h"
#include "EngineUtils.h"
#include <AlphaCore.h>

class ALPHACORERUNTIME_API AlphaCoreUtils
{
public:

	static void ConvertUnrealPositionToAlphaCore(const FVector3f& inData, AxVector3& outData);
	static AxVector3 ConvertUnrealPositionToAlphaCore(const FVector3f& inData);
	static AxVector3 ConvertUnrealPositionToAlphaCore(const FVector& inData);
	static void ConvertUnrealPositionToAlphaCore(const FVector3d& inData, AxVector3& outData);
	//static AxVector3 ConvertUnrealPositionToAlphaCore(const FVector3d& inData);
	static AxVector3 ConvertFromUnrealToAlphaCoreWithoutScale(const FVector3f& inData);
	static AxVector3 ConvertFromUnrealToAlphaCoreWithoutScale(const FVector& inData);
	static AxVector3I ConvertFromUnrealToAlphaCoreWithoutScale(const FIntVector& inData);



};