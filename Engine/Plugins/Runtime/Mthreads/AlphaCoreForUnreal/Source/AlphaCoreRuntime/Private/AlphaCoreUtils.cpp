#include "AlphaCoreUtils.h"

#define ALCORE_UNREAL_SCALE_FACTOR 100

#define Vector3DataConvertFromUnrealInternal(in, out) \
		out.x = (float)in.X / ALCORE_UNREAL_SCALE_FACTOR; \
		out.y = (float)in.Z / ALCORE_UNREAL_SCALE_FACTOR; \
		out.z = (float)in.Y / ALCORE_UNREAL_SCALE_FACTOR; \

#define Vector3DataConvertFromUnrealWithoutScaleInternal(in,out) \
		out.x = in.X; \
		out.y = in.Z; \
		out.z = in.Y; \


void AlphaCoreUtils::ConvertUnrealPositionToAlphaCore(const FVector3f& inData, AxVector3& outData)
{
	Vector3DataConvertFromUnrealInternal(inData, outData);
}

AxVector3 AlphaCoreUtils::ConvertUnrealPositionToAlphaCore(const FVector3f& inData)
{
	AxVector3 outData;
	Vector3DataConvertFromUnrealInternal(inData, outData);
	return outData;
}

AxVector3 AlphaCoreUtils::ConvertUnrealPositionToAlphaCore(const FVector& inData)
{
	AxVector3 outData;
	Vector3DataConvertFromUnrealInternal(inData, outData);
	return outData;
}


void AlphaCoreUtils::ConvertUnrealPositionToAlphaCore(const FVector3d& inData, AxVector3& outData)
{
	Vector3DataConvertFromUnrealInternal(inData, outData);
}


//AxVector3 AlphaCoreUtils::ConvertUnrealPositionToAlphaCore(const FVector3d& inData)
//{
//	AxVector3 outData;
//	Vector3DataConvertFromUnrealInternal(inData, outData);
//	return outData;
//}


AxVector3 AlphaCoreUtils::ConvertFromUnrealToAlphaCoreWithoutScale(const FVector3f& inData)
{
	AxVector3 outData;
	Vector3DataConvertFromUnrealWithoutScaleInternal(inData, outData);
	return outData;
}

AxVector3 AlphaCoreUtils::ConvertFromUnrealToAlphaCoreWithoutScale(const FVector& inData)
{
	AxVector3 outData;
	Vector3DataConvertFromUnrealWithoutScaleInternal(inData, outData);
	return outData;
}

AxVector3I AlphaCoreUtils::ConvertFromUnrealToAlphaCoreWithoutScale(const FIntVector& inData)
{
	AxVector3I outData;
	Vector3DataConvertFromUnrealWithoutScaleInternal(inData, outData)
	return outData;
}

#undef Vector3DataConvertFromUnrealWithoutScaleInternal
#undef Vector3DataConvertFromUnrealInternal


#undef ALCORE_UNREAL_SCALE_FACTOR
