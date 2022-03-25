#ifndef __AX_NOISE_DATATYPE_H__
#define __AX_NOISE_DATATYPE_H__

#include <Utility/AxStorage.h>
#include <Math/AxVectorHelper.h>

namespace AlphaCore
{
	namespace ProceduralContent
	{
		enum AxNoiseType
		{
			kPerlinNoise,
			kSimplexPerlinNoise,
			kAlligatorNoise
		};
	}
}

typedef void AxNoiseData;

struct AxNoiseBaseParam
{
	ALPHA_SHARE_FUNC AxNoiseBaseParam()
	{
		Frequency = MakeVector3(1.0f, 1.0f, 1.0f);
		Offset = MakeVector3();
		Turbulence = 3;
		Amplitude = 1.0f;
		Roughness = 0.5f;
		Attenuation = 1.0;
	}
	ALPHA_SHARE_FUNC ~AxNoiseBaseParam()
	{

	}
	AxVector3 Frequency;
	AxVector3 Offset;
	AxUInt32 Turbulence;//fbm
	AxFp32 Amplitude;
	AxFp32 Roughness;
	AxFp32 Attenuation;
	AxFp32 TimeOffset;
};


struct AxCurlNoiseParam
{
	ALPHA_SHARE_FUNC AxCurlNoiseParam()
	{
		BaseParam;
		OriginType = AlphaCore::ProceduralContent::AxNoiseType::kPerlinNoise;
		StepSize = 0.001f;
		ControlProperty = nullptr;
		Threshold = 0.001f;
		NoiseData = nullptr;
	}
	ALPHA_SHARE_FUNC ~AxCurlNoiseParam()
	{

	}
	AxNoiseBaseParam BaseParam;
	AlphaCore::ProceduralContent::AxNoiseType OriginType;
	AxFp32 StepSize;
	void* CollisionSDF;
	void* NoiseData;

	AxFp32* ControlProperty;
	AxFp32 Threshold;
};

#include <ostream>

inline std::ostream& operator <<(std::ostream& os, AxNoiseBaseParam& noiseBase)
{
	os << "  Noise Base:\n";
	os << " Frequency:" << noiseBase.Frequency << "\n";
	os << " Offset:" << noiseBase.Offset << "\n";
	os << " Turbulence:" << noiseBase.Turbulence << "\n";
	os << " Amplitude:" << noiseBase.Amplitude << "\n";
	os << " Roughness:" << noiseBase.Roughness << "\n";
	os << " Attenuation:" << noiseBase.Attenuation << "\n";
	return os;
}

inline std::ostream& operator <<(std::ostream& os, AxCurlNoiseParam& curlNoiseParam)
{
	os << "curlNoiseParam:\n";
	os << curlNoiseParam.BaseParam << "\n";
	os << "StepSize:" << curlNoiseParam.StepSize << "\n";
	os << "CollisionSDF:" << curlNoiseParam.CollisionSDF << "\n";
	os << "ControlProperty:" << curlNoiseParam.ControlProperty << "\n";
	os << "Threshold:" << curlNoiseParam.Threshold << "\n";
	return os;
}

#endif