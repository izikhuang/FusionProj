#ifndef __AXPARTICLEFLUIDEMITTER_SIMPARAM_PROTOCOL_H__
#define __AXPARTICLEFLUIDEMITTER_SIMPARAM_PROTOCOL_H__


#include "AxMacro.h"
#include "Utility/AxParameter.h"
#include "AxParameterJsonHelper.h"

//SimParameter
FSA_CLASS class AxParticleFluidEmitterSIMParam
{
public:
	AxParticleFluidEmitterSIMParam()
	{
		Init();
	}
	~AxParticleFluidEmitterSIMParam()
	{

	}
	AxVector3FParam UpDir;
	AxVector3FParam Rotate;
	AxIntParam Emission_type;
	AxVector3FParam FrontDir;
	AxVector3FParam Center;
	AxFloatParam Speed;
	AxVector2FParam Size;

	void FromJson(std::string jsonRawCode)
	{
		AX_INFO("AxParticleFluidEmitterSIMParam : Read sim parameter form json file ... ");
		rapidjson::Document doc;
		if (doc.Parse(jsonRawCode.c_str()).HasParseError())
			return;
		if (!doc.HasMember("ParamMap"))
			return;
		auto& paramMap = doc["ParamMap"];
		if (!paramMap.IsObject())
			return;
		AlphaCore::JsonHelper::AxVector3FParamDeserilization(UpDir, paramMap["upDir"]);
		AlphaCore::JsonHelper::AxVector3FParamDeserilization(Rotate, paramMap["rotate"]);
		AlphaCore::JsonHelper::AxIntParamDeserilization(Emission_type, paramMap["emission_type"]);
		AlphaCore::JsonHelper::AxVector3FParamDeserilization(FrontDir, paramMap["frontDir"]);
		AlphaCore::JsonHelper::AxVector3FParamDeserilization(Center, paramMap["center"]);
		AlphaCore::JsonHelper::AxFloatParamDeserilization(Speed, paramMap["speed"]);
		AlphaCore::JsonHelper::AxVector2FParamDeserilization(Size, paramMap["size"]);

		AX_INFO("AxParticleFluidEmitterSIMParam : Read sim parameter form json OVER !!! ");
	}

	void Init()
	{
		this->UpDir.Set(MakeVector3(0.0f, 0.0f, 1.0f));
		this->Rotate.Set(MakeVector3(0.0f, 0.0f, 0.0f));
		this->Emission_type.Set(0);
		this->FrontDir.Set(MakeVector3(0.0f, 1.0f, 0.0f));
		this->Center.Set(MakeVector3(0.0f, 0.0f, 0.0f));
		this->Speed.Set(0.0f);
		this->Size.Set(MakeVector2(10.0f, 10.0f));

	}
};

#endif