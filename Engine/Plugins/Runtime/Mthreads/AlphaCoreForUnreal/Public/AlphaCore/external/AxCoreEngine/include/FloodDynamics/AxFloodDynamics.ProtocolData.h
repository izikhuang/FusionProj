#ifndef __AXPARTICLEFLUID_SIMPARAM_PROTOCOL_H__
#define __AXPARTICLEFLUID_SIMPARAM_PROTOCOL_H__


#include "AxMacro.h"
#include "Utility/AxParameter.h"
#include "AxParameterJsonHelper.h"

//SimParameter
FSA_CLASS class AxParticleFluidSIMParam
{
public:
    AxParticleFluidSIMParam()
    {
        Init();
    }
    ~AxParticleFluidSIMParam()
    {
    
    }
    AxFloatParam Particle_radius;
AxStringParam SDFAniGeoWriteFileName;
AxIntParam SampleLever;
AxToggleParam UseField;
AxStringParam ExecuteCommand;
AxIntParam Substeps;
AxFloatParam Viscosity;
AxVector3FParam Gravity;
AxVector2IParam TaskFrameRange;
AxStringParam SDFStaicGeoWrite;
AxStringParam FieldGeoPath_FilePath;
AxIntParam ComputeArch;
AxStringParam CacheOutFilePath;
AxStringParam WorkSpace;
AxIntParam FPS;
AxVector3FParam Pivot;
AxIntParam Fluide_type;

AxFloatParam Vorticity;
AxVector3FParam Size;

    void FromJson(std::string jsonRawCode)
    {
        AX_INFO("AxParticleFluidSIMParam : Read sim parameter form json file ... ");
        rapidjson::Document doc;
        if (doc.Parse(jsonRawCode.c_str()).HasParseError())
            return;
        if (!doc.HasMember("ParamMap"))
            return;
        auto& paramMap = doc["ParamMap"];
        if (!paramMap.IsObject())
            return;
        AlphaCore::JsonHelper::AxFloatParamDeserilization(Particle_radius, paramMap["particle_radius"]);
AlphaCore::JsonHelper::AxStringParamDeserilization(SDFAniGeoWriteFileName, paramMap["sDFAniGeoWriteFileName"]);
AlphaCore::JsonHelper::AxIntParamDeserilization(SampleLever, paramMap["sampleLever"]);
AlphaCore::JsonHelper::AxToggleParamDeserilization(UseField, paramMap["useField"]);
AlphaCore::JsonHelper::AxStringParamDeserilization(ExecuteCommand, paramMap["executeCommand"]);
AlphaCore::JsonHelper::AxIntParamDeserilization(Substeps, paramMap["substeps"]);
AlphaCore::JsonHelper::AxFloatParamDeserilization(Viscosity, paramMap["viscosity"]);
AlphaCore::JsonHelper::AxVector3FParamDeserilization(Gravity, paramMap["gravity"]);
AlphaCore::JsonHelper::AxVector2IParamDeserilization(TaskFrameRange, paramMap["taskFrameRange"]);
AlphaCore::JsonHelper::AxStringParamDeserilization(SDFStaicGeoWrite, paramMap["sDFStaicGeoWrite"]);
AlphaCore::JsonHelper::AxStringParamDeserilization(FieldGeoPath_FilePath, paramMap["fieldGeoPath_FilePath"]);
AlphaCore::JsonHelper::AxIntParamDeserilization(ComputeArch, paramMap["computeArch"]);
AlphaCore::JsonHelper::AxStringParamDeserilization(CacheOutFilePath, paramMap["cacheOutFilePath"]);
AlphaCore::JsonHelper::AxStringParamDeserilization(WorkSpace, paramMap["workSpace"]);
AlphaCore::JsonHelper::AxIntParamDeserilization(FPS, paramMap["fPS"]);
AlphaCore::JsonHelper::AxVector3FParamDeserilization(Pivot, paramMap["pivot"]);
AlphaCore::JsonHelper::AxIntParamDeserilization(Fluide_type, paramMap["fluide_type"]);
AlphaCore::JsonHelper::AxFloatParamDeserilization(Vorticity, paramMap["vorticity"]);
AlphaCore::JsonHelper::AxVector3FParamDeserilization(Size, paramMap["size"]);

        AX_INFO("AxParticleFluidSIMParam : Read sim parameter form json OVER !!! ");
    }
    
    void Init()
    {
        this->Particle_radius.Set(0.1f);
this->SampleLever.Set(1);
this->UseField.Set(0);
this->ExecuteCommand.Set("");
this->Substeps.Set(1);
this->Viscosity.Set(0.01f);
this->Gravity.Set(MakeVector3(0.0f,-9.8f,0.0f));
this->TaskFrameRange.Set(MakeVector2(1,100));
this->ComputeArch.Set(0);
this->FPS.Set(24);
this->Pivot.Set(MakeVector3(0.0f,0.0f,0.0f));
this->Fluide_type.Set(0);
this->Vorticity.Set(0.01f);
this->Size.Set(MakeVector3(1.0f,1.0f,1.0f));

    }
};

#endif