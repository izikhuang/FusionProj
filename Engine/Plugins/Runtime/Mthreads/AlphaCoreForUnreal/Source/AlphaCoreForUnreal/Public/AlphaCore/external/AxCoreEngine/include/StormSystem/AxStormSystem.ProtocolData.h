#ifndef __AXSTORMSYSTEM_PROTOCOL_H__
#define __AXSTORMSYSTEM_PROTOCOL_H__


#include <AxMacro.h>
#include <Utility/AxParameter.h>
#include <AxParameterJsonHelper.h>

//SimParameter
FSA_CLASS class AxStormSysSimParam
{
public:
    AxStormSysSimParam()
    {
        Init();
    }
    ~AxStormSysSimParam()
    {

    }
    AxVector2IParam TaskFrameRange;
    AxIntParam FieldBuildType;
    AxIntParam ComputeArch;
    AxFloatParam VoxelSize;
    AxVector3FParam Pivot;
    AxVector3FParam Size;
    AxToggleParam EnableClosedBoundary;
    AxToggleParam XBoundary;
    AxToggleParam XBoundary2;
    AxToggleParam YBoundary;
    AxToggleParam YBoundary2;
    AxToggleParam ZBoundary;
    AxToggleParam ZBoundary2;
    AxFloatParam TimeScale;
    AxIntParam FPS;
    AxIntParam Substeps;
    AxIntParam SolverMehtodCore;
    AxFloatParam GSIterations;
    AxFloatParam TemperatureDiffusion;
    AxFloatParam CoolingRate;
    AxFloatParam BuoyancyScale;
    AxVector3FParam BuoyancyDirection;
    AxIntParam ReflectionType;
    AxFloatParam ReflectionAmount;
    AxIntParam SourcingType;
    AxToggleParam EnableEmitDensity;
    AxStringParam SourceDensityFieldName;
    AxFloatParam SourceDensityScale;
    AxToggleParam EnableEmitFuel;
    AxStringParam SourceFuelFieldName;
    AxFloatParam SourceFuelScale;
    AxToggleParam EnableEmitTemperature;
    AxStringParam SourceTemperatureFieldName;
    AxFloatParam SourceTemperatureScale;
    AxToggleParam EnableEmitVelocity;
    AxStringParam SourceVelFieldName;
    AxFloatParam SourceVelScale;
    AxVector3FParam SourceVelAddition;
    AxToggleParam EnableEmitDivergence;
    AxStringParam SourceDivergenceFieldName;
    AxFloatParam SourceDivergenceScale;
    AxFloatParam WindSpeed;
    AxFloatParam WindIntensity;
    AxVector3FParam WindDirection;
    AxToggleParam UseKernelFuse;
    AxStringParam AlphaCommand;
    AxIntParam PressureIterations;
    AxStringParam WorkSpace;
    AxStringParam EmitterCacheReadFilePath;
    AxStringParam CacheOutFilePath;
    AxStringParam ExecuteCommand;

    void FromJson(std::string jsonRawCode)
    {
        AX_INFO("AxStormSystem : Read sim parameter form json file ... ");
        rapidjson::Document doc;
        if (doc.Parse(jsonRawCode.c_str()).HasParseError())
            return;
        if (!doc.HasMember("ParamMap"))
            return;
        auto& paramMap = doc["ParamMap"];
        if (!paramMap.IsObject())
            return;
        AlphaCore::JsonHelper::AxVector2IParamDeserilization(TaskFrameRange, paramMap["taskFrameRange"]);
        AlphaCore::JsonHelper::AxIntParamDeserilization(FieldBuildType, paramMap["fieldBuildType"]);
        AlphaCore::JsonHelper::AxIntParamDeserilization(ComputeArch, paramMap["computeArch"]);
        AlphaCore::JsonHelper::AxFloatParamDeserilization(VoxelSize, paramMap["voxelSize"]);
        AlphaCore::JsonHelper::AxVector3FParamDeserilization(Pivot, paramMap["pivot"]);
        AlphaCore::JsonHelper::AxVector3FParamDeserilization(Size, paramMap["size"]);
        AlphaCore::JsonHelper::AxToggleParamDeserilization(EnableClosedBoundary, paramMap["enableClosedBoundary"]);
        AlphaCore::JsonHelper::AxToggleParamDeserilization(XBoundary, paramMap["xBoundary"]);
        AlphaCore::JsonHelper::AxToggleParamDeserilization(XBoundary2, paramMap["xBoundary2"]);
        AlphaCore::JsonHelper::AxToggleParamDeserilization(YBoundary, paramMap["yBoundary"]);
        AlphaCore::JsonHelper::AxToggleParamDeserilization(YBoundary2, paramMap["yBoundary2"]);
        AlphaCore::JsonHelper::AxToggleParamDeserilization(ZBoundary, paramMap["zBoundary"]);
        AlphaCore::JsonHelper::AxToggleParamDeserilization(ZBoundary2, paramMap["zBoundary2"]);
        AlphaCore::JsonHelper::AxFloatParamDeserilization(TimeScale, paramMap["timeScale"]);
        AlphaCore::JsonHelper::AxIntParamDeserilization(FPS, paramMap["fPS"]);
        AlphaCore::JsonHelper::AxIntParamDeserilization(Substeps, paramMap["substeps"]);
        AlphaCore::JsonHelper::AxIntParamDeserilization(SolverMehtodCore, paramMap["solverMehtodCore"]);
        AlphaCore::JsonHelper::AxFloatParamDeserilization(GSIterations, paramMap["gSIterations"]);
        AlphaCore::JsonHelper::AxFloatParamDeserilization(TemperatureDiffusion, paramMap["temperatureDiffusion"]);
        AlphaCore::JsonHelper::AxFloatParamDeserilization(CoolingRate, paramMap["coolingRate"]);
        AlphaCore::JsonHelper::AxFloatParamDeserilization(BuoyancyScale, paramMap["buoyancyScale"]);
        AlphaCore::JsonHelper::AxVector3FParamDeserilization(BuoyancyDirection, paramMap["buoyancyDirection"]);
        AlphaCore::JsonHelper::AxIntParamDeserilization(ReflectionType, paramMap["reflectionType"]);
        AlphaCore::JsonHelper::AxFloatParamDeserilization(ReflectionAmount, paramMap["reflectionAmount"]);
        AlphaCore::JsonHelper::AxIntParamDeserilization(SourcingType, paramMap["sourcingType"]);
        AlphaCore::JsonHelper::AxToggleParamDeserilization(EnableEmitDensity, paramMap["enableEmitDensity"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(SourceDensityFieldName, paramMap["sourceDensityFieldName"]);
        AlphaCore::JsonHelper::AxFloatParamDeserilization(SourceDensityScale, paramMap["sourceDensityScale"]);
        AlphaCore::JsonHelper::AxToggleParamDeserilization(EnableEmitFuel, paramMap["enableEmitFuel"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(SourceFuelFieldName, paramMap["sourceFuelFieldName"]);
        AlphaCore::JsonHelper::AxFloatParamDeserilization(SourceFuelScale, paramMap["sourceFuelScale"]);
        AlphaCore::JsonHelper::AxToggleParamDeserilization(EnableEmitTemperature, paramMap["enableEmitTemperature"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(SourceTemperatureFieldName, paramMap["sourceTemperatureFieldName"]);
        AlphaCore::JsonHelper::AxFloatParamDeserilization(SourceTemperatureScale, paramMap["sourceTemperatureScale"]);
        AlphaCore::JsonHelper::AxToggleParamDeserilization(EnableEmitVelocity, paramMap["enableEmitVelocity"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(SourceVelFieldName, paramMap["sourceVelFieldName"]);
        AlphaCore::JsonHelper::AxFloatParamDeserilization(SourceVelScale, paramMap["sourceVelScale"]);
        AlphaCore::JsonHelper::AxVector3FParamDeserilization(SourceVelAddition, paramMap["sourceVelAddition"]);
        AlphaCore::JsonHelper::AxToggleParamDeserilization(EnableEmitDivergence, paramMap["enableEmitDivergence"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(SourceDivergenceFieldName, paramMap["sourceDivergenceFieldName"]);
        AlphaCore::JsonHelper::AxFloatParamDeserilization(SourceDivergenceScale, paramMap["sourceDivergenceScale"]);
        AlphaCore::JsonHelper::AxFloatParamDeserilization(WindSpeed, paramMap["windSpeed"]);
        AlphaCore::JsonHelper::AxFloatParamDeserilization(WindIntensity, paramMap["windIntensity"]);
        AlphaCore::JsonHelper::AxVector3FParamDeserilization(WindDirection, paramMap["windDirection"]);
        AlphaCore::JsonHelper::AxToggleParamDeserilization(UseKernelFuse, paramMap["useKernelFuse"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(AlphaCommand, paramMap["alphaCommand"]);
        AlphaCore::JsonHelper::AxIntParamDeserilization(PressureIterations, paramMap["pressureIterations"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(WorkSpace, paramMap["workSpace"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(EmitterCacheReadFilePath, paramMap["emitterCacheReadFilePath"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(CacheOutFilePath, paramMap["cacheOutFilePath"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(ExecuteCommand, paramMap["executeCommand"]);

        AX_INFO("AxStormSystem : Read sim parameter form json OVER !!! ");
    }

    void Init()
    {
        this->TaskFrameRange.Set(MakeVector2(1, 240));
        this->FieldBuildType.Set(0);
        this->ComputeArch.Set(0);
        this->VoxelSize.Set(0.1f);
        this->Pivot.Set(MakeVector3(0.0f, 0.0f, 0.0f));
        this->Size.Set(MakeVector3(10.0f, 10.0f, 10.0f));
        this->EnableClosedBoundary.Set(0);
        this->XBoundary.Set(0);
        this->XBoundary2.Set(0);
        this->YBoundary.Set(0);
        this->YBoundary2.Set(0);
        this->ZBoundary.Set(0);
        this->ZBoundary2.Set(0);
        this->TimeScale.Set(1.0f);
        this->FPS.Set(24);
        this->Substeps.Set(1);
        this->SolverMehtodCore.Set(0);
        this->GSIterations.Set(10.0f);
        this->TemperatureDiffusion.Set(0.5f);
        this->CoolingRate.Set(0.75f);
        this->BuoyancyScale.Set(2.5f);
        this->BuoyancyDirection.Set(MakeVector3(0.0f, 1.0f, 0.0f));
        this->ReflectionType.Set(0);
        this->ReflectionAmount.Set(0.95f);
        this->SourcingType.Set(0);
        this->EnableEmitDensity.Set(0);
        this->SourceDensityFieldName.Set("density");
        this->SourceDensityScale.Set(1.0f);
        this->EnableEmitFuel.Set(0);
        this->SourceFuelFieldName.Set("fuel");
        this->SourceFuelScale.Set(1.0f);
        this->EnableEmitTemperature.Set(0);
        this->SourceTemperatureFieldName.Set("temperature");
        this->SourceTemperatureScale.Set(1.0f);
        this->EnableEmitVelocity.Set(0);
        this->SourceVelFieldName.Set("v");
        this->SourceVelScale.Set(1.0f);
        this->SourceVelAddition.Set(MakeVector3(0.0f, 1.0f, 0.0f));
        this->EnableEmitDivergence.Set(0);
        this->SourceDivergenceFieldName.Set("divergence");
        this->SourceDivergenceScale.Set(1.0f);
        this->WindSpeed.Set(1.0f);
        this->WindIntensity.Set(0.5f);
        this->WindDirection.Set(MakeVector3(0.0f, 0.0f, 0.0f));
        this->UseKernelFuse.Set(0);
        this->AlphaCommand.Set("");
        this->PressureIterations.Set(1);
        this->ExecuteCommand.Set("/AlphaCore.exe");

    }
};

#endif