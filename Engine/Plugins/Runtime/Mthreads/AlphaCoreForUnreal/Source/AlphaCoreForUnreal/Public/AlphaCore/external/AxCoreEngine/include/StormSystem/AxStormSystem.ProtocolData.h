#ifndef __AXSTORMSYS_SIMPARAM_PROTOCOL_H__
#define __AXSTORMSYS_SIMPARAM_PROTOCOL_H__


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
    AxVector3FParam Rotate;
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
    AxRampCurve32RAWData Densityramp;
    AxToggleParam Cdrangeoverride;
    AxVector2FParam Cdrange;
    AxRampCurve32RAWData Cdramp;
    AxToggleParam UseKernelFuse;
    AxStringParam AlphaCommand;
    AxIntParam PressureIterations;
    AxStringParam WorkSpace;
    AxStringParam EmitterCacheReadFilePath;
    AxStringParam CacheOutFilePath;
    AxStringParam StaticHeightFieldFielPath;
    AxStringParam ExecuteCommand;

    void FromJson(std::string jsonRawCode)
    {
        AX_INFO("AxStormSysSimParam : Read sim parameter form json file ... ");
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
        AlphaCore::JsonHelper::AxVector3FParamDeserilization(Rotate, paramMap["rotate"]);
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
        AlphaCore::JsonHelper::AxRampCurve32RAWDataDeserilization(Densityramp, paramMap["densityramp"]);
        AlphaCore::JsonHelper::AxToggleParamDeserilization(Cdrangeoverride, paramMap["cdrangeoverride"]);
        AlphaCore::JsonHelper::AxVector2FParamDeserilization(Cdrange, paramMap["cdrange"]);
        //AlphaCore::JsonHelper::AxRampCurve32RAWDataDeserilization(Cdramp, paramMap["cdramp"]);
        AlphaCore::JsonHelper::AxToggleParamDeserilization(UseKernelFuse, paramMap["useKernelFuse"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(AlphaCommand, paramMap["alphaCommand"]);
        AlphaCore::JsonHelper::AxIntParamDeserilization(PressureIterations, paramMap["pressureIterations"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(WorkSpace, paramMap["workSpace"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(EmitterCacheReadFilePath, paramMap["emitterCacheReadFilePath"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(CacheOutFilePath, paramMap["cacheOutFilePath"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(StaticHeightFieldFielPath, paramMap["staticHeightFieldFielPath"]);
        AlphaCore::JsonHelper::AxStringParamDeserilization(ExecuteCommand, paramMap["executeCommand"]);

        AX_INFO("AxStormSysSimParam : Read sim parameter form json OVER !!! ");
    }

    void Init()
    {
        this->TaskFrameRange.Set(MakeVector2(1, 540));
        this->FieldBuildType.Set(0);
        this->ComputeArch.Set(0);
        this->VoxelSize.Set(0.1f);
        this->Pivot.Set(MakeVector3(0.0f, 0.0f, 0.0f));
        this->Size.Set(MakeVector3(10.0f, 10.0f, 10.0f));
        this->Rotate.Set(MakeVector3(0.0f, 0.0f, 0.0f));
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
        AxFp32 DENSITYRAMP_RAWDATA[32] = { 0.0f, 0.032258063554763794f, 0.06451612710952759f, 0.09677419066429138f, 0.12903225421905518f, 0.16129031777381897f, 0.19354838132858276f, 0.22580644488334656f, 0.25806450843811035f, 0.29032257199287415f, 0.32258063554763794f, 0.35483869910240173f, 0.3870967626571655f, 0.4193548262119293f, 0.4516128897666931f, 0.4838709533214569f, 0.5161290168762207f, 0.5483871102333069f, 0.5806451439857483f, 0.6129032373428345f, 0.6451612710952759f, 0.6774193644523621f, 0.7096773982048035f, 0.7419354915618896f, 0.774193525314331f, 0.8064516186714172f, 0.8387096524238586f, 0.8709677457809448f, 0.9032257795333862f, 0.9354838728904724f, 0.9677419066429138f, 1.0 };
        this->Densityramp = MakeDefaultRampCurve32(DENSITYRAMP_RAWDATA);
        this->Cdrangeoverride.Set(0);
        this->Cdrange.Set(MakeVector2(0.0f, 1.0f));
        //AxFp32 CDRAMP_RAWDATA[32] = { {0.0f, 0.0f, 0.0}f, {0.032258063554763794f, 0.032258063554763794f, 0.032258063554763794}f, {0.06451612710952759f, 0.06451612710952759f, 0.06451612710952759}f, {0.09677419066429138f, 0.09677419066429138f, 0.09677419066429138}f, {0.12903225421905518f, 0.12903225421905518f, 0.12903225421905518}f, {0.16129031777381897f, 0.16129031777381897f, 0.16129031777381897}f, {0.19354838132858276f, 0.19354838132858276f, 0.19354838132858276}f, {0.22580644488334656f, 0.22580644488334656f, 0.22580644488334656}f, {0.25806450843811035f, 0.25806450843811035f, 0.25806450843811035}f, {0.29032257199287415f, 0.29032257199287415f, 0.29032257199287415}f, {0.32258063554763794f, 0.32258063554763794f, 0.32258063554763794}f, {0.35483869910240173f, 0.35483869910240173f, 0.35483869910240173}f, {0.3870967626571655f, 0.3870967626571655f, 0.3870967626571655}f, {0.4193548262119293f, 0.4193548262119293f, 0.4193548262119293}f, {0.4516128897666931f, 0.4516128897666931f, 0.4516128897666931}f, {0.4838709533214569f, 0.4838709533214569f, 0.4838709533214569}f, {0.5161290168762207f, 0.5161290168762207f, 0.5161290168762207}f, {0.5483871102333069f, 0.5483871102333069f, 0.5483871102333069}f, {0.5806451439857483f, 0.5806451439857483f, 0.5806451439857483}f, {0.6129032373428345f, 0.6129032373428345f, 0.6129032373428345}f, {0.6451612710952759f, 0.6451612710952759f, 0.6451612710952759}f, {0.6774193644523621f, 0.6774193644523621f, 0.6774193644523621}f, {0.7096773982048035f, 0.7096773982048035f, 0.7096773982048035}f, {0.7419354915618896f, 0.7419354915618896f, 0.7419354915618896}f, {0.774193525314331f, 0.774193525314331f, 0.774193525314331}f, {0.8064516186714172f, 0.8064516186714172f, 0.8064516186714172}f, {0.8387096524238586f, 0.8387096524238586f, 0.8387096524238586}f, {0.8709677457809448f, 0.8709677457809448f, 0.8709677457809448}f, {0.9032257795333862f, 0.9032257795333862f, 0.9032257795333862}f, {0.9354838728904724f, 0.9354838728904724f, 0.9354838728904724}f, {0.9677419066429138f, 0.9677419066429138f, 0.9677419066429138}f, {1.0f, 1.0f, 1.0} };
        //this->Cdramp = MakeDefaultRampCurve32(CDRAMP_RAWDATA);
        this->UseKernelFuse.Set(0);
        this->AlphaCommand.Set("");
        this->PressureIterations.Set(1);
        this->ExecuteCommand.Set("/AlphaCore.exe");

    }
};

#endif