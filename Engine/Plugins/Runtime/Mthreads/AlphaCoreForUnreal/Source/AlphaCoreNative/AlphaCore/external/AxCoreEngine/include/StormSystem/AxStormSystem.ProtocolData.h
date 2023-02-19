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
AxFloatParam WindSpeed;
AxFloatParam WindIntensity;
AxVector3FParam WindDirection;
AxIntParam WindFieldType;
AxToggleParam FieldRampX;
AxRampCurve32RAWData RampMaskX;
AxToggleParam FieldRampY;
AxRampCurve32RAWData RampMaskY;
AxToggleParam FieldRampZ;
AxRampCurve32RAWData RampMaskZ;
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
//AxFloatParam TemLapseRateLow;
//AxFloatParam TemLapseRateHigh;
//AxFloatParam TemInversionHeight;
AxFloatParam AuthenticDomainHeight;
AxFloatParam DiffusionCoeff;
AxFloatParam BuoyScale;
AxFloatParam HeatEmitterAmp;
AxFloatParam CloudPosOffset;


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
AlphaCore::JsonHelper::AxFloatParamDeserilization(WindSpeed, paramMap["windSpeed"]);
AlphaCore::JsonHelper::AxFloatParamDeserilization(WindIntensity, paramMap["windIntensity"]);
AlphaCore::JsonHelper::AxVector3FParamDeserilization(WindDirection, paramMap["windDirection"]);
AlphaCore::JsonHelper::AxIntParamDeserilization(WindFieldType, paramMap["windFieldType"]);
AlphaCore::JsonHelper::AxToggleParamDeserilization(FieldRampX, paramMap["fieldRampX"]);
AlphaCore::JsonHelper::AxRampCurve32RAWDataDeserilization(RampMaskX, paramMap["rampMaskX"]);
AlphaCore::JsonHelper::AxToggleParamDeserilization(FieldRampY, paramMap["fieldRampY"]);
AlphaCore::JsonHelper::AxRampCurve32RAWDataDeserilization(RampMaskY, paramMap["rampMaskY"]);
AlphaCore::JsonHelper::AxToggleParamDeserilization(FieldRampZ, paramMap["fieldRampZ"]);
AlphaCore::JsonHelper::AxRampCurve32RAWDataDeserilization(RampMaskZ, paramMap["rampMaskZ"]);
AlphaCore::JsonHelper::AxRampCurve32RAWDataDeserilization(Densityramp, paramMap["densityramp"]);
AlphaCore::JsonHelper::AxToggleParamDeserilization(Cdrangeoverride, paramMap["cdrangeoverride"]);
AlphaCore::JsonHelper::AxVector2FParamDeserilization(Cdrange, paramMap["cdrange"]);
//AlphaCore::JsonHelper::AxRampCurve32RAWDataDeserilization(Cdramp, paramMap["cdramp"]);
AlphaCore::JsonHelper::AxToggleParamDeserilization(UseKernelFuse, paramMap["useKernelFuse"]);
AlphaCore::JsonHelper::AxStringParamDeserilization(AlphaCommand, paramMap["alphaCommand"]);
AlphaCore::JsonHelper::AxIntParamDeserilization(PressureIterations, paramMap["pressureIterations"]);
AlphaCore::JsonHelper::AxStringParamDeserilization(WorkSpace, paramMap["workSpace"]);
//AlphaCore::JsonHelper::AxStringParamDeserilization(EmitterCacheReadFilePath, paramMap["emitterCacheReadFilePath"]);
AlphaCore::JsonHelper::AxStringParamDeserilization(CacheOutFilePath, paramMap["cacheOutFilePath"]);
AlphaCore::JsonHelper::AxStringParamDeserilization(StaticHeightFieldFielPath, paramMap["staticHeightFieldFielPath"]);
AlphaCore::JsonHelper::AxStringParamDeserilization(ExecuteCommand, paramMap["executeCommand"]);
//AlphaCore::JsonHelper::AxFloatParamDeserilization(TemLapseRateLow, paramMap["temLapseRateLow"]);
//AlphaCore::JsonHelper::AxFloatParamDeserilization(TemLapseRateHigh, paramMap["temLapseRateHigh"]);
//AlphaCore::JsonHelper::AxFloatParamDeserilization(TemInversionHeight, paramMap["temInversionHeight"]);
AlphaCore::JsonHelper::AxFloatParamDeserilization(AuthenticDomainHeight, paramMap["authenticDomainHeight"]);
AlphaCore::JsonHelper::AxFloatParamDeserilization(DiffusionCoeff, paramMap["diffusionCoeff"]);
AlphaCore::JsonHelper::AxFloatParamDeserilization(BuoyScale, paramMap["buoyScale"]);
AlphaCore::JsonHelper::AxFloatParamDeserilization(HeatEmitterAmp, paramMap["heatEmitterAmp"]);
AlphaCore::JsonHelper::AxFloatParamDeserilization(CloudPosOffset, paramMap["cloudPosOffset"]);
//AlphaCore::JsonHelper::AxFloatParamDeserilization(HeatNoiseScale, paramMap["heatNoiseScale"]);
//AlphaCore::JsonHelper::AxFloatParamDeserilization(RelHumidityGround, paramMap["relHumidityGround"]);
//AlphaCore::JsonHelper::AxFloatParamDeserilization(DensityNoiseScale, paramMap["densityNoiseScale"]);


        AX_INFO("AxStormSysSimParam : Read sim parameter form json OVER !!! ");
    }
    
    void Init()
    {
        this->TaskFrameRange.Set(MakeVector2(1,240));
this->FieldBuildType.Set(0);
this->ComputeArch.Set(0);
this->VoxelSize.Set(0.1f);
this->Pivot.Set(MakeVector3(0.0f,0.0f,0.0f));
this->Size.Set(MakeVector3(10.0f,10.0f,10.0f));
this->Rotate.Set(MakeVector3(0.0f,0.0f,0.0f));
this->TimeScale.Set(1.0f);
this->FPS.Set(24);
this->Substeps.Set(1);
this->SolverMehtodCore.Set(0);
this->GSIterations.Set(10.0f);
this->TemperatureDiffusion.Set(0.5f);
this->CoolingRate.Set(0.75f);
this->BuoyancyScale.Set(10.0f);
this->BuoyancyDirection.Set(MakeVector3(0.0f,1.0f,0.0f));
this->ReflectionType.Set(0);
this->ReflectionAmount.Set(0.95f);
this->WindSpeed.Set(1.0f);
this->WindIntensity.Set(0.5f);
this->WindDirection.Set(MakeVector3(0.0f,0.0f,0.0f));
this->WindFieldType.Set(0);
this->FieldRampX.Set(0);
AxFp32 RAMPMASKX_RAWDATA[32] = {0.0f, 0.017656339332461357f, 0.03823302313685417f, 0.06152865290641785f, 0.08734181523323059f, 0.11547110974788666f, 0.1457151472568512f, 0.17787250876426697f, 0.21174179017543793f, 0.24712160229682922f, 0.283810555934906f, 0.32160717248916626f, 0.3603101670742035f, 0.3997180163860321f, 0.4396294355392456f, 0.4798429012298584f, 0.5201570987701416f, 0.5603705644607544f, 0.6002819538116455f, 0.6396898627281189f, 0.678392767906189f, 0.7161895036697388f, 0.7528783679008484f, 0.7882581949234009f, 0.8221275210380554f, 0.8542848825454712f, 0.8845288753509521f, 0.912658154964447f, 0.9384713172912598f, 0.9617669582366943f, 0.9823436737060547f, 1.0};
this->RampMaskX = MakeDefaultRampCurve32(RAMPMASKX_RAWDATA);
this->FieldRampY.Set(0);
AxFp32 RAMPMASKY_RAWDATA[32] = {0.0f, 0.017656339332461357f, 0.03823302313685417f, 0.06152865290641785f, 0.08734181523323059f, 0.11547110974788666f, 0.1457151472568512f, 0.17787250876426697f, 0.21174179017543793f, 0.24712160229682922f, 0.283810555934906f, 0.32160717248916626f, 0.3603101670742035f, 0.3997180163860321f, 0.4396294355392456f, 0.4798429012298584f, 0.5201570987701416f, 0.5603705644607544f, 0.6002819538116455f, 0.6396898627281189f, 0.678392767906189f, 0.7161895036697388f, 0.7528783679008484f, 0.7882581949234009f, 0.8221275210380554f, 0.8542848825454712f, 0.8845288753509521f, 0.912658154964447f, 0.9384713172912598f, 0.9617669582366943f, 0.9823436737060547f, 1.0};
this->RampMaskY = MakeDefaultRampCurve32(RAMPMASKY_RAWDATA);
this->FieldRampZ.Set(0);
AxFp32 RAMPMASKZ_RAWDATA[32] = {0.0f, 0.017656339332461357f, 0.03823302313685417f, 0.06152865290641785f, 0.08734181523323059f, 0.11547110974788666f, 0.1457151472568512f, 0.17787250876426697f, 0.21174179017543793f, 0.24712160229682922f, 0.283810555934906f, 0.32160717248916626f, 0.3603101670742035f, 0.3997180163860321f, 0.4396294355392456f, 0.4798429012298584f, 0.5201570987701416f, 0.5603705644607544f, 0.6002819538116455f, 0.6396898627281189f, 0.678392767906189f, 0.7161895036697388f, 0.7528783679008484f, 0.7882581949234009f, 0.8221275210380554f, 0.8542848825454712f, 0.8845288753509521f, 0.912658154964447f, 0.9384713172912598f, 0.9617669582366943f, 0.9823436737060547f, 1.0};
this->RampMaskZ = MakeDefaultRampCurve32(RAMPMASKZ_RAWDATA);
AxFp32 DENSITYRAMP_RAWDATA[32] = {0.0f, 0.032258063554763794f, 0.06451612710952759f, 0.09677419066429138f, 0.12903225421905518f, 0.16129031777381897f, 0.19354838132858276f, 0.22580644488334656f, 0.25806450843811035f, 0.29032257199287415f, 0.32258063554763794f, 0.35483869910240173f, 0.3870967626571655f, 0.4193548262119293f, 0.4516128897666931f, 0.4838709533214569f, 0.5161290168762207f, 0.5483871102333069f, 0.5806451439857483f, 0.6129032373428345f, 0.6451612710952759f, 0.6774193644523621f, 0.7096773982048035f, 0.7419354915618896f, 0.774193525314331f, 0.8064516186714172f, 0.8387096524238586f, 0.8709677457809448f, 0.9032257795333862f, 0.9354838728904724f, 0.9677419066429138f, 1.0};
this->Densityramp = MakeDefaultRampCurve32(DENSITYRAMP_RAWDATA);
this->Cdrangeoverride.Set(0);
this->Cdrange.Set(MakeVector2(0.0f,1.0f));
//AxFp32 CDRAMP_RAWDATA[32] = {{0.0f, 0.0f, 0.0}f, {0.032258063554763794f, 0.032258063554763794f, 0.032258063554763794}f, {0.06451612710952759f, 0.06451612710952759f, 0.06451612710952759}f, {0.09677419066429138f, 0.09677419066429138f, 0.09677419066429138}f, {0.12903225421905518f, 0.12903225421905518f, 0.12903225421905518}f, {0.16129031777381897f, 0.16129031777381897f, 0.16129031777381897}f, {0.19354838132858276f, 0.19354838132858276f, 0.19354838132858276}f, {0.22580644488334656f, 0.22580644488334656f, 0.22580644488334656}f, {0.25806450843811035f, 0.25806450843811035f, 0.25806450843811035}f, {0.29032257199287415f, 0.29032257199287415f, 0.29032257199287415}f, {0.32258063554763794f, 0.32258063554763794f, 0.32258063554763794}f, {0.35483869910240173f, 0.35483869910240173f, 0.35483869910240173}f, {0.3870967626571655f, 0.3870967626571655f, 0.3870967626571655}f, {0.4193548262119293f, 0.4193548262119293f, 0.4193548262119293}f, {0.4516128897666931f, 0.4516128897666931f, 0.4516128897666931}f, {0.4838709533214569f, 0.4838709533214569f, 0.4838709533214569}f, {0.5161290168762207f, 0.5161290168762207f, 0.5161290168762207}f, {0.5483871102333069f, 0.5483871102333069f, 0.5483871102333069}f, {0.5806451439857483f, 0.5806451439857483f, 0.5806451439857483}f, {0.6129032373428345f, 0.6129032373428345f, 0.6129032373428345}f, {0.6451612710952759f, 0.6451612710952759f, 0.6451612710952759}f, {0.6774193644523621f, 0.6774193644523621f, 0.6774193644523621}f, {0.7096773982048035f, 0.7096773982048035f, 0.7096773982048035}f, {0.7419354915618896f, 0.7419354915618896f, 0.7419354915618896}f, {0.774193525314331f, 0.774193525314331f, 0.774193525314331}f, {0.8064516186714172f, 0.8064516186714172f, 0.8064516186714172}f, {0.8387096524238586f, 0.8387096524238586f, 0.8387096524238586}f, {0.8709677457809448f, 0.8709677457809448f, 0.8709677457809448}f, {0.9032257795333862f, 0.9032257795333862f, 0.9032257795333862}f, {0.9354838728904724f, 0.9354838728904724f, 0.9354838728904724}f, {0.9677419066429138f, 0.9677419066429138f, 0.9677419066429138}f, {1.0f, 1.0f, 1.0}};
//this->Cdramp = MakeDefaultRampCurve32(CDRAMP_RAWDATA);
this->UseKernelFuse.Set(0);
this->AlphaCommand.Set("");
this->PressureIterations.Set(1);
this->ExecuteCommand.Set("/AlphaCore.exe");
//this->TemLapseRateLow.Set(-0.0065f);
//this->TemLapseRateHigh.Set(0.0065f);
//this->TemInversionHeight.Set(8000.0f);
this->AuthenticDomainHeight.Set(5000.0f);
this->DiffusionCoeff.Set(0.01f);
this->BuoyScale.Set(1.f);
this->HeatEmitterAmp.Set(1.5f);
this->CloudPosOffset.Set(0.f);
//this->HeatNoiseScale.Set(1.0f);
//this->RelHumidityGround.Set(0.7f);
//this->DensityNoiseScale.Set(0.1);

AxFp32 RAMPMASKX2_RAWDATA[32] = {0.0f, 0.017656339332461357f, 0.03823302313685417f, 0.06152865290641785f, 0.08734181523323059f, 0.11547110974788666f, 0.1457151472568512f, 0.17787250876426697f, 0.21174179017543793f, 0.24712160229682922f, 0.283810555934906f, 0.32160717248916626f, 0.3603101670742035f, 0.3997180163860321f, 0.4396294355392456f, 0.4798429012298584f, 0.5201570987701416f, 0.5603705644607544f, 0.6002819538116455f, 0.6396898627281189f, 0.678392767906189f, 0.7161895036697388f, 0.7528783679008484f, 0.7882581949234009f, 0.8221275210380554f, 0.8542848825454712f, 0.8845288753509521f, 0.912658154964447f, 0.9384713172912598f, 0.9617669582366943f, 0.9823436737060547f, 1.0};

AxFp32 RAMPMASKY2_RAWDATA[32] = {0.0f, 0.017656339332461357f, 0.03823302313685417f, 0.06152865290641785f, 0.08734181523323059f, 0.11547110974788666f, 0.1457151472568512f, 0.17787250876426697f, 0.21174179017543793f, 0.24712160229682922f, 0.283810555934906f, 0.32160717248916626f, 0.3603101670742035f, 0.3997180163860321f, 0.4396294355392456f, 0.4798429012298584f, 0.5201570987701416f, 0.5603705644607544f, 0.6002819538116455f, 0.6396898627281189f, 0.678392767906189f, 0.7161895036697388f, 0.7528783679008484f, 0.7882581949234009f, 0.8221275210380554f, 0.8542848825454712f, 0.8845288753509521f, 0.912658154964447f, 0.9384713172912598f, 0.9617669582366943f, 0.9823436737060547f, 1.0};

AxFp32 RAMPMASKZ2_RAWDATA[32] = {0.0f, 0.017656339332461357f, 0.03823302313685417f, 0.06152865290641785f, 0.08734181523323059f, 0.11547110974788666f, 0.1457151472568512f, 0.17787250876426697f, 0.21174179017543793f, 0.24712160229682922f, 0.283810555934906f, 0.32160717248916626f, 0.3603101670742035f, 0.3997180163860321f, 0.4396294355392456f, 0.4798429012298584f, 0.5201570987701416f, 0.5603705644607544f, 0.6002819538116455f, 0.6396898627281189f, 0.678392767906189f, 0.7161895036697388f, 0.7528783679008484f, 0.7882581949234009f, 0.8221275210380554f, 0.8542848825454712f, 0.8845288753509521f, 0.912658154964447f, 0.9384713172912598f, 0.9617669582366943f, 0.9823436737060547f, 1.0};


    }
};

#endif