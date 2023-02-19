#ifndef __AXOP_FIELDSOURCE_SIM_PARAM_H___
#define __AXOP_FIELDSOURCE_SIM_PARAM_H___

#include "AxMacro.h"
#include "AxMicroSolverBase.h"
#include "AxSimWorld.h"

FSA_CLASS class AxFieldSourceSimParam
{
public:
    AxFieldSourceSimParam();
    ~AxFieldSourceSimParam();
    AxIntParam SourcingType;
AxIntParam SourcingEmitterType;
AxToggleParam EmitterForAtmosphere;
AxToggleParam ProjectToTerrain;
AxToggleParam Loadfromdisk;
AxVector3FParam Center;
AxVector3FParam Rotate;
AxVector2FParam RectSize;
AxIntParam Prolong;
AxToggleParam EnableEmitDensity;
AxStringParam SourceDensityFieldName;
AxFloatParam SourceDensityScale;
AxFloatParam SourceDensityInit;
AxToggleParam EnableEmitFuel;
AxFloatParam SourceFuelScale;
AxStringParam SourceFuelFieldName;
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
AxVector2FParam Frequency;
AxFloatParam NoiseAmp;
AxFloatParam NoiseSize;
AxIntParam OffsetSpeed;
AxVector4FParam Offset;
AxRampCurve32RAWData Newparameter;
//AxFloatParam TemLapseRateLow;
//AxFloatParam TemLapseRateHigh;
//AxFloatParam TemInversionHeight;
//AxFloatParam HeatEmitterAmp;
AxFloatParam HeatNoiseScale;
AxFloatParam RelHumidityGround;
AxFloatParam DensityNoiseScale;
AxStringParam EmitterFilePath;
AxFloatParam Newparameter1pos;
AxFloatParam Newparameter1value;
AxIntParam Newparameter1interp;
AxFloatParam Newparameter2pos;
AxFloatParam Newparameter2value;
AxIntParam Newparameter2interp;
//AxFloatParam HeatEmitterAmp;
//AxFloatParam HeatNoiseScale;
//AxFloatParam RelHumidityGround;
//AxFloatParam DensityNoiseScale;


    
public:

    void FromJson(std::string jsonRawCode);
    void Init();
    void PrintData();
};

#endif