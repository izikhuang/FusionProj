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
AxVector3FParam Center;
AxVector3FParam Rotate;
AxVector2FParam RectSize;
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
AxStringParam EmitterFilePath;
    
public:

    void FromJson(std::string jsonRawCode);
    void Init();
    void PrintData();
};

#endif