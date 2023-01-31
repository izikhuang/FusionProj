#ifndef __AXOP_FIELDULTRATURB_SIM_PARAM_H___
#define __AXOP_FIELDULTRATURB_SIM_PARAM_H___

#include "AxMacro.h"
#include "AxMicroSolverBase.h"
#include "AxSimWorld.h"

FSA_CLASS class AxFieldUltraTurbSimParam
{
public:
    AxFieldUltraTurbSimParam();
    ~AxFieldUltraTurbSimParam();
    AxFloatParam Scale;
AxIntParam Mode;
AxFloatParam Refscale;
AxFloatParam SpatialFrequency;
AxFloatParam TimeFrequency;
AxFloatParam RoughnessIncrease;
AxFloatParam Roughness;
AxIntParam TurbIterations;
AxToggleParam UseMaskField;
AxStringParam MaskFieldName;
AxToggleParam EnableRemapMaskfield;
AxIntParam Plane;
AxFloatParam Planeoffset;
AxRampCurve32RAWData MaskRamp;
AxToggleParam CutoffBelow;
AxVector2FParam Threshrange;
AxStringParam ThresholdField;
AxToggleParam Rotateonly;
AxFloatParam MaskRamp1pos;
AxFloatParam MaskRamp1value;
AxIntParam MaskRamp1interp;
AxFloatParam MaskRamp2pos;
AxFloatParam MaskRamp2value;
AxIntParam MaskRamp2interp;
    
public:

    void FromJson(std::string jsonRawCode);
    void Init();
    void PrintData();
};

#endif