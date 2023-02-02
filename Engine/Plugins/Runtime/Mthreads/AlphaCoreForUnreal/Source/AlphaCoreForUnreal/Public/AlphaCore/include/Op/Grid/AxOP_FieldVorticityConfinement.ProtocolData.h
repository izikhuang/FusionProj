#ifndef __AXOP_FIELDVORTICITYCONFINEMENT_SIM_PARAM_H___
#define __AXOP_FIELDVORTICITYCONFINEMENT_SIM_PARAM_H___

#include "AxMacro.h"
#include "AxMicroSolverBase.h"
#include "AxSimWorld.h"

FSA_CLASS class AxFieldVorticityConfinementSimParam
{
public:
    AxFieldVorticityConfinementSimParam();
    ~AxFieldVorticityConfinementSimParam();
    AxFloatParam Confinementscale;
AxToggleParam UseMaskField;
AxStringParam MaskFieldName;
AxToggleParam RemapMaskfield;
AxRampCurve32RAWData MaskRamp;
AxIntParam Plane;
AxFloatParam Planeoffset;
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