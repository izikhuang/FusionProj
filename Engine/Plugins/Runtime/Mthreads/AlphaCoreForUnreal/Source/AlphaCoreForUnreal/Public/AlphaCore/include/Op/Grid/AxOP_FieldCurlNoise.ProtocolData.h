#ifndef __AXOP_FIELDCURLNOISE_SIM_PARAM_H___
#define __AXOP_FIELDCURLNOISE_SIM_PARAM_H___

#include "AxMacro.h"
#include "AxMicroSolverBase.h"
#include "AxSimWorld.h"

FSA_CLASS class AxFieldCurlNoiseSimParam
{
public:
    AxFieldCurlNoiseSimParam();
    ~AxFieldCurlNoiseSimParam();
    AxFloatParam CurlTimeFrequency;
AxFloatParam MaskWeight;
AxIntParam CurlNoiseTurbulence;
AxFloatParam CurlNoiseIntensity;
AxFloatParam CurlNoiseSeed;
AxRampCurve32RAWData MaskRamp;
AxToggleParam UseMaskField;
AxStringParam ThresholdFieldName;
AxFloatParam CurlNoiseClampBelow;
AxFloatParam CurlNoiseScale;
AxFloatParam CurlSwirlSize;
AxStringParam MaskFieldName;
AxToggleParam RemapMaskfield;
    
public:

    void FromJson(std::string jsonRawCode);
    void Init();
    void PrintData();
};

#endif