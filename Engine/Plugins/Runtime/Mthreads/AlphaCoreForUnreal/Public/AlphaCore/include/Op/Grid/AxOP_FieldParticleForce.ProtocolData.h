#ifndef __AXOP_FIELDPARTICLEFORCE_SIM_PARAM_H___
#define __AXOP_FIELDPARTICLEFORCE_SIM_PARAM_H___

#include "AxMacro.h"
#include "AxMicroSolverBase.h"
#include "AxSimWorld.h"

FSA_CLASS class AxFieldParticleForceSimParam
{
public:
    AxFieldParticleForceSimParam();
    ~AxFieldParticleForceSimParam();
    AxVector3FParam Force;
AxFloatParam ForceScale;
AxVector3FParam Pivot;
AxVector3FParam Size;
AxToggleParam UseNoise;
AxFloatParam Amplitude;
AxFloatParam CurlSwirlSize;
AxFloatParam CurlNoiseIntensity;
AxFloatParam CurlTimeFrequency;
AxFloatParam CurlNoiseSeed;
AxIntParam CurlNoiseTurbulence;
AxStringParam MaskFieldName;
AxRampCurve32RAWData MaskRamp;
    
public:

    void FromJson(std::string jsonRawCode);
    void Init();
    void PrintData();
};

#endif