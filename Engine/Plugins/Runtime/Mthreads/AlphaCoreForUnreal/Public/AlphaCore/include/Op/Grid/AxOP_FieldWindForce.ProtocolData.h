#ifndef __AXOP_FIELDWINDFORCE_SIM_PARAM_H___
#define __AXOP_FIELDWINDFORCE_SIM_PARAM_H___

#include "AxMacro.h"
#include "AxMicroSolverBase.h"
#include "AxSimWorld.h"

FSA_CLASS class AxFieldWindForceSimParam
{
public:
    AxFieldWindForceSimParam();
    ~AxFieldWindForceSimParam();
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
    
public:

    void FromJson(std::string jsonRawCode);
    void Init();
    void PrintData();
};

#endif