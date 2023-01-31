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
    AxFloatParam RampMaskX1value;
AxFloatParam RampMaskX2value;
AxFloatParam RampMaskY2pos;
AxIntParam RampMaskZ2interp;
AxFloatParam RampMaskZ2value;
AxIntParam WindFieldType;
AxFloatParam WindIntensity;
AxToggleParam FieldRampZ;
AxToggleParam FieldRampX;
AxToggleParam FieldRampY;
AxFloatParam RampMaskX2pos;
AxIntParam RampMaskY1interp;
AxRampCurve32RAWData RampMaskY;
AxRampCurve32RAWData RampMaskX;
AxRampCurve32RAWData RampMaskZ;
AxFloatParam RampMaskY2value;
AxFloatParam RampMaskZ1value;
AxFloatParam RampMaskY1value;
AxIntParam RampMaskX1interp;
AxFloatParam RampMaskX1pos;
AxVector3FParam WindDirection;
AxIntParam RampMaskZ1interp;
AxIntParam RampMaskY2interp;
AxFloatParam RampMaskZ2pos;
AxFloatParam RampMaskY1pos;
AxFloatParam WindSpeed;
AxIntParam RampMaskX2interp;
AxFloatParam RampMaskZ1pos;
    
public:

    void FromJson(std::string jsonRawCode);
    void Init();
    void PrintData();
};

#endif