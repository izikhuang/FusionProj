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
    AxVector2FParam Guiderange;
AxFloatParam Timescale;
AxFloatParam Confinementscale;
AxRampCurve32RAWData Control_field_ramp;
AxStringParam Curl;
AxIntParam Guidevismode;
AxFloatParam Control_field_ramp1pos;
AxFloatParam Control_field_ramp1value;
AxToggleParam Remap_control_field;
AxIntParam Guidevistype;
AxToggleParam Opencl;
AxToggleParam Cleartemp;
AxFloatParam Guideplaneval;
AxStringParam Control_field;
AxIntParam Guideplane;
AxFloatParam Control_min;
AxFloatParam Guidestreamerminspeed;
AxStringParam Vel;
AxToggleParam Visualize_confinement;
AxToggleParam Use_control_field;
AxFloatParam Guidestreamerlen;
AxFloatParam Guidevisscale;
AxStringParam Confinement;
AxStringParam Curlmag;
AxIntParam Control_field_ramp1interp;
AxFloatParam Control_field_ramp2pos;
AxFloatParam Control_field_ramp2value;
AxIntParam Control_field_ramp2interp;
AxStringParam Vortexdir;
AxFloatParam Control_influence;
AxFloatParam Control_max;
    
public:

    void FromJson(std::string jsonRawCode);
    void Init();
    void PrintData();
};

#endif