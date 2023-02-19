#ifndef __AXOP_FIELDVECTORMAKER_SIM_PARAM_H___
#define __AXOP_FIELDVECTORMAKER_SIM_PARAM_H___

#include "AxMacro.h"
#include "AxMicroSolverBase.h"
#include "AxSimWorld.h"

FSA_CLASS class AxFieldVectorMakerSimParam
{
public:
    AxFieldVectorMakerSimParam();
    ~AxFieldVectorMakerSimParam();
    AxToggleParam Loadfromdisk;
AxIntParam AniType;
AxIntParam VectorDriveType;
AxFloatParam WindSpeed;
AxFloatParam PushScale;
AxStringParam NodePath;
AxStringParam VecFieldFilePath;
    
public:

    void FromJson(std::string jsonRawCode);
    void Init();
    void PrintData();
};

#endif