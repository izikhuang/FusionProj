#ifndef __AXOP_FIELDVORTICITYCONFINEMENT_H__
#define __AXOP_FIELDVORTICITYCONFINEMENT_H__

#include "AxOP_FieldVorticityConfinement.ProtocolData.h"

class AxOP_FieldVorticityConfinement :public AxMicroSolverBase
{
public:
    AxOP_FieldVorticityConfinement() {};
    ~AxOP_FieldVorticityConfinement() {};

    static AxMicroSolverBase* SolverConstructor()
    {
        return (AxMicroSolverBase*)new AxOP_FieldVorticityConfinement();
    }
    virtual void DeserilizationFromJson(std::string rawJSON)
    {
        simParam.FromJson(rawJSON);
    }
    
    //virtual std::string SerilizationToJson();
    virtual void BindGeometry(
        AxGeometry* geo0 = nullptr,
        AxGeometry* geo1 = nullptr,
        AxGeometry* geo2 = nullptr,
        AxGeometry* geo3 = nullptr,
        AxGeometry* geo4 = nullptr);
        
    virtual AxOpType GetOpType()
    {
        return AxOpType::kOpFromHDA;
    }
    
    AxFieldVorticityConfinementSimParam* GetSimParam()
    {
        return &simParam;
    } 
    virtual void PrintParamInfo()
    {
        simParam.PrintData();
    }
protected:
    virtual void OnUpdate(AxFp32 dt);
    virtual void OnInit();
    virtual void OnUpdateDevice(AxFp32 dt);
    virtual void OnInitDevice();

    //#BIND_GEO_DATA#
    
    AxFieldVorticityConfinementSimParam simParam;
};
#endif