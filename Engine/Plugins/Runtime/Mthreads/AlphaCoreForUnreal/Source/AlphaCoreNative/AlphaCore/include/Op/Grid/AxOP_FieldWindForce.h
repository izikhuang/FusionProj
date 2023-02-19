#ifndef __AXOP_FIELDWINDFORCE_H__
#define __AXOP_FIELDWINDFORCE_H__

#include "AxOP_FieldWindForce.ProtocolData.h"

class AxOP_FieldWindForce :public AxMicroSolverBase
{
public:
    AxOP_FieldWindForce();
    ~AxOP_FieldWindForce() {};

    static AxMicroSolverBase* SolverConstructor()
    {
        return (AxMicroSolverBase*)new AxOP_FieldWindForce();
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
    
    AxFieldWindForceSimParam* GetSimParam()
    {
        return &simParam;
    } 
    virtual void PrintParamInfo()
    {
        simParam.PrintData();
    }

    //#BIND_GEO_DATA#
    AxFieldWindForceSimParam simParam;
protected:
    virtual void OnUpdate(AxFp32 dt);
    virtual void OnInit();
    virtual void OnUpdateDevice(AxFp32 dt);
    virtual void OnInitDevice();


    AxVecFieldF32 m_VelField;

};
#endif