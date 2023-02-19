#ifndef __AXOP_FIELDULTRATURB_H__
#define __AXOP_FIELDULTRATURB_H__

#include "AxOP_FieldUltraTurb.ProtocolData.h"

class AxOP_FieldUltraTurb :public AxMicroSolverBase
{
public:
    AxOP_FieldUltraTurb();
    ~AxOP_FieldUltraTurb() {};

    static AxMicroSolverBase* SolverConstructor()
    {
        return (AxMicroSolverBase*)new AxOP_FieldUltraTurb();
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
    
    AxFieldUltraTurbSimParam* GetSimParam()
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

    AxFieldUltraTurbSimParam simParam;
    AxVecFieldF32 m_VelField;
    AxScalarFieldF32* m_MaskFieldScalar;
    AxVecFieldF32 m_VecMagMaskField;

    AxScalarFieldF32* m_ThresholdField;
    AxGeometry* m_OwnGeometry;

};
#endif