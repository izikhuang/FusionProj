#ifndef __AXOP_FIELDCURLNOISE_H__
#define __AXOP_FIELDCURLNOISE_H__

#include "AxOP_FieldCurlNoise.ProtocolData.h"

class AxOP_FieldCurlNoise :public AxMicroSolverBase
{
public:
    AxOP_FieldCurlNoise() {};
    ~AxOP_FieldCurlNoise() {};

    static AxMicroSolverBase* SolverConstructor()
    {
        return (AxMicroSolverBase*)new AxOP_FieldCurlNoise();
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
    
    AxFieldCurlNoiseSimParam* GetSimParam()
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
    
    AxFieldCurlNoiseSimParam simParam;
    AxVecFieldF32 m_VelField;
    AxScalarFieldF32* m_MaskField;
    AxScalarFieldF32* m_ThresholdField;

};
#endif