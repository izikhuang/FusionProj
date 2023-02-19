#ifndef __AXOP_FIELDPARTICLEFORCE_H__
#define __AXOP_FIELDPARTICLEFORCE_H__

#include "AxOP_FieldParticleForce.ProtocolData.h"

class AxOP_FieldParticleForce :public AxMicroSolverBase
{
public:
    AxOP_FieldParticleForce() {};
    ~AxOP_FieldParticleForce() {};

    static AxMicroSolverBase* SolverConstructor()
    {
        return (AxMicroSolverBase*)new AxOP_FieldParticleForce();
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
    
    AxFieldParticleForceSimParam* GetSimParam()
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
    
    AxFieldParticleForceSimParam simParam;
    AxBufferV3* m_posBuffer;
    AxBufferV3* m_accelBuffer;
    AxBufferF* m_maskBuffer;
    std::string m_maskName;
};
#endif