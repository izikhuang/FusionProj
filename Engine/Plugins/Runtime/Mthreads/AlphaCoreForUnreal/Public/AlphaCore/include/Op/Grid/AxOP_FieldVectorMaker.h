#ifndef __AXOP_FIELDVECTORMAKER_H__
#define __AXOP_FIELDVECTORMAKER_H__

#include "AxOP_FieldVectorMaker.ProtocolData.h"

class AxOP_FieldVectorMaker :public AxMicroSolverBase
{
public:
    AxOP_FieldVectorMaker();
    ~AxOP_FieldVectorMaker();

    static AxMicroSolverBase* SolverConstructor()
    {
        return (AxMicroSolverBase*)new AxOP_FieldVectorMaker();
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
    
    AxFieldVectorMakerSimParam* GetSimParam()
    {
        return &simParam;
    } 
    virtual void PrintParamInfo()
    {
        simParam.PrintData();
    }

    void setEmitterGeometry(AxGeometry* emitterGeo, bool readyDeviceData = false);
    AxGeometry* getEmitterGeometry();

protected:
    virtual void OnUpdate(AxFp32 dt);
    virtual void OnInit();
    virtual void OnUpdateDevice(AxFp32 dt);
    virtual void OnInitDevice();

    //#BIND_GEO_DATA#
    void preLoad(bool loadToDevice);


    AxFieldVectorMakerSimParam simParam;
    AxVecFieldF32 m_VelFieldSrc;
    AxVecFieldF32 m_VelFieldDst;
    AxGeometry* m_EmitterGeo;


};
#endif