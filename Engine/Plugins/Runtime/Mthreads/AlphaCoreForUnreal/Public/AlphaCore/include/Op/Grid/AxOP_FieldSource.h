#ifndef __AXOP_FIELDSOUCING_H__
#define __AXOP_FIELDSOUCING_H__

#include "AxOP_FieldSource.ProtocolData.h"
#include "FluidUtility/AxFluidUtility.DataType.h"
#include "include/Atmosphere/AxAtmosphere.h"

 enum AxFieldSourceGeoType
 {
     kBox,
     kPlane,
 };

class AxOP_FieldSource :public AxMicroSolverBase
{
public:
    AxOP_FieldSource();
    ~AxOP_FieldSource() {};

    static AxMicroSolverBase* SolverConstructor()
    {
        return (AxMicroSolverBase*)new AxOP_FieldSource();
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
    
    AxFieldSourceSimParam* GetSimParam()
    {
        return &simParam;
    } 
    virtual void PrintParamInfo()
    {
        simParam.PrintData();
    }

    void SetEulerRotate(AxVector3 euler);

protected:
    virtual void OnUpdate(AxFp32 dt);
    virtual void OnInit();
    virtual void OnUpdateDevice(AxFp32 dt);
    virtual void OnInitDevice();

    void preLoad(bool loadToDevice);
    void sourceFieldDevice(AxFp32 dt);
    void sourceField(AxFp32 dt);

    void sourceProcEmitterDeivce(AxFp32 dt);
    void sourceProcEmitter(AxFp32 dt);

    void setEmitterGeometry(AxGeometry* emitterGeo, bool readyDeviceData = false);
    AxGeometry* getEmitterGeometry();

    void sourceEmitterStorm(AxFp32 dt);
    void sourceEmitterStormDevice(AxFp32 dt);

    void transferNoiseTo2DField(AxCurlNoiseParam noiseParam);
    void transferNoiseTo2DFieldDevice(AxCurlNoiseParam noiseParam);

    //#BIND_GEO_DATA#
    AxFieldSourceSimParam simParam;

    //as handle can access multi field
    AxScalarFieldF32* m_DivergenceField;
    AxScalarFieldF32* m_DensityField;
    AxScalarFieldF32* m_FuelField;
    AxScalarFieldF32* m_TemperatureField;
    AxVecFieldF32 m_VelField;

    AxGeometry* m_simGeo;
    AxGeometry* m_EmitterGeo;
    AxScalarFieldF32* m_DivergenceEmitter;
    AxScalarFieldF32* m_DensityEmitter;
    AxScalarFieldF32* m_FuelEmitter;
    AxScalarFieldF32* m_TemperatureEmitter;
    AxVecFieldF32 m_VelEmitter;
    

    AxVector3 m_EulerRotate;
    AxMatrix3x3 m_RotateMatrix;
    AxMatrix3x3 m_InvRotateMatrix;


    AxFieldSourceGeoType m_SouorceGeoType;

    AxScalarFieldF32* m_noiseField = new AxScalarFieldF32();

    AxAABB box;



    //void FieldProcSourcing(AxFp32 dt);

};
#endif