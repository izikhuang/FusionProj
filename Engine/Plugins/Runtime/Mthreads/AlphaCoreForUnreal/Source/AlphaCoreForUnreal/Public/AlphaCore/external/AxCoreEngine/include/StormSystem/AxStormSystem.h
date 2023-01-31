#ifndef __AXSTORMSYSTEM_H__
#define __AXSTORMSYSTEM_H__

#include "AxSimObject.h"
#include "GridDense/AxFieldBase3D.h"
#include "GridDense/AxFieldBase2D.h"

#include "AxStormSystem.ProtocolData.h"
#include "Math/AxConjugateGradient.h"
#include "MicroSolver/AxMicroSolverBase.h"
#include "AxSimWorld.h"

FSA_CLASS class AxStormSysSimData : AxISimData
{
public:
    AxStormSysSimData()
    {
        this->m_StormSysFieldsGeometry = nullptr;
        this->Density = nullptr;
        this->Vel = new AxVecFieldF32();
        this->Vel2 = new AxVecFieldF32();
        this->CurlField = new AxVecFieldF32();
        this->Temprature = nullptr;
        this->Pressure2 = nullptr;
        this->AdvectTemp = nullptr;
        this->Divergence = nullptr;
        this->Pressure = nullptr;
        this->VelDiv = nullptr;
        this->AdvectTmp2 = nullptr;
        this->HeightField = nullptr;
        this->CollisionMaskField = nullptr;
     }

    ~AxStormSysSimData()
    {
        if (m_StormSysFieldsGeometry != nullptr)
        {
            m_StormSysFieldsGeometry->ClearAndDestory();
            m_StormSysFieldsGeometry = nullptr;
        }
    }

    AxScalarFieldF32* Density;
    AxVecFieldF32* Vel;
    AxVecFieldF32* Vel2;
    AxVecFieldF32* CurlField;
    AxScalarFieldF32* Temprature;
    AxScalarFieldF32* Pressure2;
    AxScalarFieldF32* AdvectTemp;
    AxScalarFieldF32* Divergence;
    AxScalarFieldF32* Pressure;
    AxScalarFieldF32* VelDiv;
    AxScalarFieldF32* AdvectTmp2;
    AxScalarFieldI8* CollisionMaskField;
    AxScalarFieldF32* HeightField;

    struct RAWData
    {
        ///###FSA_SIMDATA_RAW_DATA
    };

    virtual void BindProperties(
        AxGeometry* geo0 = nullptr,
        AxGeometry* geo1 = nullptr,
        AxGeometry* geo2 = nullptr,
        AxGeometry* geo3 = nullptr,
        AxGeometry* geo4 = nullptr)
    {
        if (geo0 == nullptr)
            return;
        this->m_StormSysFieldsGeometry = geo0;

        if (geo1 == nullptr)
            return;
        this->HeightField = geo1->FindFieldByName<AxFp32>("height");
    }

    virtual void LoadToDevice()
    {
        if (Density != nullptr && !Density->HasDeviceData())
            Density->DeviceMalloc();
        if (Temprature != nullptr && !Temprature->HasDeviceData())
            Temprature->DeviceMalloc();
        // if (Pressure2 != nullptr && !Pressure2->HasDeviceData())
        //     Pressure2->DeviceMalloc();
        if (AdvectTemp != nullptr && !AdvectTemp->HasDeviceData())
            AdvectTemp->DeviceMalloc();
        if (AdvectTmp2 != nullptr && !AdvectTmp2->HasDeviceData())
            AdvectTmp2->DeviceMalloc();
        if (Divergence != nullptr && !Divergence->HasDeviceData())
            Divergence->DeviceMalloc();
        if (Pressure != nullptr && !Pressure->HasDeviceData())
            Pressure->DeviceMalloc();
        if (VelDiv != nullptr && !VelDiv->HasDeviceData())
            VelDiv->DeviceMalloc();
        if (!Vel->AllFieldHashDeviceData())
            Vel->DeviceMalloc();
        if (!Vel2->AllFieldHashDeviceData())
            Vel2->DeviceMalloc();
        if (CurlField->IsValid() && !CurlField->AllFieldHashDeviceData())
            CurlField->DeviceMalloc();
        if (HeightField != nullptr)
            HeightField->DeviceMalloc();
        if (CollisionMaskField != nullptr)
            CollisionMaskField->DeviceMalloc();
    }

    virtual void LoadToHost()
    {
        if (Density != nullptr && !Density->HasDeviceData())
            Density->LoadToHost();
        if (Temprature != nullptr && !Temprature->HasDeviceData())
            Temprature->LoadToHost();
        // if (Pressure2 != nullptr && !Pressure2->HasDeviceData())
        //     Pressure2->LoadToHost();
        if (AdvectTemp != nullptr && !AdvectTemp->HasDeviceData())
            AdvectTemp->LoadToHost();
        if (AdvectTmp2 != nullptr && !AdvectTmp2->HasDeviceData())
            AdvectTmp2->LoadToHost();
        if (Divergence != nullptr && !Divergence->HasDeviceData())
            Divergence->LoadToHost();
        if (Pressure != nullptr && !Pressure->HasDeviceData())
            Pressure->LoadToHost();
        if (VelDiv != nullptr && !VelDiv->HasDeviceData())
            VelDiv->LoadToHost();
        if (!Vel->AllFieldHashDeviceData())
            Vel->LoadToHost();
        if (!Vel2->AllFieldHashDeviceData())
            Vel2->LoadToHost();
        if (CurlField->IsValid() && !CurlField->AllFieldHashDeviceData())
            CurlField->LoadToHost();

    }

    virtual void Init() {};

    void InitSimFields(AxVector3 pivot, AxVector3 size, AxFp32 voxelSize)
    {
        AxVector3UI res = MakeVector3UI(
            floor(size.x / voxelSize),
            floor(size.y / voxelSize),
            floor(size.z / voxelSize));

        AxVector3 fieldSize = MakeVector3(
            res.x * voxelSize,
            res.y * voxelSize,
            res.z * voxelSize);

        bool buildPrim = false;
        this->Density = m_StormSysFieldsGeometry->AddField<AxFp32>(AlphaProperty::DensityField, buildPrim, fieldSize, pivot, res);
        this->Temprature = m_StormSysFieldsGeometry->AddField<AxFp32>(AlphaProperty::TempratureField, buildPrim, fieldSize, pivot, res);
        this->AdvectTemp = m_StormSysFieldsGeometry->AddField<AxFp32>(AlphaProperty::AdvectTmp, buildPrim, fieldSize, pivot, res);
        this->AdvectTmp2 = m_StormSysFieldsGeometry->AddField<AxFp32>(AlphaProperty::AdvectTmp2, buildPrim, fieldSize, pivot, res);
        this->Divergence = m_StormSysFieldsGeometry->AddField<AxFp32>(AlphaProperty::DivregenceField, buildPrim, fieldSize, pivot, res);
        // this->Pressure2 = m_CatalystFieldsGeometry->AddField<AxFp32>(AlphaProperty::PressureField2, buildPrim, fieldSize, pivot, res);
        this->Pressure = m_StormSysFieldsGeometry->AddField<AxFp32>(AlphaProperty::PressureField, buildPrim, fieldSize, pivot, res);
        this->VelDiv = m_StormSysFieldsGeometry->AddField<AxFp32>(AlphaProperty::VelDivField, buildPrim, fieldSize, pivot, res);
        this->CollisionMaskField = m_StormSysFieldsGeometry->AddField<AxInt8>(AlphaProperty::VelDivField, buildPrim, fieldSize, pivot, res);

        auto vxField = m_StormSysFieldsGeometry->AddField<AxFp32>("vel.x", buildPrim, fieldSize, pivot, res);
        auto vyField = m_StormSysFieldsGeometry->AddField<AxFp32>("vel.y", buildPrim, fieldSize, pivot, res);
        auto vzField = m_StormSysFieldsGeometry->AddField<AxFp32>("vel.z", buildPrim, fieldSize, pivot, res);
        Vel->Set(vxField, vyField, vzField);

        auto vxField2 = m_StormSysFieldsGeometry->AddField<AxFp32>("vel2.x", buildPrim, fieldSize, pivot, res);
        auto vyField2 = m_StormSysFieldsGeometry->AddField<AxFp32>("vel2.y", buildPrim, fieldSize, pivot, res);
        auto vzField2 = m_StormSysFieldsGeometry->AddField<AxFp32>("vel2.z", buildPrim, fieldSize, pivot, res);
        Vel2->Set(vxField2, vyField2, vzField2);



        // auto cxField = m_CatalystFieldsGeometry->AddField<AxFp32>("curl.x", buildPrim, fieldSize, pivot, res);
        // auto cyField = m_CatalystFieldsGeometry->AddField<AxFp32>("curl.y", buildPrim, fieldSize, pivot, res);
        // auto czField = m_CatalystFieldsGeometry->AddField<AxFp32>("curl.z", buildPrim, fieldSize, pivot, res);
        // CurlField->Set(cxField, cyField, czField);

        this->Pressure->SetAllBoundaryOutsideZero();
        // this->Pressure2->SetAllBoundaryOpen();
        this->Vel->SetAllBoundaryExtesion();
    }

    void SetCloseBoundary(bool x, bool _x, bool y, bool _y, bool z, bool _z);

    AxGeometry* GetOwnGeometry() { return m_StormSysFieldsGeometry; };

protected:
private:
    AxGeometry* m_StormSysFieldsGeometry;


};


class AxStormSysObject : public AxSimObject
{
public:
    AxStormSysObject();
    ~AxStormSysObject();

    static AxSimObject *ObjectConstructor();
    virtual void ParamDeserilizationFromJson(std::string jsonRaw);
    void stormSysInit(bool initDeviceData = false);

    void SetRenderMaterial(AxGasVolumeMaterial* material);
    AxGasVolumeMaterial* GetRenderMaterial();
    AxVolumeRenderObject GetRenderObj() { return m_RenderObj; }
    void UpdateRenderData();

protected:

    virtual void OnInit();
    virtual void OnReset();

    virtual void OnPreSim(AxFp32 dt);
    virtual void OnUpdateSim(AxFp32 dt);
    virtual void OnPostSim(AxFp32 dt);

    virtual void OnInitDevice();
    virtual void OnPreSimDevice(AxFp32 dt);
    virtual void OnUpdateSimDevice(AxFp32 dt);
    virtual void OnPostSimDevice(AxFp32 dt);
    
    virtual void OnRenderUpdate();

    void stormSysPostSim(bool loadToHost = false, std::string additionExtName = "");

    AxStormSysSimParam simParam;
    AxStormSysSimData stormSimData;
    AxScalarFieldF32* m_Density = nullptr;
    AlphaCore::LinearSolver getPressureMethod();
private:

    AxGeometry* m_HeightFieldGeometry;
    AxPossionCGSolver m_CGPressureSolver;
    AxVolumeRenderObject m_RenderObj;
    std::vector< AxMicroSolverBase*> m_PostSimCallstack;
};

#endif