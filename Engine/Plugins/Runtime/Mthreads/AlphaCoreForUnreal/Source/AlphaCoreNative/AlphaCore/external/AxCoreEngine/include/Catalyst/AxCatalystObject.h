#ifndef __AX_CATALYST_OBJECT_H__
#define __AX_CATALYST_OBJECT_H__

#include "AxSimObject.h"
#include "GridDense/AxFieldBase3D.h"
#include "GridDense/AxFieldBase2D.h"

#include "AxCatalyst.ProtocolData.h"
#include "Math/AxConjugateGradient.h"
#include <unordered_map>

//TODO : SOLVER CMD PARAMETER


class AxCatalystEmitterSimData : AxISimData
{
public:
    AxCatalystEmitterSimData()
    {
        this->m_CatalystEmitterGeometry = nullptr;
        this->Fuel      = nullptr;
        this->Density   = nullptr;
        this->Temprature = nullptr;
        this->Divergence = nullptr;
        this->Vel = new AxVecFieldF32();
    }

    ~AxCatalystEmitterSimData()
    {
        if (m_CatalystEmitterGeometry != nullptr)
        {
            m_CatalystEmitterGeometry->PrintPropertyHead();
            m_CatalystEmitterGeometry->ClearAndDestory();
            m_CatalystEmitterGeometry = nullptr;
        }
    }

    AxScalarFieldF32 *Divergence;
    AxScalarFieldF32 *Fuel;
    AxScalarFieldF32 *Density;
    AxScalarFieldF32 *Temprature;
    AxVecFieldF32 *Vel;

    struct RAWData
    {
        ///###FSA_SIMDATA_RAW_DATA
    };


    virtual void BindProperties(
        AxGeometry *geo0 = nullptr,
        AxGeometry *geo1 = nullptr,
        AxGeometry *geo2 = nullptr,
        AxGeometry *geo3 = nullptr,
        AxGeometry *geo4 = nullptr)
    {
        if (geo0 == nullptr)
            return;
        this->m_CatalystEmitterGeometry = geo0;
        this->Density = m_CatalystEmitterGeometry->FindFieldByName<AxFp32>(AlphaProperty::DensityField);
        this->Temprature = m_CatalystEmitterGeometry->FindFieldByName<AxFp32>(AlphaProperty::TempratureField);
        this->Fuel = m_CatalystEmitterGeometry->FindFieldByName<AxFp32>(AlphaProperty::FuelField);
        this->Vel->FieldX = m_CatalystEmitterGeometry->FindFieldByName<AxFp32>("v.x");
        this->Vel->FieldY = m_CatalystEmitterGeometry->FindFieldByName<AxFp32>("v.y");
        this->Vel->FieldZ = m_CatalystEmitterGeometry->FindFieldByName<AxFp32>("v.z");
        this->Divergence = m_CatalystEmitterGeometry->FindFieldByName<AxFp32>(AlphaProperty::DivregenceField);
    };

    virtual void LoadToDevice()
    {
        if (Density != nullptr && !Density->HasDeviceData())
            Density->DeviceMalloc();
        if (Temprature != nullptr && !Temprature->HasDeviceData())
            Temprature->DeviceMalloc();
        if (Fuel != nullptr && !Fuel->HasDeviceData())
            Fuel->DeviceMalloc();
        if (Vel != nullptr && !Vel->AllFieldHashDeviceData())
            Vel->DeviceMalloc();
        if (Divergence != nullptr && !Divergence->HasDeviceData())
            Divergence->DeviceMalloc();
    }
    virtual void Init(){};
    void ReLoad(std::string path)
    {
        if (m_CatalystEmitterGeometry == nullptr)
            return;
        AX_WARN("LoadEmitter:{}", path);
        m_CatalystEmitterGeometry->UpdateFieldsData(path, true);
    }

    bool IsGeometryEmpty()
    {
        if (m_CatalystEmitterGeometry)
            return false;
        return true;
    }


private:

    AxGeometry* m_CatalystEmitterGeometry;

};

class AxCatalystSimData : AxISimData
{

public:
    AxCatalystSimData()
    {
        this->m_CatalystFieldsGeometry = nullptr;
        this->Density = nullptr;
        this->Vel = new AxVecFieldF32();
        this->Vel2 = new AxVecFieldF32();
        this->CurlField = new AxVecFieldF32();
        this->Temprature = nullptr;
        this->Pressure2 = nullptr;
        this->Heat = nullptr;
        this->AdvectTemp = nullptr;
        this->Divergence = nullptr;
        this->Fuel = nullptr;
        this->Pressure = nullptr;
        this->VelDiv = nullptr;
        this->AdvectTmp2 = nullptr;
        this->BurnField = nullptr;
    }

    ~AxCatalystSimData()
    {
        if (m_CatalystFieldsGeometry != nullptr)
        {
            m_CatalystFieldsGeometry->ClearAndDestory();
            m_CatalystFieldsGeometry = nullptr;
        }
    }

    AxScalarFieldF32 *Density;
    AxVecFieldF32 *Vel;
    AxVecFieldF32 *Vel2;
    AxVecFieldF32 *CurlField;
    AxScalarFieldF32 *Temprature;
    AxScalarFieldF32 *Pressure2;
    AxScalarFieldF32 *Heat;
    AxScalarFieldF32 *AdvectTemp;
    AxScalarFieldF32 *Divergence;
    AxScalarFieldF32 *Fuel;
    AxScalarFieldF32 *Pressure;
    AxScalarFieldF32 *VelDiv;
    AxScalarFieldF32 *AdvectTmp2;
    AxScalarFieldF32 *BurnField;

    struct RAWData
    {
        ///###FSA_SIMDATA_RAW_DATA
    };

    virtual void BindProperties(
        AxGeometry *geo0 = nullptr,
        AxGeometry *geo1 = nullptr,
        AxGeometry *geo2 = nullptr,
        AxGeometry *geo3 = nullptr,
        AxGeometry *geo4 = nullptr)
    {
        if (geo0 == nullptr)
            return;
        this->m_CatalystFieldsGeometry = geo0;
    }

    virtual void LoadToDevice()
    {
        if (Density != nullptr && !Density->HasDeviceData())
            Density->DeviceMalloc();
        if (Temprature != nullptr && !Temprature->HasDeviceData())
            Temprature->DeviceMalloc();
        // if (Pressure2 != nullptr && !Pressure2->HasDeviceData())
        //     Pressure2->DeviceMalloc();
        if (Heat != nullptr && !Heat->HasDeviceData())
            Heat->DeviceMalloc();
        if (AdvectTemp != nullptr && !AdvectTemp->HasDeviceData())
            AdvectTemp->DeviceMalloc();
        if (AdvectTmp2 != nullptr && !AdvectTmp2->HasDeviceData())
            AdvectTmp2->DeviceMalloc();
        if (Divergence != nullptr && !Divergence->HasDeviceData())
            Divergence->DeviceMalloc();
        if (Fuel != nullptr && !Fuel->HasDeviceData())
            Fuel->DeviceMalloc();
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
        if (BurnField != nullptr && !BurnField->HasDeviceData())
            BurnField->DeviceMalloc();
    }

    virtual void LoadToHost()
    {
        if (Density != nullptr && !Density->HasDeviceData())
            Density->LoadToHost();
        if (Temprature != nullptr && !Temprature->HasDeviceData())
            Temprature->LoadToHost();
        // if (Pressure2 != nullptr && !Pressure2->HasDeviceData())
        //     Pressure2->LoadToHost();
        if (Heat != nullptr && !Heat->HasDeviceData())
            Heat->LoadToHost();
        if (AdvectTemp != nullptr && !AdvectTemp->HasDeviceData())
            AdvectTemp->LoadToHost();
        if (AdvectTmp2 != nullptr && !AdvectTmp2->HasDeviceData())
            AdvectTmp2->LoadToHost();
        if (Divergence != nullptr && !Divergence->HasDeviceData())
            Divergence->LoadToHost();
        if (Fuel != nullptr && !Fuel->HasDeviceData())
            Fuel->LoadToHost();
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

    virtual void Init(){};

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
        this->Density = m_CatalystFieldsGeometry->AddField<AxFp32>(AlphaProperty::DensityField, buildPrim, fieldSize, pivot, res);
        this->Temprature = m_CatalystFieldsGeometry->AddField<AxFp32>(AlphaProperty::TempratureField, buildPrim, fieldSize, pivot, res);
        this->Heat = m_CatalystFieldsGeometry->AddField<AxFp32>(AlphaProperty::HeatField, buildPrim, fieldSize, pivot, res);
        this->AdvectTemp = m_CatalystFieldsGeometry->AddField<AxFp32>(AlphaProperty::AdvectTmp, buildPrim, fieldSize, pivot, res);
        this->AdvectTmp2 = m_CatalystFieldsGeometry->AddField<AxFp32>(AlphaProperty::AdvectTmp2, buildPrim, fieldSize, pivot, res);
        this->Divergence = m_CatalystFieldsGeometry->AddField<AxFp32>(AlphaProperty::DivregenceField, buildPrim, fieldSize, pivot, res);
        this->Fuel = m_CatalystFieldsGeometry->AddField<AxFp32>(AlphaProperty::FuelField, buildPrim, fieldSize, pivot, res);
        // this->Pressure2 = m_CatalystFieldsGeometry->AddField<AxFp32>(AlphaProperty::PressureField2, buildPrim, fieldSize, pivot, res);
        this->Pressure = m_CatalystFieldsGeometry->AddField<AxFp32>(AlphaProperty::PressureField, buildPrim, fieldSize, pivot, res);
        this->VelDiv = m_CatalystFieldsGeometry->AddField<AxFp32>(AlphaProperty::VelDivField, buildPrim, fieldSize, pivot, res);
        this->BurnField = m_CatalystFieldsGeometry->AddField<AxFp32>(AlphaProperty::BurnField, buildPrim, fieldSize, pivot, res);

        auto vxField = m_CatalystFieldsGeometry->AddField<AxFp32>("vel.x", buildPrim, fieldSize, pivot, res);
        auto vyField = m_CatalystFieldsGeometry->AddField<AxFp32>("vel.y", buildPrim, fieldSize, pivot, res);
        auto vzField = m_CatalystFieldsGeometry->AddField<AxFp32>("vel.z", buildPrim, fieldSize, pivot, res);
        Vel->Set(vxField, vyField, vzField);

        auto vxField2 = m_CatalystFieldsGeometry->AddField<AxFp32>("vel2.x", buildPrim, fieldSize, pivot, res);
        auto vyField2 = m_CatalystFieldsGeometry->AddField<AxFp32>("vel2.y", buildPrim, fieldSize, pivot, res);
        auto vzField2 = m_CatalystFieldsGeometry->AddField<AxFp32>("vel2.z", buildPrim, fieldSize, pivot, res);
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

    AxGeometry* GetOwnGeometry() { return m_CatalystFieldsGeometry; };

protected:
private:
    AxGeometry *m_CatalystFieldsGeometry;
};

class AxCatalystObject : public AxSimObject
{
public:
    AxCatalystObject();
    ~AxCatalystObject();

    static AxSimObject *ObjectConstructor();
    virtual void ParamDeserilizationFromJson(std::string jsonRaw);
    AxCatalystSimParam* GetCatalystSimParam()             { return &simParam; }
    AxCatalystSimData* GetCatalystSimData()               { return &catalystSimData; }
    AxCatalystEmitterSimData* GetCatalystEmitterSimData() { return &catalystEmitterSimData; }

    void ResetEmitterGeo(AxGeometry *emitter)
    {
        if (emitter == nullptr)
        {
            // this->m_OwnWorld->addErrorMessage();
            AX_ERROR("SetEmitterGeo Error");
            return;
        }
        this->catalystEmitterSimData.BindProperties(emitter);
    }

    void SetRenderMaterial(AxGasVolumeMaterial *material);
    AxGasVolumeMaterial *GetRenderMaterial();
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
    // virtual void RenderInit();

    void catalystProjectionDevice();
    void catalystPreSim(AxFp32 dt);
    void catalystInit(bool initDeviceData);
    void catalystPostSim(bool loadToHost = false, std::string additionExtName = "");
    AxCatalystSimParam simParam;
    AxCatalystSimData catalystSimData;
    AxCatalystEmitterSimData catalystEmitterSimData;
    AlphaCore::LinearSolver getPressureMethod();

private:

    AxPossionCGSolver m_CGPressureSolver;

    AxVolumeRenderObject m_RenderObj;
    //AxScalarFieldUInt8 *m_Density = nullptr;
    //AxScalarFieldUInt8 *m_Heat = nullptr;
    //AxScalarFieldUInt8 *m_Temperature = nullptr;
    AxScalarFieldF32* m_Density = nullptr;
    AxScalarFieldF32* m_Heat = nullptr;
    AxScalarFieldF32* m_Temperature = nullptr;

    std::vector< AxMicroSolverBase*> m_VelControlCallstack;
};

#endif