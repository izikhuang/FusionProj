#ifndef __AXSTORMSYSTEM_H__
#define __AXSTORMSYSTEM_H__

#include "AxSimObject.h"
#include "GridDense/AxFieldBase3D.h"
#include "GridDense/AxFieldBase2D.h"
#include "GridDense/AxGridDense.h"

#include "AxStormSystem.ProtocolData.h"
#include "Math/AxConjugateGradient.h"
#include "MicroSolver/AxMicroSolverBase.h"
#include "AxSimWorld.h"
#include "Atmosphere/AxAtmosphere.h"
#include "AxFluidUtility.h"
//#include "FluidUtility/LocalProtocol/AxStorm_Staggered.h"
//
//#include "FluidUtility/LocalProtocol/AxStorm.h"

FSA_CLASS class AxStormSysSimData : AxISimData
{
public:
    AxStormSysSimData()
    {
        this->m_StormSysFieldsGeometry = nullptr;
        // this->Density = nullptr;
        // this->Vel = new AxVecFieldF32();
        // this->Vel2 = new AxVecFieldF32();
        // this->CurlField = new AxVecFieldF32();
        // this->Temprature = nullptr;
        // this->Pressure2 = nullptr;
        // this->AdvectTemp = nullptr;
        // this->Divergence = nullptr;
        // this->Pressure = nullptr;
        // this->VelDiv = nullptr;
        // this->AdvectTmp2 = nullptr;
        this->HeightField = nullptr;
        this->CollisionMaskField = nullptr;

        this->divField = nullptr;
    this->pressureOld = nullptr;
    this->pressureNew = nullptr;
    this->advTemp = nullptr;
    this->advTemp1 = nullptr;
    this->buoyField = nullptr;
    this->combineVel = new AxVecFieldF32();
    this->combineVel2 = new AxVecFieldF32();
    this->advTempX = nullptr;
	this->advTempY = nullptr;
	this->advTempZ = nullptr;
    this->advTempVec = new AxVecFieldF32();
    this->velXField = nullptr;
    this->velYField = nullptr;
    this->velZField = nullptr;
    this->velXField2 = nullptr;
    this->velYField2 = nullptr;
    this->velZField2 = nullptr;
    
    // thermal dynamics
    this->isaTField = nullptr;
    this->isaPField = nullptr;
    this->gammaThermal = nullptr;
    this->mMThermal = nullptr;
    this->cpThermal = nullptr;
    this->temGround = nullptr;

    // atmosphere
    this->mRVapor = nullptr;
    this->mRCloud = nullptr;
    this->mRVaporSat = nullptr;
    this->mRRain = nullptr;
    this->relPhi = nullptr;
    this->ccMEc = nullptr;
	this->Er = nullptr;
	this->Ac = nullptr;
	this->Kc = nullptr;
    this->temField = nullptr;
	this->thetaField = nullptr;
    this->temThermalField = nullptr;
    
	this->mFVapor = nullptr;
	this->mFCloud = nullptr; 
	this->mFRain = nullptr;
	this->massFVapor = nullptr;

    this->markField = nullptr;

     }

    ~AxStormSysSimData()
    {
        if (m_StormSysFieldsGeometry != nullptr)
        {
            m_StormSysFieldsGeometry->ClearAndDestory();
            m_StormSysFieldsGeometry = nullptr;
        }
    }

    // AxScalarFieldF32* Density;
    
    // AxVecFieldF32* CurlField;
    // AxScalarFieldF32* Temprature;
    // AxScalarFieldF32* Pressure2;
    // AxScalarFieldF32* AdvectTemp;
    // AxScalarFieldF32* Divergence;
    // AxScalarFieldF32* Pressure;
    // AxScalarFieldF32* VelDiv;
    // AxScalarFieldF32* AdvectTmp2;
    AxScalarFieldI8* CollisionMaskField;
    AxScalarFieldF32* HeightField;

    // fluid
    AxScalarFieldF32* divField;
    AxScalarFieldF32* pressureOld;
    AxScalarFieldF32* pressureNew;
    AxScalarFieldF32* advTemp;
    AxScalarFieldF32* advTemp1;
    AxScalarFieldF32* buoyField;
    AxVecFieldF32* combineVel;
    AxVecFieldF32* combineVel2;
    AxScalarFieldF32* advTempX;
	AxScalarFieldF32* advTempY;
	AxScalarFieldF32* advTempZ;
    AxVecFieldF32* advTempVec;
    AxScalarFieldF32* velXField;
    AxScalarFieldF32* velYField;
    AxScalarFieldF32* velZField;
    AxScalarFieldF32* velXField2;
    AxScalarFieldF32* velYField2;
    AxScalarFieldF32* velZField2;
    
    // thermal dynamics
    AxScalarFieldF32* isaTField;
    AxScalarFieldF32* isaPField;
    AxScalarFieldF32* gammaThermal;
    AxScalarFieldF32* mMThermal;
    AxScalarFieldF32* cpThermal;
    AxScalarFieldF32* temGround;

    // atmosphere
    AxScalarFieldF32* mRVapor;
    AxScalarFieldF32* mRCloud;
    AxScalarFieldF32* mRVaporSat;
    AxScalarFieldF32* mRRain;
    AxScalarFieldF32* relPhi;
    AxScalarFieldF32* ccMEc;
	AxScalarFieldF32* Er;
	AxScalarFieldF32* Ac;
	AxScalarFieldF32* Kc;
    AxScalarFieldF32* temField;
	AxScalarFieldF32* thetaField;
    AxScalarFieldF32* temThermalField;
    
	AxScalarFieldF32* mFVapor;
	AxScalarFieldF32* mFCloud; 
	AxScalarFieldF32* mFRain;
	AxScalarFieldF32* massFVapor;

    //height field
    AxScalarFieldI8* markField;

    //Emitter
    // AxScalarFieldF32* heatEmitterField;
    // AxScalarFieldF32* vaporEmitterField;

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
        if (mRVapor != nullptr && !mRVapor->HasDeviceData())
            mRVapor->DeviceMalloc();
        if (thetaField != nullptr && !thetaField->HasDeviceData())
            thetaField->DeviceMalloc();
        if (buoyField != nullptr && !buoyField->HasDeviceData())
            buoyField->DeviceMalloc();
        if (gammaThermal != nullptr && !gammaThermal->HasDeviceData())  
	        gammaThermal->DeviceMalloc();
        if (mMThermal != nullptr && !mMThermal->HasDeviceData())     
	        mMThermal->DeviceMalloc();
        if (cpThermal != nullptr && !cpThermal->HasDeviceData()) 
	        cpThermal->DeviceMalloc();
        if (temThermalField != nullptr && !temThermalField->HasDeviceData()) 
	        temThermalField->DeviceMalloc();
    
        if (velXField != nullptr && !velXField->HasDeviceData()) 
	        velXField->DeviceMalloc();
        if (velYField != nullptr && !velYField->HasDeviceData()) 
	        velYField->DeviceMalloc();
        if (velZField != nullptr && !velZField->HasDeviceData()) 
            velZField->DeviceMalloc();
        if (velXField2 != nullptr && !velXField2->HasDeviceData()) 
	        velXField2->DeviceMalloc();
        if (velYField2 != nullptr && !velYField2->HasDeviceData()) 
            velYField2->DeviceMalloc();
        if (velZField2 != nullptr && !velZField2->HasDeviceData()) 
            velZField2->DeviceMalloc();
        if (divField != nullptr && !divField->HasDeviceData()) 
            divField->DeviceMalloc();
        if (pressureOld != nullptr && !pressureOld->HasDeviceData()) 
            pressureOld->DeviceMalloc();
        if (pressureNew != nullptr && !pressureNew->HasDeviceData()) 
            pressureNew->DeviceMalloc();
        if (advTemp != nullptr && !advTemp->HasDeviceData()) 
            advTemp->DeviceMalloc();
        if (advTemp1 != nullptr && !advTemp1->HasDeviceData()) 
            advTemp1->DeviceMalloc();
        if (isaPField != nullptr && !isaPField->HasDeviceData()) 
            isaPField->DeviceMalloc();
        if (isaTField != nullptr && !isaTField->HasDeviceData()) 
            isaTField->DeviceMalloc();
        
if (mFVapor != nullptr && !mFVapor->HasDeviceData()) 
	mFVapor->DeviceMalloc();
    if (mFCloud != nullptr && !mFCloud->HasDeviceData()) 
	mFCloud->DeviceMalloc();
    if (mFRain != nullptr && !mFRain->HasDeviceData()) 
	mFRain->DeviceMalloc();
    if (massFVapor != nullptr && !massFVapor->HasDeviceData()) 
	massFVapor->DeviceMalloc();
    if (mRVaporSat != nullptr && !mRVaporSat->HasDeviceData()) 
	mRVaporSat->DeviceMalloc();
    if (mRCloud != nullptr && !mRCloud->HasDeviceData()) 
	mRCloud->DeviceMalloc();
    if (mRRain != nullptr && !mRRain->HasDeviceData()) 
	mRRain->DeviceMalloc();

    if (relPhi != nullptr && !relPhi->HasDeviceData()) 
	relPhi->DeviceMalloc();
    if (ccMEc != nullptr && !ccMEc->HasDeviceData()) 
	ccMEc->DeviceMalloc();
    if (Er != nullptr && !Er->HasDeviceData()) 
	Er->DeviceMalloc();
    if (Ac != nullptr && !Ac->HasDeviceData()) 
	Ac->DeviceMalloc();
    if (Kc != nullptr && !Kc->HasDeviceData()) 
	Kc->DeviceMalloc();


    if (temField != nullptr && !temField->HasDeviceData()) 
	temField->DeviceMalloc();
    if (thetaField != nullptr && !thetaField->HasDeviceData()) 
	thetaField->DeviceMalloc();
    if (temGround != nullptr && !temGround->HasDeviceData()) 
	temGround->DeviceMalloc();

    if (advTempX != nullptr && !advTempX->HasDeviceData()) 
	advTempX->DeviceMalloc();
    if (advTempY != nullptr && !advTempY->HasDeviceData()) 
	advTempY->DeviceMalloc();
    if (advTempZ != nullptr && !advTempZ->HasDeviceData()) 
	advTempZ->DeviceMalloc();

        // if (Pressure2 != nullptr && !Pressure2->HasDeviceData())
        //     Pressure2->DeviceMalloc();
        // if (AdvectTemp != nullptr && !AdvectTemp->HasDeviceData())
        //     AdvectTemp->DeviceMalloc();
        // if (AdvectTmp2 != nullptr && !AdvectTmp2->HasDeviceData())
        //     AdvectTmp2->DeviceMalloc();
        // if (Divergence != nullptr && !Divergence->HasDeviceData())
        //     Divergence->DeviceMalloc();
        // if (Pressure != nullptr && !Pressure->HasDeviceData())
        //     Pressure->DeviceMalloc();
        // if (VelDiv != nullptr && !VelDiv->HasDeviceData())
        //     VelDiv->DeviceMalloc();
        // if (!Vel->AllFieldHashDeviceData())
        //     Vel->DeviceMalloc();
        // if (!Vel2->AllFieldHashDeviceData())
        //     Vel2->DeviceMalloc();
        // if (CurlField->IsValid() && !CurlField->AllFieldHashDeviceData())
        //     CurlField->DeviceMalloc();
        if (HeightField != nullptr)
            HeightField->DeviceMalloc();
        if (CollisionMaskField != nullptr)
            CollisionMaskField->DeviceMalloc();
        if (markField != nullptr && !markField->HasDeviceData())
        {
            markField->DeviceMalloc();
        }
    }

    virtual void LoadToHost()
    {
        // if (Density != nullptr && !Density->HasDeviceData())
        //     Density->LoadToHost();
        // if (Temprature != nullptr && !Temprature->HasDeviceData())
        //     Temprature->LoadToHost();
        // // if (Pressure2 != nullptr && !Pressure2->HasDeviceData())
        // //     Pressure2->LoadToHost();
        // if (AdvectTemp != nullptr && !AdvectTemp->HasDeviceData())
        //     AdvectTemp->LoadToHost();
        // if (AdvectTmp2 != nullptr && !AdvectTmp2->HasDeviceData())
        //     AdvectTmp2->LoadToHost();
        // if (Divergence != nullptr && !Divergence->HasDeviceData())
        //     Divergence->LoadToHost();
        // if (Pressure != nullptr && !Pressure->HasDeviceData())
        //     Pressure->LoadToHost();
        // if (VelDiv != nullptr && !VelDiv->HasDeviceData())
        //     VelDiv->LoadToHost();
        // if (!Vel->AllFieldHashDeviceData())
        //     Vel->LoadToHost();
        // if (!Vel2->AllFieldHashDeviceData())
        //     Vel2->LoadToHost();
        // if (CurlField->IsValid() && !CurlField->AllFieldHashDeviceData())
        //     CurlField->LoadToHost();

        mRCloud->LoadToHost();
        mRVapor->LoadToHost();
        thetaField->LoadToHost();

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


    this->buoyField = this->m_StormSysFieldsGeometry->AddField<AxFp32>("buoyancyField", buildPrim, fieldSize, pivot, res);
	this->gammaThermal = this->m_StormSysFieldsGeometry->AddField<AxFp32>("gammaTh", buildPrim, fieldSize, pivot, res);
	this->mMThermal = this->m_StormSysFieldsGeometry->AddField<AxFp32>("mMTh", buildPrim, fieldSize, pivot, res);
	this->cpThermal = this->m_StormSysFieldsGeometry->AddField<AxFp32>("cpTh", buildPrim, fieldSize, pivot, res);
	this->temThermalField = this->m_StormSysFieldsGeometry->AddField<AxFp32>("temperatureThermalField", buildPrim, fieldSize, pivot, res);
	this->mRVapor = this->m_StormSysFieldsGeometry->AddField<AxFp32>(AlphaProperty::DensityField, buildPrim, fieldSize, pivot, res);
	this->velXField = this->m_StormSysFieldsGeometry->AddField<AxFp32>("vel.x", buildPrim, fieldSize, pivot, res);
	this->velYField = this->m_StormSysFieldsGeometry->AddField<AxFp32>("vel.y", buildPrim, fieldSize, pivot, res);
	this->velZField = this->m_StormSysFieldsGeometry->AddField<AxFp32>("vel.z", buildPrim, fieldSize, pivot, res);
	
	this->velXField2 = this->m_StormSysFieldsGeometry->AddField<AxFp32>("vel2.x", buildPrim, fieldSize, pivot, res);
	this->velYField2 = this->m_StormSysFieldsGeometry->AddField<AxFp32>("vel2.y", buildPrim, fieldSize, pivot, res);
	this->velZField2 = this->m_StormSysFieldsGeometry->AddField<AxFp32>("vel2.z", buildPrim, fieldSize, pivot, res);

	this->divField = this->m_StormSysFieldsGeometry->AddField<AxFp32>("divergence", buildPrim, fieldSize, pivot, res);

	this->pressureOld = this->m_StormSysFieldsGeometry->AddField<AxFp32>("pressureOld", buildPrim, fieldSize, pivot, res);
	this->pressureNew = this->m_StormSysFieldsGeometry->AddField<AxFp32>("pressureNew", buildPrim, fieldSize, pivot, res);
	this->advTemp = this->m_StormSysFieldsGeometry->AddField<AxFp32>("advectTemp", buildPrim, fieldSize, pivot, res);
	this->advTemp1 = this->m_StormSysFieldsGeometry->AddField<AxFp32>("advectTemp1", buildPrim, fieldSize, pivot, res); // outfield

	this->isaPField = this->m_StormSysFieldsGeometry->AddField<AxFp32>("isaP", buildPrim, fieldSize, pivot, res);
	this->isaTField = this->m_StormSysFieldsGeometry->AddField<AxFp32>("isaT", buildPrim, fieldSize, pivot, res);

	//this->heatEmitterField = this->m_StormSysFieldsGeometry->AddField<AxFp32>("heatEmitter", buildPrim, fieldSize, pivot, res);
	//this->vaporEmitterField = this->m_StormSysFieldsGeometry->AddField<AxFp32>("vaporEmitter", buildPrim, fieldSize, pivot, res);

	this->mFVapor = this->m_StormSysFieldsGeometry->AddField<AxFp32>("mFVaporField", buildPrim, fieldSize, pivot, res);
	this->mFCloud = this->m_StormSysFieldsGeometry->AddField<AxFp32>("mFCloudField", buildPrim, fieldSize, pivot, res);
	this->mFRain = this->m_StormSysFieldsGeometry->AddField<AxFp32>("mFRainField", buildPrim, fieldSize, pivot, res);
	this->massFVapor = this->m_StormSysFieldsGeometry->AddField<AxFp32>("massFVaporField", buildPrim, fieldSize, pivot, res);

	this->mRVaporSat = this->m_StormSysFieldsGeometry->AddField<AxFp32>("mRVaporSatField", buildPrim, fieldSize, pivot, res);
	this->mRCloud = this->m_StormSysFieldsGeometry->AddField<AxFp32>("mRCloudField", buildPrim, fieldSize, pivot, res);
	this->mRRain = this->m_StormSysFieldsGeometry->AddField<AxFp32>("mRRainField", buildPrim, fieldSize, pivot, res);

	this->relPhi = this->m_StormSysFieldsGeometry->AddField<AxFp32>("relPhiField", buildPrim, fieldSize, pivot, res);

	this->ccMEc = this->m_StormSysFieldsGeometry->AddField<AxFp32>("ccMEcField", buildPrim, fieldSize, pivot, res);
	this->Er = this->m_StormSysFieldsGeometry->AddField<AxFp32>("erField", buildPrim, fieldSize, pivot, res);
	this->Ac = this->m_StormSysFieldsGeometry->AddField<AxFp32>("acField", buildPrim, fieldSize, pivot, res);
	this->Kc = this->m_StormSysFieldsGeometry->AddField<AxFp32>("kcField", buildPrim, fieldSize, pivot, res);

	this->temField = this->m_StormSysFieldsGeometry->AddField<AxFp32>("temperatureField", buildPrim, fieldSize, pivot, res);
	this->thetaField = this->m_StormSysFieldsGeometry->AddField<AxFp32>(AlphaProperty::TempratureField, buildPrim, fieldSize, pivot, res);

	this->temGround = m_StormSysFieldsGeometry->AddField<AxFp32>("temGround", buildPrim, fieldSize, pivot, res);
	this->advTempX = m_StormSysFieldsGeometry->AddField<AxFp32>("advectTemp.x", buildPrim, fieldSize, pivot, res);
	this->advTempY = m_StormSysFieldsGeometry->AddField<AxFp32>("advectTemp.y", buildPrim, fieldSize, pivot, res);
	this->advTempZ = m_StormSysFieldsGeometry->AddField<AxFp32>("advectTemp.z", buildPrim, fieldSize, pivot, res);

    this->markField = m_StormSysFieldsGeometry->AddField<AxInt8>("MarkField", buildPrim, fieldSize, pivot, res);

   
	combineVel->Set(velXField, velYField, velZField, "vel");
	combineVel2->Set(velXField2, velYField2, velZField2, "vel2");
	advTempVec->Set(advTempX, advTempY, advTempZ, "advTempVec");

    AxFp32 dx = voxelSize;
	AxFp32 dy = voxelSize;
	AxFp32 dz = voxelSize;

	AxInt32 nx = res.x;
	AxInt32 ny = res.y;
	AxInt32 nz = res.z;

	velXField->SetFieldResolution(nx + 1, ny, nz);
	velYField->SetFieldResolution(nx, ny + 1, nz);
	velZField->SetFieldResolution(nx, ny, nz + 1);

	velXField2->SetFieldResolution(nx + 1, ny, nz);
	velYField2->SetFieldResolution(nx, ny + 1, nz);
	velZField2->SetFieldResolution(nx, ny, nz + 1);

	advTempX->SetFieldResolution(nx + 1, ny, nz);
	advTempY->SetFieldResolution(nx, ny + 1, nz);
	advTempZ->SetFieldResolution(nx, ny, nz + 1);

	velXField->SetFieldSize((nx + 1) * dx, ny * dy, nz * dz);
	velYField->SetFieldSize(nx * dx, (ny + 1) * dy, nz * dz);
	velZField->SetFieldSize(nx * dx, ny * dy, (nz + 1) * dz);
	velXField->GetVoxelStorageBufferPtr()->ResizeStorage((AxUInt64)((nx + 1) * ny * nz));
	velYField->GetVoxelStorageBufferPtr()->ResizeStorage((AxUInt64)(nx * (ny + 1) * nz));
	velZField->GetVoxelStorageBufferPtr()->ResizeStorage((AxUInt64)(nx * ny * (nz + 1)));

	velXField2->SetFieldSize((nx + 1) * dx, ny * dy, nz * dz);
	velYField2->SetFieldSize(nx * dx, (ny + 1) * dy, nz * dz);
	velZField2->SetFieldSize(nx * dx, ny * dy, (nz + 1) * dz);
	velXField2->GetVoxelStorageBufferPtr()->ResizeStorage((AxUInt64)((nx + 1) * ny * nz));
	velYField2->GetVoxelStorageBufferPtr()->ResizeStorage((AxUInt64)(nx * (ny + 1) * nz));
	velZField2->GetVoxelStorageBufferPtr()->ResizeStorage((AxUInt64)(nx * ny * (nz + 1)));

	advTempX->SetFieldSize((nx + 1)* dx, ny* dy, nz* dz);
	advTempY->SetFieldSize(nx* dx, (ny + 1)* dy, nz* dz);
	advTempZ->SetFieldSize(nx* dx, ny* dy, (nz + 1)* dz);
	advTempX->GetVoxelStorageBufferPtr()->ResizeStorage((AxUInt64)((nx + 1)* ny* nz));
	advTempY->GetVoxelStorageBufferPtr()->ResizeStorage((AxUInt64)(nx* (ny + 1)* nz));
	advTempZ->GetVoxelStorageBufferPtr()->ResizeStorage((AxUInt64)(nx* ny* (nz + 1)));

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
    void ResetFiledInfo();
    void ScaleField(float scale = 10.f);
    void stormSysPostSim(bool loadToHost = false, std::string additionExtName = "");

    AxStormSysSimParam simParam;
    AxStormSysSimData stormSimData;
    AxScalarFieldF32* m_Density = nullptr;
    AlphaCore::LinearSolver getPressureMethod();
private:
    AxField3DInfo m_FieldInfo;
    AxGeometry* m_HeightFieldGeometry = nullptr;
    AxPossionCGSolver m_CGPressureSolver;
    AxVolumeRenderObject m_RenderObj;
    std::vector< AxMicroSolverBase*> m_PostSimCallstack;


// SimDatas:
	const AxFp32 Mw = 0.01802f;           // Water Malor mass  Mw
	const AxFp32 Mair = 0.02896f;         // dry air Malor Mass Mair
	const AxFp32 GammaAir = 1.4f;		 // Gamma(Dryair)
	const AxFp32 GammaVap = 1.33f;		 //  Gamma(Vapor)

	const AxFp32 tau0 = -0.0065f;			// Lapse Rate:  tau0
	const AxFp32 tau1 = 0.0065f;			// Lapse Rate:  tau1 = -tau0
	const AxFp32 z1 = 8000.0f;			// inverse Height:  z1

	const AxFp32 latetHeatWater = 0.f;	// water Latent heat  L = 2.5 * 10^6 J/kg 
	const AxFp32 possionConst = 2.f / 7.f;  // Exner function  Kappa = R_dryair / cp_dryair


	const AxFp32 P0 = 101325.0f;	
	const AxFp32 T0 = 288.15;
	const AxFp32 gasUniConstR = 8.314f;	// Universal Constant R = 8.314 J/(mol K)
//	AxFp32 gConst = 9.81f;			//  gravity acceleration

	const AxFp32 temNoiseFactor = 2.0f;   //  m， Terrain HeatEmitterMap Noise factor
	const AxFp32 vaporNoiseFactor = 2.0f; //  m， Terrain vaporEmitterMap Noise factor

};

#endif