#include "AxSimObject.h"
#include "AxFloodDynamics.ProtocolData.h"
#include "AxFloodEmitter.ProtocolData.h"
//#include "FluidUtility/LocalProtocol/ParticleFluidCollision.h"
#include "AccelTree/AxSpatialHash.h"
#include "FluidUtility/AxFluidUtility.h"
//#include "FluidUtility/LocalProtocol/SHPCoherence.h"


FSA_CLASS class AxFloodSimData : AxISimData
{
public:
	//构造函数
	AxFloodSimData::AxFloodSimData()
	{
		//给变量赋值
		this->m_FloodDynamicsGeometry = nullptr;
		this->staicSDFGeometry = nullptr;
		this->m_fieldGeometry = new AxGeometry();
		this->posBuffer = nullptr;
		this->densityBuffer = nullptr;
		this->pressureBuffer = nullptr;
		this->normOmegaBuffer = nullptr;
		this->accelerationBuffer = nullptr;
		this->velocityBuffer = nullptr;
		this->omegaBuffer = nullptr;
		this->pressureAccelBuffer = nullptr;

		this->sortDensityBuffer = nullptr;
		this->sortPressureBuffer = nullptr;
		this->sortNormOmegaBuffer = nullptr;
		this->sortPosBuffer = nullptr;
		this->sortAccelerationBuffer = nullptr;
		this->sortVelocityBuffer = nullptr;
		this->sortOmegaBuffer = nullptr;
		this->sortPressureAccelBuffer = nullptr;

		this->density0 = 1000.0f;
		this->pressureStiffness = 25000.0f;
		this->pressureExp = 1.0f;
		this->boundingBox.Max = MakeVector3(1.0f, 1.0f, 1.0f);
		this->boundingBox.Min = MakeVector3(-1.0f, -1.0f, -1.0f);
		this->mass = 0.0f;
		this->volume = 0.0f;
		this->useCoherence = false;


		this->fieldsNum = 0;
		this->Vel = new  AxVecFieldF32();
		this->weightField = nullptr;
		this->markField = nullptr;

	}
	//析构函数
	AxFloodSimData::~AxFloodSimData()
	{

	}

	//类的成员变量： need buffer
	AxUInt32 fieldsNum;

	AxGeometry* m_FloodDynamicsGeometry;
	AxGeometry* staicSDFGeometry;
	AxGeometry* m_fieldGeometry;
	AxVecFieldF32* Vel;
	AxScalarFieldF32* weightField;
	AxScalarFieldI32* markField;
	AxAABB boundingBox;
	AxFp32 density0;
	AxFp32 pressureStiffness;
	AxFp32 pressureExp;
	AxFp32 mass;
	AxFp32 volume;
	bool useCoherence;


	AxBufferF* densityBuffer;
	AxBufferF* pressureBuffer;
	AxBufferF* normOmegaBuffer;

	AxBufferV3* posBuffer;
	AxBufferV3* accelerationBuffer;
	AxBufferV3* velocityBuffer;
	AxBufferV3* omegaBuffer;
	AxBufferV3* pressureAccelBuffer;

	AxBufferF* sortDensityBuffer;
	AxBufferF* sortPressureBuffer;
	AxBufferF* sortNormOmegaBuffer;
	AxBufferV3* sortPosBuffer;
	AxBufferV3* sortAccelerationBuffer;
	AxBufferV3* sortVelocityBuffer;
	AxBufferV3* sortOmegaBuffer;
	AxBufferV3* sortPressureAccelBuffer;

	AlphaCore::FluidUtility::Param::SHPCoherenceRAWDesc coherenceRAWDesc;


	virtual void BindProperties(
		AxGeometry* geo0 = nullptr,
		AxGeometry* geo1 = nullptr,
		AxGeometry* geo2 = nullptr,
		AxGeometry* geo3 = nullptr,
		AxGeometry* geo4 = nullptr)
	{
		if (geo0 == nullptr)
			return;
		this->m_FloodDynamicsGeometry = geo0;
	}
	
	void InitGeoField(AxVector3 pivot, AxVector3 size, AxFp32 voxelSize)
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
		auto vxField = m_fieldGeometry->AddField<AxFp32>("vel.x", buildPrim, fieldSize, pivot, res);
		auto vyField = m_fieldGeometry->AddField<AxFp32>("vel.y", buildPrim, fieldSize, pivot, res);
		auto vzField = m_fieldGeometry->AddField<AxFp32>("vel.z", buildPrim, fieldSize, pivot, res);
		Vel->Set(vxField, vyField, vzField);

		this->weightField = m_fieldGeometry->AddField<AxFp32>("weight", buildPrim, fieldSize, pivot, res);
		this->markField = m_fieldGeometry->AddField<AxInt32>("mark", buildPrim, fieldSize, pivot, res);
	}

	//类的成员函数
	void InitSimBuffer()
	{
		this->m_FloodDynamicsGeometry->AddPointBlock(1);
		this->densityBuffer = m_FloodDynamicsGeometry->AddPointProperty<AxFp32>("density");
		this->pressureBuffer = m_FloodDynamicsGeometry->AddPointProperty<AxFp32>("pressure");
		this->normOmegaBuffer = m_FloodDynamicsGeometry->AddPointProperty<AxFp32>("normOmega");

		this->posBuffer = m_FloodDynamicsGeometry->GetPointProperty<AxVector3>("P");
		this->accelerationBuffer = m_FloodDynamicsGeometry->AddPointProperty<AxVector3>("acceleration");
		this->velocityBuffer = m_FloodDynamicsGeometry->AddPointProperty<AxVector3>("v");
		this->omegaBuffer = m_FloodDynamicsGeometry->AddPointProperty<AxVector3>("omega");
		this->pressureAccelBuffer = m_FloodDynamicsGeometry->AddPointProperty<AxVector3>("pressureAccel");

#ifdef __HASH_USE_SortedBuffer__
		this->useCoherence = true;
		this->sortPosBuffer = m_FloodDynamicsGeometry->AddPointProperty<AxVector3>("sortedP");
		this->sortDensityBuffer = m_FloodDynamicsGeometry->AddPointProperty<AxFp32>("sortedDensity");
		this->sortPressureBuffer = m_FloodDynamicsGeometry->AddPointProperty<AxFp32>("sortedPressure");
		this->sortNormOmegaBuffer = m_FloodDynamicsGeometry->AddPointProperty<AxFp32>("sortedNormOmega");
		this->sortAccelerationBuffer = m_FloodDynamicsGeometry->AddPointProperty<AxVector3>("sortedAcceleration");
		this->sortVelocityBuffer = m_FloodDynamicsGeometry->AddPointProperty<AxVector3>("sortedVelocity");
		this->sortOmegaBuffer = m_FloodDynamicsGeometry->AddPointProperty<AxVector3>("sortedOmega");
		this->sortPressureAccelBuffer = m_FloodDynamicsGeometry->AddPointProperty<AxVector3>("sortedPressureAccel");

#endif

	}
	AxUInt32 InitPointNum(AxUInt32 numPoint)
	{
		AxUInt32 start = m_FloodDynamicsGeometry->AddPointBlock(numPoint);
		return start;
	}
	
	void InitBoxAABB(AxVector3 pivot, AxVector3 size)
	{
		AxVector3 maxAABB = pivot + (size / 2.0f);
		AxVector3 minAABB = pivot - (size / 2.0f);
		boundingBox = AlphaCore::AccelTree::MakeAABB(maxAABB, minAABB);
	}

	void InitstaicSDFGeo(std::string path)
	{
		staicSDFGeometry = AxGeometry::Load(path);
		if (staicSDFGeometry != nullptr)
		{
			fieldsNum = staicSDFGeometry->GetNumFields();
		}

	}

	void setUseCoherence(bool turnOn)
	{
		this->useCoherence = turnOn;
	}

	//分配显存
	virtual void LoadToDevice()
	{
		if (densityBuffer != nullptr && !densityBuffer->HasDeviceData())
			densityBuffer->DeviceMalloc();
		if (pressureBuffer != nullptr && !pressureBuffer->HasDeviceData())
			pressureBuffer->DeviceMalloc();
		if (normOmegaBuffer != nullptr && !normOmegaBuffer->HasDeviceData())
			normOmegaBuffer->DeviceMalloc();
		if (posBuffer != nullptr && !posBuffer->HasDeviceData())
			posBuffer->DeviceMalloc();
		if (accelerationBuffer != nullptr && !accelerationBuffer->HasDeviceData())
			accelerationBuffer->DeviceMalloc();
		if (velocityBuffer != nullptr && !velocityBuffer->HasDeviceData())
			velocityBuffer->DeviceMalloc();
		if (omegaBuffer != nullptr && !omegaBuffer->HasDeviceData())
			omegaBuffer->DeviceMalloc();
		if (pressureAccelBuffer != nullptr && !pressureAccelBuffer->HasDeviceData())
			pressureAccelBuffer->DeviceMalloc();

		//field loadToDevice
		AX_FOR_I(fieldsNum)
		{
			AxScalarFieldF32* filedi = (AxScalarFieldF32*)staicSDFGeometry->GetField(i);
			if (filedi != nullptr && !filedi->HasDeviceData())
				filedi->DeviceMalloc();
		}

		//新增速度场
		if (!Vel->AllFieldHashDeviceData())
			Vel->DeviceMalloc();
		if (weightField != nullptr && !weightField->HasDeviceData())
			weightField->DeviceMalloc();
		if (markField != nullptr && !markField->HasDeviceData())
			markField->DeviceMalloc();


#ifdef __HASH_USE_SortedBuffer__
		if (sortPosBuffer != nullptr && !sortPosBuffer->HasDeviceData())
			sortPosBuffer->DeviceMalloc();
		if (sortDensityBuffer != nullptr && !sortDensityBuffer->HasDeviceData())
			sortDensityBuffer->DeviceMalloc();
		if (sortPressureBuffer != nullptr && !sortPressureBuffer->HasDeviceData())
			sortPressureBuffer->DeviceMalloc();
		if (sortNormOmegaBuffer != nullptr && !sortNormOmegaBuffer->HasDeviceData())
			sortNormOmegaBuffer->DeviceMalloc();
		if (sortAccelerationBuffer != nullptr && !sortAccelerationBuffer->HasDeviceData())
			sortAccelerationBuffer->DeviceMalloc();
		if (sortVelocityBuffer != nullptr && !sortVelocityBuffer->HasDeviceData())
			sortVelocityBuffer->DeviceMalloc();
		if (sortOmegaBuffer != nullptr && !sortOmegaBuffer->HasDeviceData())
			sortOmegaBuffer->DeviceMalloc();
		if (sortPressureAccelBuffer != nullptr && !sortPressureAccelBuffer->HasDeviceData())
			sortPressureAccelBuffer->DeviceMalloc();
#endif
	}

	void GetCoherenceRAWDesc()
	{
		this->coherenceRAWDesc.NumPoints = this->posBuffer->Size();
		this->coherenceRAWDesc.posRawData = this->posBuffer->GetDataRawDevice();
		this->coherenceRAWDesc.densityRawData = this->densityBuffer->GetDataRawDevice();
		this->coherenceRAWDesc.pressureRawData = this->pressureBuffer->GetDataRawDevice();
		this->coherenceRAWDesc.normOmegaRawData = this->normOmegaBuffer->GetDataRawDevice();
		this->coherenceRAWDesc.accelerationRawData = this->accelerationBuffer->GetDataRawDevice();
		this->coherenceRAWDesc.velocityRawData = this->velocityBuffer->GetDataRawDevice();
		this->coherenceRAWDesc.omegaRawData = this->omegaBuffer->GetDataRawDevice();
		this->coherenceRAWDesc.pressureAccelRawData = this->pressureAccelBuffer->GetDataRawDevice();

		this->coherenceRAWDesc.sortPosRawData = this->sortPosBuffer->GetDataRawDevice();
		this->coherenceRAWDesc.sortDensityRawData = this->sortDensityBuffer->GetDataRawDevice();
		this->coherenceRAWDesc.sortPressureRawData = this->sortPressureBuffer->GetDataRawDevice();
		this->coherenceRAWDesc.sortNormOmegaRawData = this->sortNormOmegaBuffer->GetDataRawDevice();
		this->coherenceRAWDesc.sortAccelerationRawData = this->sortAccelerationBuffer->GetDataRawDevice();
		this->coherenceRAWDesc.sortVelocityRawData = this->sortVelocityBuffer->GetDataRawDevice();
		this->coherenceRAWDesc.sortOmegaRawData = this->sortOmegaBuffer->GetDataRawDevice();
		this->coherenceRAWDesc.sortPressureAccelRawData = this->sortPressureAccelBuffer->GetDataRawDevice();
	}
	void UpdateRAWDescNumPoints()
	{
		this->coherenceRAWDesc.NumPoints = this->posBuffer->Size();
	}

protected:

private:

};


class AxFloodObject : public AxSimObject
{
public:
	//构造函数; 在类的外部进行函数定义。
	AxFloodObject();
	~AxFloodObject();

	static AxSimObject* ObjectConstructor();
	virtual void ParamDeserilizationFromJson(std::string jsonRaw);
	
	//staic 函数可以不用实例化；staic修饰的函数内不能使用 this
	static void SimParamToEmitter(AxParticleFluidEmitterSIMParam & param, AxParticleFluidEmitter& emitter, AxParticleFluidSIMParam& solverParam, AxFp32 frame);



protected:

	//CPU版本运行
	virtual void OnInit();
	virtual void OnReset();
	virtual void OnPreSim(float dt);
	virtual void OnUpdateSim(float dt);
	virtual void OnPostSim(float dt);

	//GPU 版本运行
	virtual void OnInitDevice();
	virtual void OnResetDevice();
	virtual void OnPreSimDevice(float dt);
	virtual void OnUpdateSimDevice(float dt);
	virtual void OnPostSimDevice(float dt);


	void deserilizationEmitterList(std::string jsonStr);
	void deserilizationParticleForceList(std::string jsonStr);
	void updateEmitterParam();


	//custom

	void InitHash()
	{
		AxVector3 pivot = this->simParam.Pivot.GetParamValue();
		AxVector3 size = this->simParam.Size.GetParamValue() + 0.5f;
		AxFp32 particleRadius = this->simParam.Particle_radius.GetParamValue();
		AxFp32 suportRadius = particleRadius * 4.0f;
		hash.Init(pivot, size, suportRadius);
		hash.Build(simData.m_FloodDynamicsGeometry, "P");
		AxFp32 diam = 2.0f * particleRadius;
		simData.volume = 0.8f * diam * diam * diam;
		simData.mass = simData.volume * simData.density0;
	}


	void InitG2PHashField()
	{
		AxUInt32 sampelLevel = this->simParam.SampleLever.GetParamValue();
		AxVector3 pivot = this->simParam.Pivot.GetParamValue();
		AxVector3 size = this->simParam.Size.GetParamValue() + 0.5f;
		AxFp32 particleRadius = this->simParam.Particle_radius.GetParamValue();
		AxFp32 suportRadius = particleRadius * 4.0f * sampelLevel;
		G2Phash.Init(pivot, size, suportRadius);
		G2Phash.Build(simData.m_FloodDynamicsGeometry, "P");
		simData.InitGeoField(pivot, size, suportRadius);
	}



	virtual void HashLoadToDevice(bool USEP2G = false)
	{
		hash.DeviceMalloc();
		if (USEP2G)
		{
			G2Phash.DeviceMalloc();
		}
	}

	std::vector<AxParticleFluidEmitter> emitterList;
	std::vector<AxParticleFluidEmitterSIMParam> emitterSimParamList;

	AxParticleFluidSIMParam simParam;
	AxFloodSimData simData;
	AxParticleFluidEmitter emitter;
	AxSpatialHash hash;
	AxSpatialHash G2Phash;
	

private:
	//op
	std::vector< AxMicroSolverBase*> m_VelControlCallstack;
};