// Fill out your copyright notice in the Description page of Project Settings.


#include "AxUCatalystActor.h"
#include "Render/RenderAlphaCore.h"
#include "MicroSolver/AxMicroSolverFactory.h"


// Sets default values
AAxUCatalystActor::AAxUCatalystActor()
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	//PrimaryActorTick.bCanEverTick = true;
	m_SceneManager = AxSceneManager::GetInstance();
}
AAxUCatalystActor::~AAxUCatalystActor()
{
	m_SceneManager->ClearAndDestory();
}
// Called when the game starts or when spawned
void AAxUCatalystActor::BeginPlay()
{
	Super::BeginPlay();
	if (!m_SceneManager) {m_SceneManager = AxSceneManager::GetInstance();}

	AxSimWorld* world = m_SceneManager->GetWorld();
	m_CatalystObj = new AxCatalystObject();
	m_CatalystObj->SetName("Catalyst");
	world->AddObject(m_CatalystObj);
	world->SetFrame(2);
	AddVolumeMaterial();
	LoadEmitterFromJson();
	LoadSimParmsFromJson();
	AddDefaultSceneObj();
}

void AAxUCatalystActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

void AAxUCatalystActor::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);
	if (m_SceneManager) {
		m_SceneManager->ClearAndDestory();
	}
}



void AAxUCatalystActor::AddDefaultSceneObj()
{
	AxSimWorld* world = m_SceneManager->GetWorld();
	if (world->HasSceneObject()) return;

	// Create And Init RenderScene
	AxSceneObject* SceneObj = new AxSceneObject();
	AlphaCore::Desc::AxPointLightInfo pointLight;
	AlphaCore::Desc::AxCameraInfo camInfo;
	// Light Config
	pointLight.Pivot = MakeVector3(0.f, 500.f, 50.f);
	pointLight.Intensity = 15.f;
	pointLight.LightColor = { 1.f, 1.f, 1.f, 1.f };
	pointLight.Active = true;
	// Camera Config
	camInfo.Pivot = AxVector3{ 0.f, 20.f, 100.f };
	camInfo.Forward = AxVector3{ 0.f, 0.f, -1.f };
	camInfo.UpVector = AxVector3{ 0.f, 1.f, 0.f };
	camInfo.Near = 0.01;
	camInfo.Fov = 45;
	SceneObj->SetCamera(camInfo);
	SceneObj->AddPointLight(pointLight);

	world->SetSceneObject(SceneObj);
}




bool AAxUCatalystActor::AddVolumeMaterial()
{
	m_VolumeMaterial.stepSize = StepSize;
	m_VolumeMaterial.densityScale = DensityScale;
	m_VolumeMaterial.shadowScale = ShadowScale;
	m_VolumeMaterial.usePhase = usePhase;
	m_VolumeMaterial.phase = Phase;
	m_VolumeMaterial.minMaxInputDensity = { (float)InputDensityMinMax.X, (float)InputDensityMinMax.Y };
	m_VolumeMaterial.minMaxInputHeat = { (float)InputHeatMinMax.X,(float)InputHeatMinMax.Y };
	m_VolumeMaterial.minMaxInputTemperature = { (float)InputTemperatureMinMax.X,(float)InputTemperatureMinMax.Y };
	std::memcpy(m_VolumeMaterial.lookUpTableDensity, greyUCharRamp, 128 * sizeof(AxUChar));
	std::memcpy(m_VolumeMaterial.lookUpTableDensityColor, greyColorRamp, 128 * sizeof(AxColorRGBA8));
	std::memcpy(m_VolumeMaterial.LookUpTableHeat, customColorRamp, 128 * sizeof(AxColorRGBA8));
	std::memcpy(m_VolumeMaterial.LookUpTableTemperature, greyUCharRamp, 128 * sizeof(AxUChar));
	m_VolumeMaterial.needUpdate = false;
	m_CatalystObj->SetRenderMaterial(&m_VolumeMaterial);
	return true;
}

void AAxUCatalystActor::LoadEmitterFromJson()
{
	std::string emitter = TCHAR_TO_UTF8(*EmitterJson);
	std::ifstream in(emitter);
	if (!in.is_open())
	{
		AX_WARN("Fail to read Emitter json file");
		return;
	}

	std::string jsonContent((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	in.close();

	rapidjson::Document doc;
	if (doc.Parse(jsonContent.c_str()).HasParseError())
		return;

	m_Emitter = new AxGeometry();
	auto& volumes = doc["volumes"];
	assert(volumes.IsArray());
	AX_FOR_I(int(volumes.Size()))
	{
		auto name = volumes[i]["name"].GetString();
		int rx = volumes[i]["nx"].GetInt();
		int ry = volumes[i]["ny"].GetInt();
		int rz = volumes[i]["nz"].GetInt();
		float voxelsizex = volumes[i]["voxelSize"][0].GetFloat();
		float voxelsizey = volumes[i]["voxelSize"][1].GetFloat();
		float voxelsizez = volumes[i]["voxelSize"][2].GetFloat();
		float px = volumes[i]["pivot"][0].GetFloat();
		float py = volumes[i]["pivot"][1].GetFloat();
		float pz = volumes[i]["pivot"][2].GetFloat();
		// auto data = volumes[i]["data"].GetString();

		AxVector3 fieldPivot = { px, py, pz };
		AxVector3UI fieldRes = { rx, ry, rz };
		AxVector3 fieldSize = { voxelsizex * float(rx),
						  voxelsizey * float(ry),
						  voxelsizez * float(rz) };

		AxVector3UI res = MakeVector3UI(rx, ry, rz);

		m_Emitter->AddField<AxFp32>(name, true, fieldSize, fieldPivot, fieldRes);
		AxScalarFieldF32* filed = m_Emitter->FindFieldByName<AxFp32>(name);
		AX_FOR_J(int(filed->GetNumVoxels()))
		{
			filed->SetValue(j, volumes[i]["data"][j].GetFloat());
		}
	}

	m_CatalystObj->ResetEmitterGeo(m_Emitter);

	AX_WARN("SetEmitter..");
}

void AAxUCatalystActor::LoadSimParmsFromJson()
{
	std::string jsonPath = TCHAR_TO_UTF8(*SimJson);
	//auto m_SceneManager = UUAxSceneManager::GetInstance();
	auto world = m_SceneManager->GetWorld();

	std::ifstream in(jsonPath);
	if (!in.is_open())
	{
		fprintf(stderr, "fail to read json file: %s \n", jsonPath.c_str());
		return;
	}

	std::string jsonContent((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	in.close();

	rapidjson::Document doc;
	if (doc.Parse(jsonContent.c_str()).HasParseError())
		return;
	if (!doc.HasMember("AX_SINGLE_TASK_DESC"))
		return;

	auto& taskJson = doc["AX_SINGLE_TASK_DESC"];
	auto& productionSolver = taskJson["PRODUCTION_SOLVER"];
	auto& typeName = productionSolver["TypeName"];

	AxInt32 start = taskJson["TaskStartFrame"].GetInt();
	AxInt32 end = taskJson["TaskEndFrame"].GetInt();
	AxInt32 SubSteps = taskJson["TaskSubSteps"].GetInt();
	AxUInt32 fps = taskJson["FPS"].GetInt();
	std::string Arch = taskJson["ComputeArch"].GetString();
	std::string productJsonRaw = AlphaCore::JsonHelper::RapidJsonToString(productionSolver);

	m_CatalystObj->ParamDeserilizationFromJson(productJsonRaw);
	m_CatalystObj->SetCreateFrame(start);
	if (Arch == AlphaCore::AxSPMDBackendName::x86)
	{
		world->SetSPMDBackend(AlphaCore::AxBackendAPI::CPUx86);
		m_CatalystObj->SetSPMDBackendAPI(AlphaCore::AxBackendAPI::CPUx86); // TODO move to object deserilization
	}
	if (Arch == AlphaCore::AxSPMDBackendName::CUDA)
	{
		world->SetSPMDBackend(AlphaCore::AxBackendAPI::CUDA);
		m_CatalystObj->SetSPMDBackendAPI(AlphaCore::AxBackendAPI::CUDA);
	}

	AX_INFO("productSo : {}", m_CatalystObj->GetName());

	world->SetFPS(fps);
	world->SetSubstep(SubSteps);
	
	m_CatalystObj->SetSimCacheOutputMark(false);
	// Add Micro solvers
	// ComputeArch
	//auto productSvs = AxMicroSolverFactory::GetInstance()->CreateSolverStackFromJsonContent(productJsonRaw);
	//AX_FOR_I(productSvs.size())
	//	catalystObj->PushMicroSolver(productSvs[i]);
}

//bool AAxUCatalystActor::ReadVolumeRenderObjectFromFile()
//{
//
//	bool ret = false;
//
//	std::string FieldsPath = TCHAR_TO_UTF8(*AxVolumeFieldsPath);
//	auto geo = AxGeometry::Load(FieldsPath);
//
//	AxScalarFieldF32* density = geo->FindFieldByName<float>("density");
//	AxScalarFieldF32* heat = geo->FindFieldByName<float>("heat");
//	AxScalarFieldF32* temperature = geo->FindFieldByName<float>("density");
//
//	if (density) {
//		density->DeviceMalloc();
//		m_VolumeRenderObject.density = density->GetFiedRAWDescDevice();
//		AX_INFO("density");
//		ret = true;
//	}
//
//	if (heat) {
//		heat->DeviceMalloc();
//		m_VolumeRenderObject.heat = heat->GetFiedRAWDescDevice();
//		AX_INFO("heat");
//		ret = true;
//	}
//
//	if (temperature) {
//		temperature->DeviceMalloc();
//		m_VolumeRenderObject.temperature = temperature->GetFiedRAWDescDevice();
//		AX_INFO("temperature");
//		ret = true;
//	}
//
//
//	/*AlphaCore::GridDense::ReadFields(FieldsPath, m_AxVolumeFileds);
//
//	for (AxScalarFieldF32* vol : m_AxVolumeFileds) {
//
//		if (vol->GetName() == "density") {
//			m_VolumeRenderObject.densityInfo = vol->GetFieldInfo();
//			vol->DeviceMalloc();
//			m_VolumeRenderObject.density = vol->GetRawDataDevice();
//			AX_INFO("Get Density Filed from {}, Load To Device", FieldsPath);
//			ret = true;
//		}
//		else if (vol->GetName() == "heat") {
//			m_VolumeRenderObject.heatInfo = vol->GetFieldInfo();
//			vol->DeviceMalloc();
//			m_VolumeRenderObject.heat = vol->GetRawDataDevice();
//			AX_INFO("Get Heat Filed from {}, Load To Device", FieldsPath);
//			ret = true;
//		}
//		else if (vol->GetName() == "temperature") {
//			m_VolumeRenderObject.tempInfo = vol->GetFieldInfo();
//			vol->DeviceMalloc();
//			m_VolumeRenderObject.temp = vol->GetRawDataDevice();
//			AX_INFO("Get Temperature Filed from {}, Load To Device", FieldsPath);
//			ret = true;
//		}
//	}*/
//	
//	// Get VolumeMaterial
//	m_VolumeRenderObject.material = m_VolumeMaterial;
//
//	return ret;
//}
