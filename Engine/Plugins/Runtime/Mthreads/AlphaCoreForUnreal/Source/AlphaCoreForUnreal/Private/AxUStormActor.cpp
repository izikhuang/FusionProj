// Fill out your copyright notice in the Description page of Project Settings.


#include "AxUStormActor.h"
#include "MicroSolver/AxMicroSolverFactory.h"


// Sets default values
AAxUStormActor::AAxUStormActor()
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	//PrimaryActorTick.bCanEverTick = true;
	m_StormSysObj = new AxStormSysObject();
	m_StormSysObj->SetName("Storm");
}
AAxUStormActor::~AAxUStormActor()
{
}

// Called when the game starts or when spawned
void AAxUStormActor::BeginPlay()
{
	Super::BeginPlay();
	auto SenceManger = UUAxSceneManager::GetInstance();
	if (!SenceManger) return;
	auto world = SenceManger->world;
	world->SetFrame(1);
	AddVolumeMaterial();
	LoadEmitterFromJson();
	LoadSimParmsFromJson();
	AddDefaultSceneObj();

}

void AAxUStormActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

void AAxUStormActor::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);
	
	auto SenceManger = UUAxSceneManager::GetInstance();
	SenceManger->ClearAndDestory();
	AX_WARN("ClearAndDestory SenceManger");
}


bool AAxUStormActor::AddVolumeMaterial()
{
	m_VolumeMaterial.stepSize = StepSize;
	m_VolumeMaterial.densityScale = DensityScale;
	m_VolumeMaterial.shadowScale = ShadowScale;
	m_VolumeMaterial.usePhase = usePhase;
	m_VolumeMaterial.phase = Phase;
	m_VolumeMaterial.minMaxInputDensity = { InputDensityMinMax.X, InputDensityMinMax.Y };
	m_VolumeMaterial.minMaxInputHeat = { InputHeatMinMax.X,InputHeatMinMax.Y };
	m_VolumeMaterial.minMaxInputTemperature = { InputTemperatureMinMax.X,InputTemperatureMinMax.Y };
	//m_VolumeMaterial.minMaxOuputTemperature = { 0,1 };
	std::memcpy(m_VolumeMaterial.lookUpTableDensity, greyUCharRamp, 128 * sizeof(AxUChar));
	std::memcpy(m_VolumeMaterial.lookUpTableDensityColor, greyColorRamp, 128 * sizeof(AxColorRGBA8));
	std::memcpy(m_VolumeMaterial.LookUpTableHeat, customColorRamp, 128 * sizeof(AxColorRGBA8));
	std::memcpy(m_VolumeMaterial.LookUpTableTemperature, greyUCharRamp, 128 * sizeof(AxUChar));
	m_VolumeMaterial.needUpdate = false;
	//m_StormSysObj->SetRenderMaterial(&m_VolumeMaterial);
	return true;
}

void AAxUStormActor::LoadEmitterFromJson()
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

	//m_StormSysObj->ResetEmitterGeo(m_Emitter);

	AX_WARN("SetEmitter..");
}

void AAxUStormActor::LoadSimParmsFromJson()
{
	std::string jsonPath = TCHAR_TO_UTF8(*SimTaskJson);

	auto SenceManger = UUAxSceneManager::GetInstance();
	auto world = SenceManger->world;

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
	std::string computeArch = taskJson["ComputeArch"].GetString();
	std::string productJsonRaw = AlphaCore::JsonHelper::RapidJsonToString(productionSolver);

	m_StormSysObj->ParamDeserilizationFromJson(productJsonRaw);
	m_StormSysObj->SetCreateFrame(start);
	if (computeArch == AlphaCore::AxSPMDBackendName::x86)
	{
		world->SetSPMDBackend(AlphaCore::AxBackendAPI::CPUx86);
		m_StormSysObj->SetSPMDBackendAPI(AlphaCore::AxBackendAPI::CPUx86); // TODO move to object deserilization
	}
	if (computeArch == AlphaCore::AxSPMDBackendName::CUDA)
	{
		world->SetSPMDBackend(AlphaCore::AxBackendAPI::CUDA);
		m_StormSysObj->SetSPMDBackendAPI(AlphaCore::AxBackendAPI::CUDA);
	}

	AX_INFO("productSo : {}", m_StormSysObj->GetName());

	world->AddObject(m_StormSysObj);
	world->SetFPS(fps);
	world->SetSubstep(SubSteps);
	

	// Add Micro solvers
	// ComputeArch
	//auto productSvs = AxMicroSolverFactory::GetInstance()->CreateSolverStackFromJsonContent(productJsonRaw);
	//AX_FOR_I(productSvs.size())
	//	m_StormSysObj->PushMicroSolver(productSvs[i]);

}

void AAxUStormActor::AddDefaultSceneObj()
{
	
	auto SenceManger = UUAxSceneManager::GetInstance();
	if (!SenceManger) return;
	auto world = SenceManger->world;
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

bool AAxUStormActor::ReadVolumeRenderObjectFromFile()
{
	return false;
}

