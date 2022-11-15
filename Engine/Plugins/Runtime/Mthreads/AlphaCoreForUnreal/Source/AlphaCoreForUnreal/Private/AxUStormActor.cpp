// Fill out your copyright notice in the Description page of Project Settings.

#include "AxUStormActor.h"
#include "Render/RenderAlphaCore.h"
#include "MicroSolver/AxMicroSolverFactory.h"
#include <direct.h>

// Sets default values
AAxUStormActor::AAxUStormActor()
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	m_SceneManager = AxSceneManager::GetInstance();
}

AAxUStormActor::~AAxUStormActor()
{
	m_SceneManager->ClearAndDestory();
}

// Called when the game starts or when spawned
void AAxUStormActor::BeginPlay()
{
	Super::BeginPlay();
	if (!GetRealTaskJson()) return;
	InitTaskJson();

	if (!m_SceneManager) m_SceneManager = AxSceneManager::GetInstance();
	AxSimWorld* world = m_SceneManager->GetWorld();
	m_StormSysObj = new AxStormSysObject();
	m_StormSysObj->SetName("Storm");
	world->AddObject(m_StormSysObj);
	world->SetFrame(2);
	AddVolumeMaterial();
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
	if (m_SceneManager) {
		m_SceneManager->ClearAndDestory();
	}
}

bool AAxUStormActor::GetRealTaskJson() {

	bool taskJsonExist = true;
	FString ProjectDir = FPaths::ProjectDir();
	std::string ProjectDirStr = TCHAR_TO_UTF8(*ProjectDir);
	std::string jsonPath = TCHAR_TO_UTF8(*SimTaskJson);

	struct stat buffer;
	if (stat(jsonPath.c_str(), &buffer) == 0) {
		m_RealTaskJson = jsonPath;
		taskJsonExist = true;
	}
	else {
		std::string jsonFile = ProjectDirStr + jsonPath;
		AX_WARN("jsonFile : {}", jsonFile);


		if (stat(jsonFile.c_str(), &buffer) == 0) {
			m_RealTaskJson = jsonFile;
			taskJsonExist = true;
		}
		else {
			AX_WARN("Storm Task Json Not Exist! : {}", jsonPath);
			taskJsonExist = false;
		}
	}
	return taskJsonExist;
}

void AAxUStormActor::InitTaskJson() 
{
	std::string jsonPath = m_RealTaskJson;
	std::string realWorkSpace = AlphaUtility::SplitString(jsonPath, "task.json")[0];

	std::ifstream in(jsonPath);
	if (!in.is_open())
	{
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
	auto& ParamMap = productionSolver["ParamMap"];
	if (ParamMap.HasMember("workSpace"))
	{
		auto& workSpace = ParamMap["workSpace"];
		auto& RawValue = workSpace["RawValue"];
		RawValue.SetString(jsonPath.c_str(), jsonPath.size(), doc.GetAllocator());
	}
	if (ParamMap.HasMember("emitterCacheReadFilePath"))
	{
		auto& emitterCacheReadFilePath = ParamMap["emitterCacheReadFilePath"];
		auto& RawValue = emitterCacheReadFilePath["RawValue"];
		auto charRawValue = RawValue.GetString();
		std::string strRawValue(charRawValue);
		std::vector<std::string> tokens = AlphaUtility::SplitString(strRawValue, "/");
		std::string newPath = realWorkSpace + tokens[tokens.size() - 1];
		RawValue.SetString(newPath.c_str(), newPath.size(), doc.GetAllocator());
		//RawValue.SetString(realWorkSpace.c_str());
	}
	if (ParamMap.HasMember("staticHeightFieldFielPath"))
	{
		auto& staticHeightFieldFielPath = ParamMap["staticHeightFieldFielPath"];
		auto& RawValue = staticHeightFieldFielPath["RawValue"];
		auto charRawValue = RawValue.GetString();
		std::string strRawValue(charRawValue);
		std::vector<std::string> tokens = AlphaUtility::SplitString(strRawValue, "/");
		std::string newPath = realWorkSpace + tokens[tokens.size() - 1];
		RawValue.SetString(newPath.c_str(), newPath.size(), doc.GetAllocator());
	}
	if (ParamMap.HasMember("cacheOutFilePath"))
	{
		auto& cacheOutFilePath = ParamMap["cacheOutFilePath"];
		auto& RawValue = cacheOutFilePath["RawValue"];
		auto charRawValue = RawValue.GetString();
		std::string strRawValue(charRawValue);
		std::vector<std::string> tokens = AlphaUtility::SplitString(strRawValue, "/");
		std::string newPath = realWorkSpace + tokens[tokens.size() - 1];
		RawValue.SetString(newPath.c_str(), newPath.size(), doc.GetAllocator());
	}

	auto& MicroSolverList = productionSolver["MicroSolverList"];
	auto array = MicroSolverList.GetArray();
	for (AxUInt32 i = 0; i < array.Size(); ++i) {
		auto& MicroSolverDesc = array[i]["MicroSolverDesc"];
		auto& Param = MicroSolverDesc["ParamMap"];
		if (Param.HasMember("emitterFilePath"))
		{
			auto& emitterFilePath = Param["emitterFilePath"];
			auto& RawValue = emitterFilePath["RawValue"];
			auto charRawValue = RawValue.GetString();
			std::string strRawValue(charRawValue);
			std::vector<std::string> tokens = AlphaUtility::SplitString(strRawValue, "/");
			std::string newPath = realWorkSpace + tokens[tokens.size() - 1];
			RawValue.SetString(newPath.c_str(), newPath.size(), doc.GetAllocator());
		}
	}

	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	doc.Accept(writer);

	std::ofstream out(jsonPath);
	if (!out.is_open())
	{
		return;
	}
	std::string strJson(buffer.GetString());
	out << strJson;
	out.close();

}

bool AAxUStormActor::AddVolumeMaterial()
{
	m_VolumeMaterial.stepSize = StepSize * 10.f;
	m_VolumeMaterial.densityScale = DensityScale;
	m_VolumeMaterial.shadowScale = ShadowScale * 0.1f;
	m_VolumeMaterial.usePhase = true;
	m_VolumeMaterial.phase = Phase;
	m_VolumeMaterial.minMaxInputDensity = { InputDensityMinMax.X, InputDensityMinMax.Y };
	m_VolumeMaterial.minMaxInputHeat = { InputHeatMinMax.X,InputHeatMinMax.Y };
	m_VolumeMaterial.minMaxInputTemperature = { InputTemperatureMinMax.X,InputTemperatureMinMax.Y };
	std::memcpy(m_VolumeMaterial.lookUpTableDensity, greyUCharRamp, 128 * sizeof(AxUChar));
	std::memcpy(m_VolumeMaterial.lookUpTableDensityColor, greyColorRamp, 128 * sizeof(AxColorRGBA8));
	std::memcpy(m_VolumeMaterial.LookUpTableHeat, customColorRamp, 128 * sizeof(AxColorRGBA8));
	std::memcpy(m_VolumeMaterial.LookUpTableTemperature, greyUCharRamp, 128 * sizeof(AxUChar));
	m_VolumeMaterial.needUpdate = false;
	m_StormSysObj->SetRenderMaterial(&m_VolumeMaterial);
	return true;
}

bool AAxUStormActor::LoadSimParmsFromJson()
{

	std::string jsonPath = m_RealTaskJson;

	//auto SenceManger = UUAxSceneManager::GetInstance();
	auto world = m_SceneManager->GetWorld();

	std::ifstream in(jsonPath);
	if (!in.is_open())
	{
		fprintf(stderr, "fail to read json file: %s \n", jsonPath.c_str());
		return false;
	}

	std::string jsonContent((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	in.close();

	rapidjson::Document doc;
	if (doc.Parse(jsonContent.c_str()).HasParseError())
		return false;
	if (!doc.HasMember("AX_SINGLE_TASK_DESC"))
		return false;

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

	m_StormSysObj->SetSimCacheOutputMark(false);
	return true;
}

void AAxUStormActor::AddDefaultSceneObj()
{
	auto world = m_SceneManager->GetWorld();
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


