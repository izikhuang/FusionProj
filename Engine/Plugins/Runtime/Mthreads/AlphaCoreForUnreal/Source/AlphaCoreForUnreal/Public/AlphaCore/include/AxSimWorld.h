#ifndef __ALPHA_CORE_SIM_WORLD_H__
#define __ALPHA_CORE_SIM_WORLD_H__

#include <string>
#include <vector>
#include "AxMacro.h"
#include "AxDataType.h"
#include "Utility/AxDescrition.h"
#include "GridDense/AxFieldBase2D.h"
#include "VolumeRender/AxVolumeRender.DataType.h"

//typedef AxField2DBase<AxColorRGBA8> AxTextureRGBA8;
//typedef AxField2DBase<AxColorRGBA> AxTextureRGBA;
//typedef AxField2DBase<AxFp32> AxTextureR32;
//typedef AxField2DBase<AxFp64> AxTextureR64;


class AxSceneObject;
class AxSimObject;
class AxSimWorld
{
public:
	AxSimWorld();
	~AxSimWorld();

	enum AxSimTaskLoadError
	{
		kInvalidDescFilepath,
		kJsonParserError,
		kNotAnAlphaCoreTaskDesc,
		kNonProductionInfo,
		kNonParamMap,
		kSimSucc
	};

	void Init();
	void Step(float dt);
	
	void AddObject(AxSimObject* obj);
	AxSimObject* GetObjectByIndex(AxUInt32 objId);
	void RayTracingVisualization(std::string savePath);
	int GetFrame()						{ return m_iFrame; };
	bool IsEndSubstep()					{ return m_iCurrSubstep == m_iSubstep; };
	void SetSubstep(AxUInt32 globalSubstep);
	void SetAbsolutelyFrame(int frame)  { m_iFrame = frame; };

	AxFp32 GetTime()					{ return m_fTickTime; };
	AxUInt32 GetFPS()const				{ return m_iFPS; };

	AxSimTaskLoadError LoadAndRunSingleTask(std::string singleTaskJsonPath, bool runTask = true);
	void SetFPS(AxUInt32 fps) { m_iFPS = fps; };

	AlphaCore::AxBackendAPI GetSPMDBackend()const { return m_SPMDBackend; };
	void SetSPMDBackend(AlphaCore::AxBackendAPI apiBackend) { m_SPMDBackend = apiBackend; };
	AxUInt32  GetSubSteps() const		{ return m_iSubstep; };
	AxUInt32  GetCurrentSubStep()const	{ return m_iCurrSubstep; }
	std::string GetWorkSpacePath() const;


	bool IsFirstSubstep(){
		return this->GetCurrentSubStep() == this->GetSubSteps();
	}
	void SetFrame(AxInt32 frame) { m_iFrame = frame; }

protected:
	void runSimulationTask(AxInt32 startFrame,AxInt32 endFrame);
	void clearSimObjects();

private:

	AlphaCore::AxBackendAPI m_SPMDBackend;
	std::vector<AxSimObject*> m_Objs;

	AxFp32		m_fTickTime;
	AxInt32		m_iFrame;
	AxUInt32	m_iFPS;
	AxUInt32	m_iSubstep;
	AxUInt32	m_iCurrSubstep;
	std::string m_sWorkSpacePath;

// For Render
public:

	void Render();
	void StepAndRender(bool render = true);
	void RStep() {
		AX_WARN("Frame: {}", m_iFrame);
		AxFp32 deltaTime = 1.0f / (AxFp32)m_iFPS;
		this->Step(deltaTime);
		m_fTickTime += deltaTime;
	}
	AxSceneObject* GetSceneObject();
	void SetSceneObject(AxSceneObject* scene);
	void CreateDefaultSceneObject();
	bool HasSceneObject();
	AlphaCore::Desc::AxCameraInfo GetSceneCamera();
	bool SetSceneCamera(AlphaCore::Desc::AxCameraInfo camera);
	void AddSceneLight(AlphaCore::Desc::AxPointLightInfo* light);
	void SetSceneLight(AxUInt32 i, AlphaCore::Desc::AxPointLightInfo* light);

	void InitRenderImage(AxUInt32 width, AxUInt32 height); // TODO Change Function Name
	void InitRenderImageInt8(AxUInt32 width, AxUInt32 height); // TODO Change Function Name
	int	 GetSimObjectNum() { return m_Objs.size(); }

	//void SetImageResolution(AxInt32 width, AxInt32 height);
	
	void RegisterRenderImage(AxInt32 width, AxInt32 height, bool loadDevice = true);
	void RegisterRenderImageInt8(AxInt32 width, AxInt32 height);
	void ResizeRenderImage(AxInt32 width, AxInt32 height, bool loadDevice = true);
	void ResizeRenderImageInt8(AxInt32 width, AxInt32 height);
	void RegisterDepthImage(AxInt32 width, AxInt32 height, bool loadDevice = true);
	void ResizeDepthImage(AxInt32 width, AxInt32 height, bool loadDevice = true);

	AxTextureRGBA* GetRenderImage() { return m_OutputImage; };
	AxTextureRGBA8* GetRenderImageInt8() { return m_OutputImageInt8; };
	AxTextureR32* GetDepthImage() { return  m_DepthImage; };

	//AxVector2UI GetRenderImageSize();

private:
	AxSceneObject* m_SceneObject = nullptr;
	AxUInt32 m_ImageWidth;
	AxUInt32 m_ImageHeight;

	AxTextureRGBA* m_OutputImage = nullptr;			// Unreal Use float image
	AxTextureRGBA8* m_OutputImageInt8 = nullptr;	// Unreal Use float image
	AxTextureR32* m_DepthImage = nullptr;
};


// ################
// ## For Render ##
// ################

class AxSceneObject {
public:

	AxSceneObject();
	~AxSceneObject();

	bool SetCamera(AlphaCore::Desc::AxCameraInfo& cameraInfo);
	AlphaCore::Desc::AxCameraInfo GetCamera();
	void AddPointLight(AlphaCore::Desc::AxPointLightInfo pointLight);
	bool SetLightByIndex(AxUInt32 i, AlphaCore::Desc::AxPointLightInfo lightInfo);
	AlphaCore::Desc::AxPointLightInfo* GetLightByIndex(AxUInt32 i);

	// Todo Camera is valid
	bool IsValid();
	AxSceneRenderDesc* GetSceneDesc() { return &m_SceneDesc; }
private:
	AxSceneRenderDesc m_SceneDesc;
};

#endif