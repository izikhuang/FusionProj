#ifndef __ALPHA_CORE_SIM_WORLD_H__
#define __ALPHA_CORE_SIM_WORLD_H__

#include <string>
#include <vector>
#include <AxMacro.h>
#include <AxDataType.h>

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
	int GetFrame()		{ return m_iFrame; };
	void SetAbsolutelyFrame(int frame) { m_iFrame = frame; };

	bool IsEndSubstep() { return m_iCurrSubstep == m_iSubstep; };
	void SetSubstep(AxUInt32 globalSubstep);
	AxFp32 GetTime() { return 0.0f; };

	AxSimTaskLoadError LoadAndRunSingleTask(std::string singleTaskJsonPath, bool runTask = true);
	void SetFPS(AxUInt32 fps) { m_iFPS = fps; };
	AxUInt32 GetFPS()const { return m_iFPS; };

	AlphaCore::AxSPMDBackend GetSPMDBackend()const { return m_SPMDBackend; };
	void SetSPMDBackend(AlphaCore::AxSPMDBackend apiBackend) { m_SPMDBackend = apiBackend; };
	AxUInt32  GetSubSteps() const { return m_iSubstep; };
	AxUInt32  GetCurrentSubStep()const { return m_iCurrSubstep; }

protected:
	void runSimulationTask(AxInt32 startFrame,AxInt32 endFrame);
	void clearSimObjects();
private:


	AlphaCore::AxSPMDBackend m_SPMDBackend;
	std::vector<AxSimObject*> m_Objs;

	AxUInt32 m_iFPS;
	int m_iFrame;
	AxUInt32 m_iSubstep;
	AxUInt32 m_iCurrSubstep;

};

#endif