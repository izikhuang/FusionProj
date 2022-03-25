#ifndef __ALPHA_CORE_SIMULATION_OBJECT_H__
#define __ALPHA_CORE_SIMULATION_OBJECT_H__

#include <AxMacro.h>
#include <AxGeo.h>
#include <string>
#include <map>


typedef std::vector<std::function<void(AxFp32 deltaTime)>> AxSimCallbackStack;
typedef std::function<void(AxFp32 deltaTime)> AxSimCallback;


namespace AlphaCore
{
	enum AxSimOutputMode
	{
		kSaveEveryFrame,
		kSaveEveryStep
	};
}

struct AxSimObjectDebugData
{
	AlphaCore::AxCapsuleCollider CapCollider;
	AlphaCore::AxOBBCollider OBBCollider;
	AxVector3* RenderPosition;

};

class AxMicroSolverBase;
class AxSimObject : public AxGeometry
{
public:

	AxSimObject()
	{
		Init();
	};
	virtual ~AxSimObject();


	static AxSimObject* Load(std::string path);
	static AxSimObject* Load(std::string path,AxInt32 frame);

	void Save(std::string path, AxInt32 frame);
	void Read(std::string path, AxInt32 frame);
	virtual bool Save(std::string path);
	virtual bool Read(std::string path);
	virtual int Save(std::ifstream& ifs);
	virtual int Read(std::ofstream& ofs);
	
	void Init();
 	void SetCacheWritePath(std::string cacheFrameCode){ m_sCacheSavePath = cacheFrameCode; };
	void SetCacheReadPath(std::string path){ m_sCacheReadPath = path; };
	std::string GetCacheWritePath() const { return m_sCacheSavePath; };
	std::string GetCacheReadPath() const { return m_sCacheReadPath; };

 	void SetDx(float dx);

	void SetCreateFrame(AxInt32 frame) { m_iCreateFrame = frame; };
	AxInt32 GetCreateFrame() const { return  m_iCreateFrame; };

	virtual void PrintRFSInfo() {}
 	AxUInt32 GetCookTimes() { return m_iCookTimes; };

	virtual void ParamDeserilizationFromJson(std::string jsonRaw) {};
	void PushMicroSolver(AxMicroSolverBase* solver);
	void RunMicroSolvers(AxFp32 dt);
	void SetMicroSolverList(std::vector<AxMicroSolverBase*>& list);

	void SetSPMDBackendAPI(AlphaCore::AxSPMDBackend backendAPI) { m_eDevice = backendAPI; };

	AxBufferV3* AddPredictPositionProp();
	AxBufferV3* GetPredictPositionProp();
	AxBufferV3* AddPrevPositionProp();
	AxBufferV3* GetPrevPositionProp();
	AxBufferV3* AddLastPositionProp();
	AxBufferV3* GetLastPositionProp();
	AxGeometry* FindAppendGeometry(std::string name);
 
	void* GetUserData() { return m_UserData; };
	void SetUserData(void* userData) { m_UserData = userData; }

	void AddSimEventListen(std::string msgName, AxSimCallback callbackFunc);
protected:

	void initSim();
	void preSim(float dt);
	void sim(float dt);
	void postSim(float dt);

	virtual void OnInit() {};
	virtual void OnReset() {};
	virtual void OnPreSim(float dt) {}
	virtual void OnUpdateSim(float dt) {}
	virtual void OnPostSim(float dt) {}

	virtual void OnInitDevice() {};
	virtual void OnResetDevice() {};
	virtual void OnPreSimDevice(float dt) {}
	virtual void OnUpdateSimDevice(float dt) {}
	virtual void OnPostSimDevice(float dt) {}

	void translateSimMessage(std::string msgName,AxFp32 deltaTime);

	AxUInt32 substeps;
	AxFp32   timeScale;
	AxInt32 m_iCreateFrame;
	std::string	m_sCacheReadPath;
	std::string m_sCacheSavePath;

	friend class AxSimWorld;
	AlphaCore::AxSPMDBackend m_eDevice;
	AlphaCore::AxSimOutputMode m_eAutoSaveMode;

	class AxSimWorld* m_OwnWorld;
	AxUInt32 m_iCookTimes;
	
	AxBufferV3* m_PrevPositionProp;
	AxBufferV3* m_PrdPositionProp;
	AxBufferV3* m_LastPositionProp;

	void* m_UserData;

private:

	//soverStatck
	//	std::map<std::string, RxCallback1> m_CallMap1;
	std::vector< AxMicroSolverBase*> m_MicroSolverStack;
	std::map<std::string, AxGeometry*> m_AppendGeometry;
	std::map<std::string, AxSimCallbackStack> m_SimCallbackMap;

};


typedef	AxSimObject* (*AxSimObjectConstructor)();
class AxSimObjectFactory
{
public:
	static AxSimObjectFactory* GetInstance();
	void ClearAndDestory();
	bool RegisterProduct(std::string product_name, AxSimObjectConstructor constructor);
	AxSimObject* CreateSimObject(std::string product_name);
 private:
 	AxSimObjectFactory();
	~AxSimObjectFactory();
	static AxSimObjectFactory* m_Instance;
	//ScenePath 
	std::map<std::string, AxSimObjectConstructor> m_CreatorMap;
};

#endif