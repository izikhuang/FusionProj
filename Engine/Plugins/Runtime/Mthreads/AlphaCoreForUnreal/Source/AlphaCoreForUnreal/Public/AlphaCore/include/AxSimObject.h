#ifndef __ALPHA_CORE_SIMULATION_OBJECT_H__
#define __ALPHA_CORE_SIMULATION_OBJECT_H__

#include "AxMacro.h"
#include "AxGeo.h"
#include <string>
#include <map>
#include "VolumeRender/AxVolumeRender.DataType.h"

typedef std::vector<std::function<void(AxFp32 deltaTime)>> AxSimCallbackStack;
typedef std::function<void(AxFp32 deltaTime)> AxSimCallback;

struct AxSimObjectCmdParameter
{
	AxSimObjectCmdParameter()
	{
		ExportVel = false;
		UseBlockVorticityConfinement = false;
		UseKernelFuse = false;
	}
	bool ExportVel;
	bool UseBlockVorticityConfinement;
	bool UseKernelFuse;

};

namespace AlphaCore
{
	static const char* simCmdExportVel = "export_vel";
	static const char* simCmdVorticityConfinementBlock = "vc_block";
	static const char* simCmdKernelFuse = "kernel_fuse";
}


/*

{
	std::string cmdParam = simParam.AlphaCommand.GetParamValue();
	std::vector<std::string> prmList = AlphaUtility::SplitString(cmdParam, "-");

	AX_FOR_I(prmList.size())
	{
		std::string param = prmList[i];
		if (param.size() == 0)
			continue;
		AlphaUtility::ReplaceString(param, " ", "");
		//std::cout << "CMD - Param:[" << param << "]" << std::endl;
		if (param == cmdExportVel)
			simCmdParam.ExportVel = true;
		if (param == cmdVorticityConfinementBlock)
			simCmdParam.UseBlockVorticityConfinement = true;
		if (param == cmdKernelFuse)
			simCmdParam.UseKernelFuse = true;
	}
}
*/

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


	virtual void Save(std::string path, AxInt32 frame);
	virtual void Read(std::string path, AxInt32 frame);
	virtual bool Save(std::string path);
	virtual bool Read(std::string path);
	virtual int Save(std::ifstream& ifs);
	virtual int Read(std::ofstream& ofs);
	
	void Init();

 	void SetDx(float dx);

	void SetCreateFrame(AxInt32 frame) { m_iCreateFrame = frame; };
	AxInt32 GetCreateFrame() const { return  m_iCreateFrame; };

	virtual void PrintRFSInfo() {}
 	AxUInt32 GetCookTimes() { return m_iCookTimes; };

	virtual void ParamDeserilizationFromJson(std::string jsonRaw) {};
	void PushMicroSolver(AxMicroSolverBase* solver);
	void RunMicroSolvers(AxFp32 dt);
	void SetMicroSolverList(std::vector<AxMicroSolverBase*>& list);
	void SetSPMDBackendAPI(AlphaCore::AxBackendAPI backendAPI) { m_eDevice = backendAPI; };
	void SetSimCacheOutputMark(bool e) { m_bSetSimCacheOutputMark = e; };

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

	void FormJson(std::string simObjectDesc);

	// Render
	void RegisterImage(AxUInt32 width, AxUInt32 height, bool LoadToDevice = true);
	void RegisterImageInt8(AxUInt32 width, AxUInt32 height);
	void ResizeImage(AxUInt32 width, AxUInt32 height, bool LoadToDevice = true);
	void Render();

	AxTextureRGBA*	GetRenderImage();
	AxTextureRGBA8* GetRenderImageInt8();
	AxTextureR32*	GetDepthTexture();
	AxVector2UI		GetImageSize();

	void ResizeOutputImage();

protected:
	// Render
	virtual void OnRenderUpdate() {}
	AxTextureRGBA*	m_OutputImage;
	AxTextureRGBA8* m_OutputImageInt8;	// CUDA Device
	AxTextureR32*	m_DepthTexture;	// CUDA Device
	AxVector2UI		m_ImageRes;

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
	virtual void OnPreSimDevice(float dt) {};
	virtual void OnUpdateSimDevice(float dt) {};
	virtual void OnPostSimDevice(float dt) {};

	void translateSimMessage(std::string msgName,AxFp32 deltaTime);


	void SetCmdParameters(std::string cmdParams);

	AxUInt32 substeps;
	AxFp32   timeScale;
	AxInt32 m_iCreateFrame;

	friend class AxSimWorld;
	AlphaCore::AxBackendAPI m_eDevice;
	AlphaCore::AxSimOutputMode m_eAutoSaveMode;

	class AxSimWorld* m_OwnWorld;
	AxUInt32 m_iCookTimes;
	
	AxBufferV3* m_PrevPositionProp;
	AxBufferV3* m_PrdPositionProp;
	AxBufferV3* m_LastPositionProp;


	AxSimObjectCmdParameter simCmdParam;

	void* m_UserData;
	void addKernelWorkSpace(std::string kernelPathKey,std::string kernelWorkSpace);

	bool m_bSetSimCacheOutputMark;
private:

	//soverStatck
	//	std::map<std::string, RxCallback1> m_CallMap1;
	std::vector< AxMicroSolverBase*> m_MicroSolverStack;
	std::map<std::string, AxGeometry*> m_AppendGeometry;
	std::map<std::string, AxSimCallbackStack> m_SimCallbackMap;
	std::map<std::string, std::string> m_KernelWorkSpace;
	//bool m_bSetSimCacheOutputMark;
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


class AxISimData
{
public:
	AxISimData()
	{

	}

	~AxISimData()
	{
	
	}
	virtual void BindProperties(
		AxGeometry* geo0 = nullptr,
		AxGeometry* geo1 = nullptr,
		AxGeometry* geo2 = nullptr,
		AxGeometry* geo3 = nullptr,
		AxGeometry* geo4 = nullptr) {};

	virtual void Init() {};
	virtual void DeviceMalloc(){};
	virtual void Destory() {};
	virtual void FromJson(std::string codeRaw) {};
	std::string GetSimCacheInputPath() const;
	bool HasSimCacheInput();
protected:

	std::string m_sSimCacheInputPath;
};




#endif