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


//typedef void(*AxSimCallbackFunc)(AxSimCallbackData, AxContext, AlphaCore::AxBackendAPI);

struct AxKernelLanuchInfo
{
	AxUInt32 NumThreads;
};

class ALPHA_CLASS AxSimCallbackData
{
public:
	AxSimCallbackData()
	{
		m_ExcType = AlphaCore::kNonExc;
		m_DataPath = "";
	};
	~AxSimCallbackData() {};

	AxGeometry* GetSimGeo(AxUInt32 geoID) const
	{
		return m_SimGeoList[geoID];
	}

	template<typename T>
	void AddParam(std::string name,T* rawData,AxUInt32 SIZE)
	{
		if (AlphaCore::IsFloatDataType(AlphaCore::TypeID<T>()))
		{
			this->AddFloatParam(name, (AxFp32*)rawData, SIZE);
		}

		if (AlphaCore::IsIntDataType(AlphaCore::TypeID<T>()))
		{
			this->AddIntParam(name, (AxInt32*)rawData, SIZE);
		}

		if (AlphaCore::IsStringDataType(AlphaCore::TypeID<T>()))
		{

		}
	}

	void BindGeoData(
		AxGeometry* geo0 = nullptr,
		AxGeometry* geo1 = nullptr,
		AxGeometry* geo2 = nullptr,
		AxGeometry* geo3 = nullptr,
		AxGeometry* geo4 = nullptr);

	void SetInputGeoData(AxUInt32 geoID, AxGeometry* geo);

	void ClearSimParameters()
	{
		m_FloatParams.clear();
		m_Vector2FParams.clear();
		m_Vector3FParams.clear();
		m_Vector4FParams.clear();
		m_IntParams.clear();
		m_Vector2IParams.clear();
		m_Vector3IParams.clear();
		m_Vector4IParams.clear();
	}

	void SetTaskType(AlphaCore::AxExecuteType excType)
	{
		m_ExcType = excType;
	}

	AxUInt32 GetNumExecutes()
	{
		auto _numTasks = numTasks();
		std::cout << "Run Task : " << AlphaCore::ExecuteTypeToString(m_ExcType) << "   NumTasks:" << _numTasks << std::endl;
		return _numTasks;
	}

	std::string GetDataPath()
	{
		return m_DataPath;
	}
	void SetDataPath(std::string dataPath)
	{
		m_DataPath = dataPath;
	}

	void PostCallback();

	AxFp32 GetParamFp32(std::string name)
	{
		if(m_FloatParams.find(name)!= m_FloatParams.end())
			return m_FloatParams[name];
		return 0.0f;
	}

	AxInt32 GetParamInt32(std::string name)
	{
		if (m_IntParams.find(name) != m_IntParams.end())
			return m_IntParams[name];
		return 0.0f;
	}

	void AddFloatParam(std::string paramName, AxFp32 value)
	{
		if (m_FloatParams.find(paramName) == m_FloatParams.end())
			m_FloatParams[paramName] = value;
	}

	void AddVector2FParam(std::string paramName, AxFp32 u, AxFp32 v)
	{
		if (m_Vector2FParams.find(paramName) == m_Vector2FParams.end())
			m_Vector2FParams[paramName] = MakeVector2(u, v);
	}

	void AddVector3FParam(std::string paramName, AxFp32 x, AxFp32 y, AxFp32 z)
	{
		std::cout << "    : " << x << "," << y << "," << z << std::endl;
		if (m_Vector3FParams.find(paramName) == m_Vector3FParams.end())
			m_Vector3FParams[paramName] = MakeVector3(x, y, z);
	}

	void AddVector4FParam(std::string paramName, AxFp32 x, AxFp32 y, AxFp32 z, AxFp32 w)
	{
		if (m_Vector4FParams.find(paramName) == m_Vector4FParams.end())
			m_Vector4FParams[paramName] = MakeVector4(x, y, z, w);
	}

	void AddFloatParam(std::string paramName, AxFp32* values, AxUInt32 size)
	{
		if (size == 1)
			AddFloatParam(paramName, values[0]);
		if (size == 2)
			AddVector2FParam(paramName, values[0], values[1]);
		if (size == 3)
			AddVector3FParam(paramName, values[0], values[1], values[2]);
		if (size == 4)
			AddVector4FParam(paramName, values[0], values[1], values[2], values[3]);
	}

	void AddIntParam(std::string paramName, AxInt32* values, AxUInt32 size)
	{
		if (size == 1)
			AddIntParam(paramName, values[0]);
		if (size == 2)
			AddVector2IParam(paramName, values[0], values[1]);
		if (size == 3)
			AddVector3IParam(paramName, values[0], values[1], values[2]);
		if (size == 4)
			AddVector4IParam(paramName, values[0], values[1], values[2], values[3]);
	}

	void AddIntParam(std::string paramName, AxInt32 value)
	{
		if (m_IntParams.find(paramName) == m_IntParams.end())
			m_IntParams[paramName] = value;
	}

	void AddVector2IParam(std::string paramName, AxInt32 u, AxInt32 v)
	{
		if (m_Vector2IParams.find(paramName) == m_Vector2IParams.end())
			m_Vector2IParams[paramName] = MakeVector2(u, v);
	}

	void AddVector3IParam(std::string paramName, AxInt32 x, AxInt32 y, AxInt32 z)
	{
		if (m_Vector3IParams.find(paramName) == m_Vector3IParams.end())
			m_Vector3IParams[paramName] = MakeVector3I(x, y, z);
	}

	void AddVector4IParam(std::string paramName, AxInt32 x, AxInt32 y, AxInt32 z, AxInt32 w)
	{
		if (m_Vector4IParams.find(paramName) == m_Vector4IParams.end())
			m_Vector4IParams[paramName] = MakeVector4I(x, y, z, w);
	}

	void AddRampParam(std::string paramName)
	{

	}

	void AddVector2Param(std::string paramName, AxVector2 v2)
	{
		m_Vector2FParams[paramName] = v2;
	}

	void AddVector3Param(std::string paramName, AxVector3 v3)
	{
		m_Vector3FParams[paramName] = v3;
	}

	void AddVector4Param(std::string paramName, AxVector4 v4)
	{
		m_Vector4FParams[paramName] = v4;
	}

	void AddVector3IParam(std::string paramName, AxVector3I v3i)
	{

	}

	void AddStringParam(std::string paramName, std::string val)
	{

	}

	AxVector2 GetVector2Param(std::string paramName)
	{
		if (m_Vector2FParams.find(paramName) != m_Vector2FParams.end())
			return m_Vector2FParams[paramName];
		return MakeVector2(0.0f, 0.0f);
	}

	AxVector3 GetVector3Param(std::string paramName)
	{
		if (m_Vector3FParams.find(paramName) != m_Vector3FParams.end())
			return m_Vector3FParams[paramName];
		return MakeVector3();
	}

	AxVector4 GetVector4Param(std::string paramName)
	{
		if (m_Vector4FParams.find(paramName) != m_Vector4FParams.end())
			return m_Vector4FParams[paramName];
		return MakeVector4();
	}

	AxVector3I GetVector3IParam(std::string paramName)
	{
		return MakeVector3I(0, 0, 0);
	}

	AxFp32 GetFp32Param(std::string paramName)
	{
		if (m_FloatParams.find(paramName) == m_FloatParams.end())
		{
			AX_INFO("Non-Parameter:{}", paramName);
			return 0.0f;
		}
		return m_FloatParams[paramName];
	}

	std::string  GetStringParam(std::string paramName)
	{
		return "";
	}

	void SetPivotName(std::string name)
	{
		m_sPivotName = name;
	}

	void PrintParamData()
	{
		for (auto iter = m_FloatParams.begin(); iter != m_FloatParams.end(); iter++)
			AX_INFO("Float Param  {} : {} ", iter->first, iter->second);
		for (auto iter = m_Vector2FParams.begin(); iter != m_Vector2FParams.end(); iter++)
			std::cout << "Vector2F Param  " << iter->first << ":" << iter->second << std::endl;
		for (auto iter = m_Vector3FParams.begin(); iter != m_Vector3FParams.end(); iter++)
			std::cout << "Vector3F Param  " << iter->first << ":" << iter->second << std::endl;
		for (auto iter = m_Vector4FParams.begin(); iter != m_Vector4FParams.end(); iter++)
			std::cout << "Vector4F Param  " << iter->first << ":" << iter->second << std::endl;

		for (auto iter = m_IntParams.begin(); iter != m_IntParams.end(); iter++)
			std::cout << "Float Param  " << iter->first << ":" << iter->second << std::endl;
		for (auto iter = m_Vector2IParams.begin(); iter != m_Vector2IParams.end(); iter++)
			std::cout << "Vector2I Param  " << iter->first << ":" << iter->second << std::endl;
		for (auto iter = m_Vector3IParams.begin(); iter != m_Vector3IParams.end(); iter++)
			std::cout << "Vector3I Param  " << iter->first << ":" << iter->second << std::endl;
		for (auto iter = m_Vector4IParams.begin(); iter != m_Vector4IParams.end(); iter++)
			std::cout << "Vector4I Param  " << iter->first << ":" << iter->second << std::endl;

		
	}
private:

	std::string m_DataPath;
	std::string m_sPivotName;

	std::map<std::string, AxFp32> m_FloatParams;
	std::map<std::string, AxVector2> m_Vector2FParams;
	std::map<std::string, AxVector3> m_Vector3FParams;
	std::map<std::string, AxVector4> m_Vector4FParams;
	std::map<std::string, AxInt32> m_IntParams;
	std::map<std::string, AxVector2I> m_Vector2IParams;
	std::map<std::string, AxVector3I> m_Vector3IParams;
	std::map<std::string, AxVector4I> m_Vector4IParams;
	AxUInt32 numTasks()
	{
		AxGeometry* pivotGeo = m_SimGeoList[0];
		if (pivotGeo == nullptr)
		{
			std::cout << "ERROR EXE" << std::endl;
			return 0;
		}
		switch (m_ExcType)
		{
		case AlphaCore::kExcPoints:
			return pivotGeo->GetNumPoints();
			break;
		case AlphaCore::kExcPrimitives:
			return pivotGeo->GetNumPrimitives();
			break;
		case AlphaCore::kExcVertices:
			return pivotGeo->GetNumVertex();
			break;
		case AlphaCore::kExcVoxels:
			return pivotGeo->GetNumVoxels(m_sPivotName);
			break;
		case AlphaCore::kExcTaskGroup:
			return 0;
			break;
		case AlphaCore::kExcSBPoint2PointLink:
			return pivotGeo->GetNumPoints();
			break;
		case AlphaCore::kExcSBPrim2PrimLink:
			return pivotGeo->GetNumPrimitives();
			break;
		default:
			break;
		}
		return 0;
	}

	AlphaCore::AxExecuteType m_ExcType;
	AxGeometry* m_SimGeoList[32];

};

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
	
	//by_hy
	void SetCookTimes(AxInt32 CookTimes) { m_iCookTimes = CookTimes; };
	//by_hy

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