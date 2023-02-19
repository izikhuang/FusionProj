#ifndef __AX_KERNEL_TRACE_SYSTEM_H__
#define __AX_KERNEL_TRACE_SYSTEM_H__

#include "AxMacro.h"
#include "AxDataType.h"
#include "AxTimeTick.h"
#include "Collision/AxCollision.DataType.h"
#include "AccelTree/AxAccelTree.DataType.h"
#include "FluidUtility/AxFluidUtility.DataType.h"
#include "GridDense/AxFieldBase3D.h"
#include "AccelTree/AxSpatialHash.h"
#include "Utility/AxIO.h"
#include <string>
#include <map>
#include <vector>


struct AxKernelDebugItem
{
	AxInt32 ParamID;
	std::string StackIDStr;
	std::string FileName;
	std::string ParamName;
	std::string TypeName;
	std::string FuncName;
 	std::string EvalLinkDataFileName(std::string otherPlatform = "CUDA");
};



inline std::ostream& operator << (std::ostream& out, const AxKernelDebugItem& item)
{
	std::string idStr = std::to_string(item.ParamID);
	std::stringstream sstr;
	sstr << std::setfill(' ') << std::setw(5 - idStr.size()) << "";

	std::stringstream endTypeName;
	AxInt32 alg20 = 20 - item.TypeName.size();
	AxInt32 salg20 = std::max(0, alg20);
	endTypeName << std::setfill(' ') << std::setw(salg20) << "";

	std::stringstream endParamName;
	AxInt32 alg24 = 24 - item.ParamName.size();
	AxInt32 salg24 = std::max(0, alg24);
	endParamName << std::setfill(' ') << std::setw(salg24) << "";

	out << "ID:" << idStr + sstr.str() << " | " << item.TypeName + endTypeName.str() << "|" << item.ParamName + endParamName.str() << "|";
	return out;
}


#define AX_DOP_EXECUTABLE1 

class AxKernelTraceSystem
{
public:
	static AxKernelTraceSystem* GetInstance();

	//
	// STACK + DEVICE + [ TYPE ] + __FUNCTION__ + PARAM_NAME + . aXargs
	//
	void SaveDebugParam(AxAABB box, std::string paramName, std::string funcName)
	{
		if (!m_bActive)
			return;
#ifdef AX_DOP_EXECUTABLE

		std::string r = this->compName(box, paramName, funcName, this->PushParamRecode());
		std::cout << "r:" << r << std::endl;
		if (r.size() == 0)
			return;

		std::ofstream ofs(this->GetStoragePath() + r, std::ios::binary);
		ofs.close();
#else


#endif
	}

	void SaveDebugParam(AxSpatialHash::RAWDesc hash, std::string paramName, std::string funcName)
	{
		if (!m_bActive)
			return;
#ifdef AX_DOP_EXECUTABLE
		
		std::string r = this->compName(hash, paramName, funcName, this->PushParamRecode());
		std::cout << "r:" << typeid(hash).name() << " :::: WPath" << r << std::endl;
		if (r.size() == 0)
			return;

		std::ofstream ofs(this->GetStoragePath() + r, std::ios::binary);
		ofs.close();
#else
		
#endif
	}

	void SaveDebugParam(const AxVector3& vec3, std::string paramName, std::string funcName)
	{
		if (!m_bActive)
			return;
#ifdef AX_DOP_EXECUTABLE

		std::string r = this->compName(vec3, paramName, funcName, this->PushParamRecode());
		std::cout << "r:" << typeid(vec3).name() << " :::: WPath" << r << std::endl;
		if (r.size() == 0)
			return;

		std::ofstream ofs(this->GetStoragePath() + r, std::ios::binary);
		ofs.close();

#else

#endif
	}

	template<typename T>
	void SaveDebugParam(AxStorage<T>* storage, std::string paramName, std::string funcName)
	{
		if (!m_bActive)
			return;
#ifdef AX_DOP_EXECUTABLE

		if (storage->HasDeviceData())
			storage->LoadToHost();
		std::string r = this->compName(storage, paramName, funcName, this->PushParamRecode());
		std::cout << "r:" << typeid(storage).name() << " :::: WPath" << r << std::endl;

#else
		//AxStorageBase::AlignDeviceData
#endif
		//storage->SaveRaw();
	}


	void SaveDebugParam(const bool& e, std::string paramName, std::string funcName)
	{
		if (!m_bActive)
			return;

#ifdef AX_DOP_EXECUTABLE

		std::string r = this->compName(e, paramName, funcName, this->PushParamRecode());
		std::cout << "r:" << typeid(e).name() << " :::: WPath" << r << std::endl;
		if (r.size() == 0)
			return;
		std::ofstream ofs(this->GetStoragePath() + r, std::ios::binary);
		ofs.close();

#else

#endif

	}

	void SaveDebugParam(const AxFp32& fp, std::string paramName, std::string funcName)
	{
		if (!m_bActive)
			return;

#ifdef AX_DOP_EXECUTABLE

		std::string r = this->compName(fp, paramName, funcName, this->PushParamRecode());
		std::cout << "r:" << typeid(fp).name() << " :::: WPath" << r << std::endl;
		if (r.size() == 0)
			return;
		std::ofstream ofs(this->GetStoragePath() + r, std::ios::binary);
		ofs.close();
#else


#endif
	}

	template<typename T>
	void SaveDebugParam(AxField3DBase<T>* field, std::string paramName, std::string funcName)
	{
		if (!m_bActive)
			return;
#ifdef AX_DOP_EXECUTABLE

		std::string r = this->compName(field, paramName, funcName, this->PushParamRecode());
		std::cout << "r:" << typeid(field).name() << " :::: WPath" << r << std::endl;
		if (r.size() == 0)
			return;
		if (field->HashDeviceData())
			field->LoadToHost();
		field->Save(this->GetStoragePath() + r);
#else


#endif
	}

	template<typename T>
	void SaveDebugParam(AxVectorField3DBase<T>* field, std::string paramName, std::string funcName)
	{
		if (!m_bActive)
			return;
#ifdef AX_DOP_EXECUTABLE

		std::string r = this->compName(field, paramName, funcName, this->PushParamRecode());
		std::cout << "r:" << typeid(field).name() << " :::: WPath" << r << std::endl;
		if (r.size() == 0)
			return;
		if (field->HashDeviceData())
			field->LoadToHost();
		field->Save(this->GetStoragePath() + r);
#else
		
#endif
	}

	void SaveDebugParam(AlphaCore::FluidUtility::Param::AxCombustionParam comb, std::string paramName, std::string funcName)
	{
		if (!m_bActive)
			return;
		std::string r = this->compName(comb, paramName, funcName, this->PushParamRecode());
		std::cout << "r:" << typeid(comb).name() << " :::: WPath" << r << std::endl;
		if (r.size() == 0)
			return;

		std::ofstream ofs(this->GetStoragePath() + r, std::ios::binary);
		ofs.close();
	}

	std::string GetStackIDStr();


	void PushStackRecode(std::string stackName = "");
	AxUInt32 PushParamRecode(std::string stackName = "");

	void ClearStackRecode();

	void AutoDataTracing(std::string path,std::string baseLine,std::string ground);


	bool DataComparingField(
		std::string baseFilename,
		std::string compareFilename,
		AxFp32 tolerance = 0.01f);

	

	bool DataComparingVecField(std::string baseFilename, std::string compareFilename, AxFp32 tolerance = 0.01f);

	std::string GetStoragePath()
	{
		return m_sStoragePath;
	}
	void SetActive(bool e)
	{
		m_bActive = e;
	}
	void SetStoragePath(std::string path)
	{
		if (path[path.size() - 1] != '/')
			path += "/";
		m_sStoragePath = path;
	}
protected:

	bool scalarFieldComparing(AxScalarFieldF32* assignFieldA, AxScalarFieldF32* assignFieldB, AxFp32 tolerance = 0.01f);

	std::string evalDeviceName(std::string funcName)
	{
		if (funcName.find("::CUDA::") != std::string::npos)
			return "CUDA";
		return "CPU";
	}

	template<typename T>
	std::string compName(T arg, std::string paramName, std::string funcName, AxUInt32 paramId)
	{
		// STACK + DEVICE + [ TYPE ] + __FUNCTION__ + PARAM_NAME + . aXargs
		std::string typeName = AlphaCore::TypeName<T>();
		if (typeName == "INVALID_TYPE")
		{
			AX_ERROR("DebugSystem can not parser Parameter  func : {} @ {}", funcName, paramName);
			return "";
		}
		std::string  stack = this->GetStackIDStr();
		std::string deviceName = evalDeviceName(funcName);
		std::string funcDescRaw = AlphaUtility::ReplaceString2(funcName, "::CUDA", "");
		//std::cout << "funcDescRaw:" << funcDescRaw << std::endl;
		std::string funcDesc = AlphaUtility::ReplaceString2(funcDescRaw, "::", "__");
		std::string prmId = std::to_string(paramId);
		//
		return stack + "." + prmId + "." + deviceName + "." + typeName + "." + funcDesc + ".__" + paramName + "__.aXargs";
	}



private:
	AxKernelTraceSystem();
	~AxKernelTraceSystem();

	static AxKernelTraceSystem* m_Instance;
	std::vector<std::string> m_CallStackNames;
	std::vector<std::string> m_ParamStackNames;
	std::string m_sStoragePath;

	bool m_bActive;

	AxScalarFieldF32 m_ScalarFieldBase;
	AxScalarFieldF32 m_ScalarFieldOther;
	
	AxVecFieldF32 m_VectorFieldBase;
	AxVecFieldF32 m_VectorFieldOther;
};

#define AX_DEBUG_SAVE_PARAM(arg,paramName,func)  AxKernelTraceSystem::GetInstance()->SaveDebugParam(arg,paramName,func)
#define AX_DEBUG_LOAD_PARAM(arg,paramName) 
#endif