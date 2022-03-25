#ifndef __AX_MICRO_SOLVER_BASE_H__
#define __AX_MICRO_SOLVER_BASE_H__

//param
#include <Utility/AxParameter.h>
#include <AxDataType.h>
#include <AxGeo.h>

namespace AlphaProperty
{
	namespace MicroSolver
	{
		static const char* WindForce = "windForceSolver";
		static const char* WindField = "windFieldSolver";
	}
}

typedef std::map<std::string, std::string> AxParamParentMap;

namespace AlphaCore
{
	namespace MicroSolver
	{
		struct ProtocolCode
		{
			std::string JsonParser;//"RapidJson"
			std::string ParameterBlockCPPCode;
			std::string SolverCPPCode;
		};

		bool GenerateCodeUseRapidJson(std::string jsonFilePath, std::string cppFilePath, AxParamParentMap parent = {});
		std::string SolverTypeNameTransfer(std::string solverType);
	}
}



class AxSimWorld;
class AxMicroSolverBase
{
public:
	AxMicroSolverBase();
	~AxMicroSolverBase();

	void Init(AxFp32 dt);
	void Update(AxFp32 dt);
	virtual void PrintSolverInfo();
	virtual void SerilizationToJson(std::string raw) {};
	virtual void DeserilizationFromJson(std::string raw) {};

	AxInt32 GetIntParameter(std::string name,AxInt32 frame);
	AxFp32 GetFloatParameter(std::string name, AxInt32 frame);
	AxVector2 GetVector2Parameter(std::string name, AxInt32 frame);
	AxVector3 GetVector3Parameter(std::string name, AxInt32 frame);
	AxVector4 GetVector4Parameter(std::string name, AxInt32 frame);
	std::string GetStringParameter(std::string name);
 
	void SetSimWorld(AxSimWorld* world) { m_OwnWorld = world; };
	void SetOwnGeoData(AxGeometry* geo) { m_GeoData = geo; };
	std::string GetSolverName() { return m_sName; };
	void SetName(std::string name) { m_sName = name; };
protected:
	virtual void onUpdate(AxFp32 dt) {};
	virtual void onInit(AxFp32 dt) {};

	void addParameter(std::string name,AxIParameterBase* param);
	std::map<std::string, AxIParameterBase*> m_ParameterMap;

	class AxSimWorld* m_OwnWorld;
	class AxGeometry* m_GeoData;


	std::string m_sName;

private:

	bool m_bInited;
	std::vector< AxMicroSolverBase*> m_ChildSolverStack;
	AxVector2I m_SimFrameRange;
};

struct AxFieldTrialVisualization
{
	AxToggleParam Active;
	AxFloatParam VisualizationIterations;
	AxIntParam VisualizationStepSize;
	AxFloatParam CFLCondition;
};

#endif