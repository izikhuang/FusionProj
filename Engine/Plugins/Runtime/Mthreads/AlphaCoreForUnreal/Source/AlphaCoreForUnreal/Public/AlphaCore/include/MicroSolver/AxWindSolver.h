#ifndef __AX_WIND_SOLVER_H__
#define __AX_WIND_SOLVER_H__

#include "AxMicroSolverBase.h"
#include "ProceduralContent/AxNoise.DataType.h"
namespace AlphaCore
{
	namespace NameToken
	{
		namespace MicroSolver
		{
			namespace WindForce
			{
				static const char* WindNoiseIntensity = "windNoiseIntensity";
				static const char* SwirlSize = "swirlSize";
				static const char* PushLength = "pushLength";
				static const char* AirResistance = "airResistance";
				static const char* WindNoiseAttenuation = "windNoiseAttenuation";
				static const char* WindNoiseRoughness = "windNoiseRoughness";

				static const char* WindTurblenceLevel = "windTurblenceLevel";
				static const char* SwirlScale = "swirlScale";
				static const char* WindVelocity = "windVelocity";
				static const char* Offset4 = "offset4";
				static const char* SoftMaskExtensions = "softMaskExtensions";
				static const char* MaskPoints = "maskPoints";
				static const char* InverseSelection = "inverseSelection";
			}
		}
	}
}

class AxWindForceParamBlock
{
public:
	AxVector4FParam Offset4;
	AxFloatParam WindNoiseIntensity;
	AxFloatParam PushLength;
	AxFloatParam WindNoiseAttenuation;
	AxFloatParam WindNoiseRoughness;
	AxVector3FParam SwirlScale;
	AxIntParam SoftMaskExtensions;
	AxFloatParam SwirlSize;
	AxVector3FParam WindVelocity;
	AxFloatParam AirResistance;
	AxStringParam MaskPoints;
	AxStringParam MaskAttriuteName;
	AxIntParam WindTurblenceLevel;

	void FromJson(std::string jsonRawCode);

	struct RawData
	{
		AxVector4 Offset4;
		AxFp32 WindNoiseIntensity;
		AxFp32 PushLength;
		AxFp32 WindNoiseAttenuation;
		AxFp32 WindNoiseRoughness;
		AxVector3 SwirlScale;
		AxInt32 SoftMaskExtensions;
		AxFp32 SwirlSize;
		AxVector3 WindVelocity;
		AxFp32 AirResistance;
		AxInt32 WindTurblenceLevel;
	};

	RawData GetRawDataIntFrame(int frame);
	RawData GetRawDataFloatFrame(AxFp32 floatFrame);

	AxCurlNoiseParam GetCurlNoiseParamFloatFrame(AxFp32 floatFrame);
	AxCurlNoiseParam GetCurlNoiseParamIntFrame(AxInt32 frame);

private:

	AxCurlNoiseParam asCurlNoiseParam(RawData& rawData);

};
typedef AxWindForceParamBlock AxWindForcePB;


namespace AlphaCore
{
	namespace MicroSolver
	{
		AxWindForceParamBlock MakeDefaultWindSolverParam();
	}
}

class AxWindVisualizationGeometry : public AxGeometry//AxVisualziationGeo
{
public:
	//
	virtual void DrawVisualization();
	virtual void DrawVisualizationDevice();
	//
	//	procedural content
	//	Grid Box PointField Sphere
	//
	AxWindForceParamBlock WindParam;
};

class AxWindForceSolver :public AxMicroSolverBase
{
public:
	AxWindForceSolver() {};
	~AxWindForceSolver() {};

	static AxMicroSolverBase* SolverConstructor();
	virtual void DeserilizationFromJson(std::string raw);
	virtual void SerilizationToJson(std::string raw);
	virtual void PrintParamInfo();

protected:
	virtual void onUpdate(AxFp32 dt);
	virtual void onInit(AxFp32 dt);

	void updateWind(AxFp32 dt);
	void updateWindDevice(AxFp32 dt);
	AxWindForceParamBlock m_WindParam;
	AxBufferV3* m_TargetVelocityProp;
	AxBufferV3* m_PositionInputProp;
	AxBufferF* m_AirResistProp;
	AxBufferF* m_MassProp;
	AxBufferF* m_WindMaskProp;


};





class AxWindForceFieldSolver :public AxMicroSolverBase
{
public:
	AxWindForceFieldSolver() {};
	~AxWindForceFieldSolver() {};

	static AxMicroSolverBase* SolverConstructor();

	virtual void SerilizationToJson(std::string raw) {};
	virtual void DeserilizationFromJson(std::string raw) {};
protected:
	AxWindForceParamBlock m_WindParam;

};



#endif
