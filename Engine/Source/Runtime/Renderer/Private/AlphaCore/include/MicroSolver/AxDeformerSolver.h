#include "AxMicroSolverBase.h"
#include <AxGeo.h>
class AxDeformerSolverParameterBlock
{
public:

    AxStringParam GeoDataName;
    AxIntParam Tasks;
    AxStringParam LinkPointsProperty;
    AxToggleParam ActiveCacheOutput;
    AxStringParam SaveCachePath;
    AxIntParam TargetGeometry;
    AxStringParam LinkPointWeightsProperty;


    void FromJson(std::string jsonRawCode);

    struct RawData
    {
        AxInt32 Tasks;
        AxUChar ActiveCacheOutput;
        AxInt32 TargetGeometry;
    };

    RawData GetRawDataInFrame(int frame);
    RawData GetRawDataFloatFrame(AxFp32 floatFrame);

};


class AxDeformerSolver :public AxMicroSolverBase
{
public:
	AxDeformerSolver() {};
	~AxDeformerSolver() {};

	static AxMicroSolverBase* SolverConstructor();
	virtual void DeserilizationFromJson(std::string raw);
	virtual void SerilizationToJson(std::string raw);
	virtual void PrintParamInfo();

protected:
	virtual void onUpdate(AxFp32 dt);
	virtual void onInit(AxFp32 dt);

	void updateDeformer(AxFp32 dt);
	void updateDeformerDevice(AxFp32 dt);

	AxDeformerSolverParameterBlock m_DeformerParam;
	AxIdxMapUI32 m_LinkedPoints;
	AxIdxMapFp32 m_LinkedPointWeights;

 };
