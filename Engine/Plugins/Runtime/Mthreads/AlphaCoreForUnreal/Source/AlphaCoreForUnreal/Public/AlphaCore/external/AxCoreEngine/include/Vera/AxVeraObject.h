#ifndef __AX_VERA_OBJECT_H__
#define __AX_VERA_OBJECT_H__

#include "AxSimObject.h"
#include "Collision/AxCollision.DataType.h"
#include "SolidUtility/AxSolidUtility.h"
#include "SolidUtility/AxSolidUtility.DataType.h"
#include "PBD/AxPBD.h"

#include "Utility/AxParameter.h"
#include "AccelTree/AxBVHTree.h"

struct AxVeraWindShape
{
	AxBufferV3* WindVelocity;
	AxBufferF*	WindAirresist;
	AxBufferF* NormalDrag;
	AxBufferF* TangentDrag;
};

struct AxVeraHResSimData
{
	AxBufferV3* ToCoarseUVBuf;
	AxBufferUInt32* ToCoarsePrimId;
	AxBufferV3* PtDisplacement;     // point attribute, used for wrinkle transfer
	AxBufferV3* RetargetingPos;   // point attribute, used for wrinkle transfer
	AxIdxMapUI32 Pt2PrimMap;
};




class VeraSolverParamter
{
public:
	AxIntParam Substeps;
	AxIntParam PostCollisionPasses;
	AxIntParam CollisionMethod;
	AxIntParam CollisionType;
	AxIntParam PBDIterations;
	AxIntParam CollisionPasses;
	AxFloatParam ContactSmoothIterSize;
	AxVector2IParam TaskFrameRange;
	AxStringParam ConstraintFilePath;
	AxToggleParam EnableCollision;
	AxStringParam CacheOutFileName;
	AxIntParam IntegratorType;
	AxIntParam Grouptype;
	AxIntParam LineSearchIterations;
	AxIntParam NewtwonSteps;
	AxVector3FParam Gravity;
	AxFloatParam ContactNormalDrag;
	AxFloatParam WindForceScale;
	AxStringParam SolverWorkPath;
	AxToggleParam EnableSelfCollision;
	AxIntParam PBDJacobiIterations;
	AxFloatParam DampingRate2;
	AxIntParam AttachType;
	AxStringParam TopologyFilePath;
	AxStringParam ExecuteCommand;
	AxIntParam ContactSmoothIterations;
	AxIntParam FPS;
	AxVector3FParam WindDirection;
	AxIntParam VelSmoothIterations;
	AxStringParam ColliderFilePath;
	AxIntParam ContactSet;
	AxFloatParam TimeScale;
	AxFloatParam ContactTangentDrag;
	AxIntParam SolverMehtodCore;
	AxStringParam AttachFilePath;
	AxFloatParam VelSmoothIterSize;
	AxFloatParam DampingRange;

 	AxToggleParam AttachGeoDebugEverySubstep;
	AxToggleParam ColliderGeoDebugEverySubstep;
	AxToggleParam ResultGeoDebugEverySubstep;
	AxToggleParam HasColliderGeo;//Need Butter Property
	AxToggleParam HasAttachGeo;

	AxToggleParam UseGroundCollision;
	AxVector3FParam GroundPosition;

	AxToggleParam AddDebugCapulse;
	AxFloatParam DebugCapulseRadius;
	AxFloatParam DebugCapulseHalfHeight;
	AxVector3FParam DebugCapulseDirection;
	AxVector3FParam DebugCapulsePivot;

	AxToggleParam AddDebugOBB;
	AxVector3FParam DebugOBBSize;
	AxVector3FParam DebugOBBPivot;
	AxVector3FParam DebugOBBForward;
	AxVector3FParam DebugOBBUp;

	AxStringParam DeformerMesh;

	AxIntParam CoarseMeshType;
	AxStringParam CoarseMeshFilePath;
	AxToggleParam ActiveHResMesh;

	void FromJson(std::string jsonRawCode);

	struct RawData
	{
		AxInt32 Substeps;
		AxInt32 PostCollisionPasses;
		AxInt32 CollisionMethod;
		AxInt32 CollisionType;
		AxVector2I TaskFrameRange;
		AxInt32 CollisionPasses;
		AxFp32 ContactSmoothIterSize;
		AxInt32 PBDIterations;
		AxInt32 IntegratorType;
		AxInt32 Grouptype;
		AxUChar EnableCollision;
		AxInt32 LineSearchIterations;
		AxInt32 NewtwonSteps;
		AxVector3 Gravity;
		AxFp32	ContactNormalDrag;
		AxFp32	WindForceScale;
		AxUChar EnableSelfCollision;
		AxInt32 PBDJacobiIterations;
		AxFp32	DampingRate2;
		AxInt32 AttachType;
		AxInt32 ContactSmoothIterations;
		AxInt32 FPS;
		AxVector3 WindDirection;
		AxInt32 VelSmoothIterations;
		AxInt32 ContactSet;
		AxFp32 TimeScale;
		AxFp32 ContactTangentDrag;
		AxInt32 SolverMehtodCore;
		AxFp32 VelSmoothIterSize;
		AxFp32 DampingRange;
		AxUChar AttachGeoDebugEverySubstep;
		AxUChar ColliderGeoDebugEverySubstep;
		AxUChar ResultGeoDebugEverySubstep;
    	AxUChar HasColliderGeo;
		AxUChar HasAttachGeo;
		AxUChar UseGroundCollision;
		AxVector3 GroundPosition;
	};

	RawData GetRawDataIntFrame(int frame);
	RawData GetRawDataFloatFrame(AxFp32 floatFrame);
};

class AxVeraObject : public AxSimObject
{
public:
	AxVeraObject();
	virtual ~AxVeraObject();

	virtual void ParamDeserilizationFromJson(std::string jsonPath);
	virtual void PrintRFSInfo();
	void SetSolverParameter(VeraSolverParamter veraParam);


	static AxSimObject* ObjectConstructor();
protected:

	void _init();
	void buildWindDragShape();
	void buildAxVeraHRes();

	virtual void OnInit();
	virtual void OnReset();
	virtual void OnPreSim(float dt);
	virtual void OnUpdateSim(float dt);
	virtual void OnPostSim(float dt);

#ifdef ALPHA_CUDA

	virtual void OnInitDevice();
	virtual void OnResetDevice();
	virtual void OnPreSimDevice(float dt);
	virtual void OnUpdateSimDevice(float dt);
	virtual void OnPostSimDevice(float dt);

#endif //

protected:
	VeraSolverParamter veraSolverParam;
	AlphaCore::SolidUtility::SimData::AxSolidData solidSelf;
	AlphaCore::SolidUtility::SimData::AxSolidData solidCollider;

	AlphaCore::PBD::SimData::AxPBDConstraintData pbdConstraintData;
 	AlphaCore::Collision::SimData::AxPBDCollisionResolveData pbdCollisionResolveData;

	AxSimObject* constraintGeo;
	AxSimObject* colliderGeo;
	AxSimObject* attachGeo;
	AxSimObject* coarseMeshGeo;

	//AxSimObject* coarseMeshGeo;

	//TODO:WindSOA
	AxVeraWindShape windShapeData;
	AxIdxMapUI32 m_P2PMapIdx;
	AxBVHTree m_SelfTree;
	AxBVHTree m_ColliderTree;
	AxSPMDTick m_ContactTick;
	AxBufferContact m_ContactPool;
	AxGeometry* m_TargetDeformMesh;

	AxVeraHResSimData m_HResMeshData;

private:
	void updateSceneGeometry(AxFp32 dt);
	void updateSceneGeometryDevice(AxFp32 dt);


	void postDebugCallback();
	void postDebugCallbackDevice();

	/// 
	/// hRes SimLogic
	/// 
	bool m_hResSim;
	void generalPreSim(AxFp32 dt);
	void generalPostSim(AxFp32 dt);

	void hResPreSim(AxFp32 dt);
	void hResPostSim(AxFp32 dt);

	void generalPreSimDevice(AxFp32 dt);
	void hResPreSimDevice(AxFp32 dt);

	std::vector<std::string> m_ColliderUpdateProperties;
	std::vector<std::string> m_AttachmentUpdateProperties;

	std::vector<std::string> m_CoarseMeshUpdateProperties;
};

#endif