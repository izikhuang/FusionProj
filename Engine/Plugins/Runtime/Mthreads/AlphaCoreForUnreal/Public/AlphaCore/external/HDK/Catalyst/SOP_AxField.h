#ifndef __SOP_AX_FIELD_H__
#define __SOP_AX_FIELD_H__

#include <SOP/SOP_Node.h>
#include <UT/UT_StringHolder.h>
#include <OP/OP_AutoLockInputs.h>

#include <AlphaCore.h>

static const char* UITag_FilePath		= "file_path";
static const char* UITag_ConstraintPath = "constraint_path";
static const char* UITag_DataName		= "data_name";
static const char* UITag_CreateFrame	= "create_frame";
static const char* UITag_StartFrame		= "start_frame";
static const char* UITag_CachePath		= "cache_path";
static const char* UITag_IsStaticObject = "is_static";
static const char* UITag_CacheFrameRange = "f";

static const char* UITag_SimCacheRead	= "sim_cache_read";
static const char* UITag_SimCacheWrite	= "sim_cache_write";
static const char* UITag_ClothCacheWeight = "cloth_cache_weight";

static const char* UITag_Stiffness	    = "stiffness_val";
static const char* UITag_StretchSt	    = "stretch_st_val";
static const char* UITag_ShearSt	    = "shear_st_val";
static const char* UITag_BendSt		    = "bend_st_val";
static const char* UITag_Thickness	    = "thickness_val";
static const char* UITag_SimRefFilePath = "sim_ref_path";
static const char* UITag_Substeps	    = "substeps";
static const char* UITag_Iteration	    = "iteration";
static const char* UITag_SolvingType    = "solving_type";
static const char* UITag_SimType	    = "sim_type";
static const char* UITag_Gravity	    = "gravity";
static const char* UITag_Wind		    = "wind";
static const char* UITag_Direction	    = "direction";
static const char* UITag_Scale		    = "scale";
static const char* UITag_Percentage     = "percentage";
static const char* UITag_Mass		    = "mass";
static const char* UITag_Operation      = "operation";
static const char* UITag_DistanceMin = "distance_min";
static const char* UITag_DistanceMax = "distance_max";
static const char* UITag_SelfCollision = "self_collision";

static const char* UITag_CollisionType = "collision_type";
static const char* UITag_Name = "name";
static const char* UITag_Cook = "cook";
static const char* UITag_FPS = "fps";

static const char* UITag_ForceUpdate = "force_update";
static const char* UITag_CMD = "cmd";
static const char* UITag_AutoOutputMark = "autoOutputCache";

class SOP_CatalystCORE : public SOP_Node
{
public:
	//static PRM_Template *buildTemplates();
	static OP_Node* myConstructor(OP_Network* net, const char* name, OP_Operator* op)
	{
		return new SOP_CatalystCORE(net, name, op);
	}

	static const UT_StringHolder theSOPTypeName;
	static PRM_Template          myTemplateList[];
	static int SaveLookFileCallback(void* data, int index, float time, const PRM_Template* tplate);


	void SaveCacheObject(OP_Context& context);

	std::string GET_CACHE_FILE_PATH(float t) { UT_String ret; evalString(ret, UITag_FilePath, 0, t); return ret.length() == 0 ? "" : ret.c_str(); };

	std::string GET_DATA_NAME() { UT_String ret; evalString(ret, UITag_DataName, 0, 0); return ret.length() == 0 ? "" : ret.c_str(); };
	int  GET_ROP_TYPE() { return evalInt("rd_type", 0, 0); };
	int  GET_CACHE_START_FRAME() { return evalInt(UITag_CacheFrameRange, 0, 0); };
	int  GET_CACHE_END_FRAME() { return evalInt(UITag_CacheFrameRange, 1, 0); };
	int  GET_CACHE_INC() { return evalInt(UITag_CacheFrameRange, 2, 0); };
	int  GET_CMD_MARK() { return evalInt(UITag_CMD, 0, 0); };
	int  GET_AUTO_OUTPUT_MARK() { return evalInt(UITag_AutoOutputMark, 0, 0); };

	bool CallFromCallback;

protected:

	virtual bool updateParmsFlags();

	SOP_CatalystCORE(OP_Network* net, const char* name, OP_Operator* op);
	virtual ~SOP_CatalystCORE() {}
	/// Since this SOP implements a verb, cookMySop just delegates to the verb.
	virtual OP_ERROR cookMySop(OP_Context& context) override;
	void readAlphaCoreCallback(OP_Context& context);


	std::vector<AxScalarFieldF32*> m_FieldsPool;

	void resizeFieldsPool(AxUInt32 size);

	class AxGeometry* m_GeoData;

	//GetParameter ? CodeGene?
	//Gene GET_NAME_DIFF_
	//SOP_CatalystCORE_PD
	//	simParam.Left = Right
	//  ParamChangeCallback?
	//
	class AxCatalystObject* m_CaSimObject;
};

#endif // !__SOP_AX_FIELD_H__
