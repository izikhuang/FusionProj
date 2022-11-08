#ifndef __ALPHA_CORE_MACRO_H__
#define __ALPHA_CORE_MACRO_H__

/*
 * Avoid using uint64.
 * The extra bit of precision is NOT worth the cost in pain and suffering
 * induced by use of unsigned.
 */
#if defined(WIN32)

	typedef __int64				Int64;
	typedef unsigned __int64	uInt64;

#elif defined(MBSD)

	// On MBSD, int64/uint64 are also defined in the system headers so we must
	// declare these in the same way or else we get conflicts.
	#include <stdint.h>
	typedef int64_t				Int64;
	typedef uint64_t			uInt64;

#elif defined(AMD64)

	typedef long				Int64;
	typedef unsigned long		uInt64;

#else

	typedef long long			Int64;
	typedef unsigned long long	uInt64;

#endif

/*
 * The problem with int64 is that it implies that it is a fixed 64-bit quantity
 * that is saved to disk. Therefore, we need another integral type for
 * indexing our arrays.
 */

#include <iostream>


#define AX_INVALID_INT32 0xffffffff 

#ifdef ALPHA_CUDA
#include <cuda_runtime.h>
#define ALPHA_SHARE_FUNC inline __device__ __host__ 
#define ALPHA_CUDA_THREAD_ID_INT24  __umul24(blockIdx.x, blockDim.x) + threadIdx.x
#define ALPHA_CUDA_THREAD_ID_INT32 blockIdx.x*blockDim.x + threadIdx.x
#define ALPHA_SPMD_FUNC
#define ALPHA_SPMD_CLASS
#define ALPHA_DEVICE_CONSTANT __constant__
#else
#define ALPHA_SHARE_FUNC inline 
#define ALPHA_SPMD_FUNC 
#define ALPHA_SPMD_CLASS
#define	ALPHA_DEVICE_CONSTANT
#endif

#ifdef ALPHA_UNREAL
#define RENDER_FOR_UE
#define RENDER_USE_RGBA
#else
#define RENDER_USE_RGBA8
#endif


#define AX_FOR_I(n)			for (AxInt32 i = 0; i < n; ++i)
#define AX_FOR_J(n)			for (AxInt32 j = 0; j < n; ++j)
#define AX_FOR_K(n)			for (AxInt32 k = 0; k < n; ++k)
#define AX_FOR_RANGE_I(s,n)	for (int i = s; i < n; ++i)
#define AX_FOR_RANGE_J(s,n)	for (int j = s; j < n; ++j)

#define ALPHA_EMPTY_FUNCTION_LOG	//std::cout << __FUNCTION__ << "  Empty " << std::endl
#define ALPHA_DEGREE_TO_RADIUS 0.017453292519444f
#define ALPHA_ENTROPY -1
#define ALPHA 137
#define AX_MAX_BIT   0xFFFFFFFF

#define AX_TIME_SCALE_CONSTANT 24.0f

#define AX_PI 3.14159265358979f

#define A137_API  extern "C" __declspec(dllexport)
#define FSA_CLASS

#define AX_INT_MENU enum

#define A137FUNCTION(...)
#define A137_FSA_FUNCTION
//
// AlphaCore use the 'dynamic properties' architecture design 
//
//

#define ALPHA_CLASS
#define AX_CACHE_HEAD_TOKEN 128
#include <omp.h>

#define AX_OMP 1
#define AX_KERNEL_DEBUG_LAST_ERROR 1
#define AX_KERNEL_DEBUG_FORCE_SYNC_CUDA 1
#define AX_KERNEL_DEBUG_PARAM_SAVE_FLUIDUTILITY 0

namespace AlphaCore
{
	template<typename T>
	ALPHA_SHARE_FUNC void Swap(T& a, T&b)
	{
		T tmp = a; a = b; b = tmp;
	}
}


namespace AlphaProperty
{
	static const char* EngineInfo = 
		" --------------------------------------------------------------------------------------------\n"
		" 																							\n"
		" 																							\n"
		"     //\\\\   ||     ||======|| ||      ||   //\\\\    =======  ========  ||=====\\\\  =========	\n"
		"    //==\\\\  ||     ||======|| ||======||  //==\\\\   ||       ||    ||  ||=====//  ||=====	\n"
		"   //    \\\\ ====== ||         ||      || //    \\\\  =======  ========  ||     \\\\  =========	\n"
		"																							\n"
		" \n"
		"                       阿尔法内核  アルファコア  [ version alpha 0.0.1 ]						\n"
		"  \n"
		"                             Get started breaking the row ...								\n"
		" 																							\n"
		" --------------------------------------------------------------------------------------------\n"
		"\n";
	static const char* Name = "name";
	static const char* Position			 = "P";
	static const char* PrdP				 = "PrdP";
	static const char* PrevP			 = "PrevP";
	static const char* LastP             = "LastP";
	static const char* AirResist		 = "airresist";

	static const char* PtPrevV			 = "PrevV";
	static const char* PtLastV			 = "LastV";
	static const char* PtVel			 = "v";
	static const char* PtNormal			 = "N";
	static const char* UV				 = "uv";

	static const char* PtAccelerate		 = "accel";
 	static const char* Stiffness		 = "stiffness";
	static const char* RestLength		 = "restlength";
	static const char* Mass				 = "mass";
	static const char* PrimDepth		 = "primDepth";
	static const char* PtMaxEta			 = "ptMaxEta";
	static const char* PrimMaxEta		 = "primMaxEta";
	static const char* RTriangle		 = "RTriangle";


	static const char* AdvectTmp		 = "advectTmp";
	static const char* AdvectTmp2		 = "advectTmp2";

	static const char* DensityField  	 = "density";
	static const char* DensityField2	 = "density2";
  	static const char* VelField			 = "vel";
	static const char* VelField2		 = "vel2";
 	static const char* TempratureField	 = "temperature";
	static const char* TempratureField2  = "temperature2";
 	static const char* DivregenceField   = "divergence";
	static const char* VelDivField		 = "velDiv";
	static const char* EulerRotate		 = "Euler";

	static const char* DivregenceField2  = "divergence2";
 	static const char* CurlField		 = "curl";
	static const char* CurlMagField		 = "curlMag";
	static const char* CurlField2		 = "curl2";
 	static const char* PressureField	 = "presure";
	static const char* PressureField2	 = "presure2";
	static const char* GradientField     = "gradient";
	static const char* GradientField2    = "gradient2";
										 
	static const char* BurnField		 = "burn";
	static const char* HeatField		 = "heat";
	static const char* HeatField2		 = "heat2";
	static const char* TempDivField		 = "tempDiv";
	static const char* ExplosiveDivField = "expDiv";
	static const char* FuelField	     = "fuel";

	static const char* PrimType			 = "primType";
	static const char* PrimitiveList	 = "prim2PtMap";
	static const char* ToplogyIndices	 = "indices";
	static const char* Point2PrimMap	 = "pt2PrimMap";
	static const char* Point2PrimIndices = "pt2PrimIndices";

	static const char* BvMax			= "bv_max";
    static const char* BvMin			= "bv_min";
	static const char* BvMaxEdge		= "bv_max_edge";
	static const char* BvMinEdge		= "bv_min_edge";
	static const char* BvMaxPoint		= "bv_max_pt";
	static const char* BvMinPoint		= "bv_min_pt";
    static const char* BvLeft			= "bv_NodeLeft";
    static const char* BvRight			= "bv_NodeRight";
    static const char* BvDepth			= "bv_NodeDepth";
	static const char* BvParent			= "bv_parent";
	static const char* BvSortedID		= "bv_sorted_id";
	static const char* BvNodes			= "bv_nodes";
	static const char* PointRadius		= "pscale";
	static const char* SPMDGroup		= "axSPMDGroup";
	static const char* EdgeColoring		= "color";
	static const char* MortonCode32		= "mc32";
	static const char* MortonCode64		= "mc64";
	static const char* TargetVelocity	= "target_v";
	static const char* WindVelocity     = "target_v";

	static const char* WindNormalDrag	= "dragnormal";
	static const char* WindTangentDrag	= "dragtangent";
	static const char* P2PMap			= "p2pMap2I";
	static const char* P2PIndices		= "p2pIndices";
	static const char* P2VMap			= "p2vMap2I";
	static const char* P2VIndices		= "p2vIndices";
	static const char* Inertia			= "inertia";
	static const char* RestPosition		= "restP";

	static const char* Point2VolumePtsMap = "point2VolumePtsMap";
	static const char* Point2VolumePtsIndices = "point2VolumePtsIndices";

	static const char* PtBaryCoord = "primUVW";
	static const char* PtResamplePrimId = "primID";
	static const char* PtDisplacement = "ptDisplacement";
	static const char* OriginalMesh = "originalMesh";

	static const char* Ptxform = "__xform";
	static const char* SortedPtId = "sortedPtId";
	static const char* Pt2CellId = "pt2CellId";

	namespace Collision
	{
		static const char* ContactPtFixVec4 = "ctxFix4";
		static const char* CollisionNormal = "ctxNormal";
		static const char* ContactIndices = "ctxIndices";
		static const char* ContactIdenity = "ctxIdt";
		static const char* Point2ContactVertexStart = "pt2CtxVtxStart";
		static const char* Point2ContactVertexEnd = "pt2CtxVtxEnd";
		static const char* CollisionMove = "collisionMove";
		static const char* CollisionTask = "collisionTask";
	}

	namespace AccelTree 
	{
		static const char* Cell2PtStart = "cell2PtStart";
		static const char* Cell2PtEnd = "cell2PtEnd";
	}

	namespace Constraint
	{
		static const char* Stiffness	 = "stiffness";
		static const char* CompStiffness = "compressstiffness";
		static const char* DampingRatio  = "dampingratio";
		static const char* RestLength	 = "restlength";
		static const char* Lambda		 = "L";
		static const char* TypeId		 = "axSolidConstraintType";
		static const char* AttachPos	 = "attachPos";
		static const char* AttachUVW	 = "target_uv";
		static const char* AttachId3	 = "attachPtId3";
		static const char* InvDm2x2		 = "invDm2x2";
		static const char* InvDm3x3		 = "invDm3x3";
		static const char* RestVector	 = "restvector";
 		static const char* Orient		 = "orient";
		static const char* PrevOrient	 = "orientprevious";
		static const char* Omega		 = "w";
		static const char* PrevOmega	 = "wprevious";
		static const char* PtVolume      = "volume";
		static const char* PressureGradient = "pressuregradient";
		static const char* VolumePts	 = "volumepts";
	}

	namespace Event
	{
		static const char* PreVelIntegrate = "preVelIntegrate";
		static const char* PostVelIntegrate = "postVelIntegrate";
		static const char* PreVelocityUpdate = "preVelocityUpdate";
		static const char* OnSubstepEnd = "onSubstepEnd";

	}

	namespace MicroSolver
	{
		//static const char* WindForce = "windForceSolver";
		//static const char* WindField = "windFieldSolver";
		//static const char* Blur	     = "blur";

		//static const char* Lap
		//工厂模式加
		//Msg模式
		//SimObject Msg
		//Init Computation Graph
		//Param 需要是buffer吗？？
		//Component
		//SolverComponent
		//
	}
}



#endif