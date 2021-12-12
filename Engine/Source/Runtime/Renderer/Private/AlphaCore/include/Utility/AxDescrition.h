#ifndef __ALPHA_CORE_DESC_H__
#define __ALPHA_CORE_DESC_H__

#include <Utility/AxStorage.h>
#include <Math/AxVectorBase.h>
#include <iomanip> 
#include <AxMacro.h>

namespace AlphaCore
{
	enum LinearSolver
	{
		CG				= 0x0000010,
		AMG				= 0x0000100,
		Jacobi			= 0x0001000,
		GaussSeidel		= 0x0010000,
		BiCGSTAP		= 0x0100000,
		DirectLLT		= 0x1000000
	};

	enum Preconditioner
	{
		LU,
		IncompleteCholesky,
		JacobiPrecond
	};

	namespace Flow
	{
		enum AdvectType
		{
			SemiLagrangian,
			MacCormack,
			BFECC
		};

		enum AdvectClamp
		{
			kAdvectNonClamp,
			kAdvectExtratClamp,
			kAdvectRevert,
		};

		enum AdvectTraceMethod
		{
			Trace,
			SingeStep,
			KR2,
			MidPoint
		};
 	}

	enum KernelDevice
	{
		CUDA,
		CPU,
		OpenGL,
		DirectX,
		PlayStation,
		iOSMeta
		
	};
	
	namespace Param
	{

		struct AxAdvectInfo
		{
			AlphaCore::Flow::AdvectTraceMethod AdvtTraceType;
			AlphaCore::Flow::AdvectType		   AdvtType;
			AlphaCore::Flow::AdvectClamp	   AdvtClamp;
			float CFLCondition;
		};

	};

	namespace Desc
	{

		struct AxField3DInfo
		{	
 			AxVector3	Pivot;
			AxVector3	VoxelSize;
			AxVector3	InvHalfVoxelSize; //todo : need implement
			AxVector3	FieldSize;		  //todo : need implement
			AxVector3UI	Resolution;

			int TOP;
			int DOWN;
			int LEFT;
			int RIGHT;
			int FRONT;
			int BACK;

		};

		struct AxField2DInfo
		{
			AxVector3	Pivot;
			AxVector2	VoxelSize;
			AxVector2	InvHalfVoxelSize; //todo : need implement
			AxVector2	FieldSize;		  //todo : need implement
			AxVector2UI	Resolution;

			int TOP;
			int DOWN;
			int LEFT;
			int RIGHT;
		};

		struct AxCameraInfo
		{
			AxVector3 Pivot;
			AxVector3 Euler;
			AxVector3 UpVector;
			AxVector3 Forward;
			float	  Fov;
 			float	  FocalLength;
			float	  Aperture;
			float	  Near;
			bool UseLookAt;
			bool UseFOV;
		};

		inline void PrintInfo(const char* head,const AxCameraInfo& cam)
		{
			printf("Camera Info : %s \n",head);
			printf("       Position		: %f , %f , %f \n", cam.Pivot.x, cam.Pivot.y, cam.Pivot.z);
			printf("       Euler		: %f , %f , %f \n", cam.Euler.x, cam.Euler.y, cam.Euler.z);
			printf("       FocalLength  : %f \n", cam.FocalLength);
			printf("       Aperture		: %f \n", cam.Aperture);
			printf("       Near			: %f \n", cam.Near);
			//printf();

		}

		struct AxPointLightInfo
		{
			AxVector3	 Pivot;
  			float		 Intensity;
			AxColorRGBA  LightColor;
		};

		struct AxTopologyData
		{
			AxBufferV3	   PosBuffer;
			AxBuffer2I	   Prim2PointMap;
			AxBufferUInt32 Indices;
			AxBufferI	   PrimTypeBuffer;
		};		

		struct AxBoundingBox
		{
			AxVector3	 Max;
			AxVector3	 Min;
		};
	}

	static AlphaCore::Desc::AxCameraInfo MakeDefaultCamera() 
	{ 
		AlphaCore::Desc::AxCameraInfo cam;
		cam.Pivot.x = 0;	cam.Pivot.y = 0;	cam.Pivot.z = 0;
		cam.Euler.x = 0;	cam.Euler.y = 0;	cam.Euler.z = 0;
		cam.Aperture = 41.4214f;
		cam.FocalLength = 50;
		cam.Near = 0.01f;
		cam.UseLookAt = false;
		return cam;
	};

	static float Fov2FocalLength(float fov, float aperture = 41.4214f)
	{
		return atan(90.0f - 0.5 * fov) * (aperture * 0.5f);
	}

	static float FocalLength2Fov(float focal, float aperture) {
		return atan(aperture / focal / 2.f) * 180 / 3.14;
	}

	namespace Param
	{
		struct AxCombustionParam
		{
			float IgnitionTemperature;
			float BurnRate;
			float FuelInefficiency;
			float TemperatureOutput;
			float GasRelease;
			float GasHeatInfluence;
			float GasBurnInfluence;
			float TempHeatInfluence;
			float TempBurnInfluence;

			bool FuelCreateSomke;


		};

		struct FlowSolverParam
		{
			LinearSolver SolverType;
			float		 RelativeTolerance;
			float		 CFLCondition;
			uInt32		 MaxInterations;
			float		 Dissipation;
			float		 Vorticity;

			AxCombustionParam Combustion;
		};


		static FlowSolverParam MakeDefaultFlowSolver()
		{
			FlowSolverParam parm;
			parm.SolverType = LinearSolver::Jacobi;

			parm.Combustion.BurnRate = 0.9f;
			parm.Combustion.GasRelease = 10;
			parm.CFLCondition = 1.5f;
			

			return parm;

		}

		static AxCombustionParam MakdeDefualtCombustionParam()
		{
			AxCombustionParam param;
			param.IgnitionTemperature = 0.1f;
			param.BurnRate			= 0.9f;
			param.FuelInefficiency  = 0.3f;
			param.TemperatureOutput = 0.3f;
			param.GasRelease		= 166.0f;
			param.GasHeatInfluence  = 0.2f;
			param.GasBurnInfluence  = 1.0f;
			param.TempHeatInfluence = 0.0f;
			param.TempBurnInfluence = 1.0f;
			return param;
		}

	}

}


#endif