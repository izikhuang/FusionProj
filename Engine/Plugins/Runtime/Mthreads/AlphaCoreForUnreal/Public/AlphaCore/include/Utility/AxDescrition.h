#ifndef __ALPHA_CORE_DESC_H__
#define __ALPHA_CORE_DESC_H__

#include "Utility/AxStorage.h"
#include "Math/AxVectorBase.h"
#include "AxMacro.h"
#include <iomanip>

namespace AlphaCore
{
	AX_INT_MENU LinearSolver
	{
		CG				= 0,
		AMG				= 1,
		Jacobi			= 2,
		GaussSeidel		= 3,
		DirectLLT		= 4
	};

	AX_INT_MENU Preconditioner
	{
		LU,
		IncompleteCholesky,
		JacobiPrecond
	};

	
	namespace Param
	{
		/*
		struct AxAdvectInfo
		{
			AlphaCore::Flow::AdvectTraceMethod AdvtTraceType;
			AlphaCore::Flow::AdvectType		   AdvtType;
			AlphaCore::Flow::AdvectClamp	   AdvtClamp;
			float CFLCondition;
		};
		*/

	};

	namespace Desc
	{
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
			bool		 Active;
			AxVector3	 Pivot;
  			float		 Intensity;
			AxColorRGBA  LightColor;
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
			AxUInt32	 MaxInterations;
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