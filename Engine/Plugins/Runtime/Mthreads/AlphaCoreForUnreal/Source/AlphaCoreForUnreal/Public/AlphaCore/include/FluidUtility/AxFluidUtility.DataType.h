#ifndef __AX_FLUIDUTILITY_DATATYPE_H__
#define __AX_FLUIDUTILITY_DATATYPE_H__

#include "AxDataType.h"
#include "AxMacro.h"
#include "AxStorage.h"
#include "AxNoise.DataType.h"


enum AxUltraTurbType{
	kSimpleUTurb,
	kComplexUTurb,
	kHighFrequencyUTurb
};

enum AxFieldMixType
{
	kFieldSet,
	kFieldAdd,
	kFieldSubtract,
	kFieldMax,
	kFieldMin
};

struct AxUltraTurbulenceParam
{
	AxUltraTurbType Type;
	AxFp32 Intensity;
	AxVector2 ThresholdRange;
	AxFp32 UTrubFrequency;
	AxFp32 TimeFrequency;
	AxFp32 Roughness;
	AxFp32 RoughnessIncrease;
	AxInt32 MaxOctaves;
};



struct AxProceduralFluidEmitter
{
	AxVector3 Size;
	AxVector3 Pivot;
	AxVector3 Direction;
	AxFieldMixType MixType;
	//add
	//voxelScale 1.5
	AxFp32 RectThickness;
	AxFp32 Speed;
	AxFp32 EmitIntensity;

	//Noise discretization with voxel size
	//TODO : Dot not bake emitter noise ?
	AxCurlNoiseParam IntensityCurlNoise;
};

static AxUltraTurbulenceParam MakedefultUTurbParam()
{
	AxUltraTurbulenceParam ret;
	ret.Type = kComplexUTurb;
	ret.Intensity = 25;
	ret.ThresholdRange = MakeVector2(0.05f, 0.0f);
	ret.UTrubFrequency = 5.0f;
	ret.TimeFrequency = 5;
	ret.Roughness = 0.5f;
	ret.RoughnessIncrease = 2.1f;
	ret.MaxOctaves = 1;
	return ret;
}


namespace AlphaCore
{
	namespace FluidUtility
	{

		namespace Param
		{
			struct AxCombustionParam
			{
				AxFp32 IgnitionTemperature;
				AxFp32 BurnRate;
				AxFp32 FuelInefficiency;
				AxFp32 TemperatureOutput;
				AxFp32 GasRelease;
				AxFp32 GasHeatInfluence;
				AxFp32 GasBurnInfluence;
				AxFp32 TempHeatInfluence;
				AxFp32 TempBurnInfluence;
				bool FuelCreateSomke;
				AxFp32 DenseSmokeIntensity;
			};

			///*
			struct SHPCoherenceRAWDesc
			{
				AxInt32 NumPoints;
				AxVector3* posRawData = nullptr;
				AxFp32* massRawData = nullptr;
				AxFp32* volumeRawData = nullptr;
				AxFp32* densityRawData = nullptr;
				AxFp32* pressureRawData = nullptr;
				AxFp32* staticDensityRawData = nullptr;
				AxFp32* normOmegaRawData = nullptr;
				AxFp32* lambdaRawData = nullptr;
				AxVector3* accelerationRawData = nullptr;
				AxVector3* velocityRawData = nullptr;
				AxVector3* omegaRawData = nullptr;
				AxVector3* pressureAccelRawData = nullptr;
				AxVector3* deltaRawData = nullptr;
				AxVector3* oldPosRawData = nullptr;

				AxVector3* sortPosRawData = nullptr;
				AxFp32* sortMassRawData = nullptr;
				AxFp32* sortVolumeRawData = nullptr;
				AxFp32* sortDensityRawData = nullptr;
				AxFp32* sortPressureRawData = nullptr;
				AxFp32* sortStaticDensityRawData = nullptr;
				AxFp32* sortNormOmegaRawData = nullptr;
				AxFp32* sortLambdaRawData = nullptr;
				AxVector3* sortAccelerationRawData = nullptr;
				AxVector3* sortVelocityRawData = nullptr;
				AxVector3* sortOmegaRawData = nullptr;
				AxVector3* sortPressureAccelRawData = nullptr;
				AxVector3* sortDeltaPRawData = nullptr;
				AxVector3* sortOldPosRawData = nullptr;
			};


			//*/
			static AxCombustionParam MakdeDefualtCombustionParam()
			{
				AxCombustionParam param;
				param.IgnitionTemperature = 0.1f;
				param.BurnRate = 0.9f;
				param.FuelInefficiency = 0.3f;
				param.TemperatureOutput = 0.3f;
				param.GasRelease = 166.0f;
				param.GasHeatInfluence = 0.2f;
				param.GasBurnInfluence = 1.0f;
				param.TempHeatInfluence = 0.0f;
				param.TempBurnInfluence = 1.0f;
				param.DenseSmokeIntensity = 3.0f;
				return param;
			}
		}

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

		enum AdvectTraceType
		{
			ForwardEuler = 0,
			MidPoint = 1,
			RK3 = 2,
			RK4 = 3
		};
	}
}
#endif