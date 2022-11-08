#ifndef __AXFLUIDUTILITY__H__
#define __AXFLUIDUTILITY__H__

#include "GridDense/AxGridDense.h"
#include "AxParameter.h"
#include "AxNoise.DataType.h"

#define __FSA_ARCH_PROTOCOL_AXFLUIDUTILITY__H__ 1
namespace AlphaCore
{
	namespace FluidUtility
	{

		struct AxFlameColdDownParam {

			//AxFp32 ColddownTime;
			//bool UseControlField
			//Ax ControlFieldInfo
			//AxRamp ControlRamp;
		};

	}

}

class AxParticleFluidEmitter
{
public:
	AxParticleFluidEmitter(
		AxVector3 dir = MakeVector3(0.0f, 0.0f, 1.0f),
		AxVector3 up = MakeVector3(0.0f, 1.0f, 0.0f),
		AxVector3 size = MakeVector3(1.0f, 1.0f, 0.0f),
		AxVector3 pivot = MakeVector3(0.0f, 0.0f, 0.0f),
		AxFp32 rad = 0.1f,
		AxFp32 speed = 1.0f);
	~AxParticleFluidEmitter();

	struct RAWDesc
	{
		AxVector3UI Res;
		AxVector3 Forward;
		AxVector3 Right;
		AxVector3 Up;
		AxVector3 Size;
		AxVector3 Origin;
		AxFp32 Radius;
		AxFp32 Speed;
		AxUInt32 NumPoints;
		AxFp32 RestDensity;
		AxFp32 Mass;
		AxFp32 Viscosity;
	};

	RAWDesc GetEmitterRAWDesc();
	RAWDesc GetEmitterRAWDescFLIP(AxVector3 voxelSize, AxFp32 dt);

	void SetDirection(AxVector3 dir);
	void SetUp(AxVector3 up);
	void SetSpeed(AxFp32 speed);
	void Init(AxVector3 dir, AxVector3 up, AxVector3 size, AxVector3 pivot, AxFp32 rad, AxFp32 speed);


	void SetMass(AxFp32 mass) {
		m_fEmitterPointMass = mass;
	}

	void SetViscosity(AxFp32 viscosity) {
		m_fEmitterPointViscosity = viscosity;
	}

	void SetRestDensity(AxFp32 restDensity) {
		m_fEmitterPointRestDensity = restDensity;
	}

	void SetRadius(AxFp32 rad) {
		this->m_fRadius = rad;
	}
	void SetSize(AxVector3 size) {
		this->m_Size = size;
	}

	void SetSize(AxFp32 sx, AxFp32 sy, AxFp32 sz = 0.0f) {
		this->SetSize(MakeVector3(sx, sy, sz));
	}

	void SetCenter(AxVector3 center) {
		this->m_Center = center;
	}

	void SetCenter(AxFp32 cx, AxFp32 cy, AxFp32 cz) {
		this->m_Center = MakeVector3(cx, cy, cz);
	}

	void Tick(AxFp32 dt)
	{
		if (m_bNeedEmit)
			m_bNeedEmit = false;
		m_fCurrTickTime += dt;

		if (m_fCurrTickTime * m_fSpeed > m_fRadius * 2.0f)
		{
			std::cout << "Emit : " << m_fCurrTickTime * m_fSpeed << " m_fSpeed : " << m_fRadius * 2.0f << "  " << std::endl;
			m_fCurrTickTime = 0.0f;
			m_bNeedEmit = true;
		}
	}


private:

	AxVector3 m_Forward;
	AxVector3 m_Size;
	AxVector3 m_Up;
	AxVector3 m_Center;

	AxFp32 m_fRadius;
	AxFp32 m_fSpeed;
	AxFp32 m_fEmitterPointMass;
	AxFp32 m_fEmitterPointViscosity;
	AxFp32 m_fEmitterPointRestDensity;
	AxFp32 m_fCurrTickTime;
	bool m_bNeedEmit;

};

inline std::ostream& operator<<(std::ostream& out, const AxParticleFluidEmitter::RAWDesc& emitter)
{
	out << " Res        : " << emitter.Res << std::endl;
	out << " Forward    : " << emitter.Forward << std::endl;
	out << " Right      : " << emitter.Right << std::endl;
	out << " Up         : " << emitter.Up << std::endl;
	out << " Size       : " << emitter.Size << std::endl;
	out << " Origin     : " << emitter.Origin << std::endl;
	out << " Seperation : " << emitter.Radius << std::endl;
	out << " Speed      : " << emitter.Speed << std::endl;
	out << " NumPoints  : " << emitter.NumPoints << std::endl;
	out << " PointMass  : " << emitter.Mass << std::endl;
	out << " Viscosity  : " << emitter.Viscosity << std::endl;
	out << " ResDensity : " << emitter.RestDensity << std::endl;
	out << std::endl;
	return out;
}


namespace AlphaCore
{
	namespace FluidUtility
	{
		namespace CUDA
		{
			ALPHA_SPMD_FUNC void VorticityConfinementBlock(
				AxVecFieldF32* velField,
				AxVecFieldF32* forceField,
				bool useMaskField,
				AxScalarFieldF32* maskField,
				bool useMaskRamp,
				AxRampCurve32RAWData maskRamp,
				AxFp32 confinementScale,
				AxFp32 deltaTime,
				AxUInt32 blockSize = 1000);
		}
	}
}

#if __FSA_ARCH_PROTOCOL_AXFLUIDUTILITY__H__ 1
namespace AlphaCore
{
namespace FluidUtility  
{
 ALPHA_SPMD_FUNC void CombustionColdDown (
AxFp32 heatInfluence,
AxFp32 cooldownTime,
AxScalarFieldF32* heatField,
bool useMaskField,
AxScalarFieldF32* maskField,
AxRampCurve32RAWData maskRamp,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void UltraTurblenceScalar (
AxScalarFieldF32* targetField,
AxFp32 turblenceScale,
AxFp32 turblenceSize,
AxScalarFieldF32* thresholdField,
AxFp32 maxThreshold,
bool useMaskField,
AxScalarFieldF32* maskField,
bool useMaskRamp,
AxRampCurve32RAWData maskRamp,
AxFp32 maskRampWeight,
AxInt32 timeSeed,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void UltraTurblenceVector (
AxVecFieldF32* targetField,
AxFp32 turblenceScale,
AxFp32 turblenceSize,
AxScalarFieldF32* thresholdField,
AxFp32 maxThreshold,
bool useMaskField,
AxScalarFieldF32* maskField,
bool useMaskRamp,
AxRampCurve32RAWData maskRamp,
AxFp32 maskRampWeight,
AxInt32 timeSeed,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void UltraTurblenceBlockVector (
AxVecFieldF32* targetField,
AxScalarFieldF32* thresholdField,
AxVector2 thresholdRange,
bool useMaskField,
AxScalarFieldF32* maskField,
bool useMaskRamp,
AxRampCurve32RAWData maskRamp,
AxUltraTurbulenceParam ultraTurbParam,
AxFp32 time,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void FlameSplitTurblence (
AxScalarFieldF32* temperatureField,
AxVecFieldF32* outVelField,
AxFp32 scale,
AxFp32 fade,
AxFp32 pullSize,
AxFp32 pushSize,
AxFp32 temperatureThreshold,
AxFp32 gradientMagClamp,
bool activeFieldControl,
AxScalarFieldF32* controlField,
bool activeRampControl,
AxRampCurve32RAWData rampData,
AxFp32 rampWeight,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void CurlNoiseTurblence (
AxVecFieldF32* velocityField,
AxScalarFieldF32* thresholdField,
AxFp32 threshold,
AxScalarFieldF32* maskField,
bool useMaskRamp,
AxRampCurve32RAWData maskRamp,
AxCurlNoiseParam curlNoise,
AxFp32 forceScale,
AxFp32 time,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void Dissipation (
AxFp32 rate,
AxScalarFieldF32* targetField,
AxScalarFieldF32* maskField,
bool useDissipationRamp,
AxRampCurve32RAWData dissipationRamp,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void Attenuation (
AxFp32 rate,
AxScalarFieldF32* inputSclarField,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void TemperatureExpCoolingDown (
AxScalarFieldF32* temperatureField,
AxFp32 coolingRate,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void SimpleWind (
AxVecFieldF32* velField,
AxVector3 windDir,
AxFp32 windSpeed,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void WindForce (
AxVecFieldF32* velField,
AxVector3 windDir,
AxFp32 windSpeed,
AxFp32 windIntensity,
AxFp32 deltaTime) ;

  }

namespace FluidUtility  
{
  namespace CUDA 
{
 ALPHA_SPMD_FUNC void CombustionColdDown (
AxFp32 heatInfluence,
AxFp32 cooldownTime,
AxScalarFieldF32* heatField,
bool useMaskField,
AxScalarFieldF32* maskField,
AxRampCurve32RAWData maskRamp,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void UltraTurblenceScalar (
AxScalarFieldF32* targetField,
AxFp32 turblenceScale,
AxFp32 turblenceSize,
AxScalarFieldF32* thresholdField,
AxFp32 maxThreshold,
bool useMaskField,
AxScalarFieldF32* maskField,
bool useMaskRamp,
AxRampCurve32RAWData maskRamp,
AxFp32 maskRampWeight,
AxInt32 timeSeed,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void UltraTurblenceVector (
AxVecFieldF32* targetField,
AxFp32 turblenceScale,
AxFp32 turblenceSize,
AxScalarFieldF32* thresholdField,
AxFp32 maxThreshold,
bool useMaskField,
AxScalarFieldF32* maskField,
bool useMaskRamp,
AxRampCurve32RAWData maskRamp,
AxFp32 maskRampWeight,
AxInt32 timeSeed,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void UltraTurblenceBlockVector (
AxVecFieldF32* targetField,
AxScalarFieldF32* thresholdField,
AxVector2 thresholdRange,
bool useMaskField,
AxScalarFieldF32* maskField,
bool useMaskRamp,
AxRampCurve32RAWData maskRamp,
AxUltraTurbulenceParam ultraTurbParam,
AxFp32 time,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FlameSplitTurblence (
AxScalarFieldF32* temperatureField,
AxVecFieldF32* outVelField,
AxFp32 scale,
AxFp32 fade,
AxFp32 pullSize,
AxFp32 pushSize,
AxFp32 temperatureThreshold,
AxFp32 gradientMagClamp,
bool activeFieldControl,
AxScalarFieldF32* controlField,
bool activeRampControl,
AxRampCurve32RAWData rampData,
AxFp32 rampWeight,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CurlNoiseTurblence (
AxVecFieldF32* velocityField,
AxScalarFieldF32* thresholdField,
AxFp32 threshold,
AxScalarFieldF32* maskField,
bool useMaskRamp,
AxRampCurve32RAWData maskRamp,
AxCurlNoiseParam curlNoise,
AxFp32 forceScale,
AxFp32 time,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void Dissipation (
AxFp32 rate,
AxScalarFieldF32* targetField,
AxScalarFieldF32* maskField,
bool useDissipationRamp,
AxRampCurve32RAWData dissipationRamp,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void Attenuation (
AxFp32 rate,
AxScalarFieldF32* inputSclarField,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void TemperatureExpCoolingDown (
AxScalarFieldF32* temperatureField,
AxFp32 coolingRate,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void SimpleWind (
AxVecFieldF32* velField,
AxVector3 windDir,
AxFp32 windSpeed,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void WindForce (
AxVecFieldF32* velField,
AxVector3 windDir,
AxFp32 windSpeed,
AxFp32 windIntensity,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

  
}
 }



namespace FluidUtility  
{
  namespace DX 
{
 ALPHA_SPMD_FUNC void CombustionColdDown (
AxFp32 heatInfluence,
AxFp32 cooldownTime,
AxScalarFieldF32* heatField,
bool useMaskField,
AxScalarFieldF32* maskField,
AxRampCurve32RAWData maskRamp,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void UltraTurblenceScalar (
AxScalarFieldF32* targetField,
AxFp32 turblenceScale,
AxFp32 turblenceSize,
AxScalarFieldF32* thresholdField,
AxFp32 maxThreshold,
bool useMaskField,
AxScalarFieldF32* maskField,
bool useMaskRamp,
AxRampCurve32RAWData maskRamp,
AxFp32 maskRampWeight,
AxInt32 timeSeed,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void UltraTurblenceVector (
AxVecFieldF32* targetField,
AxFp32 turblenceScale,
AxFp32 turblenceSize,
AxScalarFieldF32* thresholdField,
AxFp32 maxThreshold,
bool useMaskField,
AxScalarFieldF32* maskField,
bool useMaskRamp,
AxRampCurve32RAWData maskRamp,
AxFp32 maskRampWeight,
AxInt32 timeSeed,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void UltraTurblenceBlockVector (
AxVecFieldF32* targetField,
AxScalarFieldF32* thresholdField,
AxVector2 thresholdRange,
bool useMaskField,
AxScalarFieldF32* maskField,
bool useMaskRamp,
AxRampCurve32RAWData maskRamp,
AxUltraTurbulenceParam ultraTurbParam,
AxFp32 time,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FlameSplitTurblence (
AxScalarFieldF32* temperatureField,
AxVecFieldF32* outVelField,
AxFp32 scale,
AxFp32 fade,
AxFp32 pullSize,
AxFp32 pushSize,
AxFp32 temperatureThreshold,
AxFp32 gradientMagClamp,
bool activeFieldControl,
AxScalarFieldF32* controlField,
bool activeRampControl,
AxRampCurve32RAWData rampData,
AxFp32 rampWeight,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CurlNoiseTurblence (
AxVecFieldF32* velocityField,
AxScalarFieldF32* thresholdField,
AxFp32 threshold,
AxScalarFieldF32* maskField,
bool useMaskRamp,
AxRampCurve32RAWData maskRamp,
AxCurlNoiseParam curlNoise,
AxFp32 forceScale,
AxFp32 time,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void Dissipation (
AxFp32 rate,
AxScalarFieldF32* targetField,
AxScalarFieldF32* maskField,
bool useDissipationRamp,
AxRampCurve32RAWData dissipationRamp,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void Attenuation (
AxFp32 rate,
AxScalarFieldF32* inputSclarField,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void TemperatureExpCoolingDown (
AxScalarFieldF32* temperatureField,
AxFp32 coolingRate,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void SimpleWind (
AxVecFieldF32* velField,
AxVector3 windDir,
AxFp32 windSpeed,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void WindForce (
AxVecFieldF32* velField,
AxVector3 windDir,
AxFp32 windSpeed,
AxFp32 windIntensity,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

  
}
 }



}
#endif //@FSA:[TOKEN]  __FSA_ARCH_PROTOCOL_AXFLUIDUTILITY__H__
#endif