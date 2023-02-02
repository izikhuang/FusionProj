#ifndef __AXATMOSPHERE__H__
#define __AXATMOSPHERE__H__

#define __FSA_ARCH_PROTOCOL_AXATMOSPHERE__H__ 1

#include "Utility/AxStorage.h"
#include "AxFieldBase3D.h"
#include <AxTimeTick.h>
#include "AxNoise.h"

#if __FSA_ARCH_PROTOCOL_AXATMOSPHERE__H__ == 1
namespace AlphaCore
{
namespace Atmosphere  
{
 ALPHA_SPMD_FUNC void ApplyBuoyForceStaggered (
AxScalarFieldF32* velYField,
AxScalarFieldF32* buoyField,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void Q2X (
AxScalarFieldF32* moleFraction,
AxScalarFieldF32* massRatio) ;

ALPHA_SPMD_FUNC void Q2Y (
AxScalarFieldF32* massFraction,
AxScalarFieldF32* moleFraction,
AxFp32 molarMassWater,
AxFp32 molarMassDryAir) ;

ALPHA_SPMD_FUNC void T2PT (
AxScalarFieldF32* theta,
AxScalarFieldF32* tem,
AxScalarFieldF32* pressureN,
AxFp32 pressureZero,
AxFp32 possionConst) ;

ALPHA_SPMD_FUNC void PT2T (
AxScalarFieldF32* tem,
AxScalarFieldF32* theta,
AxScalarFieldF32* pressureN,
AxFp32 pressureZero,
AxFp32 possionConst) ;

ALPHA_SPMD_FUNC void CalcMolarMassThermal (
AxScalarFieldF32* molarMassThermal,
AxScalarFieldF32* moleFractionVapor,
AxFp32 molarMassWater,
AxFp32 molarMassDryAir) ;

ALPHA_SPMD_FUNC void CalcIsentropicExponentThermal (
AxScalarFieldF32* isentropicExponentThermal,
AxScalarFieldF32* massFractionVapor,
AxFp32 isentropicExponentVapor,
AxFp32 isentropicExponentDryAir) ;

ALPHA_SPMD_FUNC void CalcHeatCapacityThermal (
AxScalarFieldF32* heatCapacityThermal,
AxScalarFieldF32* isentropicExponentThermal,
AxScalarFieldF32* molarMassThermal,
AxFp32 gasConstantR) ;

ALPHA_SPMD_FUNC void CalcAirTemProfile (
AxScalarFieldF32* isaTemConst,
AxFp32 domainAltitudeHeight,
AxFp32 temZero,
AxFp32 inversionHeight,
AxFp32 temLapseRate0,
AxFp32 temLapseRate1) ;

ALPHA_SPMD_FUNC void CalcHydroStaticPressure (
AxScalarFieldF32* isaPressureConst,
AxFp32 domainAltitudeHeight,
AxFp32 temZero,
AxFp32 pressureZero,
AxFp32 temLapseRate0,
AxFp32 gasConstantR,
AxFp32 molarMassDryAir,
AxFp32 gravityAccelConst) ;

ALPHA_SPMD_FUNC void CalcIsaP (
AxScalarFieldF32* isaPressure,
AxScalarFieldF32* Tem,
AxFp32 domainAltitudeHeight,
AxFp32 pressureZero,
AxFp32 temLapseRate0,
AxFp32 gasConstantR,
AxFp32 molarMassDryAir,
AxFp32 gravityAccelConst) ;

ALPHA_SPMD_FUNC void CalcIsaT (
AxScalarFieldF32* isaTem,
AxScalarFieldF32* Tem,
AxFp32 domainAltitudeHeight,
AxFp32 inversionHeight,
AxFp32 temLapseRate0,
AxFp32 temLapseRate1) ;

ALPHA_SPMD_FUNC void CalcThermalTemProfile (
AxScalarFieldF32* temThermal,
AxScalarFieldF32* isaPressure,
AxScalarFieldF32* isentropicExponentThermal,
AxScalarFieldF32* heatCapacityThermal,
AxScalarFieldF32* moleFractionCloud,
AxScalarFieldF32* tem,
AxFp32 pressureZero,
AxFp32 latentHeatWater) ;

ALPHA_SPMD_FUNC void CalcSaturatedVaporRatio (
AxScalarFieldF32* massRatioVaporSat,
AxScalarFieldF32* pressureN,
AxScalarFieldF32* tem) ;

ALPHA_SPMD_FUNC void CalcCcMinusEc (
AxScalarFieldF32* ccMinusEc,
AxScalarFieldF32* massRatioVapor,
AxScalarFieldF32* massRatioCloud,
AxScalarFieldF32* massRatioVaporSat) ;

ALPHA_SPMD_FUNC void CalcAc (
AxScalarFieldF32* ac,
AxScalarFieldF32* massRatioCloud,
AxFp32 alphaA,
AxFp32 aT) ;

ALPHA_SPMD_FUNC void CalcKc (
AxScalarFieldF32* kc,
AxScalarFieldF32* massRatioCloud,
AxScalarFieldF32* massRatioRain,
AxFp32 alphaK) ;

ALPHA_SPMD_FUNC void CalcEr (
AxScalarFieldF32* er,
AxScalarFieldF32* massRatioCloud,
AxFp32 alphaE) ;

ALPHA_SPMD_FUNC void CalcLinearBuoy (
AxScalarFieldF32* buoyAccelLinear,
AxScalarFieldF32* isaTem,
AxScalarFieldF32* isaPressure,
AxFp32 pressureZero,
AxFp32 temZero,
AxFp32 qvConstFor1DBuoy,
AxFp32 molarMassWater,
AxFp32 molarMassDryAir,
AxFp32 isentropicExponentVapor,
AxFp32 isentropicExponentDryAir,
AxFp32 gasConstantR,
AxFp32 gravityAccelConst) ;

ALPHA_SPMD_FUNC void CalcBuoyancyForce (
AxScalarFieldF32* buoyAccel,
AxScalarFieldF32* molarMassThermal,
AxScalarFieldF32* temThermal,
AxScalarFieldF32* isaTem,
AxScalarFieldF32* massRatioVapor,
AxScalarFieldF32* massRatioCloud,
AxFp32 molarMassDryAir,
AxFp32 gravityAccelConst) ;

ALPHA_SPMD_FUNC void ApplyBuoyForce (
AxVecFieldF32* velocityN,
AxScalarFieldF32* buoyAccel,
AxFp32 dt) ;

ALPHA_SPMD_FUNC void EnforceWindField (
AxVecFieldF32* velocityN,
AxVecFieldF32* velocityWind) ;

ALPHA_SPMD_FUNC void BoundaryTem (
AxScalarFieldF32* theta,
AxScalarFieldF32* isaTem,
AxScalarFieldF32* isaPressure,
AxFp32 pressureZero) ;

ALPHA_SPMD_FUNC void BoundaryMixingRatio (
AxScalarFieldF32* massRatioVapor,
AxScalarFieldF32* massRatioCloud,
AxScalarFieldF32* massRatioRain) ;

ALPHA_SPMD_FUNC void TerrainSourceEmitter (
AxScalarFieldF32* heightMap,
AxScalarFieldF32* theta,
AxScalarFieldF32* massRatioVapor,
AxScalarFieldF32* terrainHeatEmitterMap,
AxScalarFieldF32* terrainVaporEmitterMap,
AxScalarFieldF32* isaTem,
AxScalarFieldF32* isaPressure,
AxFp32 emitterConst,
AxFp32 relHumidityGroundConst,
AxFp32 temMixScale,
AxFp32 temNoiseFactor,
AxFp32 vaporMixScale,
AxFp32 vaporNoiseFactor,
AxInt32 prolong) ;

ALPHA_SPMD_FUNC void MassRatioAddSource (
AxScalarFieldF32* massRatioVapor,
AxScalarFieldF32* massRatioCloud,
AxScalarFieldF32* massRatioRain,
AxScalarFieldF32* ccMinusEc,
AxScalarFieldF32* ac,
AxScalarFieldF32* kc,
AxScalarFieldF32* er,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void PotentialTemAddSource (
AxScalarFieldF32* theta,
AxScalarFieldF32* ccMinusEc,
AxScalarFieldF32* heatCapacityThermal,
AxScalarFieldF32* tem,
AxFp32 latentHeatWater,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void CalcTemGround (
AxScalarFieldF32* temGround,
AxScalarFieldF32* emtTemp,
AxFp32 T0,
AxFp32 emitterConst,
AxFp32 temMixScale,
AxFp32 temNoiseFactor) ;

ALPHA_SPMD_FUNC void FieldProcSource (
AxScalarFieldF32* thetaField,
AxScalarFieldF32* vaporField,
AxScalarFieldF32* isaT,
AxScalarFieldF32* isaP,
AxScalarFieldI8* markField,
AxInt32 prolong,
AxAABB box,
AxFp32 emitterConst,
AxFp32 temMixScale,
AxFp32 temNoiseFactor,
AxFp32 relHumidityGroundConst,
AxFp32 vaporMixScale,
AxFp32 vaporNoiseFactor,
AxScalarFieldF32* noiseField) ;

ALPHA_SPMD_FUNC void CalcTemGround2 (
AxScalarFieldF32* temGround,
AxFp32 T0,
AxFp32 emitterConst,
AxFp32 temMixScale,
AxFp32 temNoiseFactor,
AxScalarFieldF32* noiseField) ;

ALPHA_SPMD_FUNC void transferNoiseTo2DField (
AxScalarFieldF32* m_noiseField,
AxCurlNoiseParam noiseParam) ;

ALPHA_SPMD_FUNC void FieldProcSourcePlane (
AxScalarFieldF32* thetaField,
AxScalarFieldF32* vaporField,
AxScalarFieldF32* isaT,
AxScalarFieldF32* isaP,
AxInt32 prolong,
AxAABB box,
AxFp32 emitterConst,
AxFp32 temMixScale,
AxFp32 temNoiseFactor,
AxFp32 relHumidityGroundConst,
AxFp32 vaporMixScale,
AxFp32 vaporNoiseFactor,
AxScalarFieldF32* noiseField) ;

ALPHA_SPMD_FUNC void CalcTemGround3 (
AxScalarFieldF32* temGround,
AxFp32 T0,
AxBufferAABB* emitterBox,
AxFp32 emitterConst,
AxInt32 numOfEmitter) ;

  }

namespace Atmosphere  
{
  namespace CUDA 
{
 ALPHA_SPMD_FUNC void ApplyBuoyForceStaggered (
AxScalarFieldF32* velYField,
AxScalarFieldF32* buoyField,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void Q2X (
AxScalarFieldF32* moleFraction,
AxScalarFieldF32* massRatio,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void Q2Y (
AxScalarFieldF32* massFraction,
AxScalarFieldF32* moleFraction,
AxFp32 molarMassWater,
AxFp32 molarMassDryAir,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void T2PT (
AxScalarFieldF32* theta,
AxScalarFieldF32* tem,
AxScalarFieldF32* pressureN,
AxFp32 pressureZero,
AxFp32 possionConst,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void PT2T (
AxScalarFieldF32* tem,
AxScalarFieldF32* theta,
AxScalarFieldF32* pressureN,
AxFp32 pressureZero,
AxFp32 possionConst,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcMolarMassThermal (
AxScalarFieldF32* molarMassThermal,
AxScalarFieldF32* moleFractionVapor,
AxFp32 molarMassWater,
AxFp32 molarMassDryAir,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcIsentropicExponentThermal (
AxScalarFieldF32* isentropicExponentThermal,
AxScalarFieldF32* massFractionVapor,
AxFp32 isentropicExponentVapor,
AxFp32 isentropicExponentDryAir,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcHeatCapacityThermal (
AxScalarFieldF32* heatCapacityThermal,
AxScalarFieldF32* isentropicExponentThermal,
AxScalarFieldF32* molarMassThermal,
AxFp32 gasConstantR,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcAirTemProfile (
AxScalarFieldF32* isaTemConst,
AxFp32 domainAltitudeHeight,
AxFp32 temZero,
AxFp32 inversionHeight,
AxFp32 temLapseRate0,
AxFp32 temLapseRate1,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcHydroStaticPressure (
AxScalarFieldF32* isaPressureConst,
AxFp32 domainAltitudeHeight,
AxFp32 temZero,
AxFp32 pressureZero,
AxFp32 temLapseRate0,
AxFp32 gasConstantR,
AxFp32 molarMassDryAir,
AxFp32 gravityAccelConst,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcIsaP (
AxScalarFieldF32* isaPressure,
AxScalarFieldF32* Tem,
AxFp32 domainAltitudeHeight,
AxFp32 pressureZero,
AxFp32 temLapseRate0,
AxFp32 gasConstantR,
AxFp32 molarMassDryAir,
AxFp32 gravityAccelConst,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcIsaT (
AxScalarFieldF32* isaTem,
AxScalarFieldF32* Tem,
AxFp32 domainAltitudeHeight,
AxFp32 inversionHeight,
AxFp32 temLapseRate0,
AxFp32 temLapseRate1,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcThermalTemProfile (
AxScalarFieldF32* temThermal,
AxScalarFieldF32* isaPressure,
AxScalarFieldF32* isentropicExponentThermal,
AxScalarFieldF32* heatCapacityThermal,
AxScalarFieldF32* moleFractionCloud,
AxScalarFieldF32* tem,
AxFp32 pressureZero,
AxFp32 latentHeatWater,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcSaturatedVaporRatio (
AxScalarFieldF32* massRatioVaporSat,
AxScalarFieldF32* pressureN,
AxScalarFieldF32* tem,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcCcMinusEc (
AxScalarFieldF32* ccMinusEc,
AxScalarFieldF32* massRatioVapor,
AxScalarFieldF32* massRatioCloud,
AxScalarFieldF32* massRatioVaporSat,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcAc (
AxScalarFieldF32* ac,
AxScalarFieldF32* massRatioCloud,
AxFp32 alphaA,
AxFp32 aT,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcKc (
AxScalarFieldF32* kc,
AxScalarFieldF32* massRatioCloud,
AxScalarFieldF32* massRatioRain,
AxFp32 alphaK,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcEr (
AxScalarFieldF32* er,
AxScalarFieldF32* massRatioCloud,
AxFp32 alphaE,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcLinearBuoy (
AxScalarFieldF32* buoyAccelLinear,
AxScalarFieldF32* isaTem,
AxScalarFieldF32* isaPressure,
AxFp32 pressureZero,
AxFp32 temZero,
AxFp32 qvConstFor1DBuoy,
AxFp32 molarMassWater,
AxFp32 molarMassDryAir,
AxFp32 isentropicExponentVapor,
AxFp32 isentropicExponentDryAir,
AxFp32 gasConstantR,
AxFp32 gravityAccelConst,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcBuoyancyForce (
AxScalarFieldF32* buoyAccel,
AxScalarFieldF32* molarMassThermal,
AxScalarFieldF32* temThermal,
AxScalarFieldF32* isaTem,
AxScalarFieldF32* massRatioVapor,
AxScalarFieldF32* massRatioCloud,
AxFp32 molarMassDryAir,
AxFp32 gravityAccelConst,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void ApplyBuoyForce (
AxVecFieldF32* velocityN,
AxScalarFieldF32* buoyAccel,
AxFp32 dt,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void EnforceWindField (
AxVecFieldF32* velocityN,
AxVecFieldF32* velocityWind,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void BoundaryTem (
AxScalarFieldF32* theta,
AxScalarFieldF32* isaTem,
AxScalarFieldF32* isaPressure,
AxFp32 pressureZero,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void BoundaryMixingRatio (
AxScalarFieldF32* massRatioVapor,
AxScalarFieldF32* massRatioCloud,
AxScalarFieldF32* massRatioRain,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void TerrainSourceEmitter (
AxScalarFieldF32* heightMap,
AxScalarFieldF32* theta,
AxScalarFieldF32* massRatioVapor,
AxScalarFieldF32* terrainHeatEmitterMap,
AxScalarFieldF32* terrainVaporEmitterMap,
AxScalarFieldF32* isaTem,
AxScalarFieldF32* isaPressure,
AxFp32 emitterConst,
AxFp32 relHumidityGroundConst,
AxFp32 temMixScale,
AxFp32 temNoiseFactor,
AxFp32 vaporMixScale,
AxFp32 vaporNoiseFactor,
AxInt32 prolong,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void MassRatioAddSource (
AxScalarFieldF32* massRatioVapor,
AxScalarFieldF32* massRatioCloud,
AxScalarFieldF32* massRatioRain,
AxScalarFieldF32* ccMinusEc,
AxScalarFieldF32* ac,
AxScalarFieldF32* kc,
AxScalarFieldF32* er,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void PotentialTemAddSource (
AxScalarFieldF32* theta,
AxScalarFieldF32* ccMinusEc,
AxScalarFieldF32* heatCapacityThermal,
AxScalarFieldF32* tem,
AxFp32 latentHeatWater,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcTemGround (
AxScalarFieldF32* temGround,
AxScalarFieldF32* emtTemp,
AxFp32 T0,
AxFp32 emitterConst,
AxFp32 temMixScale,
AxFp32 temNoiseFactor,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldProcSource (
AxScalarFieldF32* thetaField,
AxScalarFieldF32* vaporField,
AxScalarFieldF32* isaT,
AxScalarFieldF32* isaP,
AxScalarFieldI8* markField,
AxInt32 prolong,
AxAABB box,
AxFp32 emitterConst,
AxFp32 temMixScale,
AxFp32 temNoiseFactor,
AxFp32 relHumidityGroundConst,
AxFp32 vaporMixScale,
AxFp32 vaporNoiseFactor,
AxScalarFieldF32* noiseField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcTemGround2 (
AxScalarFieldF32* temGround,
AxFp32 T0,
AxFp32 emitterConst,
AxFp32 temMixScale,
AxFp32 temNoiseFactor,
AxScalarFieldF32* noiseField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void transferNoiseTo2DField (
AxScalarFieldF32* m_noiseField,
AxCurlNoiseParam noiseParam,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldProcSourcePlane (
AxScalarFieldF32* thetaField,
AxScalarFieldF32* vaporField,
AxScalarFieldF32* isaT,
AxScalarFieldF32* isaP,
AxInt32 prolong,
AxAABB box,
AxFp32 emitterConst,
AxFp32 temMixScale,
AxFp32 temNoiseFactor,
AxFp32 relHumidityGroundConst,
AxFp32 vaporMixScale,
AxFp32 vaporNoiseFactor,
AxScalarFieldF32* noiseField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcTemGround3 (
AxScalarFieldF32* temGround,
AxFp32 T0,
AxBufferAABB* emitterBox,
AxFp32 emitterConst,
AxInt32 numOfEmitter,
AxUInt32 blockSize = 512) ;

  
}
 }



namespace Atmosphere  
{
  namespace DX 
{
 ALPHA_SPMD_FUNC void ApplyBuoyForceStaggered (
AxScalarFieldF32* velYField,
AxScalarFieldF32* buoyField,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void Q2X (
AxScalarFieldF32* moleFraction,
AxScalarFieldF32* massRatio,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void Q2Y (
AxScalarFieldF32* massFraction,
AxScalarFieldF32* moleFraction,
AxFp32 molarMassWater,
AxFp32 molarMassDryAir,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void T2PT (
AxScalarFieldF32* theta,
AxScalarFieldF32* tem,
AxScalarFieldF32* pressureN,
AxFp32 pressureZero,
AxFp32 possionConst,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void PT2T (
AxScalarFieldF32* tem,
AxScalarFieldF32* theta,
AxScalarFieldF32* pressureN,
AxFp32 pressureZero,
AxFp32 possionConst,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcMolarMassThermal (
AxScalarFieldF32* molarMassThermal,
AxScalarFieldF32* moleFractionVapor,
AxFp32 molarMassWater,
AxFp32 molarMassDryAir,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcIsentropicExponentThermal (
AxScalarFieldF32* isentropicExponentThermal,
AxScalarFieldF32* massFractionVapor,
AxFp32 isentropicExponentVapor,
AxFp32 isentropicExponentDryAir,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcHeatCapacityThermal (
AxScalarFieldF32* heatCapacityThermal,
AxScalarFieldF32* isentropicExponentThermal,
AxScalarFieldF32* molarMassThermal,
AxFp32 gasConstantR,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcAirTemProfile (
AxScalarFieldF32* isaTemConst,
AxFp32 domainAltitudeHeight,
AxFp32 temZero,
AxFp32 inversionHeight,
AxFp32 temLapseRate0,
AxFp32 temLapseRate1,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcHydroStaticPressure (
AxScalarFieldF32* isaPressureConst,
AxFp32 domainAltitudeHeight,
AxFp32 temZero,
AxFp32 pressureZero,
AxFp32 temLapseRate0,
AxFp32 gasConstantR,
AxFp32 molarMassDryAir,
AxFp32 gravityAccelConst,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcIsaP (
AxScalarFieldF32* isaPressure,
AxScalarFieldF32* Tem,
AxFp32 domainAltitudeHeight,
AxFp32 pressureZero,
AxFp32 temLapseRate0,
AxFp32 gasConstantR,
AxFp32 molarMassDryAir,
AxFp32 gravityAccelConst,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcIsaT (
AxScalarFieldF32* isaTem,
AxScalarFieldF32* Tem,
AxFp32 domainAltitudeHeight,
AxFp32 inversionHeight,
AxFp32 temLapseRate0,
AxFp32 temLapseRate1,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcThermalTemProfile (
AxScalarFieldF32* temThermal,
AxScalarFieldF32* isaPressure,
AxScalarFieldF32* isentropicExponentThermal,
AxScalarFieldF32* heatCapacityThermal,
AxScalarFieldF32* moleFractionCloud,
AxScalarFieldF32* tem,
AxFp32 pressureZero,
AxFp32 latentHeatWater,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcSaturatedVaporRatio (
AxScalarFieldF32* massRatioVaporSat,
AxScalarFieldF32* pressureN,
AxScalarFieldF32* tem,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcCcMinusEc (
AxScalarFieldF32* ccMinusEc,
AxScalarFieldF32* massRatioVapor,
AxScalarFieldF32* massRatioCloud,
AxScalarFieldF32* massRatioVaporSat,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcAc (
AxScalarFieldF32* ac,
AxScalarFieldF32* massRatioCloud,
AxFp32 alphaA,
AxFp32 aT,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcKc (
AxScalarFieldF32* kc,
AxScalarFieldF32* massRatioCloud,
AxScalarFieldF32* massRatioRain,
AxFp32 alphaK,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcEr (
AxScalarFieldF32* er,
AxScalarFieldF32* massRatioCloud,
AxFp32 alphaE,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcLinearBuoy (
AxScalarFieldF32* buoyAccelLinear,
AxScalarFieldF32* isaTem,
AxScalarFieldF32* isaPressure,
AxFp32 pressureZero,
AxFp32 temZero,
AxFp32 qvConstFor1DBuoy,
AxFp32 molarMassWater,
AxFp32 molarMassDryAir,
AxFp32 isentropicExponentVapor,
AxFp32 isentropicExponentDryAir,
AxFp32 gasConstantR,
AxFp32 gravityAccelConst,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcBuoyancyForce (
AxScalarFieldF32* buoyAccel,
AxScalarFieldF32* molarMassThermal,
AxScalarFieldF32* temThermal,
AxScalarFieldF32* isaTem,
AxScalarFieldF32* massRatioVapor,
AxScalarFieldF32* massRatioCloud,
AxFp32 molarMassDryAir,
AxFp32 gravityAccelConst,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void ApplyBuoyForce (
AxVecFieldF32* velocityN,
AxScalarFieldF32* buoyAccel,
AxFp32 dt,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void EnforceWindField (
AxVecFieldF32* velocityN,
AxVecFieldF32* velocityWind,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void BoundaryTem (
AxScalarFieldF32* theta,
AxScalarFieldF32* isaTem,
AxScalarFieldF32* isaPressure,
AxFp32 pressureZero,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void BoundaryMixingRatio (
AxScalarFieldF32* massRatioVapor,
AxScalarFieldF32* massRatioCloud,
AxScalarFieldF32* massRatioRain,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void TerrainSourceEmitter (
AxScalarFieldF32* heightMap,
AxScalarFieldF32* theta,
AxScalarFieldF32* massRatioVapor,
AxScalarFieldF32* terrainHeatEmitterMap,
AxScalarFieldF32* terrainVaporEmitterMap,
AxScalarFieldF32* isaTem,
AxScalarFieldF32* isaPressure,
AxFp32 emitterConst,
AxFp32 relHumidityGroundConst,
AxFp32 temMixScale,
AxFp32 temNoiseFactor,
AxFp32 vaporMixScale,
AxFp32 vaporNoiseFactor,
AxInt32 prolong,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void MassRatioAddSource (
AxScalarFieldF32* massRatioVapor,
AxScalarFieldF32* massRatioCloud,
AxScalarFieldF32* massRatioRain,
AxScalarFieldF32* ccMinusEc,
AxScalarFieldF32* ac,
AxScalarFieldF32* kc,
AxScalarFieldF32* er,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void PotentialTemAddSource (
AxScalarFieldF32* theta,
AxScalarFieldF32* ccMinusEc,
AxScalarFieldF32* heatCapacityThermal,
AxScalarFieldF32* tem,
AxFp32 latentHeatWater,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcTemGround (
AxScalarFieldF32* temGround,
AxScalarFieldF32* emtTemp,
AxFp32 T0,
AxFp32 emitterConst,
AxFp32 temMixScale,
AxFp32 temNoiseFactor,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldProcSource (
AxScalarFieldF32* thetaField,
AxScalarFieldF32* vaporField,
AxScalarFieldF32* isaT,
AxScalarFieldF32* isaP,
AxScalarFieldI8* markField,
AxInt32 prolong,
AxAABB box,
AxFp32 emitterConst,
AxFp32 temMixScale,
AxFp32 temNoiseFactor,
AxFp32 relHumidityGroundConst,
AxFp32 vaporMixScale,
AxFp32 vaporNoiseFactor,
AxScalarFieldF32* noiseField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcTemGround2 (
AxScalarFieldF32* temGround,
AxFp32 T0,
AxFp32 emitterConst,
AxFp32 temMixScale,
AxFp32 temNoiseFactor,
AxScalarFieldF32* noiseField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void transferNoiseTo2DField (
AxScalarFieldF32* m_noiseField,
AxCurlNoiseParam noiseParam,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldProcSourcePlane (
AxScalarFieldF32* thetaField,
AxScalarFieldF32* vaporField,
AxScalarFieldF32* isaT,
AxScalarFieldF32* isaP,
AxInt32 prolong,
AxAABB box,
AxFp32 emitterConst,
AxFp32 temMixScale,
AxFp32 temNoiseFactor,
AxFp32 relHumidityGroundConst,
AxFp32 vaporMixScale,
AxFp32 vaporNoiseFactor,
AxScalarFieldF32* noiseField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CalcTemGround3 (
AxScalarFieldF32* temGround,
AxFp32 T0,
AxBufferAABB* emitterBox,
AxFp32 emitterConst,
AxInt32 numOfEmitter,
AxUInt32 blockSize = 512) ;

  
}
 }



}
#endif //@FSA:[TOKEN]  __FSA_ARCH_PROTOCOL_AXATMOSPHERE__H__
#endif