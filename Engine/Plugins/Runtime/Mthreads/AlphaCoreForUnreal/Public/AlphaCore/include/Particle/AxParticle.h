#ifndef __AXPARTICLE__H__
#define __AXPARTICLE__H__
#define __FSA_ARCH_PROTOCOL_AXPARTICLE__H__ 1

#include "AxMacro.h"
#include "AxForce.h"
#include "Utility/AxStorage.h"
#include "AxDataType.h"
#include "AxParameter.h"

#if __FSA_ARCH_PROTOCOL_AXPARTICLE__H__ 1
namespace AlphaCore
{
namespace Particle  
{
 ALPHA_SPMD_FUNC void ParticleAdvect (
AxBufferV3* positionProp,
AxBufferV3* accelerationProp,
AxBufferV3* velocityProp,
AxBufferV3* prdPosProp,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void Force (
AxBufferV3* position,
AxBufferV3* acceleration,
AxVector3 force,
AxFp32 forceScale,
bool useNoise,
AxCurlNoiseParam curlNoise,
AxBufferF* maskProperty,
AxRampCurve32RAWData maskPropertyRamp,
AxFp32 time,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void ParticleStep (
AxBufferV3* positionProp,
AxBufferV3* accelerationProp,
AxBufferV3* velocityProp,
AxBufferF* massProp,
AxBufferF* lifespanProp,
AxBufferF* ageProp,
AxBufferI* diedProp,
AxFp32 ignoreMass,
AxFp32 deltaTime) ;

  }

namespace Particle  
{
  namespace CUDA 
{
 ALPHA_SPMD_FUNC void ParticleAdvect (
AxBufferV3* positionProp,
AxBufferV3* accelerationProp,
AxBufferV3* velocityProp,
AxBufferV3* prdPosProp,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void Force (
AxBufferV3* position,
AxBufferV3* acceleration,
AxVector3 force,
AxFp32 forceScale,
bool useNoise,
AxCurlNoiseParam curlNoise,
AxBufferF* maskProperty,
AxRampCurve32RAWData maskPropertyRamp,
AxFp32 time,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void ParticleStep (
AxBufferV3* positionProp,
AxBufferV3* accelerationProp,
AxBufferV3* velocityProp,
AxBufferF* massProp,
AxBufferF* lifespanProp,
AxBufferF* ageProp,
AxBufferI* diedProp,
AxFp32 ignoreMass,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

  
}
 }



namespace Particle  
{
  namespace DX 
{
 ALPHA_SPMD_FUNC void ParticleAdvect (
AxBufferV3* positionProp,
AxBufferV3* accelerationProp,
AxBufferV3* velocityProp,
AxBufferV3* prdPosProp,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void Force (
AxBufferV3* position,
AxBufferV3* acceleration,
AxVector3 force,
AxFp32 forceScale,
bool useNoise,
AxCurlNoiseParam curlNoise,
AxBufferF* maskProperty,
AxRampCurve32RAWData maskPropertyRamp,
AxFp32 time,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void ParticleStep (
AxBufferV3* positionProp,
AxBufferV3* accelerationProp,
AxBufferV3* velocityProp,
AxBufferF* massProp,
AxBufferF* lifespanProp,
AxBufferF* ageProp,
AxBufferI* diedProp,
AxFp32 ignoreMass,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

  
}
 }



}
#endif //@FSA:[TOKEN]  __FSA_ARCH_PROTOCOL_AXPARTICLE__H__

#include <Particle/AxParticleEmitter.h>


#endif