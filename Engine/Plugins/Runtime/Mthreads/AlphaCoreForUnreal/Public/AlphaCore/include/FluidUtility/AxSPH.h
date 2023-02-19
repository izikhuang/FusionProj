#ifndef __AXSPH__H__
#define __AXSPH__H__

#include "AxMacro.h"
#include "AxStorage.h"
#include "AxAccelTree.DataType.h"
#include "AxSpatialHash.h"
#include "ProceduralContent/AxNoise.h"
#include "FluidUtility/AxFluidUtility.h"
#define __FSA_ARCH_PROTOCOL_AXSPH__H__ 1
#if __FSA_ARCH_PROTOCOL_AXSPH__H__ 1
namespace AlphaCore
{
	namespace FluidUtility
	{
		ALPHA_SPMD_FUNC void SPHBoundingBoxCollision(
			AxAABB boundingBox,
			AxBufferV3* position,
			AxBufferV3* velocity,
			AxFp32 particleRadius);

		ALPHA_SPMD_FUNC void SPHComputeDensityCubic(
			AxSpatialHash::RAWDesc sptailHashDesc,
			AxBufferF* density,
			AxFp32 staticDensity,
			AxFp32 volume,
			AxBufferV3* position,
			AxFp32 kernelRadius,
			bool useCoherence);

		ALPHA_SPMD_FUNC void SPHDensityToPressure(
			AxSpatialHash::RAWDesc sptailHashDesc,
			AxBufferF* density,
			AxFp32 staticDensity,
			AxBufferF* pressure,
			AxFp32 pressureStiffness,
			AxFp32 exp,
			bool useCoherence);

		ALPHA_SPMD_FUNC void SPHComputeVolumeAndMass(
			AxFp32 particleRadius,
			AxBufferF* mass,
			AxBufferF* volume,
			AxBufferF* staticDensity);

		ALPHA_SPMD_FUNC void SPHStandardViscosity(
			AxSpatialHash::RAWDesc sptailHashDesc,
			AxBufferF* density,
			AxBufferV3* position,
			AxBufferV3* velocity,
			AxFp32 mass,
			AxFp32 viscosity,
			AxBufferV3* acceleration,
			AxFp32 kernelRadius,
			bool useCoherence);

		ALPHA_SPMD_FUNC void SPHComputeOmega(
			AxSpatialHash::RAWDesc sptailHashDesc,
			AxBufferV3* omega,
			AxBufferF* normOmega,
			AxBufferF* density,
			AxBufferV3* position,
			AxBufferV3* velocity,
			AxFp32 mass,
			AxFp32 KernelRaidus,
			bool useCoherence);

		ALPHA_SPMD_FUNC void SPHVorticityConfinement(
			AxSpatialHash::RAWDesc sptailHashDesc,
			AxBufferV3* omega,
			AxBufferF* normOmega,
			AxBufferF* density,
			AxBufferV3* position,
			AxBufferV3* velocity,
			AxBufferV3* acceleration,
			AxFp32 mass,
			AxFp32 kernelRaidus,
			AxFp32 vorticityScale,
			bool useCoherence);

		ALPHA_SPMD_FUNC void SPHPressureToAccel(
			AxSpatialHash::RAWDesc sptailHashDesc,
			AxBufferF* pressure,
			AxBufferV3* pressureAccel,
			AxBufferF* density,
			AxFp32 staticDensity,
			AxBufferV3* position,
			AxFp32 volume,
			AxFp32 kernelRadius,
			bool useCoherence);

		ALPHA_SPMD_FUNC void SPHUpdatePosition(
			AxBufferV3* position,
			AxBufferV3* velocity,
			AxBufferV3* acceleration,
			AxBufferV3* pressureAccel,
			AxVector3 gravity,
			AxFp32 deltaTime);

		ALPHA_SPMD_FUNC void SPHEmitterWithCurlNoise(
			AxParticleFluidEmitter::RAWDesc emitter,
			AxBufferV3* positionProp,
			AxBufferV3* velProp,
			AxBufferV3* accelProp,
			AxUInt32 bufferStart,
			AxCurlNoiseParam curlNoise,
			AxFp32 deltaTime);

	}

	namespace FluidUtility
	{
		namespace CUDA
		{
			ALPHA_SPMD_FUNC void SPHBoundingBoxCollision(
				AxAABB boundingBox,
				AxBufferV3* position,
				AxBufferV3* velocity,
				AxFp32 particleRadius,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHComputeDensityCubic(
				AxSpatialHash::RAWDesc sptailHashDesc,
				AxBufferF* density,
				AxFp32 staticDensity,
				AxFp32 volume,
				AxBufferV3* position,
				AxFp32 kernelRadius,
				bool useCoherence,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHDensityToPressure(
				AxSpatialHash::RAWDesc sptailHashDesc,
				AxBufferF* density,
				AxFp32 staticDensity,
				AxBufferF* pressure,
				AxFp32 pressureStiffness,
				AxFp32 exp,
				bool useCoherence,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHComputeVolumeAndMass(
				AxFp32 particleRadius,
				AxBufferF* mass,
				AxBufferF* volume,
				AxBufferF* staticDensity,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHStandardViscosity(
				AxSpatialHash::RAWDesc sptailHashDesc,
				AxBufferF* density,
				AxBufferV3* position,
				AxBufferV3* velocity,
				AxFp32 mass,
				AxFp32 viscosity,
				AxBufferV3* acceleration,
				AxFp32 kernelRadius,
				bool useCoherence,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHComputeOmega(
				AxSpatialHash::RAWDesc sptailHashDesc,
				AxBufferV3* omega,
				AxBufferF* normOmega,
				AxBufferF* density,
				AxBufferV3* position,
				AxBufferV3* velocity,
				AxFp32 mass,
				AxFp32 KernelRaidus,
				bool useCoherence,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHVorticityConfinement(
				AxSpatialHash::RAWDesc sptailHashDesc,
				AxBufferV3* omega,
				AxBufferF* normOmega,
				AxBufferF* density,
				AxBufferV3* position,
				AxBufferV3* velocity,
				AxBufferV3* acceleration,
				AxFp32 mass,
				AxFp32 kernelRaidus,
				AxFp32 vorticityScale,
				bool useCoherence,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHPressureToAccel(
				AxSpatialHash::RAWDesc sptailHashDesc,
				AxBufferF* pressure,
				AxBufferV3* pressureAccel,
				AxBufferF* density,
				AxFp32 staticDensity,
				AxBufferV3* position,
				AxFp32 volume,
				AxFp32 kernelRadius,
				bool useCoherence,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHUpdatePosition(
				AxBufferV3* position,
				AxBufferV3* velocity,
				AxBufferV3* acceleration,
				AxBufferV3* pressureAccel,
				AxVector3 gravity,
				AxFp32 deltaTime,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHEmitterWithCurlNoise(
				AxParticleFluidEmitter::RAWDesc emitter,
				AxBufferV3* positionProp,
				AxBufferV3* velProp,
				AxBufferV3* accelProp,
				AxUInt32 bufferStart,
				AxCurlNoiseParam curlNoise,
				AxFp32 deltaTime,
				AxUInt32 blockSize = 512);


		}
	}



	namespace FluidUtility
	{
		namespace DX
		{
			ALPHA_SPMD_FUNC void SPHBoundingBoxCollision(
				AxAABB boundingBox,
				AxBufferV3* position,
				AxBufferV3* velocity,
				AxFp32 particleRadius,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHComputeDensityCubic(
				AxSpatialHash::RAWDesc sptailHashDesc,
				AxBufferF* density,
				AxFp32 staticDensity,
				AxFp32 volume,
				AxBufferV3* position,
				AxFp32 kernelRadius,
				bool useCoherence,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHDensityToPressure(
				AxSpatialHash::RAWDesc sptailHashDesc,
				AxBufferF* density,
				AxFp32 staticDensity,
				AxBufferF* pressure,
				AxFp32 pressureStiffness,
				AxFp32 exp,
				bool useCoherence,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHComputeVolumeAndMass(
				AxFp32 particleRadius,
				AxBufferF* mass,
				AxBufferF* volume,
				AxBufferF* staticDensity,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHStandardViscosity(
				AxSpatialHash::RAWDesc sptailHashDesc,
				AxBufferF* density,
				AxBufferV3* position,
				AxBufferV3* velocity,
				AxFp32 mass,
				AxFp32 viscosity,
				AxBufferV3* acceleration,
				AxFp32 kernelRadius,
				bool useCoherence,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHComputeOmega(
				AxSpatialHash::RAWDesc sptailHashDesc,
				AxBufferV3* omega,
				AxBufferF* normOmega,
				AxBufferF* density,
				AxBufferV3* position,
				AxBufferV3* velocity,
				AxFp32 mass,
				AxFp32 KernelRaidus,
				bool useCoherence,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHVorticityConfinement(
				AxSpatialHash::RAWDesc sptailHashDesc,
				AxBufferV3* omega,
				AxBufferF* normOmega,
				AxBufferF* density,
				AxBufferV3* position,
				AxBufferV3* velocity,
				AxBufferV3* acceleration,
				AxFp32 mass,
				AxFp32 kernelRaidus,
				AxFp32 vorticityScale,
				bool useCoherence,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHPressureToAccel(
				AxSpatialHash::RAWDesc sptailHashDesc,
				AxBufferF* pressure,
				AxBufferV3* pressureAccel,
				AxBufferF* density,
				AxFp32 staticDensity,
				AxBufferV3* position,
				AxFp32 volume,
				AxFp32 kernelRadius,
				bool useCoherence,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHUpdatePosition(
				AxBufferV3* position,
				AxBufferV3* velocity,
				AxBufferV3* acceleration,
				AxBufferV3* pressureAccel,
				AxVector3 gravity,
				AxFp32 deltaTime,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SPHEmitterWithCurlNoise(
				AxParticleFluidEmitter::RAWDesc emitter,
				AxBufferV3* positionProp,
				AxBufferV3* velProp,
				AxBufferV3* accelProp,
				AxUInt32 bufferStart,
				AxCurlNoiseParam curlNoise,
				AxFp32 deltaTime,
				AxUInt32 blockSize = 512);


		}
	}



}
#endif //@FSA:[TOKEN]  __FSA_ARCH_PROTOCOL_AXSPH__H__
#endif