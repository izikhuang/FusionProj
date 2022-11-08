#ifndef __AXATOMSPHERE__H__
#define __AXATOMSPHERE__H__

#include "AxMacro.h"
#include "AxStorage.h"
#include "GridDense/AxGridDense.DataType.h"
#include "GridDense/AxFieldBase3D.h"


#define __FSA_ARCH_PROTOCOL_AXATOMSPHERE__H__ 1

#if __FSA_ARCH_PROTOCOL_AXATOMSPHERE__H__ 1
namespace AlphaCore
{
	namespace GridDense
	{
		ALPHA_SPMD_FUNC void UpdateMoleFraction(AxScalarFieldF32* massRaitoField);

		ALPHA_SPMD_FUNC void UpdateSaturationRatio(
			AxScalarFieldF32* tempetureField,
			AxScalarFieldF32* pressureGround);

		ALPHA_SPMD_FUNC void UpdateAverageMolarMass(AxScalarFieldF32* moleFractionVapor);

		ALPHA_SPMD_FUNC void UpdateIsentropicExponent(
			AxScalarFieldF32* moleFractionVapor,
			AxScalarFieldF32* molarMassThermal,
			AxScalarFieldF32* molarMassWater);

		ALPHA_SPMD_FUNC void UpdateHeatCapacity(
			AxScalarFieldF32* isentropicExponentThermal,
			AxScalarFieldF32* molarMassThermal);

	}

	namespace GridDense
	{
		namespace CUDA
		{
			ALPHA_SPMD_FUNC void UpdateMoleFraction(
				AxScalarFieldF32* massRaitoField,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void UpdateSaturationRatio(
				AxScalarFieldF32* tempetureField,
				AxScalarFieldF32* pressureGround,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void UpdateAverageMolarMass(
				AxScalarFieldF32* moleFractionVapor,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void UpdateIsentropicExponent(
				AxScalarFieldF32* moleFractionVapor,
				AxScalarFieldF32* molarMassThermal,
				AxScalarFieldF32* molarMassWater,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void UpdateHeatCapacity(
				AxScalarFieldF32* isentropicExponentThermal,
				AxScalarFieldF32* molarMassThermal,
				AxUInt32 blockSize = 512);


		}
	}



	namespace GridDense
	{
		namespace DX
		{
			ALPHA_SPMD_FUNC void UpdateMoleFraction(
				AxScalarFieldF32* massRaitoField,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void UpdateSaturationRatio(
				AxScalarFieldF32* tempetureField,
				AxScalarFieldF32* pressureGround,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void UpdateAverageMolarMass(
				AxScalarFieldF32* moleFractionVapor,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void UpdateIsentropicExponent(
				AxScalarFieldF32* moleFractionVapor,
				AxScalarFieldF32* molarMassThermal,
				AxScalarFieldF32* molarMassWater,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void UpdateHeatCapacity(
				AxScalarFieldF32* isentropicExponentThermal,
				AxScalarFieldF32* molarMassThermal,
				AxUInt32 blockSize = 512);


		}
	}



}

#endif //@FSA:[TOKEN]  __FSA_ARCH_PROTOCOL_AXATOMSPHERE__H__
#endif