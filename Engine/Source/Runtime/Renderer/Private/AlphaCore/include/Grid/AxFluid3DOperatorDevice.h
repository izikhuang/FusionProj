#ifndef __ALPHA_CORE_FIELD3D_OPERATOR_DEVICE_H__
#define __ALPHA_CORE_FIELD3D_OPERATOR_DEVICE_H__


#include "AxFluid3DOperator.h"

namespace AlphaCore
{
	
	namespace GridDense
	{
#ifdef ALPHA_CUDA

		namespace CUDA
		{
			void Advect(
				AxVecFieldF32* src,
				AxVecFieldF32* dst,
				AxVecFieldF32* vel,
				AxScalarFieldF32* advectTmp,
				AlphaCore::Flow::AdvectType type,
				float dt,
				bool loadBack,
				uInt32 blockSize = 512);

			void Advect(
				AxScalarFieldF32* src,
				AxScalarFieldF32* dst,
				AxVecFieldF32* vel,
				AxScalarFieldF32* advectTmp,
				AlphaCore::Flow::AdvectType type,
				float dt, 
				bool loadBack,
				uInt32 blockSize=512);

			void ClampExtrema(
				AxScalarFieldF32 * fieldOld,
				AxScalarFieldF32 * fieldNew,
				AxVecFieldF32 * velField,
				float dt);

			void Divergence(AxVecFieldF32 * srcVecField, AxScalarFieldF32 * dstDivField, uInt32 blockSize = 512);

			void Curl(AxVecFieldF32 * srcVecField, AxVecFieldF32 * dstCurlField,uInt32 blockSize=512);

			void Length(AxVecFieldF32 * srcVecField, AxScalarFieldF32 * dstLenField, uInt32 blockSize = 512);

			void Gradient(AxScalarFieldF32 * srcScalarField,
				AxVecFieldF32 * dstLenField,
				bool normalize = false, 
				uInt32 blockSize = 512);

			void FieldCross(AxVecFieldF32 * a, AxVecFieldF32 * b, AxVecFieldF32 * ret, uInt32 blockSize = 512);

			void SubtractGradient(AxVecFieldF32* velField,
				AxScalarFieldF32 * pressureField, 
				uInt32 blockSize = 512);


			void VorticityConfinement(AxVecFieldF32* velField,
				AxVecFieldF32* curlField,
				AxScalarFieldF32* curlMagField,
				AxVecFieldF32* vortexDirField,
				float confinementScale,
				float dt,uInt32 blockSize=512);

			void FieldMix(AxScalarFieldF32* dst,
				AxScalarFieldF32* a, float coeffA,
				AxScalarFieldF32* b, float coeffB,
				float totalCoeff = 1.0f,
				uInt32 blockSize = 512);

			void FieldMixV3(AxVecFieldF32* dst,
				AxVecFieldF32* a, float coeffA,
				AxVecFieldF32* b, float coeffB,
				float totalCoeff = 1.0f,
				uInt32 blockSize = 512);

			void ProjectNonDivergence(AxVecFieldF32* velField,
				AxScalarFieldF32 * divField,
				AxScalarFieldF32 * pressureOld,
				AxScalarFieldF32 * pressureNew,
				uInt32 iterations,
				AlphaCore::LinearSolver solverType,
				uInt32 blockSize = 512);


			void PressureSolverJacobi(AxScalarFieldF32 * pressureOld,
				AxScalarFieldF32 * pressureNew,
				AxScalarFieldF32 * divField,
				uInt32 blockSize = 512);

			void PressureSolverGaussSeidel(AxScalarFieldF32 * pressureOld,
				AxScalarFieldF32 * pressureNew,
				AxScalarFieldF32 * divField,
				uInt32 blockSize = 512);

			void Combustion_GridAlign(
				AxScalarFieldF32* temperatureField,
				AxScalarFieldF32* fuelField,
				AxScalarFieldF32* densityField,
				AxScalarFieldF32* divField,
				AxScalarFieldF32* burnField,
				AxScalarFieldF32* heatField,
				float dt,
				AlphaCore::Param::AxCombustionParam combParam,
				uInt32 blockSize = 512);

			void SimpleGPUStep(uInt32 substeps);

			void AddSourceField_Scalar(
				AxScalarFieldF32* dstField,
				AxScalarFieldF32* srcField,
				float scale,
				float dt,
				uInt32 blockSize=512);


			void ApplyBuoyancy(
				AxScalarFieldF32* density,
				AxScalarFieldF32* temperature,
				AxVecFieldF32* vel,
				AxVector3 buoyancyDir,
				float alpha,
				float beta,
				float dt,
				uInt32 blockSize=512);


		}
#endif 
	}
}

#endif