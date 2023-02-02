#ifndef __AX_GRIDDENSE_H__
#define __AX_GRIDDENSE_H__

#include "FluidUtility/AxFluidUtility.DataType.h"
#include "GridDense/AxFieldBase3D.h"
#include "GridDense/AxGridDense.DataType.h"

#define __FSA_ARCH_PROTOCOL_AXGRIDDENSE__H__

namespace AlphaCore
{
	namespace GridDense
	{

		ALPHA_SPMD_FUNC void FieldAdvect(
			AlphaCore::FluidUtility::AdvectType advectType,/*advectType*/
			AxVecFieldF32* srcField,/*old field*/
			AxVecFieldF32* outField,/*new field*/
			AxVecFieldF32* velField,/*velocity field*/
			AxVecFieldF32* advectTempField,/*temp field*/
			AlphaCore::FluidUtility::AdvectTraceType traceMethod,
			AxFp32 deltaTime,/*advect stepSize*/
			bool loadBack = false);

		ALPHA_SPMD_FUNC void FieldAdvect(
			AlphaCore::FluidUtility::AdvectType advectType,/*advectType*/
			AxScalarFieldF32* srcField,/*old field*/
			AxScalarFieldF32* outField,/*new field*/
			AxVecFieldF32* velField,/*velocity field*/
			AxScalarFieldF32* advectTempField,/*temp field*/
			AlphaCore::FluidUtility::AdvectTraceType traceMethod,
			AxFp32 deltaTime,/*advect stepSize*/
			bool loadBack = false);

		ALPHA_SPMD_FUNC void FieldAdvect(
			AlphaCore::FluidUtility::AdvectType advectType,/*advectType*/
			AxVecFieldF32* inputVecField,/*old field*/
			AxVecFieldF32* srcVelField,/*new field*/
			AxVecFieldF32* outVecField,/*velocity field*/
			AxScalarFieldF32* advectTempField,/*temp field*/
			AlphaCore::FluidUtility::AdvectTraceType traceMethod,
			AxFp32 deltaTime,/*advect stepSize*/
			bool loadBack = false);


		ALPHA_SPMD_FUNC void VorticityConfinementNEW(
			AxVecFieldF32* velField,
			AxVecFieldF32* curlField,
			AxScalarFieldF32* curlMagField,
			AxVecFieldF32* vortexDirField,
			AxFp32 confinementScale,
			AxFp32 deltaTime);

		ALPHA_SPMD_FUNC void ProjectNonDivergence(
			AxVecFieldF32* velField,
			AxScalarFieldF32* divField,
			AxScalarFieldF32* pressureOldField,
			AxScalarFieldF32* pressureNewField,
			AxInt16 iterations,
			AlphaCore::LinearSolver solverType);


		ALPHA_SPMD_FUNC void SetToZero(AxScalarFieldF32* scalarField);

		ALPHA_SPMD_FUNC void ScalarDiffusionJacobi(
			AxScalarFieldF32* inputScalarField,
			AxScalarFieldF32* outScalarField,
			AxFp32 rate,
			AxUInt32 iterations,
			AxFp32 deltaTime);

		ALPHA_SPMD_FUNC void ProjectNonDivergence(
			AxVecFieldF32* velField,
			AxScalarFieldF32* divField,
			AxScalarFieldF32* pressureOldField,
			AxScalarFieldF32* pressureNewField,
			AxScalarFieldI8* markField,
			AxInt16 iterations,
			AlphaCore::LinearSolver solverType);

		namespace CUDA
		{
			ALPHA_SPMD_FUNC void FieldAdvect(
				AlphaCore::FluidUtility::AdvectType advectType,/*advectType*/
				AxVecFieldF32* srcField,/*old field*/
				AxVecFieldF32* outField,/*new field*/
				AxVecFieldF32* velField,/*velocity field*/
				AxVecFieldF32* advectTempField,/*temp field*/
				AlphaCore::FluidUtility::AdvectTraceType traceMethod,
				AxFp32 deltaTime,/*advect stepSize*/
				bool loadBack = false,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void FieldAdvect(
				AlphaCore::FluidUtility::AdvectType advectType,/*advectType*/
				AxScalarFieldF32* srcField,/*old field*/
				AxScalarFieldF32* outField,/*new field*/
				AxVecFieldF32* velField,/*velocity field*/
				AxVecFieldF32* advectTempField,/*temp field*/
				AlphaCore::FluidUtility::AdvectTraceType traceMethod,
				AxFp32 deltaTime,/*advect stepSize*/
				bool loadBack = false);

			ALPHA_SPMD_FUNC void FieldAdvect(
				AlphaCore::FluidUtility::AdvectType advectType,/*advectType*/
				AxScalarFieldF32* srcField,/*old field*/
				AxScalarFieldF32* outField,/*new field*/
				AxVecFieldF32* velField,/*velocity field*/
				AxScalarFieldF32* advectTempField,/*temp field*/
				AlphaCore::FluidUtility::AdvectTraceType traceMethod,
				AxFp32 deltaTime,/*advect stepSize*/
				bool loadBack = false,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void FieldAdvect(
				AlphaCore::FluidUtility::AdvectType advectType,/*advectType*/
				AxVecFieldF32* inputVecField,/*old field*/
				AxVecFieldF32* srcVelField,/*new field*/
				AxVecFieldF32* outVecField,/*velocity field*/
				AxScalarFieldF32* advectTempField,/*temp field*/
				AlphaCore::FluidUtility::AdvectTraceType traceMethod,
				AxFp32 deltaTime,/*advect stepSize*/
				bool loadBack = false,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void VorticityConfinementNEW(
				AxVecFieldF32* velField,
				AxVecFieldF32* curlField,
				AxScalarFieldF32* curlMagField,
				AxVecFieldF32* vortexDirField,
				AxFp32 confinementScale,
				AxFp32 deltaTime,
				AxUInt32 blockSize = 512);


			ALPHA_SPMD_FUNC void ProjectNonDivergence(
				AxVecFieldF32* velField,
				AxScalarFieldF32* divField,
				AxScalarFieldF32* pressureOldField,
				AxScalarFieldF32* pressureNewField,
				AxInt16 iterations,
				AlphaCore::LinearSolver solverType,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void ScalarDiffusionJacobi(
				AxScalarFieldF32* inputScalarField,
				AxScalarFieldF32* outScalarField,
				AxFp32 rate,
				AxUInt32 iterations,
				AxFp32 deltaTime,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void SetToZero(AxScalarFieldF32* scalarField);

			ALPHA_SPMD_FUNC void FieldDivergenceBlock(
				AxVecFieldF32* vecField,
				AxScalarFieldF32* outDivField,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void ProjectNonDivergence(
				AxVecFieldF32* velField,
				AxScalarFieldF32* divField,
				AxScalarFieldF32* pressureOldField,
				AxScalarFieldF32* pressureNewField,
				AxScalarFieldI8* markField,
				AxInt16 iterations,
				AlphaCore::LinearSolver solverType,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void FieldDivergenceMemoryBit16(
				AxVecFieldF32* vecField,
				AxScalarFieldF32* outDivField,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void FieldDivergenceMemory(
				AxVecFieldF32* vecField,
				AxScalarFieldF32* outDivField,
				AxUInt32 blockSize = 512);

		}
	}
}


#if __FSA_ARCH_PROTOCOL_AXGRIDDENSE__H__ 1
namespace AlphaCore
{
namespace GridDense  
{
 ALPHA_SPMD_FUNC void AdvectSemiLagrangian (
AxScalarFieldF32* srcField,
AxScalarFieldF32* dstField,
AxFieldVector3F32* velField,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void AdvectSemiLagrangian (
AxScalarFieldF32* srcField,
AxScalarFieldF32* dstField,
AxVecFieldF32* velField,
AlphaCore::FluidUtility::AdvectTraceType traceMethod,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void ClampToExtrema (
AxScalarFieldF32* oldField,
AxScalarFieldF32* newField,
AxVecFieldF32* velField,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void ApplyBuoyancy (
AxScalarFieldF32* densityField,
AxScalarFieldF32* temperatureField,
AxVecFieldF32* velField,
AxFp32 alpha,
AxFp32 beta,
AxVector3 bouyancyDir,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void PressureSolverGaussSeidel (
AxScalarFieldF32* pressureOldField,
AxScalarFieldF32* pressureNewField,
AxScalarFieldF32* divergenceField,
bool redBlack) ;

ALPHA_SPMD_FUNC void pressureSolverJacobi (
AxScalarFieldF32* pressureOldField,
AxScalarFieldF32* pressureNewField,
AxScalarFieldF32* divField) ;

ALPHA_SPMD_FUNC void FieldDivergence (
AxVecFieldF32* vecField,
AxScalarFieldF32* outDivField) ;

ALPHA_SPMD_FUNC void FieldLaplacian (
AxScalarFieldF32* inputScalarField,
AxScalarFieldF32* outLaplacianField,
bool isStaggeredGrid) ;

ALPHA_SPMD_FUNC void FieldLaplacianTex (
AxScalarFieldF32* inputScalarField,
AxScalarFieldF32* outLaplacianField) ;

ALPHA_SPMD_FUNC void FieldLaplacianWithBoundary (
AxScalarFieldF32* inputScalarField,
AxScalarFieldF32* outLaplacianField,
bool isStaggeredGrid) ;

ALPHA_SPMD_FUNC void FieldSourcing (
AxScalarFieldF32* targetField,
AxScalarFieldF32* srcField,
AxFp32 scale,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void FieldSourcing (
AxVecFieldF32* targetField,
AxVecFieldF32* srcField,
AxFp32 scale,
AxVector3 additionVel,
AxMatrix3x3 additionRotation,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void CombustionWithGridAlign (
AxScalarFieldF32* temprature,
AxScalarFieldF32* fuel,
AxScalarFieldF32* density,
AxScalarFieldF32* divergence,
AxScalarFieldF32* heat,
AxScalarFieldF32* burn,
AxFp32 deltaTime,
AlphaCore::FluidUtility::Param::AxCombustionParam combustionParam) ;

ALPHA_SPMD_FUNC void CombustionWithReduceBurnAndFuel (
AxScalarFieldF32* temprature,
AxScalarFieldF32* fuel,
AxScalarFieldF32* density,
AxScalarFieldF32* divergence,
AxScalarFieldF32* heat,
AxScalarFieldF32* burn,
AxFp32 deltaTime,
AlphaCore::FluidUtility::Param::AxCombustionParam combustionParam) ;

ALPHA_SPMD_FUNC void fieldMix (
AxFieldVector3F32* outField,
AxFieldVector3F32* aField,
AxFp32 coeffA,
AxFieldVector3F32* bField,
AxFp32 coeffB,
AxFp32 totalCoeff) ;

ALPHA_SPMD_FUNC void fieldMix (
AxScalarFieldF32* outField,
AxScalarFieldF32* aField,
AxFp32 coeffA,
AxScalarFieldF32* bField,
AxFp32 coeffB,
AxFp32 totalCoeff) ;

ALPHA_SPMD_FUNC void fieldMix (
AxVecFieldF32* outField,
AxVecFieldF32* aField,
AxFp32 coeffA,
AxVecFieldF32* bField,
AxFp32 coeffB,
AxFp32 totalCoeff) ;

ALPHA_SPMD_FUNC void curl (
AxVecFieldF32* vecField,
AxVecFieldF32* curlField) ;

ALPHA_SPMD_FUNC void length (
AxVecFieldF32* vecField,
AxScalarFieldF32* magField) ;

ALPHA_SPMD_FUNC void gradient (
AxScalarFieldF32* inputScalarField,
AxVecFieldF32* outputVecField,
bool normalized) ;

ALPHA_SPMD_FUNC void cross (
AxVecFieldF32* vecFieldA,
AxVecFieldF32* vecFieldB,
AxVecFieldF32* outRetField,
bool normalized) ;

ALPHA_SPMD_FUNC void subtractGradient (
AxVecFieldF32* velField,
AxScalarFieldF32* pressureField) ;

ALPHA_SPMD_FUNC void ScalarDiffusionJacobi (
AxScalarFieldF32* inputScalarField,
AxScalarFieldF32* outScalarField,
AxFp32 rate,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void VectorDiffusionJacobi (
AxVecFieldF32* inputVecField,
AxVecFieldF32* outVecField,
AxFp32 rate,
AxFp32 deltaTime) ;

ALPHA_SPMD_FUNC void ClearFieldByMask (
AxVecFieldF32* velField,
AxScalarFieldF32* densityField,
AxScalarFieldF32* temperatureField,
AxScalarFieldF32* heatField,
AxScalarFieldI8* markField) ;

ALPHA_SPMD_FUNC void HeightFieldToMark (
AxScalarFieldI8* markField,
AxScalarFieldF32* heightField) ;

ALPHA_SPMD_FUNC void PressureSolverJacobiFast (
AxScalarFieldF32* pressureOldField,
AxScalarFieldF32* pressureNewField,
AxScalarFieldF32* divField,
AxScalarFieldI8* markField) ;

ALPHA_SPMD_FUNC void PressureSolverGaussSeidelFast (
AxScalarFieldF32* pressureOldField,
AxScalarFieldF32* pressureNewField,
AxScalarFieldF32* divField,
AxScalarFieldI8* markField,
bool redBlack) ;

ALPHA_SPMD_FUNC void SubstractGradient (
AxVecFieldF32* velField,
AxScalarFieldF32* pressureOldField,
AxScalarFieldI8* markField) ;

ALPHA_SPMD_FUNC void FieldDivergence (
AxVecFieldF32* velField,
AxScalarFieldF32* divField,
AxVecFieldF32* colliderVelField,
AxScalarFieldI8* mark) ;

ALPHA_SPMD_FUNC void ProjectVelToCollider (
AxVecFieldF32* velInput,
AxVecFieldF32* velOutput,
AxScalarFieldI8* markField) ;

  }

namespace GridDense  
{
  namespace CUDA 
{
 ALPHA_SPMD_FUNC void AdvectSemiLagrangian (
AxScalarFieldF32* srcField,
AxScalarFieldF32* dstField,
AxFieldVector3F32* velField,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void AdvectSemiLagrangian (
AxScalarFieldF32* srcField,
AxScalarFieldF32* dstField,
AxVecFieldF32* velField,
AlphaCore::FluidUtility::AdvectTraceType traceMethod,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void ClampToExtrema (
AxScalarFieldF32* oldField,
AxScalarFieldF32* newField,
AxVecFieldF32* velField,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void ApplyBuoyancy (
AxScalarFieldF32* densityField,
AxScalarFieldF32* temperatureField,
AxVecFieldF32* velField,
AxFp32 alpha,
AxFp32 beta,
AxVector3 bouyancyDir,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void PressureSolverGaussSeidel (
AxScalarFieldF32* pressureOldField,
AxScalarFieldF32* pressureNewField,
AxScalarFieldF32* divergenceField,
bool redBlack,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void pressureSolverJacobi (
AxScalarFieldF32* pressureOldField,
AxScalarFieldF32* pressureNewField,
AxScalarFieldF32* divField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldDivergence (
AxVecFieldF32* vecField,
AxScalarFieldF32* outDivField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldLaplacian (
AxScalarFieldF32* inputScalarField,
AxScalarFieldF32* outLaplacianField,
bool isStaggeredGrid,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldLaplacianTex (
AxScalarFieldF32* inputScalarField,
AxScalarFieldF32* outLaplacianField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldLaplacianWithBoundary (
AxScalarFieldF32* inputScalarField,
AxScalarFieldF32* outLaplacianField,
bool isStaggeredGrid,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldSourcing (
AxScalarFieldF32* targetField,
AxScalarFieldF32* srcField,
AxFp32 scale,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldSourcing (
AxVecFieldF32* targetField,
AxVecFieldF32* srcField,
AxFp32 scale,
AxVector3 additionVel,
AxMatrix3x3 additionRotation,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CombustionWithGridAlign (
AxScalarFieldF32* temprature,
AxScalarFieldF32* fuel,
AxScalarFieldF32* density,
AxScalarFieldF32* divergence,
AxScalarFieldF32* heat,
AxScalarFieldF32* burn,
AxFp32 deltaTime,
AlphaCore::FluidUtility::Param::AxCombustionParam combustionParam,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CombustionWithReduceBurnAndFuel (
AxScalarFieldF32* temprature,
AxScalarFieldF32* fuel,
AxScalarFieldF32* density,
AxScalarFieldF32* divergence,
AxScalarFieldF32* heat,
AxScalarFieldF32* burn,
AxFp32 deltaTime,
AlphaCore::FluidUtility::Param::AxCombustionParam combustionParam,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void fieldMix (
AxFieldVector3F32* outField,
AxFieldVector3F32* aField,
AxFp32 coeffA,
AxFieldVector3F32* bField,
AxFp32 coeffB,
AxFp32 totalCoeff,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void fieldMix (
AxScalarFieldF32* outField,
AxScalarFieldF32* aField,
AxFp32 coeffA,
AxScalarFieldF32* bField,
AxFp32 coeffB,
AxFp32 totalCoeff,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void fieldMix (
AxVecFieldF32* outField,
AxVecFieldF32* aField,
AxFp32 coeffA,
AxVecFieldF32* bField,
AxFp32 coeffB,
AxFp32 totalCoeff,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void curl (
AxVecFieldF32* vecField,
AxVecFieldF32* curlField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void length (
AxVecFieldF32* vecField,
AxScalarFieldF32* magField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void gradient (
AxScalarFieldF32* inputScalarField,
AxVecFieldF32* outputVecField,
bool normalized,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void cross (
AxVecFieldF32* vecFieldA,
AxVecFieldF32* vecFieldB,
AxVecFieldF32* outRetField,
bool normalized,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void subtractGradient (
AxVecFieldF32* velField,
AxScalarFieldF32* pressureField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void ScalarDiffusionJacobi (
AxScalarFieldF32* inputScalarField,
AxScalarFieldF32* outScalarField,
AxFp32 rate,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void VectorDiffusionJacobi (
AxVecFieldF32* inputVecField,
AxVecFieldF32* outVecField,
AxFp32 rate,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void ClearFieldByMask (
AxVecFieldF32* velField,
AxScalarFieldF32* densityField,
AxScalarFieldF32* temperatureField,
AxScalarFieldF32* heatField,
AxScalarFieldI8* markField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void HeightFieldToMark (
AxScalarFieldI8* markField,
AxScalarFieldF32* heightField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void PressureSolverJacobiFast (
AxScalarFieldF32* pressureOldField,
AxScalarFieldF32* pressureNewField,
AxScalarFieldF32* divField,
AxScalarFieldI8* markField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void PressureSolverGaussSeidelFast (
AxScalarFieldF32* pressureOldField,
AxScalarFieldF32* pressureNewField,
AxScalarFieldF32* divField,
AxScalarFieldI8* markField,
bool redBlack,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void SubstractGradient (
AxVecFieldF32* velField,
AxScalarFieldF32* pressureOldField,
AxScalarFieldI8* markField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldDivergence (
AxVecFieldF32* velField,
AxScalarFieldF32* divField,
AxVecFieldF32* colliderVelField,
AxScalarFieldI8* mark,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void ProjectVelToCollider (
AxVecFieldF32* velInput,
AxVecFieldF32* velOutput,
AxScalarFieldI8* markField,
AxUInt32 blockSize = 512) ;

  
}
 }



namespace GridDense  
{
  namespace DX 
{
 ALPHA_SPMD_FUNC void AdvectSemiLagrangian (
AxScalarFieldF32* srcField,
AxScalarFieldF32* dstField,
AxFieldVector3F32* velField,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void AdvectSemiLagrangian (
AxScalarFieldF32* srcField,
AxScalarFieldF32* dstField,
AxVecFieldF32* velField,
AlphaCore::FluidUtility::AdvectTraceType traceMethod,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void ClampToExtrema (
AxScalarFieldF32* oldField,
AxScalarFieldF32* newField,
AxVecFieldF32* velField,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void ApplyBuoyancy (
AxScalarFieldF32* densityField,
AxScalarFieldF32* temperatureField,
AxVecFieldF32* velField,
AxFp32 alpha,
AxFp32 beta,
AxVector3 bouyancyDir,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void PressureSolverGaussSeidel (
AxScalarFieldF32* pressureOldField,
AxScalarFieldF32* pressureNewField,
AxScalarFieldF32* divergenceField,
bool redBlack,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void pressureSolverJacobi (
AxScalarFieldF32* pressureOldField,
AxScalarFieldF32* pressureNewField,
AxScalarFieldF32* divField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldDivergence (
AxVecFieldF32* vecField,
AxScalarFieldF32* outDivField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldLaplacian (
AxScalarFieldF32* inputScalarField,
AxScalarFieldF32* outLaplacianField,
bool isStaggeredGrid,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldLaplacianTex (
AxScalarFieldF32* inputScalarField,
AxScalarFieldF32* outLaplacianField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldLaplacianWithBoundary (
AxScalarFieldF32* inputScalarField,
AxScalarFieldF32* outLaplacianField,
bool isStaggeredGrid,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldSourcing (
AxScalarFieldF32* targetField,
AxScalarFieldF32* srcField,
AxFp32 scale,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldSourcing (
AxVecFieldF32* targetField,
AxVecFieldF32* srcField,
AxFp32 scale,
AxVector3 additionVel,
AxMatrix3x3 additionRotation,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CombustionWithGridAlign (
AxScalarFieldF32* temprature,
AxScalarFieldF32* fuel,
AxScalarFieldF32* density,
AxScalarFieldF32* divergence,
AxScalarFieldF32* heat,
AxScalarFieldF32* burn,
AxFp32 deltaTime,
AlphaCore::FluidUtility::Param::AxCombustionParam combustionParam,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void CombustionWithReduceBurnAndFuel (
AxScalarFieldF32* temprature,
AxScalarFieldF32* fuel,
AxScalarFieldF32* density,
AxScalarFieldF32* divergence,
AxScalarFieldF32* heat,
AxScalarFieldF32* burn,
AxFp32 deltaTime,
AlphaCore::FluidUtility::Param::AxCombustionParam combustionParam,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void fieldMix (
AxFieldVector3F32* outField,
AxFieldVector3F32* aField,
AxFp32 coeffA,
AxFieldVector3F32* bField,
AxFp32 coeffB,
AxFp32 totalCoeff,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void fieldMix (
AxScalarFieldF32* outField,
AxScalarFieldF32* aField,
AxFp32 coeffA,
AxScalarFieldF32* bField,
AxFp32 coeffB,
AxFp32 totalCoeff,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void fieldMix (
AxVecFieldF32* outField,
AxVecFieldF32* aField,
AxFp32 coeffA,
AxVecFieldF32* bField,
AxFp32 coeffB,
AxFp32 totalCoeff,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void curl (
AxVecFieldF32* vecField,
AxVecFieldF32* curlField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void length (
AxVecFieldF32* vecField,
AxScalarFieldF32* magField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void gradient (
AxScalarFieldF32* inputScalarField,
AxVecFieldF32* outputVecField,
bool normalized,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void cross (
AxVecFieldF32* vecFieldA,
AxVecFieldF32* vecFieldB,
AxVecFieldF32* outRetField,
bool normalized,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void subtractGradient (
AxVecFieldF32* velField,
AxScalarFieldF32* pressureField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void ScalarDiffusionJacobi (
AxScalarFieldF32* inputScalarField,
AxScalarFieldF32* outScalarField,
AxFp32 rate,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void VectorDiffusionJacobi (
AxVecFieldF32* inputVecField,
AxVecFieldF32* outVecField,
AxFp32 rate,
AxFp32 deltaTime,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void ClearFieldByMask (
AxVecFieldF32* velField,
AxScalarFieldF32* densityField,
AxScalarFieldF32* temperatureField,
AxScalarFieldF32* heatField,
AxScalarFieldI8* markField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void HeightFieldToMark (
AxScalarFieldI8* markField,
AxScalarFieldF32* heightField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void PressureSolverJacobiFast (
AxScalarFieldF32* pressureOldField,
AxScalarFieldF32* pressureNewField,
AxScalarFieldF32* divField,
AxScalarFieldI8* markField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void PressureSolverGaussSeidelFast (
AxScalarFieldF32* pressureOldField,
AxScalarFieldF32* pressureNewField,
AxScalarFieldF32* divField,
AxScalarFieldI8* markField,
bool redBlack,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void SubstractGradient (
AxVecFieldF32* velField,
AxScalarFieldF32* pressureOldField,
AxScalarFieldI8* markField,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void FieldDivergence (
AxVecFieldF32* velField,
AxScalarFieldF32* divField,
AxVecFieldF32* colliderVelField,
AxScalarFieldI8* mark,
AxUInt32 blockSize = 512) ;

ALPHA_SPMD_FUNC void ProjectVelToCollider (
AxVecFieldF32* velInput,
AxVecFieldF32* velOutput,
AxScalarFieldI8* markField,
AxUInt32 blockSize = 512) ;

  
}
 }



}
#endif //@FSA:[TOKEN]  __FSA_ARCH_PROTOCOL_AXGRIDDENSE__H__
#endif