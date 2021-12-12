#ifndef __ALPHA_CORE_FLUID3D_OPERATOR_H__
#define __ALPHA_CORE_FLUID3D_OPERATOR_H__

#include "AxFieldBase3D.h"
#include <Math/AxMath101.h>
namespace AlphaCore
{
	namespace GridDense
	{
		using namespace AlphaCore::Desc;
		ALPHA_KERNEL_FUNC AxVector3UI IndexToIndex3(uInt32 index, AxField3DInfo& info)
		{
			return MakeVector3UI(0);
		}

		ALPHA_KERNEL_FUNC AxVector3 Index3ToPos(
			int idx, int idy, int idz, 
			AxField3DInfo& info)
		{
			AxVector3 size = MakeVector3(
				info.VoxelSize.x * info.Resolution.x,
				info.VoxelSize.y * info.Resolution.y,
				info.VoxelSize.z * info.Resolution.z);
			AxVector3 origin = info.Pivot - size * 0.5f;
			AxVector3 relPos = MakeVector3(
				((float)idx + 0.5f) * info.VoxelSize.x,
				((float)idy + 0.5f) * info.VoxelSize.y,
				((float)idz + 0.5f) * info.VoxelSize.z);
			return origin + relPos;
		}
		
		ALPHA_KERNEL_FUNC AxVector3 Index3ToPos(AxVector3I id3, AxField3DInfo& info)
		{
			return AlphaCore::GridDense::Index3ToPos(id3.x, id3.y, id3.z, info);
		}

		ALPHA_KERNEL_FUNC uInt32 Index3ToIndex(int idx, int idy, int idz, AxField3DInfo& info)
		{
			return idz * info.Resolution.x *info.Resolution.y + idy * info.Resolution.x + idx;
		}

		ALPHA_KERNEL_FUNC uInt32 Index3ToIndex(AxVector3I id3, AxField3DInfo& info)
		{
			return Index3ToIndex(id3.x, id3.y, id3.z, info);
		}

		ALPHA_KERNEL_FUNC uInt32 PositionToIndex(AxVector3 pos, AxField3DInfo& info)
		{
			return 1;
		}

		ALPHA_KERNEL_FUNC AxVector3UI PositionToIndex3(AxVector3 pos, AxField3DInfo& info)
		{
			return  MakeVector3UI(0);
		}

		ALPHA_KERNEL_FUNC bool IsInside(int ix, int iy, int iz, const AxField3DInfo& info)
		{
			if (ix < 0 || ix >= (int)info.Resolution.x ||
				iy < 0 || iy >= (int)info.Resolution.y ||
				iz < 0 || iz >= (int)info.Resolution.z)
				return false;
			return true;
		}

		ALPHA_KERNEL_FUNC float GetValue(int idx, 
										 int idy, 
										 int idz, 
										 float* scalarData,
										 AxField3DInfo& info, 
										 float backgroundValue = 0)
		{
			if (!IsInside(idx, idy, idz, info))
				return 0;
			idx = AlphaCore::Math::Clamp(idx, 0, info.Resolution.x - 1);
			idy = AlphaCore::Math::Clamp(idy, 0, info.Resolution.y - 1);
			idz = AlphaCore::Math::Clamp(idz, 0, info.Resolution.z - 1);
			return scalarData[Index3ToIndex(idx, idy, idz, info)];
		}

		ALPHA_KERNEL_FUNC float GetValue(AxVector3I id3,
			float* scalarData,
			AxField3DInfo& info,
			float backgroundValue = 0)
		{
			return AlphaCore::GridDense::GetValue(id3.x, id3.y, id3.z, scalarData, info);
		}


		ALPHA_KERNEL_FUNC void  SetValue(int idx,
										 int idy,
										 int idz,
										 float value,
										 float* dataRaw,
										 AxField3DInfo& info)
		{
			dataRaw[Index3ToIndex(idx, idy, idz, info)] = value;
		}


		ALPHA_KERNEL_FUNC AxVector3 GetValueVector(
										 int idx,
										 int idy,
										 int idz,
										 float* dataRawX,
										 float* dataRawY,
										 float* dataRawZ,
										 AxField3DInfo& info,
										 float backgroundValue = 0)
		{
			return MakeVector3(
				GetValue(idx, idy, idz, dataRawX, info, backgroundValue),
				GetValue(idx, idy, idz, dataRawY, info, backgroundValue),
				GetValue(idx, idy, idz, dataRawZ, info, backgroundValue));
		}


		ALPHA_KERNEL_FUNC void SetValueVector(int idx,
										 int idy,
										 int idz,
										 const AxVector3& value,
										 float* dataRawX,
										 float* dataRawY,
										 float* dataRawZ,
										 AxField3DInfo& info)

		{
			SetValue(idx, idy, idz, value.x, dataRawX, info);
			SetValue(idx, idy, idz, value.y, dataRawY, info);
			SetValue(idx, idy, idz, value.z, dataRawZ, info);
		}

		ALPHA_KERNEL_FUNC AxVector3 GetGradient(
										 int idx,
										 int idy,
										 int idz,
										 float* dataRaw,
										 AxField3DInfo& info,
										 float backgroundValue = 0)

		{
			float vr = GetValue(idx + 1, idy, idz, dataRaw, info);
			float vl = GetValue(idx - 1, idy, idz, dataRaw, info);
			float vt = GetValue(idx, idy + 1, idz, dataRaw, info);
			float vb = GetValue(idx, idy - 1, idz, dataRaw, info);
			float vq = GetValue(idx, idy, idz + 1, dataRaw, info);
			float vh = GetValue(idx, idy, idz - 1, dataRaw, info);
			return MakeVector3((vr - vl) / (info.VoxelSize.x * 2),
				(vt - vb) / (info.VoxelSize.y * 2),
				(vq - vh) / (info.VoxelSize.z * 2));
		}

		ALPHA_KERNEL_FUNC bool IsInside(AxVector3 pos, AxField3DInfo& info)
		{
			AxVector3 size = MakeVector3(
				info.VoxelSize.x * info.Resolution.x,
				info.VoxelSize.y * info.Resolution.y,
				info.VoxelSize.z * info.Resolution.z);
			AxVector3 origin = info.Pivot - size * 0.5f;
			AxVector3 bboxSpace = MakeVector3(
				((pos.x - origin.x) / size.x),
				((pos.y - origin.y) / size.y),
				((pos.z - origin.z) / size.z));
			if (bboxSpace.x<0 || bboxSpace.x>1.0f ||
				bboxSpace.y<0 || bboxSpace.y>1.0f ||
				bboxSpace.z<0 || bboxSpace.z>1.0f)
				return false;
			return true;
		}

		/*------------------------------------------------------
		*
		*	SMPD Function 
		*
		*------------------------------------------------------*/

		ALPHA_SPMD AxVector3 SampleValueVector(AxVector3 pos,
							   AxVecFieldF32* vecField,
 							   bool activeBackground = false,
							   float backgroundValue = 0);

		ALPHA_SPMD AxVector3 TraceRK3(AxVector3 & pos, AxVecFieldF32* velField, float dt);

  
		ALPHA_SPMD void ClampExtrema(
							   AxScalarFieldF32* fieldOld,
							   AxScalarFieldF32* fieldNew,
							   AxVecFieldF32* velField,float dt);

		//seim-lagrangian advect method
		ALPHA_SPMD void Advect(AxVecFieldF32* src, 
							   AxVecFieldF32* dst, 
							   AxVecFieldF32* vel,
							   AxScalarFieldF32* advectTmp,
							   AlphaCore::Flow::AdvectType type,
							   float dt,bool loadBack=false);

		ALPHA_SPMD void Advect(AxScalarFieldF32* src,
 							   AxVecFieldF32* vel,
							   AxScalarFieldF32* advectTmp,
							   AlphaCore::Flow::AdvectType type,
							   float dt);

		ALPHA_SPMD void Advect(AxScalarFieldF32* src, 
							   AxScalarFieldF32* dst, 
							   AxVecFieldF32* velField, 
							   AxScalarFieldF32* advectTmp,
							   AlphaCore::Flow::AdvectType type,
							   float dt, bool loadBack = false);
		
		ALPHA_SPMD void Advect_MacCormack(AxScalarFieldF32* src,
							   AxScalarFieldF32* dst,
							   AxScalarFieldF32* advectTmp,
							   AxVecFieldF32* velField, 
 							   float dt);

	

		ALPHA_SPMD void Advect_BFECC(AxScalarFieldF32* src,
							   AxScalarFieldF32* dst, 
							   AxScalarFieldF32* advectTmp,
							   AxVecFieldF32* velField, 
 							   float dt);

		ALPHA_SPMD void Advect_SemiLagrangian(AxScalarFieldF32* src,
							   AxScalarFieldF32* dst, 
							   AxVecFieldF32* velField, 
 							   float dt);

		ALPHA_SPMD void ApplyBuoyancy(AxScalarFieldF32* density,
							   AxScalarFieldF32* tmp, 
							   AxVecFieldF32* vel,
							   float alpha,
							   float beta,
							   float dt);

		ALPHA_SPMD void SetBoundaryCondition(AxVecFieldF32* vel,
							   bool openX,bool open_X,
							   bool openY,bool open_Y,
					     	   bool openZ,bool open_Z);


		ALPHA_SPMD void FieldMix(AxScalarFieldF32* dst,
							   AxScalarFieldF32* aField,float coeffA,
							   AxScalarFieldF32* bField, float coeffB,
							   float totalCoeff = 1.0f);
 		///ALPHA_SPMD void FieldSubtract(AxScalarFieldF32* a, AxScalarFieldF32* b,float coffe);

		ALPHA_SPMD void ProjectNonDivergence(AxVecFieldF32* velField,
							   AxScalarFieldF32 * divField,
							   AxScalarFieldF32 * pressureOld,
							   AxScalarFieldF32 * pressureNew,
							   uInt32 iterations, 
 							   AlphaCore::LinearSolver solverType);

		ALPHA_SPMD void PressureSolverJacobi(AxScalarFieldF32 * pressureOld, 
							   AxScalarFieldF32 * pressureNew,
							   AxScalarFieldF32 * divField);

		ALPHA_SPMD void PressureSolverGaussSeidel(AxScalarFieldF32 * pressureOld,
							   AxScalarFieldF32 * pressureNew,
							   AxScalarFieldF32 * divField);

		ALPHA_SPMD void VorticityConfinement(AxVecFieldF32* velField, 
							   AxVecFieldF32* curlField, 
							   AxScalarFieldF32* curlMagField,
							   AxVecFieldF32* vortexDirField,
							   float confinementScale,
							   float dt);

		ALPHA_SPMD void AddSourceField_Scalar(
							   AxScalarFieldF32* dstField,
							   AxScalarFieldF32* src,
							   float scale,
							   float dt);

		ALPHA_SPMD float CFLCondition(AxVecFieldF32* vecField, float dt);
		///ALPHA_SPMD void VectorFieldMultiplyVector3(AxVecFieldF32* a,AxVecFieldF32* b, AxVector3 vec);

		namespace Internal
		{
			ALPHA_KERNEL_FUNC float SampleCubeAndLerp(
				AxVector3& pos, float* fieldRaw,
				AlphaCore::Desc::AxField3DInfo info,
				float &v0, float &v1, float &v2, float &v3,
				float &v4, float &v5, float &v6, float &v7)
			{
				AxVector3 size = MakeVector3(
					info.VoxelSize.x * info.Resolution.x,
					info.VoxelSize.y * info.Resolution.y,
					info.VoxelSize.z * info.Resolution.z);
				AxVector3 origin = info.Pivot - size * 0.5f;
				AxVector3 bboxSpace = MakeVector3(
					((pos.x - origin.x) / size.x),
					((pos.y - origin.y) / size.y),
					((pos.z - origin.z) / size.z));

				AxVector3 voxelCoord = MakeVector3(
					bboxSpace.x*(float)info.Resolution.x,
					bboxSpace.y*(float)info.Resolution.y,
					bboxSpace.z*(float)info.Resolution.z);

				AxVector3 coordB = voxelCoord - 0.5f;

				AxVector3I lowerLeft = MakeVector3I(
					floor(coordB.x),
					floor(coordB.y),
					floor(coordB.z));

				float cx = coordB.x - (float)lowerLeft.x;
				float cy = coordB.y - (float)lowerLeft.y;
				float cz = coordB.z - (float)lowerLeft.z;

				v0 = AlphaCore::GridDense::GetValue(lowerLeft.x, lowerLeft.y, lowerLeft.z, fieldRaw, info);
				v1 = AlphaCore::GridDense::GetValue(lowerLeft.x + 1, lowerLeft.y, lowerLeft.z, fieldRaw, info);
				v2 = AlphaCore::GridDense::GetValue(lowerLeft.x, lowerLeft.y + 1, lowerLeft.z, fieldRaw, info);
				v3 = AlphaCore::GridDense::GetValue(lowerLeft.x + 1, lowerLeft.y + 1, lowerLeft.z, fieldRaw, info);
				v4 = AlphaCore::GridDense::GetValue(lowerLeft.x, lowerLeft.y, lowerLeft.z + 1, fieldRaw, info);
				v5 = AlphaCore::GridDense::GetValue(lowerLeft.x + 1, lowerLeft.y, lowerLeft.z + 1, fieldRaw, info);
				v6 = AlphaCore::GridDense::GetValue(lowerLeft.x, lowerLeft.y + 1, lowerLeft.z + 1, fieldRaw, info);
				v7 = AlphaCore::GridDense::GetValue(lowerLeft.x + 1, lowerLeft.y + 1, lowerLeft.z + 1, fieldRaw, info);

				float iv1 = AlphaCore::Math::LerpF32(AlphaCore::Math::LerpF32(v0, v1, cx), AlphaCore::Math::LerpF32(v2, v3, cx), cy);
				float iv2 = AlphaCore::Math::LerpF32(AlphaCore::Math::LerpF32(v4, v5, cx), AlphaCore::Math::LerpF32(v6, v7, cx), cy);
				return  AlphaCore::Math::LerpF32(iv1, iv2, cz);
			}

			ALPHA_KERNEL_FUNC float SampleValue(AxVector3& pos, float* rawData, AxField3DInfo info)
			{
				AxVector3 size = MakeVector3(
					info.VoxelSize.x * info.Resolution.x,
					info.VoxelSize.y * info.Resolution.y,
					info.VoxelSize.z * info.Resolution.z);
				AxVector3 origin = info.Pivot - size * 0.5f;
				AxVector3 bboxSpace = MakeVector3(
					((pos.x - origin.x) / size.x),
					((pos.y - origin.y) / size.y),
					((pos.z - origin.z) / size.z));

				AxVector3 voxelCoord = MakeVector3(
					bboxSpace.x*(float)info.Resolution.x,
					bboxSpace.y*(float)info.Resolution.y,
					bboxSpace.z*(float)info.Resolution.z);

				AxVector3 coordB = voxelCoord - 0.5f;
				AxVector3I lowerLeft = MakeVector3I(
					floor(coordB.x),
					floor(coordB.y),
					floor(coordB.z));

				float cx = coordB.x - (float)lowerLeft.x;
				float cy = coordB.y - (float)lowerLeft.y;
				float cz = coordB.z - (float)lowerLeft.z;

				float v0 = AlphaCore::GridDense::GetValue(lowerLeft.x, lowerLeft.y, lowerLeft.z, rawData, info);
				float v1 = AlphaCore::GridDense::GetValue(lowerLeft.x + 1, lowerLeft.y, lowerLeft.z, rawData, info);
				float v2 = AlphaCore::GridDense::GetValue(lowerLeft.x, lowerLeft.y + 1, lowerLeft.z, rawData, info);
				float v3 = AlphaCore::GridDense::GetValue(lowerLeft.x + 1, lowerLeft.y + 1, lowerLeft.z, rawData, info);
				float v4 = AlphaCore::GridDense::GetValue(lowerLeft.x, lowerLeft.y, lowerLeft.z + 1, rawData, info);
				float v5 = AlphaCore::GridDense::GetValue(lowerLeft.x + 1, lowerLeft.y, lowerLeft.z + 1, rawData, info);
				float v6 = AlphaCore::GridDense::GetValue(lowerLeft.x, lowerLeft.y + 1, lowerLeft.z + 1, rawData, info);
				float v7 = AlphaCore::GridDense::GetValue(lowerLeft.x + 1, lowerLeft.y + 1, lowerLeft.z + 1, rawData, info);

				float iv1 = AlphaCore::Math::LerpF32(AlphaCore::Math::LerpF32(v0, v1, cx), AlphaCore::Math::LerpF32(v2, v3, cx), cy);
				float iv2 = AlphaCore::Math::LerpF32(AlphaCore::Math::LerpF32(v4, v5, cx), AlphaCore::Math::LerpF32(v6, v7, cx), cy);
				return  AlphaCore::Math::LerpF32(iv1, iv2, cz);
			}

			ALPHA_KERNEL_FUNC float SampleValueOld(
				AxVector3 pos,
				float* scalarData,
				AxField3DInfo& info,
				bool activeBackground = false,
				float backgroundValue = 0)
			{
				AxVector3 size = MakeVector3(
					info.VoxelSize.x * info.Resolution.x,
					info.VoxelSize.y * info.Resolution.y,
					info.VoxelSize.z * info.Resolution.z);
				AxVector3 origin = info.Pivot - size * 0.5f;
				AxVector3 bboxSpace = MakeVector3(
					((pos.x - origin.x) / size.x),
					((pos.y - origin.y) / size.y),
					((pos.z - origin.z) / size.z));

				AxVector3 voxelCoord = MakeVector3(
					bboxSpace.x*(float)info.Resolution.x,
					bboxSpace.y*(float)info.Resolution.y,
					bboxSpace.z*(float)info.Resolution.z);

				AxVector3 coordB = voxelCoord - 0.5f;
				AxVector3I lowerLeft = MakeVector3I(
					floor(coordB.x),
					floor(coordB.y),
					floor(coordB.z));

 				AxVector3 weight = MakeVector3();
				float value = 0;

				if (!IsInside(lowerLeft.x, lowerLeft.y, lowerLeft.z, info))
					return activeBackground ? backgroundValue : value;
				if (!IsInside(lowerLeft.x + 1, lowerLeft.y + 1, lowerLeft.z + 1, info))
					return activeBackground ? backgroundValue : value;

				for (int i = 0; i < 2; i++)
				{
					int idx = lowerLeft.x + i;
					weight.x = 1.0 - fabsf(coordB.x - idx);
					for (int j = 0; j < 2; j++)
					{
						int idy = lowerLeft.y + j;
						weight.y = 1.0 - fabsf(coordB.y - idy);
						for (int k = 0; k < 2; k++)
						{
							int idz = lowerLeft.z + k;
							weight.z = 1.0 - fabsf(coordB.z - idz);
							float sampleValue = GetValue(idx, idy, idz, scalarData, info);
							value += weight.x * weight.y * weight.z * sampleValue;
						}
					}
				}

				return value;
			}

			ALPHA_KERNEL_FUNC void Combustion_GridAlign(uInt32 x, uInt32 y, uInt32 z,
				float* divRaw, float* fuelRaw,   float* heatRaw,
				float* tempRaw,float* densityRaw,float dt,
				const AlphaCore::Param::AxCombustionParam& combParam,
				AxField3DInfo& info)
			{
				float ignitionTemp	= combParam.IgnitionTemperature;
				float burnRate		= combParam.BurnRate;
				float fuelIneff		= combParam.FuelInefficiency;
				float gasRelease	= combParam.GasRelease;
				float burn2Div		= combParam.GasBurnInfluence;
				float heat2Div		= combParam.GasHeatInfluence;
				float burn2Temp		= combParam.TempBurnInfluence;
				float heat2Temp		= combParam.TempHeatInfluence;
				float heatOutput	= combParam.TemperatureOutput;
				//fuel to burn
				float divVal = GetValue(x, y, z, divRaw, info);
				float fuelVal = GetValue(x, y, z, fuelRaw, info);
				float heatVal = GetValue(x, y, z, heatRaw, info);
				float tempVal = GetValue(x, y, z, tempRaw, info);
				float densityVal = GetValue(x, y, z, densityRaw, info);

				float burnVal = tempVal - ignitionTemp;
				//clamp negative
				burnVal = fmaxf(fmaxf(burnVal, burnVal * 100), 0);
				burnVal = fminf(fuelVal, burnVal); //each frame?

				//disspzition
				burnVal -= burnVal * pow(1.0f - burnRate, dt);

				//fuel cost 
				fuelVal -= (1 - fuelIneff)*burnVal;

				//normalize burn ?
				//burnVal *= dt;

				// burn to hear
				heatVal = fmaxf(burnVal, heatVal);
				float newCombustion = (heatVal - burnVal)*dt;

				//build divergences
				divVal += gasRelease * burn2Div * burnVal +
						  gasRelease * heat2Div * newCombustion;

				//Set Temp / divergence / heat
				tempVal += heatOutput * burn2Temp * burnVal +		//	24 *dt
						   heatOutput * heat2Temp * newCombustion;	//	24 *dt

				SetValue(x, y, z, tempVal, tempRaw, info);
				SetValue(x, y, z, divVal, divRaw, info);
				SetValue(x, y, z, heatVal, heatRaw, info);
				SetValue(x, y, z, fuelVal, fuelRaw, info);

				if (heatVal > 0 && heatVal < 1.0)
					densityVal += (1.0 - heatVal)*dt*heatVal*3.0f;

				SetValue(x, y, z, densityVal, densityRaw, info);
			}

			ALPHA_KERNEL_FUNC void SubstractGradient(
				uInt32 x, uInt32 y, uInt32 z,
				float* vxRaw, float* vyRaw, float* vzRaw,
				AxField3DInfo infoVx, 
				AxField3DInfo infoVy, 
				AxField3DInfo infoVz,
				float* scalarRaw,
				AxField3DInfo infoScalar)
			{
				AxVector3 grad = GetGradient(x, y, z, scalarRaw, infoScalar);
				AxVector3 v = GetValueVector(x, y, z, vxRaw, vyRaw, vzRaw, infoScalar);
				v -= grad;
				SetValueVector(x, y, z, v, vxRaw, vyRaw, vzRaw, infoScalar); 
			}
			ALPHA_KERNEL_FUNC AxVector3 Gradient(
				uInt32 x, uInt32 y, uInt32 z,
				float* dataRaw,
				float* dstRawX,
				float* dstRawY,
				float* dstRawZ,
				AlphaCore::Desc::AxField3DInfo info,
				bool normalize)
			{
				float vr = GetValue(x + 1, y, z, dataRaw, info);
				float vl = GetValue(x - 1, y, z, dataRaw, info);
				float vt = GetValue(x, y + 1, z, dataRaw, info);
				float vb = GetValue(x, y - 1, z, dataRaw, info);
				float vq = GetValue(x, y, z + 1, dataRaw, info);
				float vh = GetValue(x, y, z - 1, dataRaw, info);
				AxVector3 grad = MakeVector3(
					(vr - vl) / (info.VoxelSize.x * 2),
					(vt - vb) / (info.VoxelSize.y * 2),
					(vq - vh) / (info.VoxelSize.z * 2));
				if (normalize)
					Normalize(grad);
				return grad;
			}

		}


		ALPHA_SPMD void Divergence(AxVecFieldF32 * srcVecField, AxScalarFieldF32 * dstDivField);
		ALPHA_SPMD void Curl(AxVecFieldF32 * srcVecField, AxVecFieldF32 * dstCurlField);
		ALPHA_SPMD void Length(AxVecFieldF32 * srcVecField, AxScalarFieldF32 * dstLenField);
		ALPHA_SPMD void Gradient(AxScalarFieldF32 * srcScalarField, AxVecFieldF32 * dstLenField,bool normalize=false);
		ALPHA_SPMD void FieldCross(AxVecFieldF32 * a, AxVecFieldF32 * b,AxVecFieldF32 * ret);
		ALPHA_SPMD void SubtractGradient(AxVecFieldF32* velField,AxScalarFieldF32 * pressureField);
		ALPHA_SPMD void LinearCombation(
			AxScalarFieldF32* ret,
			AxScalarFieldF32* a,
			AxScalarFieldF32* b,
			float aCoeff,
			float bCoeff,
			float totalCoeff=1.0f);

		ALPHA_SPMD void Combustion_GridAlign(
			AxScalarFieldF32* temperatureField,
			AxScalarFieldF32* fuelField,
			AxScalarFieldF32* densityField,
			AxScalarFieldF32* divField,
			AxScalarFieldF32* burnField,
			AxScalarFieldF32* heatField,
			float dt,
			AlphaCore::Param::AxCombustionParam combParam);

		ALPHA_SPMD void Combustion_GridSample();

		ALPHA_SPMD void Combustion_ParticleSample();


		void SaveFields(std::string path,std::vector<AxScalarFieldF32*>& fieldList);

		void ReadFields(std::string path, std::vector<AxScalarFieldF32*>& fieldList);


	}
}

#endif