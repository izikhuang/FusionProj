#ifndef __ALPHA_CORE_FLUID3D_OPERATOR_KERNEL_H__
#define __ALPHA_CORE_FLUID3D_OPERATOR_KERNEL_H__

#include <Grid/AxFluid3DOperatorDevice.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

namespace AlphaCore
{
	namespace GridDense
	{
		namespace CUDA
		{
			namespace Kernel
			{
				using namespace AlphaCore::Math;
				template<class T>
				__global__ void advect_SemiLagrangian(
					T* srcRaw,	 //field to advect
					T* dstRaw,	 //advect result
					T* u,		 //vel.x field
					T* v,		 //vel.y field
					T* w,		 //vel.z field
					AlphaCore::Desc::AxField3DInfo srcInfo,
					float dt,
					uInt64 numVoxels)
				{
					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;

					AxVector3UI res = srcInfo.Resolution;
					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;
					AxVector3 pos = Index3ToPos(x, y, z, srcInfo);
					AxVector3 vel = MakeVector3(u[index], v[index], w[index]);
					pos -= vel * dt;//TraceMethod
					AlphaCore::GridDense::SetValue(
						x, y, z,
						AlphaCore::GridDense::Internal::SampleValue(pos, srcRaw, srcInfo),
						dstRaw,
						srcInfo);

				}

				__global__ void subtractGradient(
					float* vxRaw, float* vyRaw, float* vzRaw,
					AlphaCore::Desc::AxField3DInfo infoX,
					AlphaCore::Desc::AxField3DInfo infoY,
					AlphaCore::Desc::AxField3DInfo infoZ,
					float* scalarRaw,
					AlphaCore::Desc::AxField3DInfo scalarInfo,
					uInt64 numVoxels)
				{
					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;

					AxVector3UI res = scalarInfo.Resolution;
					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;

					AlphaCore::GridDense::Internal::SubstractGradient(x,y,z,
						vxRaw,vyRaw,vzRaw,
						infoX,infoY,infoZ, 
						scalarRaw,scalarInfo);

				}

				__global__ void addSourceField_Scalar(
					float* srcRaw,
					float* dstRaw,
					float scale,
					AlphaCore::Desc::AxField3DInfo srcInfo,
					AlphaCore::Desc::AxField3DInfo dstInfo,
					uInt64 numVoxels)
				{
					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;
					
					AxVector3UI res = dstInfo.Resolution;
					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;

					AxVector3 pos = AlphaCore::GridDense::Index3ToPos(x, y, z, dstInfo);
					float srcVal = AlphaCore::GridDense::Internal::SampleValueOld(pos, srcRaw, srcInfo);
					float currVal = AlphaCore::GridDense::GetValue(x, y, z, dstRaw, dstInfo);
					currVal += srcVal * scale;
					AlphaCore::GridDense::SetValue(x, y, z, currVal, dstRaw, dstInfo);
				}

				__global__  void addBuoyancy(
					float* tmpRaw,
					float* densityRaw,
					float* velRawX,//vel.x field
					float* velRawY,//vel.y field
					float* velRawZ,//vel.z field
					AxVector3 buoyancyDir,
					float alpha,
					float beta,
					float dt,
					AlphaCore::Desc::AxField3DInfo fieldInfo,
					uInt64 numVoxels)
				{

					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;

					AxVector3UI res = fieldInfo.Resolution;
					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;

					float temprature = GetValue(x, y, z, tmpRaw, fieldInfo);
					float density = GetValue(x, y, z, densityRaw, fieldInfo);
					AxVector3 v = GetValueVector(x, y, z, velRawX, velRawY, velRawZ, fieldInfo);
					// 正确的浮力公式：?
					//v.y += dt * (-alpha * density + beta * temprature);
					v += buoyancyDir * (dt * (-alpha * density + beta * temprature));
					v.x += 10.0*dt;
					SetValueVector(x, y, z, v, velRawX, velRawY, velRawZ, fieldInfo);
 				}

				__global__ void vorticityConfinement_Align_SHARE(
					float* velXRaw,
					float* velYRaw,
					float* velZRaw,
					AlphaCore::Desc::AxField3DInfo fieldInfo,
					uInt64 numVoxels)
				{
					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;

					/*
					AxVector3UI res = fieldInfo.Resolution;
					AxVector3 vs = fieldInfo.VoxelSize;
					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;
					*/
					//curl.length
					
					//gradient

					//cross

					//fieldMix
				}

				__global__ void curl(
					float* srcX,
					float* srcY,
					float* srcZ,
					float* curlX,
					float* curlY,
					float* curlZ,
					AlphaCore::Desc::AxField3DInfo fieldInfo,
					uInt64 numVoxels)
				{
					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;

					AxVector3UI res = fieldInfo.Resolution;
					AxVector3 vs = fieldInfo.VoxelSize;
					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;

					float cxx = (GetValue(x, y + 1, z, srcZ, fieldInfo) -
								 GetValue(x, y - 1, z, srcZ, fieldInfo)) / (vs.y * 2);
					float cxy = (GetValue(x, y, z + 1, srcY, fieldInfo) -
								 GetValue(x, y, z - 1, srcY, fieldInfo)) / (vs.z * 2);
					float cyx = (GetValue(x, y, z + 1, srcX, fieldInfo) -
								 GetValue(x, y, z - 1, srcX, fieldInfo)) / (vs.z * 2);
					float cyy = (GetValue(x + 1, y, z, srcZ, fieldInfo) -
								 GetValue(x - 1, y, z, srcZ, fieldInfo)) / (vs.x * 2);
					float czx = (GetValue(x + 1, y, z, srcY, fieldInfo) -
								 GetValue(x - 1, y, z, srcY, fieldInfo)) / (vs.x * 2);
					float czy = (GetValue(x, y + 1, z, srcX, fieldInfo) -
								 GetValue(x, y - 1, z, srcX, fieldInfo)) / (vs.y * 2);

					AxVector3 curl = MakeVector3(cxx - cxy, cyx - cyy, czx - czy);
					SetValueVector(x, y, z, curl, curlX, curlY, curlZ, fieldInfo);
				}


				__global__ void length(
					float* srcX,
					float* srcY,
					float* srcZ,
					float* dstRaw,
					AlphaCore::Desc::AxField3DInfo fieldInfo,
					uInt64 numVoxels)
				{
					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;

					AxVector3UI res = fieldInfo.Resolution;
					AxVector3 vs = fieldInfo.VoxelSize;
					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;

					AxVector3 vec = GetValueVector(x, y, z, srcX, srcY, srcZ, fieldInfo);
					SetValue(x, y, z, Length(vec), dstRaw, fieldInfo);
				}


				__global__ void gradient(
					float* srcRaw,
					float* dstRawX,
					float* dstRawY,
					float* dstRawZ,
					AlphaCore::Desc::AxField3DInfo fieldInfo,
					bool normalize,
					uInt64 numVoxels)
				{

					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;

					AxVector3UI res = fieldInfo.Resolution;
 					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;

					AxVector3 grad =  Internal::Gradient(x,y,z,srcRaw, dstRawX, dstRawY, dstRawZ,fieldInfo,normalize);
					SetValueVector(x, y, z, grad, dstRawX, dstRawY, dstRawZ, fieldInfo);

				}

				__global__ void divergence(
					float* srcVecRawX,
					float* srcVecRawY,
					float* srcVecRawZ,
					float* divRaw,
					AlphaCore::Desc::AxField3DInfo fieldInfo,
					uInt64 numVoxels)
				{
					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;

					AxVector3UI res = fieldInfo.Resolution;
					AxVector3 vs = fieldInfo.VoxelSize;
					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;

					float right  = GetValue(x + 1, y, z, srcVecRawX, fieldInfo);
					float left   = GetValue(x - 1, y, z, srcVecRawX, fieldInfo);
					float top    = GetValue(x, y + 1, z, srcVecRawY, fieldInfo);
					float bottom = GetValue(x, y - 1, z, srcVecRawY, fieldInfo);
					float front	 = GetValue(x, y, z + 1, srcVecRawZ, fieldInfo);
					float back	 = GetValue(x, y, z - 1, srcVecRawZ, fieldInfo);
					AxVector3 vc = GetValueVector(
						x, y, z,
						srcVecRawX,
						srcVecRawY, 
						srcVecRawZ, 
						fieldInfo);
		
					float div = (right-left)/(vs.x*2) + 
						(top-bottom)/(vs.y*2) +
						(front-back)/(vs.z*2);
					SetValue(x, y, z, div, divRaw, fieldInfo);
				}

				__global__ void fieldCross(
					float* aRawX,
					float* aRawY,
					float* aRawZ,
					float* bRawX,
					float* bRawY,
					float* bRawZ,
					float* retRawX,
					float* retRawY,
					float* retRawZ,
					AlphaCore::Desc::AxField3DInfo fieldInfo,
					uInt64 numVoxels)
				{
					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;

					AxVector3UI res = fieldInfo.Resolution;
					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;

					AxVector3 a = GetValueVector(x, y, z, aRawX, aRawY, aRawZ, fieldInfo);
					AxVector3 b = GetValueVector(x, y, z, bRawX, bRawY, bRawZ, fieldInfo);
					AxVector3 c = Cross(a, b);
					SetValueVector(x, y, z, c, retRawX, retRawY, retRawZ, fieldInfo);
				}


				__global__ void fieldMixV3(
					float* aRawX,
					float* aRawY,
					float* aRawZ,
					float coeffA,
					float* bRawX,
					float* bRawY,
					float* bRawZ,
					float coeffB,
					float* retRawX,
					float* retRawY,
					float* retRawZ,
					float totalCoeff,
					AlphaCore::Desc::AxField3DInfo fieldInfo,
					uInt64 numVoxels)
				{
					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;

					AxVector3UI res = fieldInfo.Resolution;
					AxVector3 vs = fieldInfo.VoxelSize;
					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;

					AxVector3 a = GetValueVector(x, y, z, aRawX, aRawY, aRawZ, fieldInfo);
					AxVector3 b = GetValueVector(x, y, z, bRawX, bRawY, bRawZ, fieldInfo);
					AxVector3 c = a * coeffA + b * coeffB;
					c *= totalCoeff;
					SetValueVector(x, y, z, c, retRawX, retRawY, retRawZ, fieldInfo);

				}


				__global__ void clampExtrema(
					float * fieldOldRaw,
					float * fieldNewRaw,
					float * vxRaw,
					float * vyRaw,
					float * vzRaw,
					float dt,
					AlphaCore::Desc::AxField3DInfo fieldInfo,
					AlphaCore::Desc::AxField3DInfo velFieldInfoX,
					AlphaCore::Desc::AxField3DInfo velFieldInfoY,
					AlphaCore::Desc::AxField3DInfo velFieldInfoZ,
					uInt64 numVoxels)
				{
					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;

					float v0, v1, v2, v3, v4, v5, v6, v7;
					AxVector3UI res = fieldInfo.Resolution;
					AxVector3 vs = fieldInfo.VoxelSize;
					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;

					AxVector3 pos = Index3ToPos(x, y, z, fieldInfo);
					AxVector3 vel = MakeVector3(
						GetValue(x, y, z, vxRaw, velFieldInfoX),
						GetValue(x, y, z, vyRaw, velFieldInfoY),
						GetValue(x, y, z, vzRaw, velFieldInfoZ));
					pos -= vel * dt;

					float newValue = GetValue(x, y, z, fieldNewRaw, fieldInfo);
					float SLv = AlphaCore::GridDense::Internal::SampleCubeAndLerp(pos, fieldOldRaw, fieldInfo, v0, v1, v2, v3, v4, v5, v6, v7);
					float minValue = MinF(v0, MinF(v1, MinF(v2, MinF(v3, MinF(v4, MinF(v5, MinF(v6, v7)))))));
					float maxValue = MaxF(v0, MaxF(v1, MaxF(v2, MaxF(v3, MaxF(v4, MaxF(v5, MaxF(v6, v7)))))));
					if (newValue < minValue || newValue > maxValue)
						SetValue(x, y, z, SLv, fieldNewRaw, fieldInfo);

					//f_np1(i,j,k) = max(min(max_value, f_np1(i,j,k)),min_value);

				}


				__global__ void fieldMix(
					float* retRaw,
					float* aRaw,
					float coeffA,
					float* bRaw,
					float coeffB,
					float totalCoeff,
					AlphaCore::Desc::AxField3DInfo fieldInfo,
					uInt64 numVoxels)
				{
					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;

					AxVector3UI res = fieldInfo.Resolution;
					AxVector3 vs = fieldInfo.VoxelSize;
					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;

					float a = GetValue(x, y, z, aRaw, fieldInfo);
					float b = GetValue(x, y, z, bRaw, fieldInfo);
					float c = a * coeffA + b * coeffB;
					c *= totalCoeff;
					SetValue(x, y, z, c, retRaw, fieldInfo);

				}

				__global__ void pressureSolverJacobi(
					float* pressureOldRaw,
					float* pressureNewRaw,
					float* divRaw,
					AlphaCore::Desc::AxField3DInfo fieldInfo,
					uInt64 numVoxels)
				{
					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;

					AxVector3UI res = fieldInfo.Resolution;
					AxVector3 vs = fieldInfo.VoxelSize;
					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;

					float wi = 1.0f / 6.0f;
					float pAlpha = -vs.x * vs.x;

					float pl = GetValue(x - 1, y, z, pressureOldRaw, fieldInfo);
					float pr = GetValue(x + 1, y, z, pressureOldRaw, fieldInfo);
					float pb = GetValue(x, y - 1, z, pressureOldRaw, fieldInfo);
					float pt = GetValue(x, y + 1, z, pressureOldRaw, fieldInfo);
					float ph = GetValue(x, y, z - 1, pressureOldRaw, fieldInfo);
					float pq = GetValue(x, y, z + 1, pressureOldRaw, fieldInfo);
					float div = GetValue(x, y, z, divRaw, fieldInfo);
					float pNew = (pl + pr + pb + pt + ph + pq + pAlpha * div) * wi;
					SetValue(x, y, z, pNew, pressureNewRaw, fieldInfo);

				}


				__global__ void combustion_GridAlign(
					float* divRaw, float* fuelRaw, float* heatRaw,
					float* tempRaw, float* densityRaw, float dt,
					AlphaCore::Param::AxCombustionParam combParam,
					AxField3DInfo info,
					uInt64 numVoxels)
				{
					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;
					AxVector3UI res = info.Resolution;
					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;
					Internal::Combustion_GridAlign(x, y, z, divRaw, fuelRaw, heatRaw, tempRaw, densityRaw, dt, combParam, info);
				}


				__global__ void pressureSolverGaussSeidel(
					float* pressOldRaw,
					float* pressNewRaw,
					float* divRaw,
					AlphaCore::Desc::AxField3DInfo info,
					bool gridSwith,
					uInt64 numVoxels)
				{
					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;

					AxVector3UI res = info.Resolution;
					AxVector3 vs = info.VoxelSize;
 					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;
					float pAlpha = -vs.x * vs.x;
					float wi = 1.0f / 6.0f;

					if (gridSwith)
					{
						if ((x + y + z) % 2 == 1)
							return;
						float pl = GetValue(x - 1, y, z, pressOldRaw, info);
						float pr = GetValue(x + 1, y, z, pressOldRaw, info);
						float pb = GetValue(x, y - 1, z, pressOldRaw, info);
						float pt = GetValue(x, y + 1, z, pressOldRaw, info);
						float ph = GetValue(x, y, z - 1, pressOldRaw, info);
						float pq = GetValue(x, y, z + 1, pressOldRaw, info);
						float div = GetValue(x, y, z, divRaw, info);
						float pNew = (pl + pr + pb + pt + ph + pq + pAlpha * div) * wi;
						SetValue(x, y, z, pNew, pressNewRaw, info);
					}
					else 
					{
						if ((x + y + z) % 2 == 0)
							return;
						float pl = GetValue(x - 1, y, z, pressNewRaw, info);
						float pr = GetValue(x + 1, y, z, pressNewRaw, info);
						float pb = GetValue(x, y - 1, z, pressNewRaw, info);
						float pt = GetValue(x, y + 1, z, pressNewRaw, info);
						float ph = GetValue(x, y, z - 1, pressNewRaw, info);
						float pq = GetValue(x, y, z + 1, pressNewRaw, info);
						float div = GetValue(x, y, z, divRaw, info);
						float pNew = (pl + pr + pb + pt + ph + pq + pAlpha * div) * wi;
						SetValue(x, y, z, pNew, pressNewRaw, info);
					}

				}

				__global__ void combustion_GridAlign(
					float* tempRaw,
					float* fuelRaw,
					float* densityRaw,
					float* divRaw,
					float* burnRaw,
					float* heatRaw,
					float dt,
					AlphaCore::Param::AxCombustionParam combParam,
					AlphaCore::Desc::AxField3DInfo fieldInfo,
					uInt64 numVoxels)
				{
					unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
					if (index >= numVoxels)
						return;

					AxVector3UI res = fieldInfo.Resolution;
					AxVector3 vs = fieldInfo.VoxelSize;
					uInt64 nvSlice = res.x*res.y;
					uInt32 x = index % res.x;
					uInt32 y = (index % nvSlice) / res.x;
					uInt32 z = index / nvSlice;

					Internal::Combustion_GridAlign(x, y, z, divRaw, fuelRaw, heatRaw, tempRaw, densityRaw, dt, combParam, fieldInfo);
		
				}


				template<class T>
				void advect_MacCormack(T* src, T* dst, T* u, T* v, T* w, float dt)
				{

				}

			}

		}
	}
}

#endif 
