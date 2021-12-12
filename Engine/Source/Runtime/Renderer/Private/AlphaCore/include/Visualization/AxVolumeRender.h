#ifndef __ALPHA_CORE_VOLUME_RENDER_H__
#define __ALPHA_CORE_VOLUME_RENDER_H__

#include <Utility/AxDescrition.h>
#include <Grid/AxFieldBase3D.h>
#include <Grid/AxFieldBase2D.h>

#include <Math/AxVectorHelper.h>
#include <Utility/AxImage.h>

#include <Grid/AxFluid3DOperator.h>

namespace AlphaCore
{
	namespace Visualization
	{
		void RenderField3D(
			AxScalarFieldF32* field,
			AlphaCore::Desc::AxCameraInfo& cam,
			AlphaCore::Desc::AxPointLightInfo& light,
			float stepSize,
			AxImageRGBA8& image,
			float ovFov = -1);
 
		namespace Internal
		{
			// case 1: ���ߴ��ⲿ�ཻ (0 <= min <= max)
			//
			//		dstA��dst������Ľ���㣬dstB dst��Զ����
			//
			// case 2: ���ߴ��ڲ��ཻ (min < 0 < max)
			//
			//		dstA��dst�����ߺ��ཻ��, dstB��dst�����򽻼�
			//
			// case 3: ����û���ཻ (min > max)
			//
			ALPHA_KERNEL_FUNC int RayBox(
				const AxVector3& pivot,
				const AxVector3& dir,
				const AxVector3& boxmin,
				const AxVector3& boxmax,
				float& tnear,
				float& tfar)
			{
				
				AxVector3 invR = MakeVector3(1.0f) / dir;
				AxVector3 tbot = invR * (boxmin - pivot);
				AxVector3 ttop = invR * (boxmax - pivot);

				// re-order intersections to find smallest and largest on each axis
				AxVector3 tmin = AlphaCore::Math::Min(ttop, tbot);
				AxVector3 tmax = AlphaCore::Math::Max(ttop, tbot);

				// find the largest tmin and the smallest tmax
				float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
				float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

				tnear = largest_tmin;
				tfar = smallest_tmax;

				return smallest_tmax > largest_tmin;
			}
			
			ALPHA_KERNEL_FUNC AxColorRGBA8 RayMarching(
				float* fieldRaw,
				const AlphaCore::Desc::AxField3DInfo& fieldInfo,
				const uInt32& x,
				const uInt32& y,
				const AxVector2UI& imgRes,
				const AxVector3& camPivot,
				const AxVector3& LUp,
				const AxVector3& LDown,
				const AxVector3& RUp,
				const AxVector3& RDown,
				const AxVector3& fieldBoxMin,
				const AxVector3& fieldBoxMax,
				float stepSize,
				uInt32 maxSteps)
			{
				//return MakeColorRGBA8(0, 0, 0, 0);

 				float v = (float)y / (float)(imgRes.y - 1);
				float u = (float)x / (float)(imgRes.x - 1);

				AxVector3 entryPoint = AlphaCore::Math::LerpV3(
					AlphaCore::Math::LerpV3(LUp, LDown, v),
					AlphaCore::Math::LerpV3(RUp, RDown, v),u);

				auto tmp = entryPoint - camPivot;
				AxVector3 worldViewDir = Normalize(tmp);
				float tnear, tfar;

				//TRACE_HDA_RAY(camPivot, worldViewDir);
				int hit = RayBox(
					camPivot,
					worldViewDir,
					fieldBoxMin,
					fieldBoxMax,
					tnear,
					tfar);
				if (!hit)
					return MakeColorRGBA8(0, 0, 0, 0);

				float  sumDensity = 0;
				float dstTravelled = tnear;
				float dstLimit = 10000.0;
				for (uInt32 i = 0; i < maxSteps; i++)
				{
					if (dstTravelled < dstLimit + tnear)
					{
						AxVector3 rayPos = entryPoint + (worldViewDir * dstTravelled);
						float val = AlphaCore::GridDense::Internal::SampleValue(rayPos, fieldRaw, fieldInfo);
						sumDensity += (val * 2.0f) * (val * 2.f);
					}
					dstTravelled += stepSize;
				}
				if (sumDensity > 255)
					sumDensity = 255;
				//printf(" Density:%f ", sumDensity);
				return MakeColorRGBA8(sumDensity, sumDensity, sumDensity, sumDensity);
			}
		}


		namespace CUDA
		{
			void RenderField3D(
				AxScalarFieldF32* field,
				AlphaCore::Desc::AxCameraInfo cam,
				AlphaCore::Desc::AxPointLightInfo light,
				float stepSize,
				AxImageRGBA8& image,
				float ovFov = -1);
		}
	}
}

#endif // !__ALPHA_CORE_VOLUME_RENDER_H__
