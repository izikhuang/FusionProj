#ifndef __AX_VOLUMERENDER_SHARECODE_H__
#define __AX_VOLUMERENDER_SHARECODE_H__

#include <Math/AxVectorBase.h>
#include <Math/AxVectorHelper.h>
#include <Utility/AxStorage.h>
#include <Math/AxMatrixBase.h>
#include <Math/AxMath101.h>
#include <Grid/AxFluid3DOperator.h>

namespace AlphaCore {
	namespace VolumeRender {
		namespace ShareCode {

            ALPHA_SHARE_FUNC int rayBox(
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

            ALPHA_SHARE_FUNC float densityMap(float density, const AxGasVolumeMaterial& material) {
                density = AlphaCore::Math::ClampF32(density, material.minMaxInputDensity.x, material.minMaxInputDensity.y);
                density = AlphaCore::Math::RemapF32(density, material.minMaxInputDensity.x, material.minMaxInputDensity.y, 0.f, 1.f);
                density *= material.densityScale;
                density *= 127;
                int idx = int(density);
                float ret = material.lookUpTableDensity[idx];
                ret *= material.densityScale;
                return ret;
            }

            ALPHA_SHARE_FUNC AxColorRGBA densityColorMap(float density, const AxGasVolumeMaterial& material) {
                AxColorRGBA8 tmp;
                density = AlphaCore::Math::ClampF32(density, material.minMaxInputDensity.x, material.minMaxInputDensity.y);
                density = AlphaCore::Math::RemapF32(density, material.minMaxInputDensity.x, material.minMaxInputDensity.y, 0, 1);
                density *= 127;
                int idx = int(density);
                tmp = material.lookUpTableDensityColor[idx];
                AxColorRGBA ret;
                ret.r = material.minMaxOuputTemperature.x + tmp.r / 255.f;
                ret.g = material.minMaxOuputTemperature.x + tmp.g / 255.f;
                ret.b = material.minMaxOuputTemperature.x + tmp.b / 255.f;
                ret.a = material.minMaxOuputTemperature.x + tmp.a / 255.f;

                return ret;
            }


            ALPHA_SHARE_FUNC float heatMap(float heat, const AxGasVolumeMaterial& material) {

                heat = AlphaCore::Math::ClampF32(heat, material.minMaxInputHeat.x, material.minMaxInputHeat.y);
                heat = AlphaCore::Math::RemapF32(heat, material.minMaxInputHeat.x, material.minMaxInputHeat.y, 0.f, 1.f);
                heat *= 127;
                int idx = int(heat);
                float ret = material.LookUpTableHeat[idx];
                return ret;
            }

            ALPHA_SHARE_FUNC AxColorRGBA temperatureMap(float temperature, const AxGasVolumeMaterial& material) {
                AxColorRGBA8 tmp;
                temperature = AlphaCore::Math::ClampF32(temperature, material.minMaxInputTemperature.x, material.minMaxInputTemperature.y);
                temperature = AlphaCore::Math::RemapF32(temperature, material.minMaxInputTemperature.x, material.minMaxInputTemperature.y, 0, 1);
                temperature *= 127;
                int idx = int(temperature);
                tmp = material.LookUpTableTemperature[idx];

                AxColorRGBA ret;
                ret.r = material.minMaxOuputTemperature.x + tmp.r / 255.f * (material.minMaxOuputTemperature.y - material.minMaxOuputTemperature.x);
                ret.g = material.minMaxOuputTemperature.x + tmp.g / 255.f * (material.minMaxOuputTemperature.y - material.minMaxOuputTemperature.x);
                ret.b = material.minMaxOuputTemperature.x + tmp.b / 255.f * (material.minMaxOuputTemperature.y - material.minMaxOuputTemperature.x);
                ret.a = material.minMaxOuputTemperature.x + tmp.a / 255.f * (material.minMaxOuputTemperature.y - material.minMaxOuputTemperature.x);

                return ret;
            }



            ALPHA_SHARE_FUNC AxColorRGBA lightMarching(const AxVolumeRenderObject& volume, const AlphaCore::Desc::AxPointLightInfo& lightInfo,
                const AxVector3& rayPos,const AxVector3& boxMin, const AxVector3& boxMax, AxMatrix4x4 xform)
            {

                AxVector3 lightPivot;
                lightPivot = lightInfo.Pivot;
                lightPivot *= xform;

                float tNear, tFar;
                auto lightDir = rayPos - lightPivot;
                Normalize(lightDir);
                rayBox(lightPivot, lightDir, boxMin, boxMax, tNear, tFar);
                constexpr int maxStep = 128;
                auto t0 = ((rayPos - lightPivot) / lightDir).x;
                auto density = 0.f;
                auto dt = (t0 - tNear) / maxStep;
                for (int i = 0; i < maxStep; ++i)
                {
                    auto t = t0 - i * dt;
                    auto pos = lightPivot + t * lightDir;
                    density += densityMap(AlphaCore::GridDense::Internal::SampleValue(pos, volume.density, volume.densityInfo), volume.material);
                }
                float shadowTerms = exp(-density * dt * volume.material.shadowScale);
                return lightInfo.LightColor * shadowTerms * lightInfo.Intensity;
            }


			ALPHA_SHARE_FUNC void GasVolumeRender(
				AxUInt32 idx,
				AxVolumeRenderObject gasVolumeRenderObject,
				AxSceneRenderDesc sceneDesc,
				AxColorRGBA8* outputTexRaw,
				AxFp32* depthTexRaw,
				AxFp32 stepSize,
				int width,
				int height,
				AxMatrix4x4 postXform)
			{
                if (idx >= width * height) return;
               
                //
                //return;
                int index = (int)idx;
                int x = width - idx % width;
                int y = height - idx / (width);
                //if (idx<30)printf("idx:%d index:%d  x:%d y:%d\n", idx, index,x,y);
                //if (x != 88 || y != 100) { outputTexRaw[idx] = MakeColorRGBA8(0, 0, 0, 1); return; }
                float ratio = (float)width / (float)height;
                float U = tan(AlphaCore::Math::DegreesToRadians(sceneDesc.camInfo.Fov * 0.5f)) * sceneDesc.camInfo.Near;
                float V = U / ratio;

                AxVector3 L = Cross(sceneDesc.camInfo.UpVector, sceneDesc.camInfo.Forward);
                AxVector3 center = sceneDesc.camInfo.Pivot + sceneDesc.camInfo.Forward * sceneDesc.camInfo.Near;
                AxVector3 offsetX = L * U;
                AxVector3 offsetY = sceneDesc.camInfo.UpVector * V;
                AxVector3 RUp = center + offsetX + offsetY;
                AxVector3 RDown = center + offsetX - offsetY;
                AxVector3 LUp = center - offsetX + offsetY;
                AxVector3 LDown = center - offsetX - offsetY;

                AxVector3 tmpBoxMin = gasVolumeRenderObject.densityInfo.Pivot - gasVolumeRenderObject.densityInfo.FieldSize * 0.5;
                AxVector3 tmpBoxMax = gasVolumeRenderObject.densityInfo.Pivot + gasVolumeRenderObject.densityInfo.FieldSize * 0.5;


                AxVector3 tBoxMin = tmpBoxMin * postXform;
                AxVector3 tBoxMax = tmpBoxMax * postXform;
                AxVector3 boxMin = { fminf(tBoxMin.x,tBoxMax.x), fminf(tBoxMin.y,tBoxMax.y), fminf(tBoxMin.z,tBoxMax.z) };
                AxVector3 boxMax = { fmaxf(tBoxMin.x,tBoxMax.x), fmaxf(tBoxMin.y,tBoxMax.y), fmaxf(tBoxMin.z,tBoxMax.z) };


                float v = (float)y / (float)(height - 1);
                float u = (float)x / (float)(width - 1);

                AxVector3 entryPoint = AlphaCore::Math::LerpV3(
                    AlphaCore::Math::LerpV3(LUp, LDown, v),
                    AlphaCore::Math::LerpV3(RUp, RDown, v), u);

                auto tmp = entryPoint - sceneDesc.camInfo.Pivot;
                AxVector3 worldViewDir = Normalize(tmp);


                float tNear, tFar;
                int hit = rayBox(sceneDesc.camInfo.Pivot, worldViewDir, boxMin, boxMax, tNear, tFar);
                
                

                if (!hit)
                {
                    //printf("not hit\n");
                    //outputTexRaw[(height - y - 1) * width + (width - x - 1)] = make_uchar4(0, 0, 0, 0);
                    outputTexRaw[idx] = MakeColorRGBA8(0,0,0,0);
                    return;
                }
                //else {
                //    //printf("hit\n");
                //    outputTexRaw[idx] = MakeColorRGBA8(255, 0, 255, 255);
                //    return;
                //}
                float depth = depthTexRaw[idx];
                if (AlphaCore::Math::NearZero(depth))
                {
                    depth = -1.f;
                }
                float sumDensity = 0.f;
                const int maxStep = 500000;
                AxColorRGBA lightEnergy = { 0.f, 0.f, 0.f, 0.f };
                float transmittance = 1.f;

                AxMatrix4x4 xformInverse = postXform;
                Inverse(xformInverse);
                AxVector3 tmpRayPos = tNear * worldViewDir + entryPoint;
                AxVector3 rayPos = tmpRayPos * xformInverse;
                AxColorRGBA sumEmis = { 0,0,0,0 };

                

                for (int i = 0; i < maxStep; ++i)
                {
                    auto t = i * stepSize + tNear;
                    if (t > tFar)
                        break;
                    tmpRayPos = t * worldViewDir + entryPoint;
                    rayPos = tmpRayPos * xformInverse;

                    if (depth > 0.f && Length(tmpRayPos - sceneDesc.camInfo.Pivot) > depth)
                        break;
                    float density = AlphaCore::GridDense::Internal::SampleValue(rayPos, gasVolumeRenderObject.density, gasVolumeRenderObject.densityInfo);
                    density = densityMap(density, gasVolumeRenderObject.material);
                    sumDensity += density;

                    float heat = AlphaCore::GridDense::Internal::SampleValue(rayPos, gasVolumeRenderObject.heat, gasVolumeRenderObject.heatInfo);
                    float temperature  = AlphaCore::GridDense::Internal::SampleValue(rayPos, gasVolumeRenderObject.temp, gasVolumeRenderObject.tempInfo);
                    float emisStrength = heatMap(heat, gasVolumeRenderObject.material);
                    AxColorRGBA emisColor = temperatureMap(temperature, gasVolumeRenderObject.material);
                    emisColor *= emisStrength;
                    emisColor *= (1 - density);
                    sumEmis += emisColor;
                    AxColorRGBA shadowTerm = { 0,0,0,0 };
                    for (AxUInt32 i = 0; i < sceneDesc.lightNum; ++i) {
                        shadowTerm+= lightMarching(gasVolumeRenderObject, sceneDesc.lightInfo[i], rayPos, boxMin, boxMax, postXform);
                    }
                    lightEnergy += shadowTerm * transmittance * density;
                    transmittance *= 1.f - density;
                    if (transmittance <= 0.f || sumDensity >= 1.f)
                        break;
                }
                sumDensity = sumDensity > 1.f ? 1.f : sumDensity;
                float avrLight = (lightEnergy.r + lightEnergy.g + lightEnergy.b) / 3;
                lightEnergy = densityColorMap(avrLight, gasVolumeRenderObject.material);
                lightEnergy += sumEmis;
                lightEnergy = AlphaCore::Math::Clamp(lightEnergy, 0.f, 1.f);
                lightEnergy *= 255;
                //outputTexRaw[(height - y - 1) * width + (width - x - 1)] = { (Byte)lightEnergy.r, (Byte)lightEnergy.g, (Byte)lightEnergy.b, (Byte)lightEnergy.a };
                outputTexRaw[idx] = MakeColorRGBA8((UCHAR)lightEnergy.r, (UCHAR)lightEnergy.g, (UCHAR)lightEnergy.b, (UCHAR)lightEnergy.a);
                return;
			}
		}
	}//@namespace end of : VolumeRender
}
#endif