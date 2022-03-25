#ifndef __AX_PLASTIC_SHARECODE_H__
#define __AX_PLASTIC_SHARECODE_H__

#include "AxConstraintType.h"
#include <Math/AxMath101.h>


#define RAD_TO_DEG 57.295779513082003f
#define DEG_TO_RAD  0.017453292519943f

///*
ALPHA_SHARE_FUNC AxVector4 operator*(AxVector4 a, AxVector4 b)
{
	return MakeVector4(a.x * b.x, a.y * b.y,a.z * b.z, a.w * b.w);
}
//*/

template<typename T>
ALPHA_SHARE_FUNC AxVector4T<T> operator+(AxVector4T<T> a, AxVector4T<T> b)
{

	return MakeVector4(a.x + b.x, a.y + b.y,a.z + b.z, a.w + b.w);
}

template<typename T>
ALPHA_SHARE_FUNC AxVector4T<T> operator-(AxVector4T<T> a, AxVector4T<T> b)
{
	return MakeVector4(a.x - b.x, a.y - b.y,a.z - b.z, a.w - b.w);
}


ALPHA_SHARE_FUNC AxVector4 QuatMultiply(const AxVector4& q0, const AxVector4& q1)
{
	AxVector3 v1 = MakeVector3(q0.x, q0.y, q0.z);
	AxVector3 v2 = MakeVector3(q1.x, q1.y, q1.z);
	AxFp32 s1 = q0.w;
	AxFp32 s2 = q1.w;
	AxFp32 w = s1 * s2 - Dot(v1, v2);
	AxVector3 v3 = s1 * v2 + s2 * v1 + Cross(v1, v2);
	return MakeVector4(v3, w);
}
ALPHA_SHARE_FUNC AxFp32 Dot(const AxVector4& a, const AxVector4& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
ALPHA_SHARE_FUNC AxFp32 Length(const AxVector4& v)
{
	return sqrtf(Dot(v, v));
}


ALPHA_SHARE_FUNC AxVector4 Normalize(AxVector4& a)
{
	float invLen = 1.0f / (Length(a) + 1e-9f);
	a *= invLen;
	return a;
}



namespace AlphaCore {
	namespace SolidUtility {
		namespace ShareCode {
			
			namespace Internal
			{
				ALPHA_SHARE_FUNC AxVector3 qconvert(AxVector4 restvector)
				{
					AxVector4 norm_restvector = Normalize(restvector);//少vector4的normalize
					AxFp32 angle = acos(norm_restvector.w) * 2;
					AxFp32 s = sin(angle*0.5f);
					AxVector3 axis = MakeVector3(norm_restvector.x, norm_restvector.y, norm_restvector.z);
					axis.x = s == 0 ? 0 : norm_restvector.x / s;
					axis.y = s == 0 ? 0 : norm_restvector.y / s;
					axis.z = s == 0 ? 0 : norm_restvector.z / s;
					axis = Normalize(axis) * angle;
					return axis;
				}

				ALPHA_SHARE_FUNC AxVector4 makeQuat(AxVector3 axis, AxFp32 angle)
				{
					AxFp32 s = sin(angle*0.5f);
					AxVector4 rest;
					rest.x = axis.x * s;
					rest.y = axis.y * s;
					rest.z = axis.z * s;
					rest.w = cos(angle*0.5f);
					return rest;
				}

				ALPHA_SHARE_FUNC AxVector4 quaternion(AxVector3 axisAngle)
				{
					AxVector3 input_axis = axisAngle;
					AxFp32 angle = Length(axisAngle);
					return makeQuat(Normalize(input_axis), Length(axisAngle));
				}

				ALPHA_SHARE_FUNC AxVector4 computeBendTwistRestVector(AxUInt32 idx, 
					AxVector2UI* primList2IRaw, 
					AxUInt32* topologyIndicesRaw, 
					Quat* orientRaw)
				{
					AxVector2UI primList2I = primList2IRaw[idx];
					AxUInt32 pt0 = topologyIndicesRaw[primList2I.x];
					AxUInt32 pt1 = topologyIndicesRaw[primList2I.x + 1];
					AxVector4 orientPt0 = ((AxVector4*)orientRaw)[pt0];
					AxVector4 orientPt1 = ((AxVector4*)orientRaw)[pt1];


					AxVector4 q0conj = orientPt0 * MakeVector4(-1.0f, -1.0f, -1.0f, 1.0f);
					AxVector4 restDarbeaux = QuatMultiply(q0conj, orientPt1);
					AxVector4 omegaplus = restDarbeaux + MakeVector4(0.0f, 0.0f, 0.0f, 1.0f);
					AxVector4 omegaminus = restDarbeaux - MakeVector4(0.0f, 0.0f, 0.0f, 1.0f);
					AxFp32 dot1 = Dot(omegaminus, omegaminus);
					AxFp32 dot2 = Dot(omegaplus, omegaplus);
					if (Dot(omegaminus, omegaminus) > Dot(omegaplus, omegaplus))
					{
						restDarbeaux *= -1.0f;
					}
					return restDarbeaux;

				}

				ALPHA_SHARE_FUNC AxUInt32 orientedRestDifference(AxUInt32 idx,
					AxVector2UI* primList2IRaw,
					AxUInt32* topologyIndicesRaw,
					Quat* orientRaw,
					AxInt32 primType,
					AxVector4* restvectorRaw,
					AxVector3 &aadiff)
				{
					AxUInt32 isorient = 0;
					AxVector4 curorient;

					if (primType == AlphaCore::SolidUtility::AxSolidConstraint::kBendTwist)
					{
						curorient = computeBendTwistRestVector(idx, primList2IRaw, topologyIndicesRaw,orientRaw);
						isorient = 1;
					}

					if (isorient)
					{
						AxVector4 restconj = restvectorRaw[idx] * MakeVector4(-1, -1, -1, 1);
						aadiff = qconvert(QuatMultiply(restconj, curorient));
						//PrintInfo("aadiff : ", aadiff);						
						//printf("isorient : %f \n", isorient);

					}
					return isorient;
				}

				ALPHA_SHARE_FUNC  AxFp32 updateRestVectorOrient(AxUInt32 idx,
					AxFp32 inamount,
					AxUInt32 isratio,
					AxVector3 &aadiff,
					AxVector4* restvectorRaw) 
				{
					PrintInfo("restvectorRaw[idx] : ", restvectorRaw[idx]);//一致
					AxFp32 amount = inamount;
					AxFp32 degdiff = RAD_TO_DEG * Length(aadiff);//有一点偏差。弧度转角度：180/pai *弧度。 角度转弧度：pai/180 * 角度
					if (!isratio)
						amount /= degdiff;
					AxVector3 rest = qconvert(restvectorRaw[idx]);//一致
					rest += AlphaCore::Math::ClampF32(amount, 0, 1) * aadiff;//一致
					restvectorRaw[idx] = quaternion(rest);
					return amount * degdiff;
				}


				ALPHA_SHARE_FUNC  AxFp32 DifferenceRest(AxUInt32 idx,
					AxVector2UI* primList2IRaw,
					AxUInt32* topologyIndicesRaw,
					AxInt32 primType,
					Quat* orientRaw,
					AxFp32 restlength,
					AxVector4* restvectorRaw)
				{
					AxVector3 aadiff_init = MakeVector3(0, 0, 0);
					AxVector3 &aadiff = aadiff_init;

					if (orientedRestDifference(idx, primList2IRaw, topologyIndicesRaw, orientRaw, primType, restvectorRaw, aadiff))
					{
						return RAD_TO_DEG * Length(aadiff);
					}

				
				}


				ALPHA_SHARE_FUNC  AxFp32 PlasticFlow(AxUInt32 idx,
					AxVector2UI* primList2IRaw,
					AxUInt32* topologyIndicesRaw,
					AxInt32 primType,
					Quat* orientRaw,
					AxFp32 diff,
					AxFp32 plasticrate,
					AxFp32 plasticthreshold,
					AxFp32 plastichardening,
					AxFp32 dt,
					AxFp32 restlength,
					AxVector4* restvectorRaw,
					AxFp32* stiffnessRaw)
				{
					//获取需要的参数值
					AxFp32 threshold = plasticthreshold;

					if (threshold < 0)
						threshold = -threshold * restlength;

					if (abs(diff) <= threshold)
						return 0;

					AxVector3 aadiff = MakeVector3(0, 0, 0);//??????这里是不是直接 aadiff = MakeVector3(0, 0, 0)就可以

					//计算系数
					AxFp32 u = exp(-plasticrate * dt);
					AxFp32 v = 1 - u;
					AxFp32 flow = 0;

					if (orientedRestDifference(idx, primList2IRaw, topologyIndicesRaw, orientRaw, primType, restvectorRaw, aadiff))
					{
						flow = updateRestVectorOrient(idx, v, 1, aadiff, restvectorRaw);
					}
			

					//更新修改stiffness
					AxFp32 stiffness = stiffnessRaw[idx];
					float k = u + v * plastichardening;
					stiffness  = AlphaCore::Math::ClampF32(exp(k * log(stiffness + 1)) - 1, 0, 1e+10);

					stiffnessRaw[idx] = stiffness;//数据存回去

					return flow;
				}
			}

			ALPHA_SHARE_FUNC void ApplyFastPlasticDeform(AxUInt32 idx,
				AxVector2UI* primList2IRaw,
				AxUInt32* topologyIndicesRaw,
				AxInt32* primTypeRaw,
				Quat* orientRaw,
				AxVector4* restvectorRaw,
				AxFp32* restlengthRaw,
				AxFp32* stiffnessRaw,
				AxFp32* plastichardeningRaw,
				AxFp32* plasticthresholdRaw,
				AxFp32* plasticrateRaw,
				AxFp32* flowRaw,
				AxFp32 dt)
			{
 
				AxInt32 primType = primTypeRaw[idx];
				AxFp32 restlength = restlengthRaw[idx];
				AxFp32 plastichardening = plastichardeningRaw[idx];
				AxFp32 plasticthreshold = plasticthresholdRaw[idx];
				AxFp32 plasticrate = plasticrateRaw[idx];
				AxFp32 flowVal = flowRaw[idx];
				
				dt = 0.041667;

				AxFp32 diff = Internal::DifferenceRest(idx, primList2IRaw, topologyIndicesRaw, primType, orientRaw, restlength, restvectorRaw);
				flowVal += Internal::PlasticFlow(idx, primList2IRaw, topologyIndicesRaw, primType, orientRaw, diff, plasticrate, plasticthreshold, plastichardening, dt, restlength, restvectorRaw, stiffnessRaw);

				flowRaw[idx] = flowVal;
 			}


		}
	}//@namespace end of : SolidUtility
}
#endif