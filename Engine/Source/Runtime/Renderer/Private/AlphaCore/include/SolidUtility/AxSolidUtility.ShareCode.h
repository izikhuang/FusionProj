#ifndef __AX_SOLIDUTILITY_SHARECODE_H__
#define __AX_SOLIDUTILITY_SHARECODE_H__

#include <Math/AxVectorBase.h>
#include <Math/AxVectorHelper.h>
#include <Math/AxMat.ShareCode.h>

namespace AlphaCore {
	namespace SolidUtility {
		namespace ShareCode {

			ALPHA_SHARE_FUNC void AdvectOrd1(AxUInt32 idx,
				AxVector3* accelRaw,
				AxVector3* velRaw,
				AxVector3* prdPRaw,
				float dt)
			{
				AxVector3 p = prdPRaw[idx];
				AxVector3 a = accelRaw[idx];
				AxVector3 v = velRaw[idx];
				v += a * dt;
				p += v * dt;
				prdPRaw[idx] = p;
				velRaw[idx] = v;
			}

			ALPHA_SHARE_FUNC void PredictOrient1Ord(AxUInt32 idx,
				Quat* orientRaw,
				AxFp32* inertiaRaw,
				AxVector3* omegaRaw,
				AxFp32 dt)
			{
				//if (@inertia) cloth Ã«·¢»ìºÏ
				AxFp32 inertia = inertiaRaw[idx];
				AxVector3 omega = omegaRaw[idx];
				Quat orient = orientRaw[idx];
				if (inertia > 0)
				{
					Quat wq = MakeQuat(omega,1.0f);
					orient += (dt * 0.5f) * QuatMultiply(wq, orient);
					orient = Normalize(orient);
				}
				orientRaw[idx] = orient;
			}

			ALPHA_SHARE_FUNC void IntegrateOmegaBDF1(AxUInt32 idx,
				Quat* orientRaw,
				Quat* orientpreviousRaw,
				AxVector3* omegaRaw,
				AxFp32 dt)
			{
				/// The result of solving BDF1 for w:
				/// q = q_prev + t * dqdt
				/// q = q_prev + t * 1/2 * w * q
				/// w = (q - qprev) * conj(q) * (2 / t)
				Quat orient = orientRaw[idx];
				Quat orientprevious = orientpreviousRaw[idx];
				Quat qconj = orient * MakeQuat(-1, -1, -1, 1);
				Quat q = orient - orientprevious;
				q = QuatMultiply(q, qconj);
				omegaRaw[idx] = QuatXYZ(q) * (2.0f / dt);
			}


			ALPHA_SHARE_FUNC void AdvectBDF2(
				AxUInt32 idx,
				AxVector3* prdPosRaw,
				AxVector3* prevPosRaw,
				AxVector3* lastPosRaw,
				AxVector3* velRaw,
				AxVector3* prevVelRaw,
				AxVector3* lastVelRaw,
				AxVector3* accelRaw,
				AxFp32 dt)
			{
				AxVector3 p = prdPosRaw[idx];
				AxVector3 a = accelRaw[idx];
				AxVector3 v = velRaw[idx];
				
				AxVector3 vprevious = prevVelRaw[idx];
				AxVector3 vlast		= lastVelRaw[idx];
				AxVector3 pprevious = prevPosRaw[idx];
				AxVector3 plast		= lastPosRaw[idx];

				//apply external force here 
				//	v intergrated drag force beforce previous solver
				//	v += a * dt;
				// dv/dt represents force and drag from POP Solver.
				//
				AxVector3 dvdt = v - vprevious;
				// vprevious is v at start of timestep, vlast is previous timestep.
				// BDF2 integration.
				AxVector3 vNew = (4.0f * vprevious - vlast + 2.0f * dvdt) / 3.0f;
				prdPosRaw[idx] = (4.0f * pprevious - plast + 2.0f * dt * vNew) / 3.0f;
			}

			ALPHA_SHARE_FUNC void IntergratorOrd1(AxUInt32 idx,
				AxVector3* posRaw,
				AxVector3* prdPRaw,
				AxVector3* hitNormalRaw,
				AxVector3* velRaw,
				AxVector3* accelRaw,
				float dampRange,
				float dampRate,
				float ctxNormalDrag,
				float ctxTangentDrag,
				float dt,
				float invDt)
			{
				//IMPLEMENT CODE HERE
				AxVector3 pos = posRaw[idx];
				AxVector3 prdP = prdPRaw[idx];
				AxVector3 hitN = MakeVector3(0.0f, 0.0f, 0.0f);
				if(hitNormalRaw != nullptr)
					hitN = hitNormalRaw[idx];
				AxVector3 v = (prdP - pos)*invDt;
				v *= powf(1.0f - dampRange, dt / dampRate);
				if (Length(hitN) > 1e-7f)
				{
					hitN = Normalize(hitN);
					AxFp32 proN = Dot(hitN, v);
					AxVector3 nProj = hitN * proN;
					AxVector3 tangent = v - nProj;
					tangent *= fmaxf(0.0f, 1.0f - dt * ctxTangentDrag);
					nProj *= fmaxf(0.0f, 1.0f - dt * ctxNormalDrag);
					v = tangent + nProj;
				}
				velRaw[idx] = v;
				posRaw[idx] = prdP;
			}


			ALPHA_SHARE_FUNC void IntergratorBDF2(
				AxUInt32 idx,
				AxVector3* prdPRaw,
				AxVector3* prevPRaw,
				AxVector3* lastPRaw,
				AxVector3* velRaw,
				AxFp32 dt)
			{
				AxVector3 prdP = prdPRaw[idx];
				AxVector3 prevP = prevPRaw[idx];
				AxVector3 lastP = lastPRaw[idx];
				AxVector3 v = ((2.0f * prdP + (prdP + lastP)) - 4.0f * prevP) / (2.0f * dt);
				velRaw[idx] = v;
			}


		}
	}//@namespace end of : SolidUtility
}
#endif