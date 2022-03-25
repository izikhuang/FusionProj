#ifndef __AX_PBD_SHARECODE_H__
#define __AX_PBD_SHARECODE_H__

#include <Math/AxVectorBase.h>
#include <Math/AxVectorHelper.h>
#include "AxConstraintType.h"
#include <Math/AxMat.h>
#include <Math/AxMat.ShareCode.h>

namespace AlphaCore 
{
	namespace PBD 
	{
		namespace ShareCode 
		{
			ALPHA_SHARE_FUNC void ProjectAttach_XPBD(
				AxVector3&	pos,
				AxVector3&	prev1,
				AxVector3&	lambda,
				AxFp32	restLength,
				AxVector3	targetPos,
				AxFp32	stiffness,
				AxFp32 kdampratio,
				float dt)
			{
				//distancePosUpdateXPBD()
				if (stiffness < 1e-4)
					return;
				AxFp32 wsum = 1.0f;
				AxVector3 n = pos - targetPos;
				AxFp32 d = Length(n);
				if (d < 1e-7f)
					return;
				AxFp32 l = lambda.x;
				AxFp32 alpha = 1.0f / stiffness;
				alpha /= dt * dt;

				AxFp32 C = d - restLength;
				n /= d;
 				AxVector3 gradC = n;
				AxFp32 dsum = 0.0f;
				AxFp32 gamma = 1.0f;
				if (kdampratio > 1e-6f)
				{
					// Compute damping terms.
					AxFp32 beta = stiffness * kdampratio * dt * dt;
					gamma = alpha * beta / dt;
					dsum = gamma * Dot(gradC, pos - prev1);
					gamma += 1.0f;
				}
				AxFp32 dL = (-C - alpha * l - dsum) / (gamma * wsum + alpha);
				AxVector3 dp = n * -dL;
				pos -= 1.0f * dp;
				lambda.x += dL;
			}

			ALPHA_SHARE_FUNC void ProjectDistance_XPBD(
				AxVector3 & p0,
				AxVector3 & p1,
				AxVector3 & prev0,
				AxVector3 & prev1,
				AxVector3 & lambda,
				AxFp32 restLength,
				AxFp32 invMass0,
				AxFp32 invMass1,
				AxFp32 stiff,
				AxFp32 kdampratio,
				AxFp32 dt)
			{
				if (stiff < 1e-4)
					return;
				AxFp32 wsum = invMass0 + invMass1;
				if (wsum == 0.0f)
					return;

				AxVector3 n = p1 - p0;
				AxFp32 d = Length(n);
				if (d < 1e-6f)
					return;

				AxFp32 kstiff = stiff;
				AxFp32 l = lambda.x;
				// XPBD term.
				AxFp32 alpha = 1.0f / kstiff;
				alpha /= dt * dt;

				// Constraint calc.
				AxFp32 C = d - restLength;
				n /= d;

				AxVector3 gradC = n;
				AxFp32 dsum = 0.0f;
				AxFp32 gamma = 1.0f;
				if (kdampratio > 1e-6f)
				{
					// Compute damping terms.
					AxFp32 beta = kstiff * kdampratio * dt * dt;
					gamma = alpha * beta / dt;
					dsum = gamma * (-Dot(gradC, p0 - prev0) + Dot(gradC, p1 - prev1));
					gamma += 1.0f;
				}
				AxFp32 dL = (-C - alpha * l - dsum) / (gamma * wsum + alpha);
				AxVector3 dp = n * -dL;
				p0 += invMass0 * dp;
				p1 -= invMass1 * dp;
				lambda.x += dL;
			}

			ALPHA_SHARE_FUNC void ProjectDihedral_XPBD(
				AxVector3 & pos0,  AxVector3 & pos1,  AxVector3 & pos2,  AxVector3 & pos3,
				AxVector3 & prev0, AxVector3 & prev1, AxVector3 & prev2, AxVector3 & prev3,
				AxVector3 & lambda,
				AxFp32 restAngle,
				AxFp32 invMass0, AxFp32 invMass1, AxFp32 invMass2, AxFp32 invMass3,
				AxFp32 stiff,
				AxFp32 kdampratio,
				AxFp32 dt)
			{
				AxVector3 e = pos3 - pos2;
				AxFp32 elen = Length(e);

				// M¨¹ller M, Chentanez N, Kim T, Macklin M. Strain Based Dynamics
				AxVector3 n1 = Cross(pos3 - pos0, pos2 - pos0);
				AxVector3 n2 = Cross(pos2 - pos1, pos3 - pos1);
				AxFp32 n1lenSq = Dot(n1, n1);
				AxFp32 n2lenSq = Dot(n2, n2);

				// if the triangle is too small, return
				if (n1lenSq < 1e-12f || n2lenSq < 1e-12f || elen < 1e-12f) {
					return;
				}
				n1 /= n1lenSq;
				n2 /= n2lenSq;
				AxFp32 invElen = 1.0f / elen;
 				AxFp32 s = -1.0f;
				if (Dot(Cross(n1, n2), e) > 0.0f) {
					s = 1.0f;
				}

				AxVector3 grad0 = s * elen *n1;
				AxVector3 grad1 = s * elen *n2;
				AxVector3 grad2 = s * (Dot(pos0 - pos3, e)*invElen*n1 + Dot(pos1 - pos3, e)*invElen*n2);
				AxVector3 grad3 = s * (Dot(pos2 - pos0, e)*invElen*n1 + Dot(pos2 - pos1, e)*invElen*n2);

				AxFp32 wsum = invMass0 * Dot(grad0, grad0) +
							  invMass1 * Dot(grad1, grad1) +
							  invMass2 * Dot(grad2, grad2) +
							  invMass3 * Dot(grad3, grad3);

				if (wsum == 0.0f)
					return;

				AxFp32 alpha = 1.0f / stiff;
				alpha /= dt * dt;
				n1 = Normalize(n1);
				n2 = Normalize(n2);
				AxFp32 n1dotn2 = Dot(n1, n2);
				if (n1dotn2 >= 1.0f) {
					n1dotn2 = 1.0f;
				}
				if (n1dotn2 <= -1.0f) {
					n1dotn2 = -1.0f;
				}
				AxFp32 phi = acosf(n1dotn2);
				phi *= s;
				AxFp32 C = phi - restAngle / 180.0f * 3.1415926f;
				C *= s;
				AxFp32 dsum = 0.0f;
				AxFp32 gamma = 1.0f;
				if (kdampratio > 1e-6f)
				{
					// compute damping terms.
					AxFp32 beta = stiff * kdampratio * dt * dt;
					gamma = alpha * beta / dt;
					dsum = Dot(grad0, pos0 - prev0) +
						   Dot(grad1, pos1 - prev1) +
						   Dot(grad2, pos2 - prev2) +
						   Dot(grad3, pos3 - prev3);
					dsum *= gamma;
					gamma += 1.0f;
				}
				// Change in Lagrange multiplier.
				AxFp32 dL = (-C - alpha * lambda.x - dsum) / (gamma*wsum + alpha);
				pos0 += dL * invMass0 * grad0;
				pos1 += dL * invMass1 * grad1;
				pos2 += dL * invMass2 * grad2;
				pos3 += dL * invMass3 * grad3;
				lambda.x += dL;
			}


			//
			//	FEM
			//
			//
			// 
			ALPHA_SHARE_FUNC void stretchShearUpdateXPBD(
				int idx,
				AxInt32 pt0, 
				AxInt32 pt1,
				//int ptidx,               //?
				//int *pts,   // pts
				AxVector3* Ls,
				AxVector3* P,
				AxVector3 *pprev,
				AxFp32 *masses,
				Quat *orient,
				Quat *orientprev,
				AxFp32 *inertias,
				AxFp32 restlength,
				AxFp32 kstiff,
				AxFp32 kdampratio,
				AxFp32 timeinc)
			{

				AxVector3 p0 = P[pt0];
				AxVector3 p1 = P[pt1];
				
				Quat oi = orient[pt0];
				AxVector4 q0 =  MakeVector4(oi.mm[0], oi.mm[1], oi.mm[2], oi.mm[3]);//
				AxFp32 mass0 = masses[pt0];
				AxFp32 mass1 = masses[pt1];
				AxFp32 invmassp0 = 1.0f / mass0;
				AxFp32 invmassp1 = 1.0f / mass1;
				AxFp32 invmassq0 = 1.0f / inertias[pt0];
				AxVector3 L = Ls[idx];

				AxVector3 d3;    //third director d3 = q0 * e_3 * q0_conjugate
				d3.x = 2.0f * (q0.x * q0.z + q0.w * q0.y);
				d3.y = 2.0f * (q0.y * q0.z - q0.w * q0.x);
				d3.z = q0.w * q0.w - q0.x * q0.x - q0.y * q0.y + q0.z * q0.z;

				// If restlength is zero we'll get NAN's, but if we just return when restlength==0,
				// we won't get any stretch stiffness keeping the points in place.  We could enforce
				// in constraint creation, but users could always override / scale restlength.
				// So punt and enforce a minimum restlength at constraint solve time.
				restlength = fmaxf(restlength, 1e-6f);

				AxVector3 gradp = MakeVector3(-1.0f / restlength);
				// ||gradp|| =  1/restlength^2
				// ||gradq0|| = 4;
				AxFp32 wsum = (invmassp0 + invmassp1) / (restlength * restlength) + invmassq0 * 4;
				if (wsum == 0.0f)
					return;

				// XPBD term
				AxFp32 alpha = 1.0f / kstiff;
				alpha /= timeinc * timeinc;

				// Vector-valued constraint function.
				AxVector3 C = (p1 - p0) / restlength - d3;
				AxVector3 dsum = MakeVector3();
				AxFp32 gamma = 1.0f;
				if (kdampratio > 0)
				{
					// Compute damping terms.
					AxVector3 prevp0 = pprev[pt0];
					AxVector3 prevp1 = pprev[pt1];
					Quat prevq0 = orientprev[pt0];
					AxFp32 beta = kstiff * kdampratio * timeinc * timeinc;
					gamma = alpha * beta / timeinc;

					// Damping for linear part of constraint on points.
					dsum = gradp * ((p0 - prevp0) - (p1 - prevp1));

					// Damping for orientation part of constraint.
					// dq/dt with timeinc factored out (reapplied in gamma)
					Quat dq0_dt = (q0 - prevq0);
					// dC/dt = dC/dq0 * dq0/dt = dq0/dt * e3 * q0.conjugate
					Quat e3_qconj = QuatYXWZ(MakeQuat(q0)) * MakeQuat(1.0f, -1.0f, 1.0f, 1.0f);
					// Ignore scalar part.
					dq0_dt = QuatMultiply(dq0_dt, e3_qconj);
					dsum -= 2.0f * QuatXYZ(dq0_dt);

					dsum *= gamma;
					gamma += 1.0f;
				}

				AxVector3 dL = (Neg(C) - alpha * L - dsum) / (gamma * wsum + alpha);

				// Update points.
				p0 += invmassp0 * dL * gradp;
				p1 -= invmassp1 * dL * gradp;

				// Compute q*e_3.conjugate (cheaper than quaternion product)
				// Y X W Z
				Quat q_e3_conj = QuatYXWZ(MakeQuat(q0)) * MakeQuat(-1, 1, -1, 1);

				// Update orientation.
				// gradq0^T * dL = dL * q * e_3.conjugate
				q0 -= (2.0f * invmassq0) * QuatMultiply(MakeQuat(dL, 0.0f), q_e3_conj);
				
				//TODO : .....
				Quat _qq0 = Normalize(MakeQuat(q0));
				q0 = MakeVector4(_qq0);
				L += dL;
				

				P[pt0] = p0;
				P[pt1] = p1;
				orient[pt0] = MakeQuat(q0);
				Ls[idx] = L;

				//PrintInfo("Orient:", q0);

				//vstore3(p0, pt0, P);
				//vstore3(p1, pt1, P);
				//vstore4(q0, pt0, orient);//TODO : ¿´²»¶®....
				//vstore3(L, idx, Ls);
			}

			
				ALPHA_SHARE_FUNC void bendTwistUpdateXPBD(
					AxInt32 idx,
					AxInt32 pt0,
					AxInt32 pt1,
 					AxVector3 *Ls,
					Quat *orient,
					Quat *orientprev,
					AxFp32 *inertias,
					Quat *restvectors,
					AxFp32 kstiff,
					AxFp32 kdampratio,
					AxFp32 timeinc)
				{
					Quat q0 = orient[pt0];
					Quat q1 = orient[pt1];
					Quat q0conj = QuatConjugate(q0);

					AxFp32 invmassq0 = 1.0f/inertias[pt0];
					AxFp32 invmassq1 = 1.0f/inertias[pt1];
					AxVector3 L =  Ls[idx];

					Quat restvector = restvectors[idx];
					Quat omega = QuatMultiply(q0conj, q1);   //darboux vector
					omega = QuatCloser(omega, restvector);
 
					// XPBD term
					AxFp32 alpha = 1.0f / kstiff;
					alpha /= timeinc * timeinc;
					// Vector constraint function:
					// Zero in w since discrete Darboux vector does not have
					// vanishing scalar part.
					AxVector3 C = QuatXYZ(omega);
					// ||gradq0|| = ||gradq1|| = 1
					AxFp32 wsum = invmassq0 + invmassq1;
					if (wsum == 0.0f)
						return;

					AxVector3 dsum = MakeVector3();
					AxFp32 gamma = 1.0f;
					//printf("kdamp = %g\n", kdamp);
					if (kdampratio > 0)
					{
						// Compute damping terms.
						Quat prevq0 = orientprev[pt0];
						Quat prevq1 = orientprev[pt1];
						AxFp32 beta = kstiff * kdampratio * timeinc * timeinc;
						gamma = alpha * beta / timeinc;

						// Angular velocities with timeinc factored out
						// (reapplied in gamma)
						// w = (q0 * prevq0_conj) * (2 / timeinc)
						Quat w0 = QuatMultiply(q0, QuatConjugate(prevq0));// * 2.0f;
						Quat w1 = QuatMultiply(q1, QuatConjugate(prevq1));// * 2.0f;
						// Zero out scalar part.
						w0.mm[3] = w1.mm[3] = 0;

						// dq/dt = 1/2 * w * q
						Quat dq0_dt = QuatMultiply(w0, q0);// * 0.5f;
						Quat dq1_dt = QuatMultiply(w1, q1);// * 0.5f;

						// dC/dt = dC/dq0 * dq0/dt = -q1_conj * 1/2 * w * q
						// dC/dt = dC/dq1 * dq1/dt = q0_conj * 1/2 * w * q
						Quat dsum4 = QuatMultiply(q0conj, dq1_dt) - QuatMultiply(QuatConjugate(q1), dq0_dt);
						// Ignore scalar part.
						dsum = QuatXYZ(dsum4) * gamma;
						gamma += 1.0f;
					}

					AxVector3 dL = (Neg(C) - alpha * L - dsum) / (gamma * wsum + alpha);
					// gradq0^T * dL = -q1 * dL
					// gradq1^T * dL = q0 * dL
					Quat dl4 = MakeQuat(dL, 0);
					Quat dq0 = -invmassq0 * QuatMultiply(q1, dl4);
					Quat dq1 = invmassq1 * QuatMultiply(q0, dl4);

					q0 = Normalize(q0 + dq0);
					q1 = Normalize(q1 + dq1);
					L += dL;
					//PrintInfo("L BendTwist:",L);

					orient[pt0] = q0;
					orient[pt1] = q1;
					Ls[idx]=L;
				}

			ALPHA_SHARE_FUNC void UpdteXPBD(
				AxUInt32 constraintID,
				AxVector2UI* primList2IRaw,
				AxUInt32* topologyIndices,
				int* primTypeRaw,//Need UChar?
				AxVector3* prdPRaw,
				AxVector3* posRaw,
				AxFp32* restLengthRaw,
				AxVector4* restVectorRaw,
				AxFp32* massRaw,
				AxFp32* stiffnessRaw,
				AxFp32* dampingRatioRaw,
				AxVector3* lambdaRaw,
				AxVector3* attachUVWRaw,
				AxVector3* attachPosRaw,
				AxVector3UI* attachPt3IRaw,
				Quat *orientRaw,
				Quat *orientprevRaw,
				AxFp32 *inertiasRaw,
				float dt)
			{
				AxFp32 stiff = stiffnessRaw[constraintID];
				if (stiff < 1e-5f)
					return;
#ifndef __CUDA_ARCH__
				///std::cout << "con:" << constraintID << std::endl;
#endif
				int cType = primTypeRaw[constraintID];
				AxVector2UI prim2I	= primList2IRaw[constraintID];
				AxVector3 lambda	= lambdaRaw[constraintID];
				AxFp32 restLen		= restLengthRaw[constraintID];
				AxFp32 dampRatio	= dampingRatioRaw[constraintID];
				if (cType == AlphaCore::SolidUtility::AxSolidConstraint::kAttach)
				{
 					AxUInt32 id0 = topologyIndices[prim2I.x];
					AxVector3 pos0 = prdPRaw[id0];
					AxVector3 prev0 = posRaw[id0];
					AxVector3 targetUV = attachUVWRaw[constraintID];
					AxVector3UI attachPtId = attachPt3IRaw[constraintID];
					//TODO:Need AXAttachUVW Properties !!!!!
					AxVector3 targetUVW = MakeVector3(1.0f - targetUV.x - targetUV.y, targetUV.x, targetUV.y);
					AxVector3 targetPos = attachPosRaw[attachPtId.x] * targetUVW.x +
										  attachPosRaw[attachPtId.y] * targetUVW.y +
										  attachPosRaw[attachPtId.z] * targetUVW.z;

					AlphaCore::PBD::ShareCode::ProjectAttach_XPBD(
						pos0, prev0,
						lambda,
						restLen,
						targetPos,
						stiff,
						dampRatio,
						dt);
					prdPRaw[id0] = pos0;
					lambdaRaw[constraintID] = lambda;
					return;
				}
				if (cType == AlphaCore::SolidUtility::AxSolidConstraint::kDistance)
				{
					AxUInt32 id0	= topologyIndices[prim2I.x];
					AxUInt32 id1	= topologyIndices[prim2I.x + 1];
					AxVector3 pos0	= prdPRaw[id0];
					AxVector3 pos1	= prdPRaw[id1];
					AxVector3 prev0 = posRaw[id0];
					AxVector3 prev1 = posRaw[id1];
					AxFp32 mass0 = massRaw[id0];
					AxFp32 mass1 = massRaw[id1];
					AlphaCore::PBD::ShareCode::ProjectDistance_XPBD(
						pos0, pos1,
						prev0, prev1,
						lambda,
						restLen,
						1.0f / mass0,
						1.0f / mass1,
						stiff,
						dampRatio,
						dt);
					prdPRaw[id0] = pos0;
					prdPRaw[id1] = pos1;
					lambdaRaw[constraintID] = lambda;
					return;
				}

				if (cType == AlphaCore::SolidUtility::AxSolidConstraint::kStretchShear)
				{
					AxUInt32 id0 = topologyIndices[prim2I.x];
					AxUInt32 id1 = topologyIndices[prim2I.x + 1];
#ifndef __CUDA_ARCH__
					///std::cout << " kStretchShear constraintID:" << constraintID << std::endl;
#endif
					AlphaCore::PBD::ShareCode::stretchShearUpdateXPBD(
						constraintID,
						id0,
						id1,
						lambdaRaw,
						prdPRaw,
						posRaw,
						massRaw,
						orientRaw,//2
						orientprevRaw,
						inertiasRaw,
						restLen,
						stiff,
						dampRatio,
						dt);
				}

				if (cType == AlphaCore::SolidUtility::AxSolidConstraint::kBendTwist)
				{
					AxUInt32 id0 = topologyIndices[prim2I.x];
					AxUInt32 id1 = topologyIndices[prim2I.x + 1];
#ifndef __CUDA_ARCH__
					///std::cout << " kBendTwist constraintID:" << constraintID << std::endl;
#endif
					AlphaCore::PBD::ShareCode::bendTwistUpdateXPBD(
						constraintID,
						id0,
						id1,
						lambdaRaw,
						orientRaw,
						orientprevRaw,
						inertiasRaw,
						(Quat*)restVectorRaw,
						stiff,
						dampRatio,
						dt);
				}


				if (cType == AlphaCore::SolidUtility::AxSolidConstraint::kBend)
				{
					AxUInt32 id0 = topologyIndices[prim2I.x];
					AxUInt32 id1 = topologyIndices[prim2I.x + 1];
					AxUInt32 id2 = topologyIndices[prim2I.x + 2];
					AxUInt32 id3 = topologyIndices[prim2I.x + 3];
					AxVector3 p0 = prdPRaw[id0];
					AxVector3 p1 = prdPRaw[id1];
					AxVector3 p2 = prdPRaw[id2];
					AxVector3 p3 = prdPRaw[id3];
					AxVector3 prev0 = posRaw[id0];
					AxVector3 prev1 = posRaw[id1];
					AxVector3 prev2 = posRaw[id2];
					AxVector3 prev3 = posRaw[id3];
					AxFp32 invmass0 = 1.0f / massRaw[id0];
					AxFp32 invmass1 = 1.0f / massRaw[id1];
					AxFp32 invmass2 = 1.0f / massRaw[id2];
					AxFp32 invmass3 = 1.0f / massRaw[id3];

					AlphaCore::PBD::ShareCode::ProjectDihedral_XPBD(
						p0, p1, p2, p3,
						prev0, prev1, prev2, prev3,
						lambda,
						restLen,
						invmass0,
						invmass1,
						invmass2,
						invmass3,
						stiff,
						dampRatio,
						dt);

					prdPRaw[id0] = p0;
					prdPRaw[id1] = p1;
					prdPRaw[id2] = p2;
					prdPRaw[id3] = p3;
					lambdaRaw[constraintID] = lambda;
				}
			}
		}
	}//@namespace end of : PBD
}
#endif