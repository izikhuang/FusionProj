#ifndef __AX_COLLISIONRESOLVE_SHARECODE_H__
#define __AX_COLLISIONRESOLVE_SHARECODE_H__

#include <Collision/AxCollision.DataType.h>
#include <Math/AxVectorHelper.h>
#include <Math/AxMath101.h>
#include <AxMacro.h>

namespace AlphaCore 
{
	namespace Collision 
	{
		namespace ShareCode 
		{

			ALPHA_SHARE_FUNC void EvaluatePoint2ContactVertexMap(
				AxUInt32 idx,
				AxUInt32* p2CtxVStartRaw,
				AxUInt32* p2CtxVEndRaw, 
				AxUInt32* contactIndicesRaw,
				int contactIndicesSize)
			{
				AxUInt32 indices = AX_INVALID_INT32;
				AxUInt32 indicesPrev = AX_INVALID_INT32;
				AxUInt32 indicesNext = AX_INVALID_INT32;
				indices = contactIndicesRaw[idx];//-1
				if (idx != 0)
					indicesPrev = contactIndicesRaw[idx - 1];
				if (idx != (contactIndicesSize - 1))
					indicesNext = contactIndicesRaw[idx + 1];
				if (indices == AX_INVALID_INT32)
					return;
				if (contactIndicesSize == 1)
				{
					p2CtxVStartRaw[indices] = idx;
					p2CtxVEndRaw[indices] = idx;
					return;
				}
				if ((indices == indicesPrev && indices == indicesNext))
					return;
				if (indices != indicesPrev)
					p2CtxVStartRaw[indices] = idx;
				if (indices != indicesNext)
					p2CtxVEndRaw[indices] = idx;
			}

			ALPHA_SHARE_FUNC bool VertexVertexResolvePBD(
				AxVector3& v0PrdP,
				AxVector3& v1PrdP,
				AlphaCore::AxContact& contact, 
				AxFp32* thickness, 
				AxVector4* contactFixBuffer,
				int ctxIndicesStart)
			{
				return true;
			}

			ALPHA_SHARE_FUNC bool VertexFaceResolvePBD(
				AxVector3& vtxPrdP, 
				AxVector3& p1PrdP, 
				AxVector3& p2PrdP, 
				AxVector3& p3PrdP,
				AxContact& contact, 
				AxFp32* thickness,
				AxVector4* contactFixBuffer, 
				AxUInt32 ctxIndicesStart)
			{
				//if (Reflex::Internal::IsVStaticF(contact.Type))
				//	return;
				AxVector3 weight;
				if (!AlphaCore::Math::BaryCenterCoordinate(vtxPrdP, p1PrdP, p2PrdP, p3PrdP, weight, 10.0f))
					return false;
				//Clamp(weight, 0, 1);
				AxVector3 prdN = Cross(p3PrdP - p2PrdP, p2PrdP - p1PrdP);
				if (Length(prdN) < 1e-6f)  // if triangle become a line then pass it! fix constraint solver
					return false;
				prdN = Normalize(prdN);
				//PrintInfo("prdN:", prdN);

				AxFp32 thicknessF = (thickness[1] + thickness[2] + thickness[3]) * 0.33333f;
				AxFp32 constraintValue = Dot(prdN * (contact.NormalDir ? 1 : -1), vtxPrdP - p1PrdP) - (thickness[0] + thicknessF); //float constraintValue = RxMath::dot(prdN, vtxPrdP - p1PrdP)*(contact.NormalDir ? 1 : -1) - (thickness[0] + thicknessF);
				if (constraintValue >= -1e-6f)
					return false;
				float weightSq = Dot(weight, weight);
				if (weightSq < 1e-8f)
					return false;
				AxFp32 invSw = 1.0f / weightSq;
				AxFp32 coeffV = AlphaCore::Collision::IsVStaticF(contact.Token) ? 0.0f : (AlphaCore::Collision::IsVFSelf(contact.Token) ? 0.5f : 1.0f);
				AxFp32 coeffF = AlphaCore::Collision::IsVStaticF(contact.Token) ? 1.0f : (AlphaCore::Collision::IsVFSelf(contact.Token) ? 0.5f : 0.0f);
				AxVector3 move = prdN * constraintValue * (contact.NormalDir ? 1.0f : -1.0f);
				AlphaCore::Math::ClampMin(weight, 0.00001f);
				//PrintInfo("Move:", move);
				if (coeffV > 0.0f) {
					contactFixBuffer[ctxIndicesStart + 0] = MakeVector4(move * coeffV * -1.0f, (AlphaCore::Collision::IsVFSelf(contact.Token)) ? -1.0f : 1.0f);
				}
				if (coeffF > 0.0f)
				{
					contactFixBuffer[ctxIndicesStart + 1] = MakeVector4(move * coeffF * weight.x * invSw, (AlphaCore::Collision::IsVFSelf(contact.Token)) ? -1.0f : 1.0f);
					contactFixBuffer[ctxIndicesStart + 2] = MakeVector4(move * coeffF * weight.y * invSw, (AlphaCore::Collision::IsVFSelf(contact.Token)) ? -1.0f : 1.0f);
					contactFixBuffer[ctxIndicesStart + 3] = MakeVector4(move * coeffF * weight.z * invSw, (AlphaCore::Collision::IsVFSelf(contact.Token)) ? -1.0f : 1.0f);
				}
				return true;
			}

 			ALPHA_SHARE_FUNC bool EdgeEdgeResolvePBD(
				AxVector3& e0PrdP, 
				AxVector3& e1PrdP,
				AxVector3& e2PrdP,
				AxVector3& e3PrdP,
				AxContact& contact,
				AxFp32* thickness,
				AxVector4* contactFixBuffer, 
				AxUInt32 ctxIndicesStart)
			{
				AxVector3 e1 = Normalize(e1PrdP - e0PrdP);
				AxVector3 e2 = Normalize(e3PrdP - e2PrdP);
				if (fabs(Dot(e1, e2)) > 0.9f)//FCCD for work?????
					return false;
				AxVector3 prdN = Cross(e1, e2);
				if (Dot(prdN, prdN) <= 1e-3f)
					return false;
				prdN = Normalize(prdN);
				
				AxFp32 a0 = AlphaCore::Math::Stp(e3PrdP - e1PrdP, e2PrdP - e1PrdP, prdN);
				AxFp32 a1 = AlphaCore::Math::Stp(e2PrdP - e0PrdP, e3PrdP - e0PrdP, prdN);
				AxFp32 b0 = AlphaCore::Math::Stp(e0PrdP - e3PrdP, e1PrdP - e3PrdP, prdN);
				AxFp32 b1 = AlphaCore::Math::Stp(e1PrdP - e2PrdP, e0PrdP - e2PrdP, prdN);
				AxFp32 w0 = a0 / (a0 + a1);
				AxFp32 w1 = a1 / (a0 + a1);
				AxFp32 w2 = b0 / (b0 + b1);
				AxFp32 w3 = b1 / (b0 + b1);
				if (w0 < 0.0f || w0>1.0f || w1 < 0.0f || w1>1.0f || w2 < 0.0f || w2>1.0f || w3 < 0.0f || w3>1.0f)
					return false;
				float thickness0 = w0 * thickness[0] + w1 * thickness[1];
				float thickness1 = w2 * thickness[2] + w3 * thickness[3];
				AxVector3 rPos = (w0 * e0PrdP + w1 * e1PrdP) - (w2 * e2PrdP + w3 * e3PrdP);
				float constraintValue = Dot(prdN * (contact.NormalDir ? 1 : -1), rPos) - (thickness0 + thickness1);
				if (constraintValue >= -0.0f)
					return false;
				if (AlphaCore::Collision::IsEEStatic(contact.Token))
					w2 = w3 = 0.0f;
				AxFp32 wSum = (w0 * w0 + w1 * w1 + w2 * w2 + w3 * w3);
				if (wSum < 1e-8f)
					return false;
				float invSw = 1.0f / wSum;
				AxVector3 move = constraintValue * prdN * (contact.NormalDir ? 1.0f : -1.0f);
				contactFixBuffer[ctxIndicesStart + 0] = MakeVector4(move * w0 * invSw * -1.0f, (AlphaCore::Collision::IsEESelf(contact.Token)) ? -1.0f : 1.0f);
				contactFixBuffer[ctxIndicesStart + 1] = MakeVector4(move * w1 * invSw * -1.0f, (AlphaCore::Collision::IsEESelf(contact.Token)) ? -1.0f : 1.0f);
				contactFixBuffer[ctxIndicesStart + 2] = MakeVector4(move * w2 * invSw, (AlphaCore::Collision::IsEESelf(contact.Token)) ? -1.0f : 1.0f);
				contactFixBuffer[ctxIndicesStart + 3] = MakeVector4(move * w3 * invSw, (AlphaCore::Collision::IsEESelf(contact.Token)) ? -1.0f : 1.0f);
				return true;
			}
			ALPHA_SHARE_FUNC void PBDContactResolve(
				AxUInt32 ctxId,
				AlphaCore::AxContact* contactRaw,
				AxVector3* selfPosRaw,
				AxVector3* selfPrdPRaw,
				AxVector3* colliderPrdPRaw,
				AxFp32* pointThicknessRaw,
				AlphaCore::Collision::SimData::AxPBDCollisionResolveData::RawData resolveData,
				bool updateIndicesBuffer = false)
			{
				AxVector3 zero = MakeVector3();
				AxVector4 zeroV4 = MakeVector4();

 				AxVector4* contactFixRaw = resolveData.ContactFixRaw;//
				AxUInt32* contactIndicesRaw = resolveData.Contact2PtIndicesRaw;
				AxFp32 ticknesses[4];
				AlphaCore::AxContact ctx = contactRaw[ctxId];
				//PrintInfo(ctx);
				AxUInt32 i0 = ctx.Points[0];	bool useSelf0 = AlphaCore::Collision::UseSelfBuf(ctx.Token, 0);
				AxUInt32 i1 = ctx.Points[1];	bool useSelf1 = AlphaCore::Collision::UseSelfBuf(ctx.Token, 1);
				AxUInt32 i2 = ctx.Points[2];	bool useSelf2 = AlphaCore::Collision::UseSelfBuf(ctx.Token, 2);
				AxUInt32 i3 = ctx.Points[3];	bool useSelf3 = AlphaCore::Collision::UseSelfBuf(ctx.Token, 3);

				AxUInt32 ctxIndicesStart = ctxId * 4; //
				if (updateIndicesBuffer)
				{
					contactIndicesRaw[ctxIndicesStart] = useSelf0 ? i0 : -1;
					contactIndicesRaw[ctxIndicesStart + 1] = useSelf1 ? i1 : -1;
					contactIndicesRaw[ctxIndicesStart + 2] = useSelf2 ? i2 : -1;
					contactIndicesRaw[ctxIndicesStart + 3] = useSelf3 ? i3 : -1;
				}
				contactFixRaw[ctxIndicesStart] = zeroV4; // 
				contactFixRaw[ctxIndicesStart + 1] = zeroV4;
				contactFixRaw[ctxIndicesStart + 2] = zeroV4;
				contactFixRaw[ctxIndicesStart + 3] = zeroV4;
				ticknesses[0] = useSelf0 ? pointThicknessRaw[i0] : 0;
				ticknesses[1] = useSelf1 ? pointThicknessRaw[i1] : 0;
				ticknesses[2] = (i2 < 0) ? 0 : (useSelf2 ? pointThicknessRaw[i2] : 0);
				ticknesses[3] = (i3 < 0) ? 0 : (useSelf3 ? pointThicknessRaw[i3] : 0);
				AxVector3& p0 = useSelf0 ? selfPrdPRaw[i0] : colliderPrdPRaw[i0];
				AxVector3& p1 = useSelf1 ? selfPrdPRaw[i1] : colliderPrdPRaw[i1];
				AxVector3& p2 = (i2 < 0) ? zero : (useSelf2 ? selfPrdPRaw[i2] : colliderPrdPRaw[i2]);
				AxVector3& p3 = (i3 < 0) ? zero : (useSelf3 ? selfPrdPRaw[i3] : colliderPrdPRaw[i3]);
				bool isResolved = false;
				if (AlphaCore::Collision::IsVF(ctx.Token))
					isResolved = VertexFaceResolvePBD(p0, p1, p2, p3, ctx, ticknesses, contactFixRaw, ctxIndicesStart);
				if (AlphaCore::Collision::IsEE(ctx.Token))
					isResolved = EdgeEdgeResolvePBD(p0, p1, p2, p3, ctx, ticknesses, contactFixRaw, ctxIndicesStart);

				/*
				if (AlphaCore::Collision::IsVE(ctx.Token))
					isResolved = VertexEdge<T>(p0, p1, p2, ctx, ticknesses, contactFixBuffer, ctxIndicesStart);
				if (AlphaCore::Collision::IsVV(ctx.Token))
					isResolved = VertexVertexResolvePBD<T>(p0, p1, ctx, ticknesses, contactFixBuffer, ctxIndicesStart);
				*/
			}


			ALPHA_SHARE_FUNC void JacobiMovePoint(
				AxUInt32 idx,
				AxVector4* contactFixVec4Raw,
				AxUInt32* outPt2ContactVertexStartRaw,
				AxUInt32* outPt2ContactVertexEndRaw, 
				AxUInt32* sortedCtxIndicesIDRaw,
				AxVector3* selfPrdPRaw,
				AxVector3* collisionNormalDstRaw,
				int numPoints,
				AxFp32 coeff)
			{
 				AxUInt32 start = outPt2ContactVertexStartRaw[idx];
				AxUInt32 end   = outPt2ContactVertexEndRaw[idx];
				if (start == -1)
					return;
				AxVector4 retSelf = MakeVector4();
				AxVector4 retCollider = MakeVector4();
				AxUInt32 rsvSelfTimes = 0;
				AxUInt32 rsvColliderTimes = 0;
				for (AxUInt32 i = start; i <= end; ++i)
				{
					//printf("f - %d\n", sortedCtxIndicesIDRaw[i]);
					AxVector4 fix = contactFixVec4Raw[sortedCtxIndicesIDRaw[i]];
					AxVector3 fixV3 = MakeVector3(fix.x, fix.y, fix.z);
					if (Length(fixV3) < 1e-7f)
						continue;
  					if (fix.w < 0.0f)
					{
						retSelf += fix;
						rsvSelfTimes++;
					}
					if (fix.w > 0.0f)
					{
						retCollider += fix;
						rsvColliderTimes++;
					}
				}
				if (rsvSelfTimes + rsvColliderTimes == 0)
					return;
				if (rsvSelfTimes > 0)
					retSelf /= (AxFp32)(rsvSelfTimes);
				if (rsvColliderTimes > 0)
					retCollider /= (AxFp32)(rsvColliderTimes);
				///ret *= 0.25f;
				///printf("Resolve [%d] <%f,%f,%f> : \n", idx, retSelf.x, retSelf.y, retSelf.z);
				AxVector3 fixSelf = MakeVector3(retSelf.x, retSelf.y, retSelf.z) * coeff;
				AxVector3 fixCollider = MakeVector3(retCollider.x, retCollider.y, retCollider.z) * coeff;

				//printf("Fix Self :[%d] <%f,%f,%f> : \n", idx, fixSelf.x, fixSelf.y, fixSelf.z);
				//printf("Fix Collider :[%d] <%f,%f,%f> : \n", idx, fixCollider.x, fixCollider.y, fixCollider.z);
				selfPrdPRaw[idx] += fixSelf + fixCollider;
				if (collisionNormalDstRaw)
					collisionNormalDstRaw[idx] += fixSelf + fixCollider;
			}

		}

	}//@namespace end of : Collision
}
#endif