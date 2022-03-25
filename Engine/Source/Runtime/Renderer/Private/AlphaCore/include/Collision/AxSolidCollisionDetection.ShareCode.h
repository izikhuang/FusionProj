#ifndef __AX_SOLIDCOLLISIONDETECTION_SHARECODE_H__
#define __AX_SOLIDCOLLISIONDETECTION_SHARECODE_H__

#include "AxCollision.DataType.h"
#include <AccelTree/AxBVHTree.h>
#include <SolidUtility/AxSolidUtility.DataType.h>
#include <AccelTree/AxBVHTree.ShareCode.h>
#include "AxRTriangle.h"
#include "AxCCD.h"

#include <Debug/AxDebugTraceID.h>
#include <Debug/Collision/AxCollision.Debug.h>

namespace AlphaCore {
	namespace Collision {
		namespace ShareCode 
		{
			ALPHA_SHARE_FUNC void GroundCollision(
				AxUInt32 idx,
				AxVector3* posRaw,
				AxFp32 height)
			{
				AxVector3 pos = posRaw[idx];
				if (pos.y < height)
					pos.y = height + 1e-6f;
				posRaw[idx] = pos;
			}

			ALPHA_SHARE_FUNC void CapsuleCollision(
				AxUInt32 idx,
				AxVector3* posRaw,
				const AxCapsuleCollider& cap)
			{
				AxFp32 e = 0.001f;
				AxVector3 pos = posRaw[idx];
				AxVector3 capPivot = cap.Pivot;
				AxVector3 capDirection = cap.Direction;
				AxFp32 capHalfHeight = cap.HalfHeight;
				AxFp32 capRadius = cap.Radius;

				capDirection = Normalize(capDirection);
				AxVector3 A = capPivot - capDirection * capHalfHeight;
				AxVector3 B = capPivot + capDirection * capHalfHeight;
				AxVector3 AC = pos - A;
				AxVector3 AB = B - A;
				AxVector3 BC = pos - B;

				AxFp32 distance = 0.0f;
				AxFp32 direction = Dot(AC, AB) / (Length(AB));

				if (direction < 0) {
					distance = Length(AC);
					if (distance < (capRadius + e)) {
						pos = A + Normalize(AC) * (capRadius + e);
					}
				}
				else if (direction > Length(AB)) {
					distance = Length(BC);
					if (distance < capRadius + e) {
						pos = B + Normalize(BC) * (capRadius + e);
					}
				}
				else {
					AxVector3 D;
					D = A + direction * Normalize(AB);
					distance = Length(pos - D);
					if (distance < (capRadius + e)) {
						pos = D + Normalize(pos - D) * (capRadius + e);
					}
				}
				posRaw[idx] = pos;
 			}

			ALPHA_SHARE_FUNC void OBBCollision(
				AxUInt32 idx,
				AxVector3* posRaw,
				const AxOBBCollider& obb)
			{
				AxFp32 e = 0.001f;
				AxVector3 point			= posRaw[idx];
				AxVector3 obbPivot		= obb.Pivot;
				AxVector3 obbForward	= obb.Forward;
				AxVector3 obbUp			= obb.Up;
				AxFp32 obbWidth			= obb.Size.x;	// right
				AxFp32 obbHeight		= obb.Size.y;	// up
				AxFp32 obbLength		= obb.Size.z;   // forward

				Normalize(obbForward);
				Normalize(obbUp);
				AxVector3	obbRight = Normalize(Cross(obbForward, obbUp));

				AxVector3 vec = point - obbPivot;
				AxFp32 vecLength = Length(vec);

				AxFp32 w = Dot(vec, obbRight);
				AxFp32 h = Dot(vec, obbUp);
				AxFp32 l = Dot(vec, obbForward);

				AxFp32 disW = abs(w) - obbWidth / 2 - e;
				AxFp32 disH = abs(h) - obbHeight / 2 - e;
				AxFp32 disL = abs(l) - obbLength / 2 - e;

				// disW >= 0 || disH >= 0 || disL >= 0 ,so the point not inside obb
				if (disW >= 0 || disH >= 0 || disL >= 0)return;

				// here disW,disH,disL are all <0; find the max dis mean that we shuold move torward that direction;
				// disNear <0
				AxFp32 disNear = fmax(fmax(disW, disH), disL);

				// resolve
				AxVector3 newPos;
				if (disW == disNear) {
					if (w > 0) {
						newPos = point + obbRight * disNear * (-1);
					}
					else {
						newPos = point - obbRight * disNear * (-1);
					}
				}
				else if (disH == disNear) {
					if (h > 0) {
						newPos = point + obbUp * disNear * (-1);
					}
					else {
						newPos = point - obbUp * disNear * (-1);
					}
				}
				else {
					if (l > 0) {
						newPos = point + obbForward * disNear * (-1);
					}
					else {
						newPos = point - obbForward * disNear * (-1);
					}
				}

				posRaw[idx] = newPos;
			}

			//TO: redo repeat?
			template<typename T>
			ALPHA_SHARE_FUNC T CalcRestEEConstr(AxVector3T<T> e0Pos, AxVector3T<T>  e1Pos, AxVector3T<T>  e2Pos, AxVector3T<T> e3Pos)
			{
				AxVector3T<T> e1 = Normalize(e1Pos - e0Pos);
				AxVector3T<T> e2 = Normalize(e3Pos - e2Pos);
				AxVector3T<T>  currNormal = Cross(e1, e2);
				if (Dot(currNormal, currNormal) <= 1e-6)
					return 0.0f;
				T weight[4];
				Normalize(currNormal);
				T a0 = AlphaCore::Math::Stp(e3Pos - e1Pos, e2Pos - e1Pos, currNormal);
				T a1 = AlphaCore::Math::Stp(e2Pos - e0Pos, e3Pos - e0Pos, currNormal);
				T b0 = AlphaCore::Math::Stp(e0Pos - e3Pos, e1Pos - e3Pos, currNormal);
				T b1 = AlphaCore::Math::Stp(e1Pos - e2Pos, e0Pos - e2Pos, currNormal);
				weight[0] = a0 / (a0 + a1);
				weight[1] = a1 / (a0 + a1);
				weight[2] = -b0 / (b0 + b1);
				weight[3] = -b1 / (b0 + b1);
				AxVector3T<T> dist = (weight[0] * e0Pos + weight[1] * e1Pos) - ((-weight[2]) * e2Pos + (-weight[3]) * e3Pos);
				return Dot(currNormal, dist);
			}
			ALPHA_SHARE_FUNC AxUInt32 FlipCCDTrianglePair(
				AxVector3* selfPosRaw,			AxVector3* otherPosRaw,
				AxVector3* selfPrdPRaw,			AxVector3* otherPrdPRaw,
				AxUInt32 selfTriangleID,		AxUInt32 otherTriangleID,
				AxUChar selfRTriangleInfo,		AxUChar otherRTriangleInfo,
				AxFp32 selfMaxEta,				AxFp32 otherMaxEta,
				AxVector3UI selfTriaglePtID3,   AxVector3UI otherTriaglePtID3,
				const AxAABB& selfTriangleAABB,
				const AxAABB& otherTriangleAABB,
				AxContact* ctxBuffer,
				bool processSelf,
				AxSPMDTick& lock,
				bool doEETest = true,
				bool traceDebugInfo = false)
			{

				static AxRTriangleType allToken[6];
				allToken[0] = AxRTriangleType::kVtx0;
				allToken[1] = AxRTriangleType::kVtx1;
				allToken[2] = AxRTriangleType::kVtx2;
				allToken[3] = AxRTriangleType::kEdge0;
				allToken[4] = AxRTriangleType::kEdge1;
				allToken[5] = AxRTriangleType::kEdge2;

				//LoadPosition
				AxVector3 SelfPos3[3];
				AxVector3 SelfPrdPPos3[3];
				AxVector3 NbPos3[3];
				AxVector3 NbPrdPPos3[3];

				AxVector2UI eeIdA[3];
				eeIdA[0].x = selfTriaglePtID3.z;
				eeIdA[0].y = selfTriaglePtID3.y;
				eeIdA[1].x = selfTriaglePtID3.y;
				eeIdA[1].y = selfTriaglePtID3.x;
				eeIdA[2].x = selfTriaglePtID3.x;
				eeIdA[2].y = selfTriaglePtID3.z;

				AxVector2UI eeIdB[3];
				eeIdB[0].x = otherTriaglePtID3.z;
				eeIdB[0].y = otherTriaglePtID3.y;
				eeIdB[1].x = otherTriaglePtID3.y;
				eeIdB[1].y = otherTriaglePtID3.x;
				eeIdB[2].x = otherTriaglePtID3.x;
				eeIdB[2].y = otherTriaglePtID3.z;

				bool selfRtV[3];
				bool selfRtE[3];
				bool otherRtV[3];
				bool otherRtE[3];

				AxAABB selfPtAABB[3];
				AxAABB otherPtAABB[3];
				AxAABB selfEdgeAABB[3];
				AxAABB otherEdgeAABB[3];

				AxUInt32 idsA[3];
				idsA[0] = selfTriaglePtID3.x;
				idsA[1] = selfTriaglePtID3.y;
				idsA[2] = selfTriaglePtID3.z;

				AxUInt32 idsB[3];
				idsB[0] = otherTriaglePtID3.x;
				idsB[1] = otherTriaglePtID3.y;
				idsB[2] = otherTriaglePtID3.z;
				
				AxFp32 eta = selfMaxEta;
				AxFp32 etb = otherMaxEta;// other.PrimMaxEtaRaw == nullptr ? 0.00001f : other.PrimMaxEtaRaw[triangleIDB];
				
				#pragma unroll 3
				AX_FOR_I(3)
				{
					SelfPos3[i]		= selfPosRaw[idsA[i]];
					SelfPrdPPos3[i]	= selfPrdPRaw[idsA[i]];
					NbPos3[i]		= otherPosRaw[idsB[i]];
					NbPrdPPos3[i]	= otherPrdPRaw[idsB[i]];
					selfRtV[i]		= AlphaCore::Collision::ShareCode::RTriangleTest(selfRTriangleInfo, allToken[i]);
					selfRtE[i]		= AlphaCore::Collision::ShareCode::RTriangleTest(selfRTriangleInfo, allToken[i + 3]);
					otherRtV[i]		= AlphaCore::Collision::ShareCode::RTriangleTest(otherRTriangleInfo, allToken[i]);
					otherRtE[i]		= AlphaCore::Collision::ShareCode::RTriangleTest(otherRTriangleInfo, allToken[i + 3]);
					selfPtAABB[i]	= AlphaCore::AccelTree::MakeAABB(SelfPos3[i], SelfPrdPPos3[i], eta);
					otherPtAABB[i]	= AlphaCore::AccelTree::MakeAABB(NbPos3[i], NbPrdPPos3[i], etb);
				}

				//if (traceDebugInfo)
				//{
				//	printf("TRACE\n");
				//	printf("RTriangle Test A:[%d,%d,%d,%d,%d,%d]\n", selfRtV[0],  selfRtV[1],  selfRtV[2],  selfRtE[0],  selfRtE[1],  selfRtE[2]);
				//	printf("RTriangle Test B:[%d,%d,%d,%d,%d,%d]\n", otherRtV[0], otherRtV[1], otherRtV[2], otherRtE[0], otherRtE[1], otherRtE[2]);
				//	AlphaCore::Collision::Debug::Trace101::VFTest_DEBUGINFO
				//}

				AxVector2UI edgeVtxIds[3];
				edgeVtxIds[0].x = 2;	edgeVtxIds[0].y = 1;
				edgeVtxIds[1].x = 1;	edgeVtxIds[1].y = 0;
				edgeVtxIds[2].x = 0;	edgeVtxIds[2].y = 2;

				#pragma unroll 3
				AX_FOR_I(3)
				{
					int e0 = edgeVtxIds[i].x;
					int e1 = edgeVtxIds[i].y;
					selfEdgeAABB[i]  = AlphaCore::AccelTree::ShareCode::MergeAABB(selfPtAABB[e0], selfPtAABB[e1]);
					otherEdgeAABB[i] = AlphaCore::AccelTree::ShareCode::MergeAABB(otherPtAABB[e0], otherPtAABB[e1]);
				}

				//--------------------------------------//
				//               allToken               //
				//--------------------------------------//

				AxFp32 _eta = (eta + etb)*2.0f;
				AxFp32 colliTime = 0.0f;
				char contactNodeId;
				int ctxNum = 0;
				AxContact contacts[16];

 
				#pragma unroll 3
				AX_FOR_I(3)
				{
					int v0 = idsA[i];
					int vOther0 = idsB[i];
					//if (v0 != 7498)
					//	continue;
					//		  1         2
					//		   x------x
					//		  /  \   /
					//		 /    \ /
					//	  0 x------x 3
					//
					bool markSelf = ((v0 == idsB[0] || v0 == idsB[1] || v0 == idsB[2]) && processSelf);
					int ctxType = 0;
					if (selfRtV[i] && !markSelf)
					{

						//AlphaCore::Collision::Debug::Trace101::IsIncludeTracePoint(idsA[0], idsA[1], idsA[2], idsB[0], idsB[1], idsB[2]);
						if (AlphaCore::AccelTree::ShareCode::Intersect(selfPtAABB[i], otherTriangleAABB))
						{
							//if (!processSelf)
							//	AlphaCore::Collision::Debug::Trace101::VFEntryTest(idsA[i], idsB[0], idsB[1], idsB[2]);
							ctxType = AlphaCore::Collision::VertexFaceFlipCCD(
								SelfPos3[i], NbPos3[0], NbPos3[1], NbPos3[2],
								SelfPrdPPos3[i], NbPrdPPos3[0], NbPrdPPos3[1], NbPrdPPos3[2],
								nullptr, _eta, colliTime, contactNodeId);



							if (ctxType == AlphaCore::AxContactType::kVF)
							{
								contacts[ctxNum].Token = processSelf ? AlphaCore::Collision::VFSelfToken() : AlphaCore::Collision::VFStaticToken();
								contacts[ctxNum].Points[0] = v0;
								contacts[ctxNum].Points[1] = idsB[0];
								contacts[ctxNum].Points[2] = idsB[1];
								contacts[ctxNum].Points[3] = idsB[2];
 								contacts[ctxNum].NormalDir = Dot(SelfPos3[i] - NbPos3[0], Cross(NbPos3[2] - NbPos3[1], NbPos3[1] - NbPos3[0])) > 0.0f;
								ctxNum++;
							}

							
							/*
							if (AlphaCore::Collision::Debug::Trace101::VFEntryTest(idsA[i], idsB[0], idsB[1], idsB[2]))
							{
								AlphaCore::Collision::Debug::Trace103::VFTestWithDebugInfo(
									vOther0, 
									NbPos3[i], 
									NbPrdPPos3[i], 
									SelfPos3,
									SelfPrdPPos3,
									_eta,
									processSelf, 
									ctxType);
							}
							*/
						}

					}

					bool markOther = ((vOther0 == idsA[0] || vOther0 == idsA[1] || vOther0 == idsA[2]) && processSelf);

					if (otherRtV[i] && !markOther)
					{
						//AlphaCore::Collision::Debug::Trace101::IsIncludeTracePoint(idsA[0], idsA[1], idsA[2], idsB[0], idsB[1], idsB[2]);
						if (AlphaCore::AccelTree::ShareCode::Intersect(otherPtAABB[i], selfTriangleAABB))
						{
							ctxType = AlphaCore::Collision::VertexFaceFlipCCD(
								NbPos3[i], SelfPos3[0], SelfPos3[1], SelfPos3[2],
								NbPrdPPos3[i], SelfPrdPPos3[0], SelfPrdPPos3[1], SelfPrdPPos3[2],
								nullptr, _eta, colliTime, contactNodeId);

							if (ctxType == AlphaCore::AxContactType::kVF)
							{
								contacts[ctxNum].Token = processSelf ? AlphaCore::Collision::VFSelfToken() : AlphaCore::Collision::VStaticFToken();
								contacts[ctxNum].Points[0] = vOther0;
								contacts[ctxNum].Points[1] = idsA[0];
								contacts[ctxNum].Points[2] = idsA[1];
								contacts[ctxNum].Points[3] = idsA[2];
 								contacts[ctxNum].NormalDir = Dot(NbPos3[i] - SelfPos3[0], Cross(SelfPos3[2] - SelfPos3[1], SelfPos3[1] - SelfPos3[0])) > 0.0f;
								ctxNum++;
							}
							//AlphaCore::Collision::Debug::Trace101::VFTest();
							//printf("debugTrace@CCD_VF idx4:[%d,%d,%d,%d] processSelf:%d  ctxType:%d\n", vOther0, idsA[0], idsA[1], idsA[2], processSelf ? 1 : 0, ctxType);

						}
						//AlphaCore::Collision::Debug::Trace101::VFTestWithDebugInfo(vOther0, NbPos3[i], NbPrdPPos3[i],SelfPos3, SelfPrdPPos3,_eta, processSelf, ctxType);
						/*
						printf("debugTrace@CCD_VF idx4:[%d,%d,%d,%d] processSelf:%d  ctxType:%d\n", vOther0, idsA[0], idsA[1], idsA[2], processSelf ? 1 : 0, ctxType);
						*/
					}

					//	Edge - Edge
					if (!doEETest)
						continue;
					if (!selfRtE[i])
						continue;

					//TODO : NEED REMOVE : eeIdA & eeIdB
					AxAABB		edgeAabbSelf = selfEdgeAABB[i];
					AxVector2UI currEdgeVtxId = edgeVtxIds[i];
					int eVtx0 = edgeVtxIds[i].x;
					int eVtx1 = edgeVtxIds[i].y;
					int ePt0Self = idsA[eVtx0];
					int ePt1Self = idsA[eVtx1];
					AxVector3 e0Pos = SelfPos3[edgeVtxIds[i].x];
					AxVector3 e1Pos = SelfPos3[edgeVtxIds[i].y];
					AxVector3 e0PrdP = SelfPrdPPos3[edgeVtxIds[i].x];
					AxVector3 e1PrdP = SelfPrdPPos3[edgeVtxIds[i].y];

					for (short ee = 0; ee < 3; ++ee)
					{
						AxVector2UI edgeVtxIdOther = edgeVtxIds[ee];
						int eVtx0Other = edgeVtxIdOther.x;
						int eVtx1Other = edgeVtxIdOther.y;
						int ePt0Other = idsB[eVtx0Other];
						int ePt1Other = idsB[eVtx1Other];
						if (!otherRtE[ee])
							continue;
						// EdgeId:

						//RxUInt64 edgeTokenOther = (RxUInt64)max(ePt0Other, ePt1Other) * (RxUInt64)npts + (RxUInt64)min(ePt0Other, ePt1Other);
						//if (processSelf && (edgeTokenSelf <= edgeTokenOther))
						//	continue;
						AxAABB edgeAabbOther = otherEdgeAABB[ee];
						if (!AlphaCore::AccelTree::ShareCode::Intersect(edgeAabbSelf, edgeAabbOther))
							continue;

						if ((ePt0Self == ePt0Other || ePt0Self == ePt1Other || ePt1Self == ePt0Other || ePt1Self == ePt1Other) && processSelf)
							continue;

						//	print EdgeId - EdgeId BoundingBox
						AxVector3 e2Pos = NbPos3[eVtx0Other];
						AxVector3 e3Pos = NbPos3[eVtx1Other];
						AxVector3 e2PrdP = NbPrdPPos3[eVtx0Other];
						AxVector3 e3PrdP = NbPrdPPos3[eVtx1Other];

						bool isEE = AlphaCore::Collision::EdgeEdgeFlipCCD(
							e0Pos, e1Pos, e2Pos, e3Pos,
							e0PrdP, e1PrdP, e2PrdP, e3PrdP,
							_eta,colliTime);

						if (!isEE)
							continue;

 						contacts[ctxNum].Token = processSelf ? AlphaCore::Collision::EESelfToken() : AlphaCore::Collision::EEStaticToken();
						contacts[ctxNum].Points[0] = ePt0Self;
						contacts[ctxNum].Points[1] = ePt1Self;
						contacts[ctxNum].Points[2] = ePt0Other;
						contacts[ctxNum].Points[3] = ePt1Other;
						contacts[ctxNum].NormalDir = CalcRestEEConstr(e0Pos, e1Pos, e2Pos, e3Pos) >= -0.0f ? true : false;
						ctxNum++;
					}
					//other.E
				}
				int lastIndex = lock.AddIndex(ctxNum);
				int ii = 0;
				AX_FOR_RANGE_I(lastIndex, lastIndex + ctxNum) {
					ctxBuffer[i] = contacts[ii];
					ii++;
				}
				return ctxNum;
			}

			struct __DebugTask
			{
				int OtherPrimId;
			};

			ALPHA_SHARE_FUNC void FCCDWithSortedBVH__ORIGIN(AxUInt32 idx,
				AlphaCore::SolidUtility::SimData::AxSolidData::RAWDesc self,
				AlphaCore::SolidUtility::SimData::AxSolidData::RAWDesc other,
				AxBVHTree::RAWDesc bvhTreeA,
				AxBVHTree::RAWDesc bvhTreeB,
				AxContact* contactRaw,
				AxSPMDTick SPMDTick)
			{
				AxAABB*		allAABBs_ITER;
				AxBVHNode*	allNodes_ITER;
				AxUInt32	bvIDStack[256];
				AxUInt32	linkNum = 0;
				AxUInt32	aabbTestCount = 0;
				AxVector3UI idsSelf = bvhTreeA.IndicesBuffer[idx];//Heat bug
				AxUInt32* oldIdRaw = bvhTreeA.SortedPrimId;
				AxUInt32 numPrims_ITER = bvhTreeA.NumPrimitives;
				AxAABB targetBV = bvhTreeA.PrimBVs[idx];
#ifndef __CUDA_ARCH__
				//std::cout << "TargetBV:" << targetBV << std::endl;
#endif
				AX_FOR_K((other.Valid ? 2 : 1))
				{
					int currVistPtr = 0;
					int maxIter = 0;
					int LRIndex[2];
					bool processSelf = true;
					if (k == 0) {
						allAABBs_ITER = bvhTreeA.AllBVs;
						allNodes_ITER = bvhTreeA.AllNodes;
					}
					else {
						allAABBs_ITER = bvhTreeB.AllBVs;
						allNodes_ITER = bvhTreeB.AllNodes;
						numPrims_ITER = bvhTreeB.NumPrimitives;
						processSelf = false;
					}
					bvIDStack[0] = 0;
					int loop = 0;
					while (currVistPtr <= maxIter)
					{

						int _loop = currVistPtr / 256;
						AxUInt32 currVistId = bvIDStack[currVistPtr - _loop * 256];
						AxBVHNode node = allNodes_ITER[currVistId];
						AxAABB currAABB = allAABBs_ITER[currVistId];

						if (!AlphaCore::AccelTree::ShareCode::Intersect(targetBV, currAABB))
						{
							aabbTestCount++;
							currVistPtr++;
							continue;
						}

						LRIndex[0] = node.Left;
						LRIndex[1] = node.Right;
#pragma unroll 2
						AX_FOR_I(2)
						{
							if (LRIndex[i] < numPrims_ITER - 1)
							{
								maxIter++;
								int loop = maxIter / 256;
								bvIDStack[maxIter - loop * 256] = LRIndex[i];
								continue;
							}
							AxBVHNode nodeLeaf = allNodes_ITER[LRIndex[i]];
							int otherPrimId = LRIndex[i] - numPrims_ITER + 1;
							if (idx <= otherPrimId && processSelf)
								continue;
							AxAABB otherBV = allAABBs_ITER[LRIndex[i]];
							//allAABBs_ITER
							///if (AlphaCore::Collision::Debug::Trace101::VFEntryTest(idsSelf))
							///	AlphaCore::Collision::Debug::Trace102::CollisionAABBB_Search(oldIdRaw[idx], oldIdRaw[otherPrimId], targetBV, otherBV, k != 1, true);

							if (AlphaCore::AccelTree::ShareCode::Intersect(targetBV, otherBV))
							{
								AxVector3UI idsOther = processSelf ? bvhTreeA.IndicesBuffer[otherPrimId] : bvhTreeB.IndicesBuffer[otherPrimId];
								/*
								//if (!processSelf)
								{
									//Id
									printf("Sorted This <<%d,%d>> sorted old:%d-%d [%s] || idThis[%d,%d,%d]   idOther[%d,%d,%d]  \n",
										idx, otherPrimId,
										oldIdRaw[idx], oldIdRaw[otherPrimId], processSelf ? "Self" : "Collider",
										idsSelf.x, idsSelf.y, idsSelf.z,
										idsOther.x, idsOther.y, idsOther.z);
								}
								//*/

								AxUInt32 ctxNum = AlphaCore::Collision::ShareCode::FlipCCDTrianglePair(
									bvhTreeA.Pos0Buffer, processSelf ? bvhTreeA.Pos0Buffer : bvhTreeB.Pos0Buffer,
									bvhTreeA.Pos1Buffer, processSelf ? bvhTreeA.Pos1Buffer : bvhTreeB.Pos1Buffer,
									idx, otherPrimId,
									bvhTreeA.RTriangleTokens[idx], processSelf ? bvhTreeA.RTriangleTokens[otherPrimId] : bvhTreeB.RTriangleTokens[otherPrimId],
									bvhTreeA.MaxEta[idx], processSelf ? bvhTreeA.MaxEta[otherPrimId] : 0.0f,
									idsSelf, idsOther,
									targetBV, otherBV,
									contactRaw, processSelf, SPMDTick);
								linkNum++;
								//printf("ctxNum:%d", ctxNum);
							}
							//
							//AlphaCore::Collision::Debug::Trace102::CollisionAABBB_Search(oldIdRaw[idx], oldIdRaw[otherPrimId], targetBV, otherBV, k != 1,true);
							//

							aabbTestCount++;
						}
						currVistPtr++;
					}
				}
			}

			ALPHA_SHARE_FUNC void FCCDWithSortedBVH(AxUInt32 idx,
				AlphaCore::SolidUtility::SimData::AxSolidData::RAWDesc self,
				AlphaCore::SolidUtility::SimData::AxSolidData::RAWDesc other,
				AxBVHTree::RAWDesc bvhTreeA,
				AxBVHTree::RAWDesc bvhTreeB,
				AxContact* contactRaw,
				AxSPMDTick SPMDTick)
			{
				AxAABB*		allAABBs_ITER;
				AxBVHNode*	allNodes_ITER;
				AxUInt32	bvIDStack[256];
				AxUInt32	linkNumSelf = 0;
				AxUInt32	linkNumOther = 0;

				AxUInt32	aabbTestCount = 0;
				AxVector3UI idsSelf = bvhTreeA.IndicesBuffer[idx];//Heat bug
				AxUInt32* oldIdRaw	= bvhTreeA.SortedPrimId;
				AxUInt32 numPrims_ITER = bvhTreeA.NumPrimitives;
				AxAABB targetBV = bvhTreeA.PrimBVs[idx];
#ifndef __CUDA_ARCH__
				//std::cout << "TargetBV:" << targetBV << std::endl;
#endif
				__DebugTask selfTask[1024];
				__DebugTask colliderTask[1024];
				AX_FOR_K((other.Valid ? 2 : 1))
				{
					int currVistPtr = 0;
					int maxIter = 0;
					int LRIndex[2];
					bool processSelf = true;
					if (k == 0) {
						allAABBs_ITER = bvhTreeA.AllBVs;
						allNodes_ITER = bvhTreeA.AllNodes;
					}
					else {
						allAABBs_ITER = bvhTreeB.AllBVs;
						allNodes_ITER = bvhTreeB.AllNodes;
						numPrims_ITER = bvhTreeB.NumPrimitives;
						processSelf = false;
					}
					bvIDStack[0] = 0;
					int loop = 0;
					while (currVistPtr <= maxIter)
					{
						int _loop = currVistPtr / 256;
						AxUInt32 currVistId = bvIDStack[currVistPtr - _loop * 256];
						AxBVHNode node = allNodes_ITER[currVistId];
						AxAABB currAABB = allAABBs_ITER[currVistId];

  						if (!AlphaCore::AccelTree::ShareCode::Intersect(targetBV, currAABB))
						{
							aabbTestCount++;
							currVistPtr++;
							continue;
						}

						LRIndex[0] = node.Left;
						LRIndex[1] = node.Right;
						#pragma unroll 2
						AX_FOR_I(2)
						{
							if (LRIndex[i] < numPrims_ITER - 1)
							{
								maxIter++;
								int loop = maxIter / 256;
								bvIDStack[maxIter - loop * 256] = LRIndex[i];
								continue;
							}
							int otherPrimId = LRIndex[i] - numPrims_ITER + 1;
							if (idx <= otherPrimId && processSelf)
								continue;
							AxAABB otherBV = allAABBs_ITER[LRIndex[i]];
							if (AlphaCore::AccelTree::ShareCode::Intersect(targetBV, otherBV))
							{
								if (processSelf)
								{
									selfTask[linkNumSelf].OtherPrimId = otherPrimId;
									linkNumSelf++;
								}
								else {
									colliderTask[linkNumOther].OtherPrimId = otherPrimId;
									linkNumOther++;
								}
							}
							aabbTestCount++;
						}
						currVistPtr++;
					}
				}

				AX_FOR_I(linkNumSelf)
				{
					int otherPrimId = selfTask[i].OtherPrimId;
					AxVector3UI idsOther = bvhTreeA.IndicesBuffer[otherPrimId];
					AxAABB otherBV = bvhTreeA.AllBVs[otherPrimId + (bvhTreeA.NumPrimitives - 1)];

					AxUInt32 ctxNum = AlphaCore::Collision::ShareCode::FlipCCDTrianglePair(
						bvhTreeA.Pos0Buffer, bvhTreeA.Pos0Buffer,
						bvhTreeA.Pos1Buffer, bvhTreeA.Pos1Buffer,
						idx, otherPrimId,
						bvhTreeA.RTriangleTokens[idx], bvhTreeA.RTriangleTokens[otherPrimId],
						bvhTreeA.MaxEta[idx], bvhTreeA.MaxEta[otherPrimId],
						idsSelf, idsOther,
						targetBV, otherBV,
						contactRaw, true, SPMDTick);
				}

				AX_FOR_I(linkNumOther)
				{
					int otherPrimId = colliderTask[i].OtherPrimId;
					AxVector3UI idsOther = bvhTreeB.IndicesBuffer[otherPrimId];
					AxAABB otherBV = bvhTreeB.AllBVs[otherPrimId + (bvhTreeB.NumPrimitives - 1)];

					AxUInt32 ctxNum = AlphaCore::Collision::ShareCode::FlipCCDTrianglePair(
						bvhTreeA.Pos0Buffer, bvhTreeB.Pos0Buffer,
						bvhTreeA.Pos1Buffer, bvhTreeB.Pos1Buffer,
						idx, otherPrimId,
						bvhTreeA.RTriangleTokens[idx], bvhTreeB.RTriangleTokens[otherPrimId],
						bvhTreeA.MaxEta[idx], 0.0f,
						idsSelf, idsOther,
						targetBV, otherBV,
						contactRaw, false, SPMDTick);
				}
			}



			ALPHA_SHARE_FUNC void FCCDWithSortedBVHFast(AxUInt32 idx,
				AlphaCore::SolidUtility::SimData::AxSolidData::RAWDesc self,
				AlphaCore::SolidUtility::SimData::AxSolidData::RAWDesc other,
				AxBVHTree::RAWDesc bvhTreeA,
				AxBVHTree::RAWDesc bvhTreeB,
				AxContact* contactRaw,
				AxSPMDTick SPMDTick)
			{
				AxAABB*		allAABBs_ITER;
				AxBVHNode*	allNodes_ITER;
				AxUInt32	bvIDStack[256];
				AxUInt32	linkNum = 0;
				AxUInt32	aabbTestCount = 0;
				AxVector3UI idsSelf = bvhTreeA.IndicesBuffer[idx];//Heat bug
				AxUInt32* oldIdRaw = bvhTreeA.SortedPrimId;
				AxUInt32 numPrims_ITER = bvhTreeA.NumPrimitives;
				AxAABB targetBV = bvhTreeA.PrimBVs[idx];
#ifndef __CUDA_ARCH__
				//std::cout << "TargetBV:" << targetBV << std::endl;
#endif
				AX_FOR_K((other.Valid ? 2 : 1))
				{
					int currVistPtr = 0;
					int maxIter = 0;
					int LRIndex[2];
					bool processSelf = true;
					if (k == 0) {
						allAABBs_ITER = bvhTreeA.AllBVs;
						allNodes_ITER = bvhTreeA.AllNodes;
					}
					else {
						allAABBs_ITER = bvhTreeB.AllBVs;
						allNodes_ITER = bvhTreeB.AllNodes;
						numPrims_ITER = bvhTreeB.NumPrimitives;
						processSelf = false;
					}
					bvIDStack[0] = 0;
					int loop = 0;
					while (currVistPtr <= maxIter)
					{

						int _loop = currVistPtr / 256;
						AxUInt32 currVistId = bvIDStack[currVistPtr - _loop * 256];
						AxBVHNode node = allNodes_ITER[currVistId];
						AxAABB currAABB = allAABBs_ITER[currVistId];

						if (!AlphaCore::AccelTree::ShareCode::Intersect(targetBV, currAABB))
						{
							aabbTestCount++;
							currVistPtr++;
							continue;
						}

						LRIndex[0] = node.Left;
						LRIndex[1] = node.Right;
#pragma unroll 2
						AX_FOR_I(2)
						{
							if (LRIndex[i] < numPrims_ITER - 1)
							{
								maxIter++;
								int loop = maxIter / 256;
								bvIDStack[maxIter - loop * 256] = LRIndex[i];
								continue;
							}
							AxBVHNode nodeLeaf = allNodes_ITER[LRIndex[i]];
							int otherPrimId = LRIndex[i] - numPrims_ITER + 1;
							if (idx <= otherPrimId && processSelf)
								continue;
							AxAABB otherBV = allAABBs_ITER[LRIndex[i]];
							//allAABBs_ITER
							///if (AlphaCore::Collision::Debug::Trace101::VFEntryTest(idsSelf))
							///	AlphaCore::Collision::Debug::Trace102::CollisionAABBB_Search(oldIdRaw[idx], oldIdRaw[otherPrimId], targetBV, otherBV, k != 1, true);

							if (AlphaCore::AccelTree::ShareCode::Intersect(targetBV, otherBV))
							{
								AxVector3UI idsOther = processSelf ? bvhTreeA.IndicesBuffer[otherPrimId] : bvhTreeB.IndicesBuffer[otherPrimId];
								/*
								//if (!processSelf)
								{
									//Id
									printf("Sorted This <<%d,%d>> sorted old:%d-%d [%s] || idThis[%d,%d,%d]   idOther[%d,%d,%d]  \n",
										idx, otherPrimId,
										oldIdRaw[idx], oldIdRaw[otherPrimId], processSelf ? "Self" : "Collider",
										idsSelf.x, idsSelf.y, idsSelf.z,
										idsOther.x, idsOther.y, idsOther.z);
								}
								//*/

								AxUInt32 ctxNum = AlphaCore::Collision::ShareCode::FlipCCDTrianglePair(
									bvhTreeA.Pos0Buffer, processSelf ? bvhTreeA.Pos0Buffer : bvhTreeB.Pos0Buffer,
									bvhTreeA.Pos1Buffer, processSelf ? bvhTreeA.Pos1Buffer : bvhTreeB.Pos1Buffer,
									idx, otherPrimId,
									bvhTreeA.RTriangleTokens[idx], processSelf ? bvhTreeA.RTriangleTokens[otherPrimId] : bvhTreeB.RTriangleTokens[otherPrimId],
									bvhTreeA.MaxEta[idx], processSelf ? bvhTreeA.MaxEta[otherPrimId] : 0.0f,
									idsSelf, idsOther,
									targetBV, otherBV,
									contactRaw, processSelf, SPMDTick);
								linkNum++;
								//printf("ctxNum:%d", ctxNum);
							}
							//
							//AlphaCore::Collision::Debug::Trace102::CollisionAABBB_Search(oldIdRaw[idx], oldIdRaw[otherPrimId], targetBV, otherBV, k != 1,true);
							//

							aabbTestCount++;
						}
						currVistPtr++;
					}
				}
			}
		}
	}//@namespace end of : Collision
}
#endif