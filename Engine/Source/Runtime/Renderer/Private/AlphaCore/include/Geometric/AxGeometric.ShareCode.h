#ifndef __AX_GEOMETRIC_SHARECODE_H__
#define __AX_GEOMETRIC_SHARECODE_H__

//#include "AxGeometric.h"
#include <AxMacro.h>
#include <Math/AxVectorBase.h>
#include <Math/AxVectorHelper.h>
#include <Math/AxMatrixBase.h>
#include <Math/AxMath101.h>
#include <Math/AxMat.ShareCode.h>

ALPHA_SHARE_FUNC void PrintInfo(const AxMatrix3x3&  m, const char* head, bool oneline = false)
{
	if (oneline)
	{
		printf("%s {%f,%f,%f,%f,%f,%f,%f,%f,%f}\n", head, m.m[0], m.m[1], m.m[2], m.m[3], m.m[4], m.m[5], m.m[6], m.m[7], m.m[8]);
	}
	else {
		printf("%s\t\n\t     [%f,%f,%f]\n", head, m.m[0], m.m[1], m.m[2]);
		printf("\t     [%f,%f,%f]\n", m.m[3], m.m[4], m.m[5]);
		printf("\t     [%f,%f,%f]\n", m.m[6], m.m[7], m.m[8]);
	}
}


namespace AlphaCore {
	namespace Geometric {
		namespace ShareCode {

			ALPHA_SHARE_FUNC void ComputePrimNormal(AxUInt32 idx,
				AxUInt32* topologyIndicesRaw,
				AxVector3* posRaw,
				AxVector2UI* prim2IRaw,
				AxVector3* primNormalRaw)
			{
				if (prim2IRaw[idx].y != 3) {
					primNormalRaw[idx] = MakeVector3(0, 0, 0);
					return;
				}

				AxUInt32 indices0 = topologyIndicesRaw[prim2IRaw[idx].x];
				AxUInt32 indices1 = topologyIndicesRaw[prim2IRaw[idx].x + 1];
				AxUInt32 indices2 = topologyIndicesRaw[prim2IRaw[idx].x + 2];
				AxVector3 pos0 = posRaw[indices0];
				AxVector3 pos1 = posRaw[indices1];
				AxVector3 pos2 = posRaw[indices2];
				AxVector3 p01 = pos1 - pos0;
				AxVector3 p02 = pos2 - pos0;
				AxVector3 primNormal = Cross(p02, p01);
				primNormalRaw[idx] = Normalize(primNormal);

				return;
			}

			ALPHA_SHARE_FUNC void IntegratePointNormal(AxUInt32 idx,
				AxVector2UI* point2PrimMapRaw,
				AxUInt32* point2PrimIndicesRaw,
				AxVector3* primNormalRaw,
				AxVector3* pointNormalRaw)
			{
				int start = point2PrimMapRaw[idx].x;
				int num = point2PrimMapRaw[idx].y;
				AxVector3 pointNormal = MakeVector3(0,0,0);
				AX_FOR_I(num)
				{
					int primNum = point2PrimIndicesRaw[start + i];
					pointNormal += primNormalRaw[primNum];
				}
				pointNormalRaw[idx] = Normalize(pointNormal);
			}

			ALPHA_SHARE_FUNC void MakeDeformerXform(
				AxUInt32 idx,
				AxMatrix3x3* xformsRaw,
				AxVector3* restPosRaw,
				AxVector3* dynamicPosRaw,
				AxVector2UI* p2pMapRaw,
				AxUInt32* p2pIndicesRaw)
			{
				AxVector2UI p2pStartNum = p2pMapRaw[idx];
				AxUInt32 start = p2pStartNum.x;
				AxUInt32 numLinkPts = p2pStartNum.y;

				if (numLinkPts < 2) {
					xformsRaw[idx] = Make3x3Identity();
					return;
				}

				AxVector3 restP = restPosRaw[idx];
				AxVector3 dynamicP = dynamicPosRaw[idx];
				AxMatrix3x3 xform = xformsRaw[idx];
				AxUInt32 lastNeighber = p2pIndicesRaw[start + numLinkPts - 1];
				AxVector3 lastSolvedP = dynamicPosRaw[lastNeighber] - dynamicP;
				AxVector3 lastRestdP = restPosRaw[lastNeighber] - restP;
				AxMatrix3x3 totalXform = MakeMat3x3();
				float totalWeight = 0.0f;

				//PrintInfo("lastSolvedP:", lastSolvedP);
 				//PrintInfo("dynamicP:", dynamicP);
 				//printf("lastNeighber:%d\n", lastNeighber);
				//PrintInfo("lastSolvedP:", lastSolvedP);
				//PrintInfo("lastRestdP:", lastRestdP);


				AX_FOR_I(numLinkPts) 
				{
					AxUInt32 linkedId = p2pIndicesRaw[start + i];
					AxVector3 solveddP = dynamicPosRaw[linkedId] - dynamicP;
					AxVector3 restdP = restPosRaw[linkedId] - restP;
					AxVector3 restUp = Cross(restdP, lastRestdP);
					AxVector3 solvedUp = Cross(solveddP, lastSolvedP);

					//PrintInfo(" -- A restUp:", solveddP);
					//PrintInfo(" -- B lastSolvedP:", lastSolvedP);
					lastRestdP = restdP;
					lastSolvedP = solveddP;
			

					AxFp32 weight = sqrt(Length(restUp) * Length(solvedUp));
					AxMatrix3x3 restxform = MakeTransform(Normalize(restdP), Normalize(restUp));
					AxMatrix3x3 solvedXform = MakeTransform(Normalize(solveddP), Normalize(solvedUp));
					bool inv = Inverse(restxform);
					if (!inv) 
						continue;
					totalXform = totalXform + restxform * solvedXform * weight;
					totalWeight = totalWeight + weight;
				}
				if (totalWeight != 0) {
					totalXform = totalXform / totalWeight;
				}
				xformsRaw[idx] = totalXform;
				//PrintInfo(totalXform, "totalXform", true);
			}


			ALPHA_SHARE_FUNC void DeformerByMatrix(
				AxUInt32 idx,
				AxVector3* targetPosRaw,
				AxVector3* restPosRaw,
				AxVector3* dynamicPosRaw,
				AxVector2UI* linkedPtsMapRaw,
				AxUInt32* linkedPtsIndicesRaw,
				AxFp32* linkedWeightsRaw,
				AxMatrix3x3* xformsRaw)
			{
				AxFp32 totalWeight = 0.f;
				AxVector3 delta = { 0,0,0 };
				AxVector3 originP = targetPosRaw[idx];
				AxVector2UI linkedPtsMap = linkedPtsMapRaw[idx];
				AxUInt32 linkedStart = linkedPtsMap.x;
				AxUInt32 linkedNum = linkedPtsMap.y;
				AX_FOR_I(linkedNum)
				{
					AxUInt32 linkedPt = linkedPtsIndicesRaw[linkedStart + i];
					AxVector3 oldCenter = restPosRaw[linkedPt];
					AxVector3 diff = dynamicPosRaw[linkedPt] - oldCenter;
					//AX_INFO("diff{}  {}: {},{},{}", idx,i, diff.x, diff.y, diff.z);
					AxMatrix3x3 xform = xformsRaw[linkedPt];
					float weight = linkedWeightsRaw[linkedStart + i];
					AxVector3 newP = originP;
					newP -= oldCenter;
					newP = newP * xform;
					newP += oldCenter;
					newP += diff;
					diff = newP - originP;
					delta += (diff * weight);
					totalWeight += weight;
				}
				delta /= totalWeight;
				targetPosRaw[idx] = originP + delta;
			}

		}
	}//@namespace end of : Geometric
}
#endif