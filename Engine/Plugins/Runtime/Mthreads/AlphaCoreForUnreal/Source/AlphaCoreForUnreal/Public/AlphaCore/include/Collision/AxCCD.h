#ifndef __AX_CCD_H__
#define __AX_CCD_H__

#include "AxMacro.h"
#include "AxDataType.h"
#include "Math/AxVectorHelper.h"
#include "Math/AxMath101.h"
#define FCCD_ITERS 10
namespace AlphaCore 
{
	namespace Collision
	{
		ALPHA_SHARE_FUNC int VertexFaceFlipCCD(
			const AxVector3& q0start,
			const AxVector3& q1start,
			const AxVector3& q2start,
			const AxVector3& q3start,
			const AxVector3& q0end,
			const AxVector3& q1end,
			const AxVector3& q2end,
			const AxVector3& q3end,
			bool* rtMask6,
			AxFp32 eta,
			AxFp32 &t,
			char &contactNodeId,
			bool *flipTestRtn = nullptr)
		{
			// FLIP Test

			AxFp32 colliFreeDist = Dot(q0start - q1start, Normalize(Cross(q2start - q1start, q3start - q1start)));
			bool colliFreeDir = colliFreeDist >= 0.0f ? true : false;
			AxFp32 prdPTimeDist = Dot(q0end - q1end, Normalize(Cross(q2end - q1end, q3end - q1end)));
			bool prdPTimeDir = prdPTimeDist >= 0.0f ? true : false;
			//RX_LOG_INFO("colliFreeDir:%d; prdPTimeDir:%d", colliFreeDir, prdPTimeDir);
			const float vfeplsion = 0.0f;
			if (colliFreeDir == prdPTimeDir)  // same side: no collision will happen
			{
				//RX_LOG_INFO("FLIP test --> should do DCD");
				if (fabs(prdPTimeDist) - eta >= -1e-6f)
				{
					//*flipTestRtn = true;
					return AxContactType::NonContact;
				}
				///*
				AxVector3 weight;
				//barycentricCoord(q0end, q1end, q2end, q3end, weight);
				// isIndside?
				AlphaCore::Math::BaryCenterCoordinate(q0end, q1end, q2end, q3end, weight);
				//RX_LOG_WARNING("prdP: weight.x: %f, weight.y: %f, weight.z: %f", weight.x, weight.y, weight.z);
				t = 0.0F;
				if (!(weight.x<0.0F - vfeplsion || weight.x>1.0f + vfeplsion ||
					weight.y<0.0F - vfeplsion || weight.y>1.0f + vfeplsion ||
					weight.z<0.0F - vfeplsion || weight.z>1.0f + vfeplsion))
					//*/
					//if(IsInSide<T>(q0end, q1end, q2end, q3end))
				{
					//*flipTestRtn = true;
					return AxContactType::kVF;
				}
				/*
				else
				{
					float maxEdge = 0.0F;
					float edge1 = length(q3end - q2end);
					if (maxEdge < edge1)
						maxEdge = edge1;
					float edge2 = length(q1end - q3end);
					if (maxEdge < edge2)
						maxEdge = edge2;
					float edge3 = length(q2end - q1end);
					if (maxEdge < edge3)
						maxEdge = edge3;
					float epsilon = eta / maxEdge;
					if (nbMask.E[1] && weight.x < 0.0F && weight.x > -epsilon) // on edge 2-3
					{
						AxVector3 dd = q3end - q2end;
						float dd2 = dot(dd, dd);
						float tt = (dd2 == 0.0F) ? (0.5F) : dot(q0end - q2end, dd) / dd2;
						if (nbMask.V[1] && tt < 0.0F && tt > -epsilon) // on point 2
						{
							contactNodeId = 1;
							return ContactType::kVV;
						}
						if (nbMask.V[2] && tt > 1.0F && tt < 1.0F + epsilon) // on point 3
						{
							contactNodeId = 2;
							return ContactType::kVV;
						}
						contactNodeId = 1;
						return ContactType::kVE;
					}
					else if (nbMask.E[2] && weight.y < 0.0F && weight.y > -epsilon) // on edge 3-1
					{
						AxVector3 dd = q1end - q3end;
						float dd2 = dot(dd, dd);
						float tt = (dd2 == 0.0F) ? (0.5F) : dot(q0end - q3end, dd) / dd2;
						if (nbMask.V[2] && tt < 0.0F && tt > -epsilon) //on point 3
						{
							contactNodeId = 2;
							return ContactType::kVV;
						}
						if (nbMask.V[0] && tt > 1.0F && tt < 1.0F + epsilon) //on point 1
						{
							contactNodeId = 0;
							return ContactType::kVV;
						}
						contactNodeId = 2;
						return ContactType::kVE;
					}
					else if (nbMask.E[0] && weight.z < 0.0F && weight.z > -epsilon) // on edge 1-2
					{
						//printf("edge1-2: weight.z: %f\n", weight.z);
						AxVector3 dd = q2end - q1end;
						float dd2 = dot(dd, dd);
						float tt = (dd2 == 0.0F) ? (0.5F) : dot(q0end - q1end, dd) / dd2;
						if (nbMask.V[0] && tt < 0.0F && tt > -epsilon) //on point 1
						{
							contactNodeId = 0;
							return ContactType::kVV;
						}
						if (nbMask.V[1] && tt > 1.0F && tt < 1.0F + epsilon) // on point 2
						{
							contactNodeId = 1;
							return ContactType::kVV;
						}
						contactNodeId = 0;
						return ContactType::kVE;
					}
				}
				//*/ //203 Branch
				return AxContactType::NonContact;
			}
			//*flipTestRtn = true;
			//return false;  // FOR TEST ONLY

			//RX_LOG_INFO("DCD is false --> should do Flip CCD");
			int i = 0;
			int maxIteration = 10;
			AxFp32 currTime = 1.0f;
			AxFp32 targetTime = 0.0f;
			AxFp32 prdTime = 0.5f;
			AxFp32 prdTimeDist = 0.0f;
			AxVector3 nq0end, nq1end, nq2end, nq3end;

			while (i < FCCD_ITERS)  // 10 iterations
			{
				prdTime = AlphaCore::Math::Lerp(targetTime, currTime, 0.5f);
				nq0end = AlphaCore::Math::Lerp(q0start, q0end, prdTime);
				nq1end = AlphaCore::Math::Lerp(q1start, q1end, prdTime);
				nq2end = AlphaCore::Math::Lerp(q2start, q2end, prdTime);
				nq3end = AlphaCore::Math::Lerp(q3start, q3end, prdTime);
				prdTimeDist = Dot((nq0end - nq1end), Normalize(Cross(nq2end - nq1end, nq3end - nq1end)));
				int prdPTimeDir = prdTimeDist >= 0.0F ? 1 : 0;

				if (!(colliFreeDir == prdPTimeDir))
					currTime = prdTime;
				else
					targetTime = prdTime;
				i++;
			}

			AxVector3 weight;
			AlphaCore::Math::BaryCenterCoordinate(nq0end, nq1end, nq2end, nq3end, weight);
			// VF
			if (!(weight.x<0.0F - vfeplsion || weight.x>1.0F + vfeplsion ||
				weight.y<0.0F - vfeplsion || weight.y>1.0F + vfeplsion ||
				weight.z<0.0F - vfeplsion || weight.z>1.0F + vfeplsion))
				//if (IsInSide<T>(nq0end, nq1end, nq2end, nq3end))
			{
				return AxContactType::kVF;
			}
			/*
			else
			{
				float maxEdge = 0.0F;
				float edge1 = length(nq3end - nq2end);
				if (maxEdge < edge1)
					maxEdge = edge1;
				float edge2 = length(nq1end - nq3end);
				if (maxEdge < edge2)
					maxEdge = edge2;
				float edge3 = length(nq2end - nq1end);
				if (maxEdge < edge3)
					maxEdge = edge3;
				//float epsilon = eta / ((edge1 + edge2 + edge3) / 3.0F);
				float epsilon = eta / maxEdge;
				if (nbMask.E[1] && weight.x < 0.0F && weight.x > -epsilon) // on edge 2-3
				{
					AxVector3 dd = nq3end - nq2end;
					float dd2 = dot(dd, dd);
					float tt = (dd2 == 0.0F) ? (0.5F) : dot(nq0end - nq2end, dd) / dd2;
					if (nbMask.V[1] && tt < 0.0F && tt > -epsilon) // on point 2
					{
						contactNodeId = 1;
						return ContactType::kVV;
					}
					if (nbMask.V[2] && tt > 1.0F && tt < 1.0F + epsilon) // on point 3
					{
						contactNodeId = 2;
						return ContactType::kVV;
					}
					contactNodeId = 1;
					return ContactType::kVE;
				}
				else if (nbMask.E[2] && weight.y < 0.0F && weight.y > -epsilon) // on edge 3-1
				{
					AxVector3 dd = nq1end - nq3end;
					float dd2 = dot(dd, dd);
					float tt = (dd2 == 0.0F) ? (0.5F) : dot(nq0end - nq3end, dd) / dd2;
					if (nbMask.V[2] && tt < 0.0F && tt > -epsilon) //on point 3
					{
						contactNodeId = 2;
						return ContactType::kVV;
					}
					if (nbMask.V[0] && tt > 1.0F && tt < 1.0F + epsilon) //on point 1
					{
						contactNodeId = 0;
						return ContactType::kVV;
					}
					contactNodeId = 2;
					return ContactType::kVE;
				}
				else if (nbMask.E[0] && weight.z < 0.0F && weight.z > -epsilon) // on edge 1-2
				{
					//printf("edge1-2: weight.z: %f\n", weight.z);
					AxVector3 dd = nq2end - nq1end;
					float dd2 = dot(dd, dd);
					float tt = (dd2 == 0.0F) ? (0.5F) : dot(nq0end - nq1end, dd) / dd2;
					if (nbMask.V[0] && tt < 0.0F && tt > -epsilon) //on point 1
					{
						contactNodeId = 0;
						return ContactType::kVV;
					}
					if (nbMask.V[1] && tt > 1.0F && tt < 1.0F + epsilon) // on point 2
					{
						contactNodeId = 1;
						return ContactType::kVV;
					}
					contactNodeId = 0;
					return ContactType::kVE;
				}
			}
			//*/ //203 Branch
			return AxContactType::NonContact;
		}

		ALPHA_SHARE_FUNC AxFp32 CalcEEDist(
			const AxVector3 & e0Pos,
			const AxVector3 & e1Pos, 
			const AxVector3 & e2Pos,
			const AxVector3 & e3Pos,
			AxFp32* weight)
		{
			AxVector3 e1 = e1Pos - e0Pos;
			AxVector3 e2 = e3Pos - e2Pos;
			AxVector3 currNormal = Cross(Normalize(e1), Normalize(e2));
			if (Dot(currNormal, currNormal) <= 1e-3f)
				return 0.0f;
			//T weight[4];
			currNormal = Normalize(currNormal);
			AxFp32 a0 = AlphaCore::Math::Stp(e3Pos - e1Pos, e2Pos - e1Pos, currNormal);
			AxFp32 a1 = AlphaCore::Math::Stp(e2Pos - e0Pos, e3Pos - e0Pos, currNormal);
			AxFp32 b0 = AlphaCore::Math::Stp(e0Pos - e3Pos, e1Pos - e3Pos, currNormal);
			AxFp32 b1 = AlphaCore::Math::Stp(e1Pos - e2Pos, e0Pos - e2Pos, currNormal);
			weight[0] = a0 / (a0 + a1);
			weight[1] = a1 / (a0 + a1);
			weight[2] = -b0 / (b0 + b1);
			weight[3] = -b1 / (b0 + b1);
			AxVector3 dist = (weight[0] * e0Pos + weight[1] * e1Pos) - ((-weight[2]) * e2Pos + (-weight[3]) * e3Pos);
			return Dot(currNormal, dist);
		}

		ALPHA_SHARE_FUNC bool EdgeEdgeFlipCCD(
			const AxVector3 &q0start,
			const AxVector3 &p0start,
			const AxVector3 &q1start,
			const AxVector3 &p1start,
			const AxVector3 &q0end,
			const AxVector3 &p0end,
			const AxVector3 &q1end,
			const AxVector3 &p1end, 
			AxFp32 eta,
			AxFp32 &t,
			bool *angleTestRtn = nullptr)
		{
			//*angleTestRtn = true;
			//return true;
			//return false;

			AxVector3 e1start = Normalize(p0start - q0start);
			AxVector3 e2start = Normalize(p1start - q1start);
			AxVector3 e1end = Normalize(p0end - q0end);
			AxVector3 e2end = Normalize(p1end - q1end);
			if (fmaxf(fabs(Dot(e1start, e2start)), fabs(Dot(e1end, e2end))) > 0.9f)  // almost parallel
			{
				//RX_LOG_WARNING("ANGLE check -- almost parallel��%f --> return", max(abs(dot(e1start, e2start)), abs(dot(e1end, e2end))));
				//*angleTestRtn = true;
				return false;
			}

			//RX_LOG_WARNING("ANGLE check --> FLIP Test");
			// FLIP Test	
			AxFp32 colliFreeWeight[4];
			AxFp32 prdPWeight[4];

			AxFp32 colliFreeDist = CalcEEDist(q0start, p0start, q1start, p1start, colliFreeWeight);
			bool colliFreeDir = colliFreeDist >= -0.0F ? true : false;
			AxFp32 prdPTimeDist = CalcEEDist(q0end, p0end, q1end, p1end, prdPWeight);
			bool prdPTimeDir = prdPTimeDist >= -0.0F ? true : false;
			//RX_LOG_WARNING("colliFreeDist: %f; prdPTimeDist: %f", colliFreeDist, prdPTimeDist);

			if ((colliFreeWeight[0] < 0.0F || colliFreeWeight[0]>1.0F || colliFreeWeight[1] < 0.0F || colliFreeWeight[1]>1.0F ||
				-colliFreeWeight[2] < 0.0F || -colliFreeWeight[2]>1.0F || -colliFreeWeight[3] < 0.0F || -colliFreeWeight[3]>1.0F))
				return false;


			if (colliFreeDir == prdPTimeDir)  // same side: no collision will happen
			{
				// DCD Test
				AxVector3 prdN = Cross(e1end, e2end);
				if (Dot(prdN, prdN) <= 1e-3F)
					return false;

				if (fabs(prdPTimeDist) - eta >= -1e-6F)
					return false;

				if (!(prdPWeight[0]<0.0F || prdPWeight[0]>1.0F || prdPWeight[1]<0.0F || prdPWeight[1]>1.0F ||
					-prdPWeight[2]<0.0F || -prdPWeight[2]>1.0F || -prdPWeight[3]<0.0F || -prdPWeight[3]>1.0F))
				{
					return true;
				}
				return false;
			}

			//RX_LOG_INFO("DCD is false --> should do Flip CCD");

			int i = 0;
			int maxIteration = 10;
			float currTime = 1.0f;
			float targetTime = 0.0f;
			float prdTime = 0.5f;
			float prdTimeDist = 0.0f;
			AxVector3 nq0end, np0end, nq1end, np1end;

			while (i < FCCD_ITERS)  // 10 iterations
			{
				prdTime = AlphaCore::Math::Lerp(targetTime, currTime, 0.5f);
				nq0end  = AlphaCore::Math::Lerp(q0start, q0end, prdTime);
				np0end  = AlphaCore::Math::Lerp(p0start, p0end, prdTime);
				nq1end  = AlphaCore::Math::Lerp(q1start, q1end, prdTime);
				np1end  = AlphaCore::Math::Lerp(p1start, p1end, prdTime);
				prdTimeDist = CalcEEDist(nq0end, np0end, nq1end, np1end, prdPWeight);
				int prdTimeDir = prdTimeDist >= -0.0F ? 1 : 0;

				if (!(colliFreeDir == prdTimeDir))
					currTime = prdTime;
				else
					targetTime = prdTime;

				i++;
			}
			//printf("final prdTime: %d - prdTimeDist: %f\n", prdTime, prdTimeDist);
			//AxVector3 weight;
			//evalVFBary(nq0end, nq1end, nq1end, np1end, weight);
			if (!(prdPWeight[0]<0.0F || prdPWeight[0]>1.0F || prdPWeight[1]<0.0F || prdPWeight[1]>1.0F ||
				-prdPWeight[2]<0.0F || -prdPWeight[2]>1.0F || -prdPWeight[3]<0.0F || -prdPWeight[3]>1.0F))
			{
				//t = 1.0;
				//*angleTestRtn = true;
				return true;
				//return false;
			}
			//*angleTestRtn = true;
			return false;
			//printf("weight: %f", weight);

		}
	}
}


#endif
