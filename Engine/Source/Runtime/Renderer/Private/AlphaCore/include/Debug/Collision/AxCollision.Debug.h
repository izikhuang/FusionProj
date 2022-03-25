#ifndef __AX_COLLISION__DOT__DEBUG__DOT__H__
#define __AX_COLLISION__DOT__DEBUG__DOT__H__

#include <AxDataType.h>
#include <AxMacro.h>
#include <Collision/AxCollision.DataType.h>
#include <AccelTree/AxAccelTree.DataType.h>


#define DEBUG_PIVOT_NODE_ID		346
#define DEBUG_PIVOT_NODE_ID2	306
#define TRACE_POINT_ID 5


#define TRACE_POINT_ID_2 5
#define TRACE_POINT_ID_3 6
#define TRACE_POINT_ID_4 7



#define TRACE101__CollisionAABBB_Search_ENABLE 1

#define TRACE101__CollisionAABBB_Test_ENABLE 1

#define TRACE101__AABBPairTest_ENABLE 1

/* Generate VF test code */

#define TRACE101__VFTestWithDebugInfo_ENABLE 1


namespace AlphaCore
{
    namespace Collision
    {
		namespace Debug
		{
			namespace Trace101
			{
				ALPHA_SHARE_FUNC bool VFEntryTest(AxInt32 v, AxInt32 f1, AxInt32 f2, AxInt32 f3)
				{
					if (v == TRACE_POINT_ID && f1 == TRACE_POINT_ID_2 && f2 == TRACE_POINT_ID_3 &&f3 == TRACE_POINT_ID_4)
					{
						printf("TRACE101__VFEntryTest :::: DO Intersection test @ vertex : %d face <%d,%d,%d> \n", TRACE_POINT_ID, TRACE_POINT_ID_2, TRACE_POINT_ID_3, TRACE_POINT_ID_4);
						return true;
					}
					return false;
				}

				ALPHA_SHARE_FUNC bool VFEntryTest(AxVector3UI v)
				{
					if (v.x == TRACE_POINT_ID || v.y == TRACE_POINT_ID || v.z == TRACE_POINT_ID)
					{
						printf("TRACE101__VFEntryTest ID3 :::: DO Intersection test @ vertex pivot face <%d,%d,%d> \n", v.x, v.y, v.z);
						return true;
					}
					return false;
				}
				/* Pivot : primitive_AABB */
				ALPHA_SHARE_FUNC bool IsIncludeTracePoint(AxUInt32 p0, AxUInt32 p1 ,AxUInt32 p2, AxUInt32 p00, AxUInt32 p11, AxUInt32 p22)
				{
					if (p0 == TRACE_POINT_ID || p1 == TRACE_POINT_ID || p2 == TRACE_POINT_ID ||
						p00 == TRACE_POINT_ID || p11 == TRACE_POINT_ID || p22 == TRACE_POINT_ID)
					{
						printf("TRACE101__IsIncludeTracePoint@%d\n", TRACE_POINT_ID);
						return true;
					}
					return false;
				}

				ALPHA_SHARE_FUNC bool IsTracePair(AxUInt32 selfIdx, AxUInt32 otherIdx)
				{
					if (selfIdx != DEBUG_PIVOT_NODE_ID && otherIdx != DEBUG_PIVOT_NODE_ID)
						return false;
					if (selfIdx != DEBUG_PIVOT_NODE_ID2 && otherIdx != DEBUG_PIVOT_NODE_ID2)
						return false;
					return true;
				}


				ALPHA_SHARE_FUNC void CollisionAABBB_Test(
					AxUInt32 idx,
					const AxAABB& a,
					const AxAABB& b,
					bool processSelf)
				{
				#if TRACE101__CollisionAABBB_Test_ENABLE
					if (idx != DEBUG_PIVOT_NODE_ID)
						return;
					printf("TRACE101__CollisionAABBB_Test @ %d,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
						idx, processSelf ? "Self" : "Collider",
						a.Min.x, a.Min.y, a.Min.z, a.Max.x, a.Max.y, a.Max.z,
						b.Min.x, b.Min.y, b.Min.z, b.Max.x, b.Max.y, b.Max.z);
				#endif
				}

				ALPHA_SHARE_FUNC void AABBPairTest(const AxAABB& a,const AxAABB& b,bool ret)
				{
				#if TRACE101__AABBPairTest_ENABLE
					printf("TRACE101__AABBPairTest @ %s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
						ret ? "True" : "False",
						a.Min.x, a.Min.y, a.Min.z, a.Max.x, a.Max.y, a.Max.z,
						b.Min.x, b.Min.y, b.Min.z, b.Max.x, b.Max.y, b.Max.z);
				#endif
				}
			}

			namespace Trace102
			{
				ALPHA_SHARE_FUNC void CollisionAABBB_Search(
					AxUInt32 selfIdx,
					AxUInt32 otherIdx,
					AxAABB a,
					AxAABB b,
					bool processSelf,
					bool dualMatch)
				{
#if TRACE101__CollisionAABBB_Search_ENABLE
					/*
					if (selfIdx != DEBUG_PIVOT_NODE_ID && otherIdx != DEBUG_PIVOT_NODE_ID)
						return;
					
					if (selfIdx != DEBUG_PIVOT_NODE_ID2 && otherIdx != DEBUG_PIVOT_NODE_ID2 && dualMatch)
						return;
					if (selfIdx != DEBUG_PIVOT_NODE_ID)
					{
						AlphaCore::Swap(selfIdx, otherIdx);
						AlphaCore::Swap(a, b);
					}
					*/
					printf("TRACE102__CollisionAABBB_Search @ %d,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d\n",
						selfIdx, processSelf ? "Self" : "Collider",
						a.Min.x, a.Min.y, a.Min.z, a.Max.x, a.Max.y, a.Max.z,
						b.Min.x, b.Min.y, b.Min.z, b.Max.x, b.Max.y, b.Max.z,
						selfIdx, otherIdx);
#endif
				}
			}
		
			namespace Trace103
			{
				ALPHA_SHARE_FUNC void VFTestWithDebugInfo(AxUInt32 idx,
					const AxVector3& selfPos,
					const AxVector3& selfPrdPPos,
					AxVector3* otherPos3,
					AxVector3* otherPrdPos3,
					AxFp32 eta,
					bool processSelf,
					AxInt32 ctxTypeToken)
				{
				#if TRACE101__VFTestWithDebugInfo_ENABLE 
					if (idx != TRACE_POINT_ID)
						return;

					printf("TRACE103__VFTestWithDebugInfo @ "\
						"vertexId : %d |"
						"vStart:%f,%f,%f |"\
						"vEnd:%f,%f,%f |"\
						"fStart:%f,%f,%f,%f,%f,%f,%f,%f,%f |"\
						"fEnd:%f,%f,%f,%f,%f,%f,%f,%f,%f |"\
						"eta:%f |"\
						"processSelf:%d |" \
						"ctxType:%d \n", idx,
						selfPos.x, selfPos.y, selfPos.z,
						selfPrdPPos.x, selfPrdPPos.y, selfPrdPPos.z,
						otherPos3[0].x, otherPos3[0].y, otherPos3[0].z,
						otherPos3[1].x, otherPos3[1].y, otherPos3[1].z,
						otherPos3[2].x, otherPos3[2].y, otherPos3[2].z,
						otherPrdPos3[0].x, otherPrdPos3[0].y, otherPrdPos3[0].z,
						otherPrdPos3[1].x, otherPrdPos3[1].y, otherPrdPos3[1].z,
						otherPrdPos3[2].x, otherPrdPos3[2].y, otherPrdPos3[2].z,
						eta, (processSelf ? 1 : 0), ctxTypeToken);

				#endif
				}
			}
}
    }
} 



#endif