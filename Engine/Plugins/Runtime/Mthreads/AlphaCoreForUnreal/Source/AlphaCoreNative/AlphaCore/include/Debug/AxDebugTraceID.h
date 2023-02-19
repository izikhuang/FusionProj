#ifndef __AX_DEBUG_TRACE_ID_H__
#define __AX_DEBUG_TRACE_ID_H__

namespace AlphaCore
{
	namespace DebugInfoID
	{
		static const char* BVH_101_TrianglesAABB		 = "BVH_101_TrianglesAABB";
		static const char* BVH_102_SortedFinalMortonCode = "BVH_102_SortedFinalMortonCode";
		static const char* BVH_103_BVHNode				 = "BVH_103_BVHNode";
		static const char* BVH_104_FinalAABB			 = "BVH_104_FinalAABB";
	}

	namespace TraceVisualization
	{
		namespace CCD
		{
			void NarrowPhase__VF_B();
			void NarrowPhase__VF_A();
			void BroadPhase__EE();

		}
	}
}


#endif