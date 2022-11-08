#ifndef __AX_RTRIANGLE_H__
#define __AX_RTRIANGLE_H__

#include "AxDataType.h"
#include "AxMacro.h"

//token as unsigned Char
namespace AlphaCore
{
	enum AxRTriangleType
	{
		kVtx0  = 0b000001,
		kVtx1  = 0b000010,
		kVtx2  = 0b000100,
		kEdge0 = 0b001000,
		kEdge1 = 0b010000,
		kEdge2 = 0b100000,
		kNonRTriangleInfo = 0b000000
	};
}

namespace AlphaCore
{
	namespace Collision
	{
		inline void EnableRTrianglePointInfo(AxUChar& token, AxUChar index)
		{
			if (index == 0)
				token |= AxRTriangleType::kVtx0;
			if (index == 1)
				token |= AxRTriangleType::kVtx1;
			if (index == 2)
				token |= AxRTriangleType::kVtx2;
		}

		inline void EnableRTriangleEdgeInfo(AxUChar& token, AxUChar index)
		{
			if (index == 0)
				token |= AxRTriangleType::kEdge0;
			if (index == 1)
				token |= AxRTriangleType::kEdge1;
			if (index == 2)
				token |= AxRTriangleType::kEdge2;
		}

		namespace ShareCode
		{

			ALPHA_SHARE_FUNC bool RTriangleTest(const AxUChar& token, const AxRTriangleType& target)
			{
				if ((token & target) != target)
					return false;
				return true;
			}
			ALPHA_SHARE_FUNC void PrintRTriangleInfo(const AxUChar& token)
			{
				printf("Vertex: ");
				printf("[%d]",    (token & AxRTriangleType::kVtx0) == AxRTriangleType::kVtx0);
				printf("[%d]",    (token & AxRTriangleType::kVtx1) == AxRTriangleType::kVtx1);
				printf("[%d]",    (token & AxRTriangleType::kVtx2) == AxRTriangleType::kVtx2);
				printf(" Edge: ");
				printf("[%d]",    (token & AxRTriangleType::kEdge0) == AxRTriangleType::kEdge0);
				printf("[%d]",    (token & AxRTriangleType::kEdge1) == AxRTriangleType::kEdge1);
				printf("[%d] \n", (token & AxRTriangleType::kEdge2) == AxRTriangleType::kEdge2);
			}

			ALPHA_SHARE_FUNC void PrintRTriangleInfo(const AxUChar& token, AxUChar primId)
			{
				printf("	  [2]			\n"\
					"       %d				\n"\
					"     /  \\				\n"\
					"    %d   %d			\n"\
					"   /      \\			\n"\
					"  /        \\			\n"\
					" %d ---%d--- %d		\n"\
					" [1]		[0]		<%d>\n",
					(token & AxRTriangleType::kVtx2)  == AxRTriangleType::kVtx2,
					(token & AxRTriangleType::kEdge0) == AxRTriangleType::kEdge0,
					(token & AxRTriangleType::kVtx1)  == AxRTriangleType::kVtx1,
					(token & AxRTriangleType::kEdge1) == AxRTriangleType::kEdge1,
					(token & AxRTriangleType::kVtx0)  == AxRTriangleType::kVtx0,
					(token & AxRTriangleType::kEdge2) == AxRTriangleType::kEdge2,
					primId);

			}
		}
	}
}

#endif // !__AX_RTRIANGLE_H__

