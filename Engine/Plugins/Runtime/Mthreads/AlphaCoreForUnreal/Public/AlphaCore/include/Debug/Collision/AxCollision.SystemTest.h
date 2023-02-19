#ifndef __AX_COLLISION_SYSTEMTEST_H__
#define __AX_COLLISION_SYSTEMTEST_H__

#include <string>
#include "AxDataType.h"

namespace AlphaCore
{
	namespace Collision
	{
		namespace SystemTest
		{
			void SelfCollisionWithBVH(
				std::string path,
				std::string ctxRetPath,
				AxUInt32 maxCtx = 30000);

			namespace CUDA
			{
				void SelfCollisionWithBVH(
					std::string path,
					std::string ctxRetPath,
					AxUInt32 maxCtx = 30000);
			}
		}
	}
}



#endif