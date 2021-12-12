#ifndef __ALPHA_CORE_TEST_SET_H__
#define __ALPHA_CORE_TEST_SET_H__

#include <string>
#include <AxMacro.h>

namespace AlphaCore
{
	namespace TestSet
	{
		
		void ExpolosionCPU(
			std::string inputEmitter,
			std::string outputFrameCode,
			uInt32 maxDivision = 128,
			uInt32 endFrame = 240,
			uInt32 substep = 2,
			uInt32 FPS = 24);

		void ExpolosionGPU(
			std::string inputEmitter,
			std::string outputFrameCode,
			uInt32 maxDivision = 128,
			uInt32 endFrame = 240,
			uInt32 substep = 2,
			uInt32 FPS = 24);
	}
}

#endif