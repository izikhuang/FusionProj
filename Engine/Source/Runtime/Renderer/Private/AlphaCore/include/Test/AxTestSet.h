#ifndef __ALPHA_CORE_TEST_SET_H__
#define __ALPHA_CORE_TEST_SET_H__

#include <string>
#include <AxMacro.h>
#include <AxDataType.h>
namespace AlphaCore
{
	namespace TestSet
	{
		
		void ExpolosionCPU(
			std::string inputEmitter,
			std::string outputFrameCode,
			AxUInt32 maxDivision = 128,
			AxUInt32 endFrame = 240,
			AxUInt32 substep = 2,
			AxUInt32 FPS = 24);

		void ExpolosionGPU(
			std::string inputEmitter,
			std::string outputFrameCode,
			AxUInt32 maxDivision = 128,
			AxUInt32 endFrame = 240,
			AxUInt32 substep = 2,
			AxUInt32 FPS = 24);
	}
}

#endif