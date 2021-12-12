#ifndef __ALPHA_CORE_MATH_101_H__
#define __ALPHA_CORE_MATH_101_H__

#include <AxMacro.h>

//#define Clamp(val,max,min) val > max ? max : (val < min ? min : val)
namespace AlphaCore
{
	namespace Math
	{
		ALPHA_KERNEL_FUNC float LerpF32(float a, float b, float t)
		{
			return a * (1.0f - t) + b * t;
		}

		ALPHA_KERNEL_FUNC float ClampF32(float val, float max, float min)
		{
			return val > max ? max : (val < min ? min : val);
		}

		ALPHA_KERNEL_FUNC float DegreesToRadians(float degree)
		{
			return degree * ALPHA_DEGREE_TO_RADIUS;
		}
	}
}

#endif