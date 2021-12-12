#ifndef __ALPHA_CORE_VECTOR_HELPER_H__
#define __ALPHA_CORE_VECTOR_HELPER_H__

#include "AxVectorBase.h"
#include <math.h>
#include <stdio.h>
#include <Math/AxMath101.h>

ALPHA_KERNEL_FUNC void PrintInfo(const char* head,const AxVector3& v)
{
	printf("%s : [%f,%f,%f] \n", head, v.x, v.y, v.z);
}

ALPHA_KERNEL_FUNC void PrintInfo_OBJFormat(const AxVector3& v)
{
	printf("v %f %f %f\n", v.x, v.y, v.z);
}

ALPHA_KERNEL_FUNC void PrintInfo(const char* head, const AxVector2UI& v)
{
	printf("%s : [%d,%d] \n", head, v.x, v.y);
}
ALPHA_KERNEL_FUNC void PrintInfo(const char* head, const AxVector3UI& v)
{
	printf("%s : [%d,%d,%d] \n", head, v.x, v.y,v.z);
}

ALPHA_KERNEL_FUNC float Dot(const AxVector3& a, const AxVector3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

ALPHA_KERNEL_FUNC float Length(const AxVector3& v)
{
	return sqrtf(Dot(v, v));
}

//-------------------------------------------------------------
//
//		Vector 3 float / scalar Type
//
//-------------------------------------------------------------

ALPHA_KERNEL_FUNC AxVector3 MakeVector3(float x,float y,float z)
{
	AxVector3 t; t.x = x; t.y = y; t.z = z; return t;
}

ALPHA_KERNEL_FUNC AxVector3 MakeVector3(float s)
{
	return MakeVector3(s, s, s);
}

ALPHA_KERNEL_FUNC AxVector3 MakeVector3()
{
	return MakeVector3(0, 0, 0);
}

ALPHA_KERNEL_FUNC void operator+=(AxVector3 &a, AxVector3 b)
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}

ALPHA_KERNEL_FUNC void operator*=(AxVector3 &a, AxVector3 b)
{
	a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

ALPHA_KERNEL_FUNC AxVector3 operator*(AxVector3 a, AxVector3 b)
{
	return MakeVector3(a.x * b.x, a.y * b.y, a.z * b.z);
}

ALPHA_KERNEL_FUNC AxVector3 operator*(AxVector3 a, AxVector3UI b)
{
	return MakeVector3(a.x * b.x, a.y * b.y, a.z * b.z);
}

ALPHA_KERNEL_FUNC AxVector3 operator+(AxVector3 a, AxVector3 b)
{
	return MakeVector3(a.x + b.x, a.y + b.y, a.z + b.z);
}

ALPHA_KERNEL_FUNC AxVector3 operator/(AxVector3 a, AxVector3 b)
{
	return MakeVector3(a.x / b.x, a.y / b.y, a.z / b.z);
}

ALPHA_KERNEL_FUNC AxVector3 operator*(const AxVector3& a, float b)
{
	return MakeVector3(a.x * b, a.y * b, a.z * b);
}

ALPHA_KERNEL_FUNC AxVector3 operator*(float a, const AxVector3& b)
{
	return MakeVector3(b.x * a, b.y * a, b.z * a);
}

ALPHA_KERNEL_FUNC void operator*=(AxVector3 &a, float b)
{
	a.x *= b; a.y *= b; a.z *= b;
}

ALPHA_KERNEL_FUNC AxVector3 operator-(const AxVector3& a, const AxVector3& b)
{
	return MakeVector3(a.x - b.x, a.y - b.y, a.z - b.z);
}

ALPHA_KERNEL_FUNC void operator-=(AxVector3 &a, const  AxVector3& b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
ALPHA_KERNEL_FUNC AxVector3 operator-(const AxVector3 &a, const float& b)
{
	return MakeVector3(a.x - b, a.y - b, a.z - b);
}


ALPHA_KERNEL_FUNC void operator/=(AxVector3 &a, float b)
{
	a.x /= b; a.y /= b; a.z /= b;
}

ALPHA_KERNEL_FUNC AxVector3 operator/(const AxVector3 &a,const AxVector3UI& b)
{
 	return MakeVector3(a.x / (float)b.x,a.y / (float)b.y, a.z / (float)b.z);
}

ALPHA_KERNEL_FUNC AxVector3 operator/(const AxVector3 &a, const float& b)
{
	return MakeVector3(a.x / b,a.y / b,a.z / b);
}

ALPHA_KERNEL_FUNC AxVector3 Normalize(AxVector3 &a)
{
	float invLen = 1.0f / (Length(a) + 1e-8);
	a *= invLen;
	return a;
}

ALPHA_KERNEL_FUNC AxVector3 Cross(AxVector3 a, AxVector3 b)
{
	return MakeVector3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);

}

ALPHA_KERNEL_FUNC void operator/=(AxVector3 &a, const AxVector3UI& b)
{
	a.x /= (float)b.x; 
	a.y /= (float)b.y; 
	a.z /= (float)b.z;
}
//-------------------------------------------------------------
//
//		Vector 2 unsigned int 
//
//-------------------------------------------------------------
ALPHA_KERNEL_FUNC bool operator ==(AxVector2UI& a, const AxVector2UI& b)
{
	return a.x == b.x && a.y == b.y;
}

ALPHA_KERNEL_FUNC bool operator !=(AxVector2UI& a, const AxVector2UI& b)
{
	return a.x != b.x || a.y != b.y;
}

//-------------------------------------------------------------
//
//		Vector 3 unsigned int 
//
//-------------------------------------------------------------
ALPHA_KERNEL_FUNC AxVector3UI MakeVector3UI(uInt32 x, uInt32 y, uInt32 z)
{
	AxVector3UI t; t.x = x; t.y = y; t.z = z; return t;
}

ALPHA_KERNEL_FUNC AxVector3UI MakeVector3UI(uInt32 s)
{
	return MakeVector3UI(s, s, s);
}

ALPHA_KERNEL_FUNC AxVector3UI MakeVector3UI()
{
	return MakeVector3UI(0, 0, 0);
}

ALPHA_KERNEL_FUNC AxVector3I MakeVector3I(int x, int y, int z)
{
	AxVector3I t; t.x = x; t.y = y; t.z = z; return t;
}

ALPHA_KERNEL_FUNC AxVector3I MakeVector3I(int s)
{
	return MakeVector3I(s, s, s);
}

ALPHA_KERNEL_FUNC AxVector3I MakeVector3I()
{
	return MakeVector3I(0, 0, 0);
}


ALPHA_KERNEL_FUNC AxVector3 operator*(const AxVector3UI& a, float b)
{
	return MakeVector3((float)a.x * b, (float)a.y * b, (float)a.z * b);
}

ALPHA_KERNEL_FUNC AxVector3 operator+(const AxVector3UI& a, float b)
{
	return MakeVector3((float)a.x + b, (float)a.y + b, (float)a.z + b);
}

ALPHA_KERNEL_FUNC AxVector3 operator-(const AxVector3UI& a, float b)
{
	return MakeVector3((float)a.x - b, (float)a.y - b, (float)a.z - b);
}

ALPHA_KERNEL_FUNC AxVector2UI MakeVector2UI(uInt32 x, uInt32 y)
{
	AxVector2UI t; t.x = x; t.y = y; return t;
}

ALPHA_KERNEL_FUNC AxVector2UI MakeVector2UI(uInt32 x)
{
	return MakeVector2UI(x,x);
}

ALPHA_KERNEL_FUNC AxVector2UI MakeVector2UI()
{
	return MakeVector2UI(0);
}

ALPHA_KERNEL_FUNC AxVector2 MakeVector2(float x, float y)
{
	AxVector2 t; t.x = x; t.y = y; return t;
}

ALPHA_KERNEL_FUNC AxVector2 operator/(const AxVector2 &a, const AxVector2UI& b)
{
	return MakeVector2(a.x / (float)b.x, a.y / (float)b.y);
}


ALPHA_KERNEL_FUNC void operator+=(AxColorRGBA8 &a, AxColorRGBA8 b)
{
	a.r += b.r;
	a.g += b.g;
	a.b += b.b;
	a.a += b.a;
}

ALPHA_KERNEL_FUNC AxVector2UI ThreadBlockInfo(uInt32 blockSize, uInt64 numThreads)
{
	return MakeVector2UI(int(numThreads / blockSize) + 1, 
		blockSize > numThreads ? numThreads : blockSize);
}

ALPHA_KERNEL_FUNC AxColorRGBA8 MakeColorRGBA8()
{
	AxColorRGBA8 t; t.r = 0; t.g = 0; t.b = 0; t.a = 0; return t;
}

ALPHA_KERNEL_FUNC AxColorRGBA8 MakeColorRGBA8(Byte r, Byte g, Byte b, Byte a)
{
	AxColorRGBA8 t; t.r = r; t.g = g; t.b = b; t.a = a; return t;
}

namespace AlphaCore
{
	namespace Math
	{
		ALPHA_KERNEL_FUNC void ClampMaxmin(AxVector3& vec,float max,float min)
		{
			if (vec.x > max) { vec.x = max; }
			if (vec.y > max) { vec.y = max; }
			if (vec.z > max) { vec.z = max; }
			if (vec.x < min) { vec.x = min; }
			if (vec.y < min) { vec.y = min; }
			if (vec.z < min) { vec.z = min; }
		}

		ALPHA_KERNEL_FUNC int Clamp(int val, int min, int max)
		{
			if (val > max) { val = max; }
			if (val < min) { val = min; }
			return val;
		}

		template<typename T>
		ALPHA_KERNEL_FUNC T Min(T a, T b)
		{
			return a < b ? a : b;
		}

		template<typename T>
		ALPHA_KERNEL_FUNC T Max(T a, T b)
		{
			return a > b ? a : b;
		}

		ALPHA_KERNEL_FUNC float MinF(float a, float b)
		{
			return a < b ? a : b;
		}

		ALPHA_KERNEL_FUNC float MaxF(float a, float b)
		{
			return a > b ? a : b;
		}

		ALPHA_KERNEL_FUNC int MaxI(int a, int b)
		{
			return a > b ? a : b;
		}

		ALPHA_KERNEL_FUNC int Min(int a, int b)
		{
			return a < b ? a : b;
		}

		ALPHA_KERNEL_FUNC float RSqrtF(float x)
		{
			return 1.0f / sqrtf(x);
		}

		ALPHA_KERNEL_FUNC AxVector3 Min(AxVector3 a, AxVector3 b)
		{
			return MakeVector3(MinF(a.x, b.x), MinF(a.y, b.y), MinF(a.z, b.z));
		}

		ALPHA_KERNEL_FUNC AxVector3 Max(AxVector3 a, AxVector3 b)
		{
			return MakeVector3(MaxF(a.x, b.x), MaxF(a.y, b.y), MaxF(a.z, b.z));
		}

		template<typename T>
		ALPHA_KERNEL_FUNC AxColor4T<T> Clamp(const AxColor4T<T>& a, T bottom, T up)
		{
			return AxColor4T<T>{
				Min(Max(a.r, bottom), up), 
				Min(Max(a.g, bottom), up),
				Min(Max(a.b, bottom), up),
				Min(Max(a.a, bottom), up)
			};
		}

		ALPHA_KERNEL_FUNC AxVector3 LerpV3(const AxVector3& a, const AxVector3& b,float t)
		{
			return MakeVector3(
				AlphaCore::Math::LerpF32(a.x, b.x, t),
				AlphaCore::Math::LerpF32(a.y, b.y, t),
				AlphaCore::Math::LerpF32(a.z, b.z, t));
		}
	}
}

#endif