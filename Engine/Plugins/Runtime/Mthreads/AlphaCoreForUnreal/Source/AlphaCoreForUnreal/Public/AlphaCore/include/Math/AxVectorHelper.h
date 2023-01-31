#ifndef __ALPHA_CORE_VECTOR_HELPER_H__
#define __ALPHA_CORE_VECTOR_HELPER_H__

#include "AxVectorBase.h"
#include <math.h>
#include <stdio.h>

#define AX_VOXEL_ID3_CONDITION(id3, i, j, k) if (id3.x == i && id3.y == j && id3.z == k)

ALPHA_SHARE_FUNC void PrintInfo(const char *head, const AxFp32 &v)
{
	printf("%s : %f  \n", head, v);
}
ALPHA_SHARE_FUNC void PrintInfo(const char *head, const AxInt32 &v)
{
	printf("%s : %d  \n", head, v);
}
ALPHA_SHARE_FUNC void PrintInfo(const char *head, const AxUInt32 &v)
{
	printf("%s : %d  \n", head, v);
}

ALPHA_SHARE_FUNC void PrintInfo(const char *head, AxInt32 i, AxInt32 j, AxInt32 k, const AxFp32 &v)
{
	printf("%s :<%d,%d,%d> %f  \n", head, i, j, k, v);
}

ALPHA_SHARE_FUNC void PrintInfo(const char *head, const AxVector3I &id3, const AxFp32 &v)
{
	printf("%s :<%d,%d,%d> %f  \n", head, id3.x, id3.y, id3.z, v);
}

ALPHA_SHARE_FUNC void PrintInfo(const char *head, const AxVector3I &id3,
								const AxFp32 &v0,
								const AxFp32 &v1,
								const AxFp32 &v2,
								const AxFp32 &v3)
{
	printf("%s :<%d,%d,%d> %f %f %f %f \n", head, id3.x, id3.y, id3.z, v0, v1, v2, v3);
}

ALPHA_SHARE_FUNC void PrintInfo(const char *head, const AxVector3 &v)
{
	printf("%s : [%f,%f,%f] \n", head, v.x, v.y, v.z);
}

ALPHA_SHARE_FUNC void PrintInfo(const char *head, AxInt32 i, AxInt32 j, AxInt32 k, const AxVector3 &v)
{
	printf("%s :<%d,%d,%d> [%f,%f,%f] \n", head, i, j, k, v.x, v.y, v.z);
}

ALPHA_SHARE_FUNC void PrintInfo_OBJFormat(const AxVector3 &v)
{
	printf("v %f %f %f\n", v.x, v.y, v.z);
}

ALPHA_SHARE_FUNC void PrintInfo(const char *head, const AxVector2UI &v)
{
	printf("%s : [%d,%d] \n", head, v.x, v.y);
}
ALPHA_SHARE_FUNC void PrintInfo(const char *head, const AxVector3UI &v)
{
	printf("%s : [%d,%d,%d] \n", head, v.x, v.y, v.z);
}
ALPHA_SHARE_FUNC void PrintInfo(const char *head, const AxVector3I &v)
{
	printf("%s : [%d,%d,%d] \n", head, v.x, v.y, v.z);
}
ALPHA_SHARE_FUNC void PrintInfo(const char *head, const AxVector4 &v)
{
	printf("%s : [%f,%f,%f,%f] \n", head, v.x, v.y, v.z, v.w);
}

ALPHA_SHARE_FUNC float Dot(const AxVector3 &a, const AxVector3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

ALPHA_SHARE_FUNC float Length(const AxVector3 &v)
{
	return sqrtf(Dot(v, v));
}

//-------------------------------------------------------------
//
//		Vector 3 float / scalar Type
//
//-------------------------------------------------------------

ALPHA_SHARE_FUNC AxVector3 MakeVector3(float x, float y, float z)
{
	AxVector3 t;
	t.x = x;
	t.y = y;
	t.z = z;
	return t;
}

ALPHA_SHARE_FUNC AxVector3 MakeVector3(float s)
{
	return MakeVector3(s, s, s);
}

template <class T>
ALPHA_SHARE_FUNC AxVector3T<T> MakeVector3T(T x, T y, T z)
{
	AxVector3T<T> ret;
	ret.x = x;
	ret.y = y;
	ret.z = z;
	return ret;
}

ALPHA_SHARE_FUNC AxVector3 MakeVector3()
{
	return MakeVector3(0.0f, 0.0f, 0.0f);
}

ALPHA_SHARE_FUNC void operator+=(AxVector3 &a, AxVector3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

ALPHA_SHARE_FUNC void operator+=(AxVector3 &a, AxFp32 b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

template <class T>
ALPHA_SHARE_FUNC AxVector3T<T> operator/(T b, const AxVector3T<T> &a)
{
	return MakeVector3T(b / a.x, b / a.y, b / a.z);
}

ALPHA_SHARE_FUNC void operator-=(AxVector3 &a, AxFp32 b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

ALPHA_SHARE_FUNC void operator*=(AxVector3 &a, AxVector3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}

ALPHA_SHARE_FUNC AxVector3 operator*(AxVector3 a, AxVector3 b)
{
	return MakeVector3(a.x * b.x, a.y * b.y, a.z * b.z);
}

ALPHA_SHARE_FUNC AxVector3 operator*(AxVector3 a, AxVector3UI b)
{
	return MakeVector3(a.x * b.x, a.y * b.y, a.z * b.z);
}

ALPHA_SHARE_FUNC AxVector3 operator+(const AxVector3 &a, const AxVector3 &b)
{
	return MakeVector3(a.x + b.x, a.y + b.y, a.z + b.z);
}

ALPHA_SHARE_FUNC AxVector3 operator/(AxVector3 a, AxVector3 b)
{
	return MakeVector3(a.x / b.x, a.y / b.y, a.z / b.z);
}

ALPHA_SHARE_FUNC AxVector3 operator*(const AxVector3 &a, float b)
{
	return MakeVector3(a.x * b, a.y * b, a.z * b);
}

ALPHA_SHARE_FUNC AxVector3 operator*(float a, const AxVector3 &b)
{
	return MakeVector3(b.x * a, b.y * a, b.z * a);
}

ALPHA_SHARE_FUNC void operator*=(AxVector3 &a, float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

ALPHA_SHARE_FUNC AxVector3 operator-(const AxVector3 &a, const AxVector3 &b)
{
	return MakeVector3(a.x - b.x, a.y - b.y, a.z - b.z);
}

ALPHA_SHARE_FUNC void operator-=(AxVector3 &a, const AxVector3 &b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
ALPHA_SHARE_FUNC AxVector3 operator-(const AxVector3 &a, const float &b)
{
	return MakeVector3(a.x - b, a.y - b, a.z - b);
}

ALPHA_SHARE_FUNC void operator/=(AxVector3 &a, float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}

ALPHA_SHARE_FUNC AxVector3 operator/(const AxVector3 &a, const AxVector3UI &b)
{
	return MakeVector3(a.x / (float)b.x, a.y / (float)b.y, a.z / (float)b.z);
}

ALPHA_SHARE_FUNC AxVector3 operator/(const AxVector3 &a, const float &b)
{
	return MakeVector3(a.x / b, a.y / b, a.z / b);
}

ALPHA_SHARE_FUNC AxVector3 Normalize(AxVector3 &a)
{
	AxFp32 invLen = 1.0f / (Length(a) + 1e-10f);
	a *= invLen;
	return a;
}

ALPHA_SHARE_FUNC AxVector3 Normalized(AxVector3 a)
{
	AxFp32 invLen = 1.0f / (Length(a) + 1e-10f);
	a *= invLen;
	return a;
}

template <typename T>
ALPHA_SHARE_FUNC void AssignV3(AxVector3T<T> &a, T v)
{
	a.x = v;
	a.y = v;
	a.z = v;
}

ALPHA_SHARE_FUNC AxVector3 Cross(AxVector3 a, AxVector3 b)
{
	return MakeVector3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

ALPHA_SHARE_FUNC void operator/=(AxVector3 &a, const AxVector3UI &b)
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
ALPHA_SHARE_FUNC bool operator==(AxVector2UI &a, const AxVector2UI &b)
{
	return a.x == b.x && a.y == b.y;
}

ALPHA_SHARE_FUNC bool operator!=(AxVector2UI &a, const AxVector2UI &b)
{
	return a.x != b.x || a.y != b.y;
}

//-------------------------------------------------------------
//
//		Vector 3 unsigned int
//
//-------------------------------------------------------------
ALPHA_SHARE_FUNC AxVector3UI MakeVector3UI(AxUInt32 x, AxUInt32 y, AxUInt32 z)
{
	AxVector3UI t;
	t.x = x;
	t.y = y;
	t.z = z;
	return t;
}

ALPHA_SHARE_FUNC AxVector3UI MakeVector3UI(AxUInt32 s)
{
	return MakeVector3UI(s, s, s);
}

ALPHA_SHARE_FUNC AxVector3UI MakeVector3UI()
{
	return MakeVector3UI(0, 0, 0);
}

ALPHA_SHARE_FUNC AxVector3I MakeVector3I(int x, int y, int z)
{
	AxVector3I t;
	t.x = x;
	t.y = y;
	t.z = z;
	return t;
}

ALPHA_SHARE_FUNC AxVector3I MakeVector3I(int s)
{
	return MakeVector3I(s, s, s);
}

ALPHA_SHARE_FUNC AxVector3I MakeVector3I()
{
	return MakeVector3I(0, 0, 0);
}

ALPHA_SHARE_FUNC AxVector3 operator*(const AxVector3UI &a, float b)
{
	return MakeVector3((float)a.x * b, (float)a.y * b, (float)a.z * b);
}

ALPHA_SHARE_FUNC AxVector3 operator*(const AxVector3UI &a, const AxVector3 &b)
{
	return MakeVector3((float)a.x * b.x, (float)a.y * b.y, (float)a.z * b.z);
}

ALPHA_SHARE_FUNC AxVector3UI operator*(const AxVector3UI &a, AxInt32 b)
{
	return MakeVector3UI((float)a.x * b, (float)a.y * b, (float)a.z * b);
}

ALPHA_SHARE_FUNC AxVector3 operator+(const AxVector3UI &a, float b)
{
	return MakeVector3((float)a.x + b, (float)a.y + b, (float)a.z + b);
}

ALPHA_SHARE_FUNC AxVector3UI operator+(const AxVector3UI &a, const AxVector3UI &b)
{
	return MakeVector3UI(a.x + b.x, a.y + b.y, a.z + b.z);
}

ALPHA_SHARE_FUNC void operator+=(AxVector3UI &a, const AxVector3UI &b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

ALPHA_SHARE_FUNC void operator-=(AxVector3UI &a, const AxVector3UI &b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

ALPHA_SHARE_FUNC void operator/=(AxVector3UI &a, const AxVector3UI &b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
}

ALPHA_SHARE_FUNC void operator/=(AxVector3UI &a, AxInt32 b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}

ALPHA_SHARE_FUNC AxVector3 operator-(const AxVector3UI &a, float b)
{
	return MakeVector3((float)a.x - b, (float)a.y - b, (float)a.z - b);
}

ALPHA_SHARE_FUNC AxVector3I operator-(const AxVector3I &a, const AxVector3I &b)
{
	return MakeVector3I(a.x - b.x, a.y - b.y, a.z - b.z);
}

ALPHA_SHARE_FUNC AxVector3I operator+(const AxVector3I &a, const AxVector3I &b)
{
	return MakeVector3I(a.x + b.x, a.y + b.y, a.z + b.z);
}

ALPHA_SHARE_FUNC AxVector3I operator*(const AxVector3I &a, const AxVector3I &b)
{
	return MakeVector3I(a.x * b.x, a.y * b.y, a.z * b.z);
}

ALPHA_SHARE_FUNC AxVector4 MakeVector4()
{
	AxVector4 t;
	t.x = 0;
	t.y = 0;
	t.z = 0;
	t.w = 0;
	return t;
}

ALPHA_SHARE_FUNC AxVector4 MakeVector4(AxFp32 x, AxFp32 y, AxFp32 z, AxFp32 w)
{
	AxVector4 t;
	t.x = x;
	t.y = y;
	t.z = z;
	t.w = w;
	return t;
}

ALPHA_SHARE_FUNC bool IsEqual(const AxVector3I &a, const AxVector3I &b)
{
	return (a.x == b.x && b.y == b.y && a.z == a.z);
}

ALPHA_SHARE_FUNC bool IsEqual(const AxVector3UI &a, const AxVector3UI &b)
{
	return (a.x == b.x && b.y == b.y && a.z == a.z);
}

template <typename T>
ALPHA_SHARE_FUNC AxVector4 MakeVector4(const AxVector3T<T> &vec, T w)
{
	AxVector4 t;
	t.x = vec.x;
	t.y = vec.y;
	t.z = vec.z;
	t.w = w;
	return t;
}

ALPHA_SHARE_FUNC void operator*=(AxVector4 &a, AxFp32 b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

ALPHA_SHARE_FUNC void operator*=(AxVector4& a, const AxVector4& b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}

ALPHA_SHARE_FUNC void operator*=(AxVector4& a, const AxVector3& b)
{
	a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

ALPHA_SHARE_FUNC void operator-=(AxVector4& a, const AxVector3& b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

ALPHA_SHARE_FUNC void operator-=(AxVector4& a, const AxVector4& b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

ALPHA_SHARE_FUNC AxVector4 operator-=(const AxVector4 &a, const AxVector3 &b)
{
	return MakeVector4(a.x - b.x, a.y - b.y, a.z - b.z, a.w);
}

ALPHA_SHARE_FUNC AxVector3 Neg(const AxVector3 &vec3)
{
	return MakeVector3(-vec3.x, -vec3.y, -vec3.z);
}

ALPHA_SHARE_FUNC AxVector3 operator+(const AxVector3 &a, AxFp32 b)
{
	return MakeVector3(a.x + b, a.y + b, a.z + b);
}

ALPHA_SHARE_FUNC AxVector2UI MakeVector2UI(AxUInt32 x, AxUInt32 y)
{
	AxVector2UI t;
	t.x = x;
	t.y = y;
	return t;
}

ALPHA_SHARE_FUNC AxVector2UI MakeVector2UI(AxUInt32 x)
{
	return MakeVector2UI(x, x);
}

ALPHA_SHARE_FUNC AxVector2UI MakeVector2UI()
{
	return MakeVector2UI(0);
}

template <typename T>
ALPHA_SHARE_FUNC AxVector2T<T> MakeVector2(T x = 0.0f, T y = 0.0f)
{
	AxVector2T<T> t;
	t.x = x;
	t.y = y;
	return t;
}

ALPHA_SHARE_FUNC AxVector2I MakeVector2I(AxInt32 x = 0, AxInt32 y = 0)
{
	AxVector2I t;
	t.x = x;
	t.y = y;
	return t;
}

ALPHA_SHARE_FUNC AxVector2 operator/(const AxVector2 &a, const AxVector2UI &b)
{
	return MakeVector2(a.x / (float)b.x, a.y / (float)b.y);
}

ALPHA_SHARE_FUNC void operator+=(AxColorRGBA8 &a, AxColorRGBA8 b)
{
	a.r += b.r;
	a.g += b.g;
	a.b += b.b;
	a.a += b.a;
}

template <typename T>
ALPHA_SHARE_FUNC void operator+=(AxVector4T<T> &a, const AxVector3T<T> &b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

template <typename T>
ALPHA_SHARE_FUNC void operator+=(AxVector4T<T> &a, const AxVector4T<T> &b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}

template <typename T>
ALPHA_SHARE_FUNC void operator/=(AxVector4T<T> &a, const AxVector4T<T> &b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
}

// TODO huiyang
// template<typename T>
// ALPHA_SHARE_FUNC AxVector4T<T> operator*(AxVector4T<T> a, AxVector4T<T> b)
//{
//	return MakeVector4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
// }

template <typename T>
ALPHA_SHARE_FUNC void operator/=(AxVector4T<T> &a, const T &b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
	a.w /= b;
}

ALPHA_SHARE_FUNC AxVector2UI ThreadBlockInfo(AxUInt32 blockSize, uInt64 numThreads)
{
	return MakeVector2UI(int(numThreads / blockSize) + 1,
						 blockSize > numThreads ? numThreads : blockSize);
}

ALPHA_SHARE_FUNC AxColorRGBA8 MakeColorRGBA8()
{
	AxColorRGBA8 t;
	t.r = 0;
	t.g = 0;
	t.b = 0;
	t.a = 0;
	return t;
}

ALPHA_SHARE_FUNC AxColorRGBA8 MakeColorRGBA8(Byte r, Byte g, Byte b, Byte a)
{
	AxColorRGBA8 t;
	t.r = r;
	t.g = g;
	t.b = b;
	t.a = a;
	return t;
}

ALPHA_SHARE_FUNC AxColorRGBA MakeColorRGBA(float r, float g, float b, float a)
{
	AxColorRGBA t;
	t.r = r;
	t.g = g;
	t.b = b;
	t.a = a;
	return t;
}
#include <ostream>

inline std::ostream &operator<<(std::ostream &os, AxColorRGBA8 &rgba)
{
	os << "rgba:" << (short)rgba.r << "," << (short)rgba.g << "," << (short)rgba.b << "," << (short)rgba.a;
	return os;
}

inline bool operator==(AxColorRGBA8 &rgba, float scalar)
{
	if (rgba.r == scalar && rgba.g == scalar && rgba.b == scalar && rgba.a == scalar)
		return true;
	return false;
}

#endif