#ifndef __ALPHA_CORE_VECTOR_BASE_H__
#define __ALPHA_CORE_VECTOR_BASE_H__

#include <AxMacro.h>
#include <ostream>

template<class T>
struct AxVector3T
{
	T x;
	T y;
	T z;
};

template<class T>
struct AxVector2T
{
	T x;
	T y;
};

template<class T>
struct AxColor4T
{
	T r;
	T g;
	T b;
	T a;

	template<typename U>
	ALPHA_KERNEL_FUNC AxColor4T<T>& operator+=(const AxColor4T<U>& rhs) {
		this->r += rhs.r;
		this->g += rhs.g;
		this->b += rhs.b;
		this->a += rhs.a;
		return *this;
	}

	ALPHA_KERNEL_FUNC AxColor4T<T>& operator*=(const AxColor4T<T>& rhs) {
		this->r *= rhs.r;
		this->g *= rhs.g;
		this->b *= rhs.b;
		this->a *= rhs.a;
		return *this;
	}

	template<typename U>
	ALPHA_KERNEL_FUNC AxColor4T<T>& operator*=(U rhs) {
		this->r *= rhs;
		this->g *= rhs;
		this->b *= rhs;
		this->a *= rhs;
		return *this;
	}
};

typedef AxVector3T<float>	AxVector3;
typedef AxVector3T<double>	AxVector3D;
typedef AxVector3T<int>		AxVector3I;
typedef AxVector3T<uInt32>	AxVector3UI;

typedef AxVector2T<float>	AxVector2;
typedef AxVector2T<double>	AxVector2D;
typedef AxVector2T<int>		AxVector2I;
typedef AxVector2T<uInt32>	AxVector2UI;

typedef AxColor4T<Byte>		AxColorRGBA8;
typedef AxColor4T<float>    AxColorRGBA;

template<typename T, typename U>
ALPHA_KERNEL_FUNC AxColor4T<T> operator*(const AxColor4T<T>& a, U b)
{
	return AxColor4T<T>{
		a.r * b, a.g * b, a.b * b, a.a * b
	};
}

template<typename T>
ALPHA_KERNEL_FUNC AxColor4T<T> operator*(const AxColor4T<T>& a, const AxColor4T<T>& b)
{
	return AxColor4T<T>{
		a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a
	};
}

#endif
