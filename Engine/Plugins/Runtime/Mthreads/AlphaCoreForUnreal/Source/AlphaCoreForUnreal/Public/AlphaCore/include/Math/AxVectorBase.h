#ifndef __ALPHA_CORE_VECTOR_BASE_H__
#define __ALPHA_CORE_VECTOR_BASE_H__

#include "AxMacro.h"
#include "AxDataType.h"
#include <ostream>

template<class T>
struct AxVector3T
{
	T x;
	T y;
	T z;
};


template<class T>
struct AxVector4T
{
	T x;
	T y;
	T z;
	T w;
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
	ALPHA_SHARE_FUNC AxColor4T<T>& operator+=(const AxColor4T<U>& rhs) {
		this->r += rhs.r;
		this->g += rhs.g;
		this->b += rhs.b;
		this->a += rhs.a;
		return *this;
	}

	ALPHA_SHARE_FUNC AxColor4T<T>& operator*=(const AxColor4T<T>& rhs) {
		this->r *= rhs.r;
		this->g *= rhs.g;
		this->b *= rhs.b;
		this->a *= rhs.a;
		return *this;
	}

	template<typename U>
	ALPHA_SHARE_FUNC AxColor4T<T>& operator*=(U rhs) {
		this->r *= rhs;
		this->g *= rhs;
		this->b *= rhs;
		this->a *= rhs;
		return *this;
	}
};

typedef AxVector3T<float>		AxVector3;
typedef AxVector3T<double>		AxVector3D;
typedef AxVector3T<int>			AxVector3I;
typedef AxVector3T<AxUInt32>	AxVector3UI;

typedef AxVector4T<int>			AxVector4I;
typedef AxVector4T<AxUInt32>	AxVector4UI;
typedef AxVector4T<float>		AxVector4;
typedef AxVector4T<double>		AxVector4D;

typedef AxColor4T<Byte>		AxColorRGBA8;
typedef AxColor4T<float>    AxColorRGBA;
typedef AxVector2T<float>		AxVector2;
typedef AxVector2T<double>		AxVector2D;
typedef AxVector2T<int>			AxVector2I;
typedef AxVector2T<AxUInt32>	AxVector2UI;

template<typename T>
inline std::ostream& operator<<(std::ostream& out, AxVector4T<T>& c)
{
	out << c.x << "," << c.y << "," << c.z << "," << c.w;
	return out;
}

template<typename T>
inline std::ostream& operator<<(std::ostream& out, AxVector3T<T>& c)
{
	out << c.x << "," << c.y << "," << c.z;
	return out;
}
template<typename T>
inline std::ostream& operator<<(std::ostream& out, AxVector2T<T>& c)
{
	out << c.x << "," << c.y;
	return out;
}

template<typename T>
inline std::ostream& operator<<(std::ostream& out, const AxVector4T<T>& c)
{
	out << c.x << "," << c.y << "," << c.z << "," << c.w;
	return out;
}

template<typename T>
inline std::ostream& operator<<(std::ostream& out, const AxColor4T<T>& c)
{
	out << c.r << "," << c.g << "," << c.b << "," << c.a;
	return out;
}

template<typename T>
inline std::ostream& operator<<(std::ostream& out, const AxVector3T<T>& c)
{
	out << c.x << "," << c.y << "," << c.z;
	return out;
}

template<typename T>
inline std::ostream& operator<<(std::ostream& out,const AxVector2T<T>& c)
{
	out << c.x << "," << c.y;
	return out;
}

namespace AlphaCore
{
	template<>
	inline AxUInt32 TypeVecSize<AxVector2>() { return 2; }
	template<>
	inline AxUInt32 TypeVecSize<AxVector3>() { return 3; }
	template<>
	inline AxUInt32 TypeVecSize<AxVector4>() { return 4; }
	template<>
	inline AxUInt32 TypeVecSize<AxVector2UI>() { return 2; }
	template<>
	inline AxUInt32 TypeVecSize<AxVector3UI>() { return 3; }
	template<>
	inline AxUInt32 TypeVecSize<AxVector3I>() { return 3; }
	template<>
	inline AxUInt32 TypeVecSize<AxVector2I>() { return 2; }

	//TypeID
	template<>
	inline AxDataType TypeID<AxVector2>() { return AxDataType::kFP32; }
	template<>
	inline AxDataType TypeID<AxVector3>() { return AxDataType::kFP32; }
	template<>
	inline AxDataType TypeID<AxVector4>() { return AxDataType::kFP32; }
	//------------------
	template<>
	inline AxDataType TypeID<AxVector2D>() { return AxDataType::kFP64; }
	template<>
	inline AxDataType TypeID<AxVector3D>() { return AxDataType::kFP64; }
	template<>
	inline AxDataType TypeID<AxVector4D>() { return AxDataType::kFP64; }
	//------------------
	template<>
	inline AxDataType TypeID<AxVector2UI>() { return AxDataType::kUInt32; }
	template<>
	inline AxDataType TypeID<AxVector3UI>() { return AxDataType::kUInt32; }
	template<>
	inline AxDataType TypeID<AxVector4UI>() { return AxDataType::kUInt32; }
	//------------------
	template<>
	inline AxDataType TypeID<AxVector2I>() { return AxDataType::kInt32; }
	template<>
	inline AxDataType TypeID<AxVector3I>() { return AxDataType::kInt32; }
	template<>
	inline AxDataType TypeID<AxVector4I>() { return AxDataType::kInt32; }

	//TypeName
	template<>
	inline AxString TypeName<AxVector2>() { return "AxVector2"; }
	template<>
	inline AxString TypeName<AxVector3>() { return "AxVector3"; }
	template<>
	inline AxString TypeName<AxVector4>() { return "AxVector4"; }
	//------------------
	template<>
	inline AxString TypeName<AxVector2D>() { return "AxVector2D"; }
	template<>
	inline AxString TypeName<AxVector3D>() { return "AxVector3D"; }
	template<>
	inline AxString TypeName<AxVector4D>() { return "AxVector4D"; }
	//------------------
	template<>
	inline AxString TypeName<AxVector2UI>() { return "AxVector2UI"; }
	template<>
	inline AxString TypeName<AxVector3UI>() { return "AxVector3UI"; }
	template<>
	inline AxString TypeName<AxVector4UI>() { return "AxVector4UI"; }
	//------------------
	template<>
	inline AxString TypeName<AxVector2I>() { return "AxVector2I"; }
	template<>
	inline AxString TypeName<AxVector3I>() { return "AxVector3I"; }
	template<>
	inline AxString TypeName<AxVector4I>() { return "AxVector4I"; }
}

template<typename T, typename U>
ALPHA_SHARE_FUNC AxColor4T<T> operator*(const AxColor4T<T>& a, U b)
{
	return AxColor4T<T>{
		a.r * b, a.g * b, a.b * b, a.a * b
	};
}

template<typename T>
ALPHA_SHARE_FUNC AxColor4T<T> operator*(const AxColor4T<T>& a, const AxColor4T<T>& b)
{
	return AxColor4T<T>{
		a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a
	};
}



#endif
