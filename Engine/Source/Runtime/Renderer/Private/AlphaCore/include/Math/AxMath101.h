#ifndef __ALPHA_CORE_MATH_101_H__
#define __ALPHA_CORE_MATH_101_H__

#include <AxMacro.h>
#include <AxDataType.h>
#include <Math/AxVectorHelper.h>
#include <Math/AxMat.ShareCode.h>

//#define Clamp(val,max,min) val > max ? max : (val < min ? min : val)
namespace AlphaCore
{
	namespace Math
	{
		template<typename T>
		ALPHA_SHARE_FUNC T Fade(T t)
		{
			return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); // 6t^5 - 15t^4 + 10t^3
		}

		template<typename T>
		ALPHA_SHARE_FUNC T Lerp(T a, T b, T t)
		{
			return a * (1.0f - t) + b * t;
		}

		ALPHA_SHARE_FUNC AxFp32 LerpF32(AxFp32 a, AxFp32 b, AxFp32 t)
		{
			return a * (1.0f - t) + b * t;
		}

		ALPHA_SHARE_FUNC float ClampF32(float val, float min, float max)
		{
			return val > max ? max : (val < min ? min : val);
		}

		ALPHA_SHARE_FUNC float RemapF32(float val, float inputMin, float inputMax, float outputMin, float outputMax)
		{
			return (val - inputMin) / (inputMax - inputMin) * (outputMax - outputMin) + outputMin;
		}

		ALPHA_SHARE_FUNC float DegreesToRadians(float degree)
		{
			return degree * ALPHA_DEGREE_TO_RADIUS;
		}

		ALPHA_SHARE_FUNC bool NearVal(float val, float dst) {
			return  val - dst > -1e-6 && val - dst < 1e-6;
		}

		ALPHA_SHARE_FUNC bool NearZero(float val) {
			return NearVal(val, 0.f);
		}
		template<typename T>
		ALPHA_SHARE_FUNC T Stp(const AxVector3T<T>& u,const AxVector3T<T>& v,const AxVector3T<T>& w)
		{
			return Dot(u, Cross(v, w));
		}

		template<typename T>
		ALPHA_SHARE_FUNC AxVector3T<T> Lerp(AxVector3T<T> a, AxVector3T<T> b, T t)
		{
			return a + t * (b - a);
		}

		/*
		template<typename T>
		ALPHA_SHARE_FUNC T Lerp(T a, T b, T t)
		{
			return a + t * (b - a);
		}
		*/
		template <typename T>
		ALPHA_SHARE_FUNC void BaryCenterCoordinate(
			const AxVector3T<T> &p,
			const AxVector3T<T> &p0,
			const AxVector3T<T> &p1,
			const AxVector3T<T> &p2,
			AxVector3T<T>& weight)
		{
			AxVector3T<T> d1 = p1 - p0;
			AxVector3T<T> d2 = p2 - p0;
			AxVector3T<T> pp0 = p - p0;

			T a = Dot(d1, d1);
			T b = Dot(d1, d2);
			T c = Dot(d1, pp0);
			T d = b;
			T e = Dot(d2, d2);
			T f = Dot(d2, pp0);
			T det = a * e - b * d;

			if (det == 0.0f)
				det = 1e-7f;
			T inverDeno = 1.0f / det;
			T s = (c*e - b * f) * inverDeno;
			T t = (a*f - c * d) * inverDeno;
			weight.x = 1.0f - s - t;
			weight.y = s;
			weight.z = t;
		}

		//TODO:Move One?????
		template <typename T>
		ALPHA_SHARE_FUNC bool BaryCenterCoordinate(
			AxVector3T<T> pos, 
			AxVector3T<T> p0, 
			AxVector3T<T> p1, 
			AxVector3T<T> p2,
			AxVector3T<T>& ret, 
			AxFp32 eplison)
		{
			AxVector3T<T> AB = p0 - p2;
			AxVector3T<T> AC = p1 - p2;
			AxVector3T<T> AP = pos - p2;
			T dot00 = Dot(AC, AC);
			T dot01 = Dot(AC, AB);
			T dot02 = Dot(AC, AP);
			T dot11 = Dot(AB, AB);
			T dot12 = Dot(AB, AP);
			T divisor = dot00 * dot11 - dot01 * dot01;
			if (divisor < 1e-7f)
				divisor = 1e-7f;
			T inverDeno = 1.0f / divisor;
			T v = (dot11 * dot02 - dot01 * dot12) * inverDeno;
			T u = (dot00 * dot12 - dot01 * dot02) * inverDeno;
			T w = 1.0f - u - v;
			ret.x = u;
			ret.y = v;
			ret.z = w;
			if (u < 0.0f - eplison || u>1.0f + eplison || v < 0.0f - eplison || v>1.0f + eplison || (u + v > 1.0f + eplison))
				return false;
			return true;
		}
	}
}

namespace AlphaCore
{
	namespace Math
	{
		ALPHA_SHARE_FUNC void ClampMaxmin(AxVector3& vec, float max, float min)
		{
			if (vec.x > max) { vec.x = max; }
			if (vec.y > max) { vec.y = max; }
			if (vec.z > max) { vec.z = max; }
			if (vec.x < min) { vec.x = min; }
			if (vec.y < min) { vec.y = min; }
			if (vec.z < min) { vec.z = min; }
		}

		ALPHA_SHARE_FUNC void ClampMin(AxVector3& vec,AxFp32 min)
		{
			if (vec.x < min) { vec.x = min; }
			if (vec.y < min) { vec.y = min; }
			if (vec.z < min) { vec.z = min; }
		}

		template<typename T>
		ALPHA_SHARE_FUNC T Clamp(T val, T min, T max)
		{
			if (val > max) { val = max; }
			if (val < min) { val = min; }
			return val;
		}

		ALPHA_SHARE_FUNC float MinF(float a, float b)
		{
			return a < b ? a : b;
		}

		ALPHA_SHARE_FUNC float MaxF(float a, float b)
		{
			return a > b ? a : b;
		}

		ALPHA_SHARE_FUNC int MaxI(int a, int b)
		{
			return a > b ? a : b;
		}

		ALPHA_SHARE_FUNC int Min(int a, int b)
		{
			return a < b ? a : b;
		}

		ALPHA_SHARE_FUNC float RSqrtF(float x)
		{
			return 1.0f / sqrtf(x);
		}

		template<typename T>
		ALPHA_SHARE_FUNC T Min(T a, T b)
		{
			return a < b ? a : b;
		}

		template<typename T>
		ALPHA_SHARE_FUNC T Max(T a, T b)
		{
			return a > b ? a : b;
		}

		ALPHA_SHARE_FUNC AxVector3 Min(AxVector3 a, AxVector3 b)
		{
			return MakeVector3(MinF(a.x, b.x), MinF(a.y, b.y), MinF(a.z, b.z));
		}

		ALPHA_SHARE_FUNC AxVector3 Max(AxVector3 a, AxVector3 b)
		{
			return MakeVector3(MaxF(a.x, b.x), MaxF(a.y, b.y), MaxF(a.z, b.z));
		}

		template<typename T>
		ALPHA_SHARE_FUNC AxColor4T<T> Clamp(const AxColor4T<T>& a, T bottom, T up)
		{
			return AxColor4T<T>{
				Min(Max(a.r, bottom), up),
					Min(Max(a.g, bottom), up),
					Min(Max(a.b, bottom), up),
					Min(Max(a.a, bottom), up)
			};
		}

		ALPHA_SHARE_FUNC AxVector3 LerpV3(const AxVector3& a, const AxVector3& b, float t)
		{
			return MakeVector3(
				AlphaCore::Math::LerpF32(a.x, b.x, t),
				AlphaCore::Math::LerpF32(a.y, b.y, t),
				AlphaCore::Math::LerpF32(a.z, b.z, t));
		}

		ALPHA_SHARE_FUNC AxFp32 Fit(AxFp32 old, AxFp32 min, AxFp32 max, AxFp32 minNew, AxFp32 maxNew)
		{
			float _old = Clamp(old, min, max);
			float sp = (_old - min) / (max - min);
			return 	minNew + sp * (maxNew - minNew);
		}
		ALPHA_SHARE_FUNC AxVector3 Fit(AxVector3 old, AxFp32 min, AxFp32 max, AxFp32 minNew, AxFp32 maxNew)
		{
			AxVector3 ret;
			ret.x = Fit(old.x, min, max, minNew, maxNew);
			ret.y = Fit(old.y, min, max, minNew, maxNew);
			ret.z = Fit(old.z, min, max, minNew, maxNew);
			return ret;
		}
	}
}
#endif