#ifndef __AX_DATA_TYPE_SHARECODE_H__
#define __AX_DATA_TYPE_SHARECODE_H__

#include "AxDataType.h"
#include "AxMacro.h"

ALPHA_SHARE_FUNC unsigned short InternalToAxFp16(const float f, unsigned int& sign, unsigned int& remainder)
{
	unsigned int x;
	unsigned int u;
	unsigned int result;

#ifdef __CUDA_ARCH__
	//TODO : 
#else
	std::memcpy(&x, &f, sizeof(f));
#endif

	u = (x & 0x7fffffffU);
	sign = ((x >> 16U) & 0x8000U);
	// NaN/+Inf/-Inf
	if (u >= 0x7f800000U) {
		remainder = 0U;
		result = ((u == 0x7f800000U) ? (sign | 0x7c00U) : 0x7fffU);
	}
	else if (u > 0x477fefffU) { // Overflows
		remainder = 0x80000000U;
		result = (sign | 0x7bffU);
	}
	else if (u >= 0x38800000U) { // Normal numbers
		remainder = u << 19U;
		u -= 0x38000000U;
		result = (sign | (u >> 13U));
	}
	else if (u < 0x33000001U) { // +0/-0
		remainder = u;
		result = sign;
	}
	else { // Denormal numbers
		const unsigned int exponent = u >> 23U;
		const unsigned int shift = 0x7eU - exponent;
		unsigned int mantissa = (u & 0x7fffffU);
		mantissa |= 0x800000U;
		remainder = mantissa << (32U - shift);
		result = (sign | (mantissa >> shift));
		result &= 0x0000FFFFU;
	}
	return static_cast<unsigned short>(result);
}

ALPHA_SHARE_FUNC AxFp16 ToAxFp16(const float f)
{
	AxFp16 ret;
	unsigned int sign = 0U;
	unsigned int remainder = 0U;
	ret.x = InternalToAxFp16(f, sign, remainder);
	if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((ret.x & 0x1U) != 0U))) {
		ret.x++;
	}
	return ret;
}


ALPHA_SHARE_FUNC float InternalToAxFp32(const unsigned short h)
{
	unsigned int sign = ((static_cast<unsigned int>(h) >> 15U) & 1U);
	unsigned int exponent = ((static_cast<unsigned int>(h) >> 10U) & 0x1fU);
	unsigned int mantissa = ((static_cast<unsigned int>(h) & 0x3ffU) << 13U);
	float f;
	if (exponent == 0x1fU) { /* NaN or Inf */
		/* discard sign of a NaN */
		sign = ((mantissa != 0U) ? (sign >> 1U) : sign);
		mantissa = ((mantissa != 0U) ? 0x7fffffU : 0U);
		exponent = 0xffU;
	}
	else if (exponent == 0U) { /* Denorm or Zero */
		if (mantissa != 0U) {
			unsigned int msb;
			exponent = 0x71U;
			do {
				msb = (mantissa & 0x400000U);
				mantissa <<= 1U; /* normalize */
				--exponent;
			} while (msb == 0U);
			mantissa &= 0x7fffffU; /* 1.mantissa is implicit */
		}
	}
	else {
		exponent += 0x70U;
	}
	const unsigned int u = ((sign << 31U) | (exponent << 23U) | mantissa);
	std::memcpy(&f, &u, sizeof(u));
	return f;
}

ALPHA_SHARE_FUNC float ToAxFp32(const AxFp16 v) {
	float val = InternalToAxFp32(v.x);
	return val;
}

/* Make AxFp16 */
ALPHA_SHARE_FUNC AxFp16 MakeAxFp16(const float& f) { return ToAxFp16(f); }



/* Some basic arithmetic operations */
/* add */
ALPHA_SHARE_FUNC AxFp16 operator+(const AxFp16 lh, const AxFp16 rh) {
	return ToAxFp16(ToAxFp32(lh) + ToAxFp32(rh));
}

ALPHA_SHARE_FUNC AxFp16 operator+(const AxFp16& lh, const float& rh) {
	return ToAxFp16(ToAxFp32(lh) + rh);
}

ALPHA_SHARE_FUNC AxFp16& operator+=(AxFp16& lh, const AxFp16& rh) {
	lh = ToAxFp16(ToAxFp32(lh) + ToAxFp32(rh));
	return lh;
}

ALPHA_SHARE_FUNC AxFp16& operator+=(AxFp16& lh, const float& rh) {
	lh = ToAxFp16(ToAxFp32(lh) + rh);
	return lh;
}

/* subtraction */
ALPHA_SHARE_FUNC AxFp16 operator-(const AxFp16 lh, const AxFp16 rh) {
	return ToAxFp16(ToAxFp32(lh) - ToAxFp32(rh));
}

ALPHA_SHARE_FUNC AxFp16 operator-(const AxFp16& lh, const float& rh) {
	return ToAxFp16(ToAxFp32(lh) - rh);
}

ALPHA_SHARE_FUNC AxFp16& operator-=(AxFp16& lh, const AxFp16& rh) {
	lh = ToAxFp16(ToAxFp32(lh) - ToAxFp32(rh));
	return lh;
}

ALPHA_SHARE_FUNC AxFp16& operator-=(AxFp16& lh, const float& rh) {
	lh = ToAxFp16(ToAxFp32(lh) - rh);
	return lh;
}

/* multiplication */
ALPHA_SHARE_FUNC AxFp16 operator*(const AxFp16 lh, const AxFp16 rh) {
	return ToAxFp16(ToAxFp32(lh) * ToAxFp32(rh));
}

ALPHA_SHARE_FUNC AxFp16 operator*(const AxFp16& lh, const float& rh) {
	return ToAxFp16(ToAxFp32(lh) * rh);
}

ALPHA_SHARE_FUNC AxFp16& operator*=(AxFp16& lh, const AxFp16& rh) {
	lh = ToAxFp16(ToAxFp32(lh) * ToAxFp32(rh));
	return lh;
}

ALPHA_SHARE_FUNC AxFp16& operator*=(AxFp16& lh, const float& rh) {
	lh = ToAxFp16(ToAxFp32(lh) * rh);
	return lh;
}

/* division */
ALPHA_SHARE_FUNC AxFp16 operator/(const AxFp16 lh, const AxFp16 rh) {
	return ToAxFp16(ToAxFp32(lh) / ToAxFp32(rh));
}

ALPHA_SHARE_FUNC AxFp16 operator/(const AxFp16& lh, const float& rh) {
	return ToAxFp16(ToAxFp32(lh) / rh);
}

ALPHA_SHARE_FUNC AxFp16& operator/=(AxFp16& lh, const AxFp16& rh) {
	lh = ToAxFp16(ToAxFp32(lh) / ToAxFp32(rh));
	return lh;
}

ALPHA_SHARE_FUNC AxFp16& operator/=(AxFp16& lh, const float& rh) {
	lh = ToAxFp16(ToAxFp32(lh) / rh);
	return lh;
}

namespace AlphaCore
{
	template<typename T>
	ALPHA_SHARE_FUNC AxInt32 ToInt32(T* ptr)
	{
		AxInt32* ret = (AxInt32*)ptr;
		return *ret;
	}

	template<typename T>
	ALPHA_SHARE_FUNC AxUInt32 ToUInt32(T* ptr)
	{
		AxUInt32* ret = (AxUInt32*)ptr;
		return *ret;
	}

	template<typename T>
	ALPHA_SHARE_FUNC AxFp32 ToFp32(T* ptr)
	{
		AxFp32* ret = (AxFp32*)ptr;
		return *ret;
	}
}

#endif