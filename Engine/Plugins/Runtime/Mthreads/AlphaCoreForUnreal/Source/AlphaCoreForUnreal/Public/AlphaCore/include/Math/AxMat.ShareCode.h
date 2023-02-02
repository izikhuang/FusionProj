#ifndef __AX_MATRIX_SHARECODE_H__
#define __AX_MATRIX_SHARECODE_H__

#include "AxMat.h"
#include "AxMacro.h"
#include "AxVectorBase.h"
#include "Math/AxVectorHelper.h"


ALPHA_SHARE_FUNC void PrintInfo(const char* head, const Quat&  m)
{
	printf("%s {%f,%f,%f,%f}\n", head, m.mm[0], m.mm[1], m.mm[2], m.mm[3]);
}

ALPHA_SHARE_FUNC Quat MakeQuat(AxFp32 defaultValue = 0.0f)
{
	Quat ret;
	ret.mm[0] = defaultValue;
	ret.mm[1] = defaultValue;
	ret.mm[2] = defaultValue;
	ret.mm[3] = defaultValue;
	return ret;
}

ALPHA_SHARE_FUNC Quat MakeQuat(AxFp32 q0, AxFp32 q1, AxFp32 q2, AxFp32 q3)
{
	Quat ret;
	ret.mm[0] = q0;
	ret.mm[1] = q1;
	ret.mm[2] = q2;
	ret.mm[3] = q3;
	return ret;
}

ALPHA_SHARE_FUNC Quat MakeQuat(AxVector3 vec3, AxFp32 w)
{
	Quat ret;
	ret.mm[0] = vec3.x;
	ret.mm[1] = vec3.y;
	ret.mm[2] = vec3.z;
	ret.mm[3] = w;
	return ret;
}

template<typename T>
ALPHA_SHARE_FUNC AxMatT<3, T> MakeMat3x3T()
{
	AxMatT<3, T> ret;
	ret.mm[0] = 0; ret.mm[3] = 0; ret.mm[6] = 0;
	ret.mm[1] = 0; ret.mm[4] = 0; ret.mm[7] = 0;
	ret.mm[2] = 0; ret.mm[5] = 0; ret.mm[8] = 0;
	return ret;
}

template<typename T>
ALPHA_SHARE_FUNC AxMatT<3, T> MakeMat3x3T(
	T m11, T m12, T m13,
	T m21, T m22, T m23,
	T m31, T m32, T m33)
{
	AxMatT<3, T> ret;
	ret.mm[0] = m11; ret.mm[3] = m12; ret.mm[6] = m13;
	ret.mm[1] = m21; ret.mm[4] = m22; ret.mm[7] = m23;
	ret.mm[2] = m31; ret.mm[5] = m32; ret.mm[8] = m33;
	return ret;
}

ALPHA_SHARE_FUNC AxMat3x3F MakeMat3x3F()
{
	AxMat3x3F ret;
	ret.mm[0] = 0; ret.mm[3] = 0; ret.mm[6] = 0;
	ret.mm[1] = 0; ret.mm[4] = 0; ret.mm[7] = 0;
	ret.mm[2] = 0; ret.mm[5] = 0; ret.mm[8] = 0;
	return ret;
}


ALPHA_SHARE_FUNC AxMat3x3F MakeMat3x3Identity()
{
	AxMat3x3F ret;
	ret.mm[0] = 1.0f; ret.mm[3] = 0;	ret.mm[6] = 0;
	ret.mm[1] = 0;	  ret.mm[4] = 1.0f; ret.mm[7] = 0;
	ret.mm[2] = 0;	  ret.mm[5] = 0;	ret.mm[8] = 1.0f;
	return ret;
}

template<typename T>
ALPHA_SHARE_FUNC T& At(T raw[3][3], AxUInt32 i, AxUInt32 j)
{
	return raw[i][j];
}

template<typename T>
ALPHA_SHARE_FUNC T& At(AxMatT<3, T>& mat3, AxUInt32 i, AxUInt32 j)
{
	return At((T(*)[3])mat3.mm, i, j);
}

template<typename T>
ALPHA_SHARE_FUNC T& At(AxVector3T<T>& raw, AxUInt32 i)
{
	return (&raw.x)[i];
}

ALPHA_SHARE_FUNC AxVector3 operator *(Quat quat, AxVector3 vec)
{
	float num = quat.mm[0] * 2.0f;
	float num2 = quat.mm[1] * 2.0f;
	float num3 = quat.mm[2] * 2.0f;
	float num4 = quat.mm[0] * num;
	float num5 = quat.mm[1] * num2;
	float num6 = quat.mm[2] * num3;
	float num7 = quat.mm[0] * num2;
	float num8 = quat.mm[0] * num3;
	float num9 = quat.mm[1] * num3;
	float num10 = quat.mm[3] * num;
	float num11 = quat.mm[3] * num2;
	float num12 = quat.mm[3] * num3;
	AxVector3 result;
	result.x = (1.0f - (num5 + num6)) * vec.x + (num7 - num12) * vec.y + (num8 + num11) * vec.z;
	result.y = (num7 + num12) * vec.x + (1.0f - (num4 + num6)) * vec.y + (num9 - num10) * vec.z;
	result.z = (num8 - num11) * vec.x + (num9 + num10) * vec.y + (1.0f - (num4 + num5)) * vec.z;
	return result;
}

ALPHA_SHARE_FUNC Quat operator /(Quat quat, float s)
{
	Quat ret = quat;
	ret.mm[0] /= s;
	ret.mm[1] /= s;
	ret.mm[2] /= s;
	ret.mm[3] /= s;
	return ret;
}

ALPHA_SHARE_FUNC Quat operator +(const Quat& a, const Quat& b)
{
	Quat ret = a;
	ret.mm[0] += b.mm[0];
	ret.mm[1] += b.mm[1];
	ret.mm[2] += b.mm[2];
	ret.mm[3] += b.mm[3];
	return ret;
}

ALPHA_SHARE_FUNC void operator+=(Quat& a, const Quat& b)
{
	a.mm[0] += b.mm[0];
	a.mm[1] += b.mm[1];
	a.mm[2] += b.mm[2];
	a.mm[3] += b.mm[3];
}


ALPHA_SHARE_FUNC Quat operator -(const Quat& a, const Quat& b)
{
	Quat ret = a;
	ret.mm[0] -= b.mm[0];
	ret.mm[1] -= b.mm[1];
	ret.mm[2] -= b.mm[2];
	ret.mm[3] -= b.mm[3];
	return ret;
}

ALPHA_SHARE_FUNC void operator -=(Quat& a, const Quat& b)
{
 	a.mm[0] -= b.mm[0];
	a.mm[1] -= b.mm[1];
	a.mm[2] -= b.mm[2];
	a.mm[3] -= b.mm[3];
 }

ALPHA_SHARE_FUNC Quat operator *(AxFp32 s, Quat quat)
{
	Quat ret = quat;
	ret.mm[0] *= s;
	ret.mm[1] *= s;
	ret.mm[2] *= s;
	ret.mm[3] *= s;
	return ret;
}

/// Dot product
ALPHA_SHARE_FUNC AxFp32 Dot(const Quat &q, const Quat &b)
{
	return b.mm[0] * q.mm[0] + 
		   b.mm[1] * q.mm[1] +
		   b.mm[2] * q.mm[2] +
		   b.mm[3] * q.mm[3];
}
//TODO Test
ALPHA_SHARE_FUNC Quat QuatCloser(Quat q0, Quat q1)
{
	Quat q0plus = q0 + q1;
	q0 -= q1;
	///PrintInfo("........q0:", q0);
	///PrintInfo("........q0plus:", q0plus);
 	return Dot(q0, q0) > Dot(q0plus, q0plus) ? q0plus : q0;
}

ALPHA_SHARE_FUNC Quat QuatInverse(const Quat& src)
{
	AxFp32 d = src.mm[0] * src.mm[0] + src.mm[1] * src.mm[1] + src.mm[2] * src.mm[2] + src.mm[3] * src.mm[3];
	Quat result = src / -d;
	result.mm[3] = -result.mm[3];
	return result;
}
ALPHA_SHARE_FUNC AxVector3 abs(const AxVector3& v)
{
	return MakeVector3(abs(v.x), abs(v.y), abs(v.z));
}


ALPHA_SHARE_FUNC Quat MakeQuatFormMat3x3(AxMat3x3F a)
{
	Quat q;
	float trace = At(a, 0, 0) + At(a, 1, 1) + At(a, 2, 2); // I removed + 1.0f; see discussion with Ethan
	if (trace > 0) {// I changed M_EPSILON to 0
		float s = 0.5f / sqrtf(trace + 1.0f);
		q.mm[3] = 0.25f / s;
		q.mm[0] = (At(a, 2, 1) - At(a, 1, 2)) * s;
		q.mm[1] = (At(a, 0, 2) - At(a, 2, 0)) * s;
		q.mm[2] = (At(a, 1, 0) - At(a, 0, 1)) * s;
	}
	else {
		if (At(a, 0, 0) > At(a, 1, 1) && At(a, 0, 0) > At(a, 2, 2)) {
			float s = 2.0f * sqrtf(1.0f + At(a, 0, 0) - At(a, 1, 1) - At(a, 2, 2));
			q.mm[3] = (At(a, 2, 1) - At(a, 1, 2)) / s;
			q.mm[0] = 0.25f * s;
			q.mm[1] = (At(a, 0, 1) + At(a, 1, 0)) / s;
			q.mm[2] = (At(a, 0, 2) + At(a, 2, 0)) / s;
		}
		else if (At(a, 1, 1) > At(a, 2, 2)) {
			float s = 2.0f * sqrtf(1.0f + At(a, 1, 1) - At(a, 0, 0) - At(a, 2, 2));
			q.mm[3] = (At(a, 0, 2) - At(a, 2, 0)) / s;
			q.mm[0] = (At(a, 0, 1) + At(a, 1, 0)) / s;
			q.mm[1] = 0.25f * s;
			q.mm[2] = (At(a, 1, 2) + At(a, 2, 1)) / s;
		}
		else {
			float s = 2.0f * sqrtf(1.0f + At(a, 2, 2) - At(a, 0, 0) - At(a, 1, 1));
			q.mm[3] = (At(a, 1, 0) - At(a, 0, 1)) / s;
			q.mm[0] = (At(a, 0, 2) + At(a, 2, 0)) / s;
			q.mm[1] = (At(a, 1, 2) + At(a, 2, 1)) / s;
			q.mm[2] = 0.25f * s;
		}
	}
	return q;
}
template<typename T>
ALPHA_SHARE_FUNC void operator+=(AxMatT<3, T>& a, const AxMatT<3, T>& b)
{
	a.mm[0] += b.mm[0]; a.mm[1] += b.mm[1]; a.mm[2] += b.mm[2];
	a.mm[3] += b.mm[3]; a.mm[4] += b.mm[4]; a.mm[5] += b.mm[5];
	a.mm[6] += b.mm[6]; a.mm[7] += b.mm[7]; a.mm[8] += b.mm[8];
}

template<typename T>
ALPHA_SHARE_FUNC void operator/=(AxMatT<3, T>& a, T b)
{
	AX_FOR_I(9)
		a.mm[i] /= b;
}

/// Return (this*q), e.g.   q = q1 * q2;
ALPHA_SHARE_FUNC Quat operator*(const Quat &q, const Quat &b)
{
	/*
	Quat prod;
	prod.mm[0] = b.mm[3] * q.mm[0] + b.mm[0] * q.mm[3] + b.mm[1] * q.mm[2] - b.mm[2] * q.mm[1];
	prod.mm[1] = b.mm[3] * q.mm[1] + b.mm[1] * q.mm[3] + b.mm[2] * q.mm[0] - b.mm[0] * q.mm[2];
	prod.mm[2] = b.mm[3] * q.mm[2] + b.mm[2] * q.mm[3] + b.mm[0] * q.mm[1] - b.mm[1] * q.mm[0];
	prod.mm[3] = b.mm[3] * q.mm[3] - b.mm[0] * q.mm[0] - b.mm[1] * q.mm[1] - b.mm[2] * q.mm[2];
	return prod;
	*/ 
	Quat prod;
	prod.mm[0] = q.mm[0] * b.mm[0];
	prod.mm[1] = q.mm[1] * b.mm[1];
	prod.mm[2] = q.mm[2] * b.mm[2];
	prod.mm[3] = q.mm[3] * b.mm[3];
	return prod;
}

ALPHA_SHARE_FUNC AxVector3 QuatXYZ(const Quat& a)
{
	return MakeVector3(a.mm[0], a.mm[1], a.mm[2]);
}

ALPHA_SHARE_FUNC Quat QuatYXWZ(const Quat& a)
{
	return MakeQuat(a.mm[1], a.mm[0], a.mm[3],a.mm[2]);
}



ALPHA_SHARE_FUNC Quat QuatMultiply(Quat q0, Quat q1)
{
	AxVector3 v1 = QuatXYZ(q0);
	AxVector3 v2 = QuatXYZ(q1);
	AxFp32 s1 = q0.mm[3];
	AxFp32 s2 = q1.mm[3];
	AxFp32 w = s1 * s2 - Dot(v1, v2);
	AxVector3 v3 = s1 * v2 + s2 * v1 + Cross(v1, v2);
	return MakeQuat(v3, w);
}

//ALPHA_SHARE_FUNC Quat QuatMultiply(AxVector4 q0, AxVector4 q1)
//{
//	AxVector3 v1 = MakeVector3(q0.x, q0.y, q0.z);
//	AxVector3 v2 = MakeVector3(q1.x, q1.y, q1.z);
//	AxFp32 s1 = q0.w;
//	AxFp32 s2 = q1.w;
//	AxFp32 w = s1 * s2 - Dot(v1, v2);
//	AxVector3 v3 = s1 * v2 + s2 * v1 + Cross(v1, v2);
//	return MakeQuat(v3, w);
//}



ALPHA_SHARE_FUNC Quat operator-(const AxVector4& a, Quat b)
{
	return MakeQuat(
		a.x - b.mm[0],
		a.y - b.mm[1],
		a.z - b.mm[2],
		a.w - b.mm[3]);
}

ALPHA_SHARE_FUNC Quat MakeQuat(AxVector4 vec4)
{
	Quat ret;
	ret.mm[0] = vec4.x;
	ret.mm[1] = vec4.y;
	ret.mm[2] = vec4.z;
	ret.mm[3] = vec4.w;
	return ret;
}

ALPHA_SHARE_FUNC void operator-=(AxVector4& a, const Quat& b)
{
	a.x -= b.mm[0];
	a.y -= b.mm[1];
	a.z -= b.mm[2];
	a.w -= b.mm[3];
}

ALPHA_SHARE_FUNC AxVector4 MakeVector4(const Quat& q)
{
	AxVector4 t;
	t.x = q.mm[0]; t.y = q.mm[1]; t.z = q.mm[2]; t.w = q.mm[3];
	return t;
}

ALPHA_SHARE_FUNC Quat QuatConjugate(Quat q)
{
	return MakeQuat(Neg(QuatXYZ(q)), q.mm[3]);
}



ALPHA_SHARE_FUNC Quat Normalize(Quat q)
{
	AxFp32 d = AxFp32(sqrtf(q.mm[0] * q.mm[0] + q.mm[1] * q.mm[1] + q.mm[2] * q.mm[2] + q.mm[3] * q.mm[3]));
	// TODO  d : != 0 
	//if (isApproxEqual(d, T(0.0), eps)) return false;
	return q / d;
}

/*
ALPHA_SHARE_FUNC Quat QuatInverse(const Quat& src)
{
	AxFp32 d = src.mm[0] * src.mm[0] + src.mm[1] * src.mm[1] + src.mm[2] * src.mm[2] + src.mm[3] * src.mm[3];
	Quat result = src / -d;
	result.mm[3] = -result.mm[3];
	return result;
}

ALPHA_SHARE_FUNC void AssignXYZ(AxVector4* a,AxVector3 v)
{
	a->x = v.x;
	a->y = v.y;
	a->z = v.z;
}



ALPHA_SHARE_FUNC void QuatMultiply(AxVector4* dst, Quat a, Quat b)
{
	AxVector3 a_pt = QuatXYZ(a);
	AxVector3 b_pt = QuatXYZ(b);
	dst->w = a.mm[3] * b.mm[3] - Dot(a_pt, b_pt);
	AxVector3 br = a.mm[3] * b_pt + b.mm[3] * a_pt + Cross(a_pt, b_pt);
	AssignXYZ(dst, br);
}

ALPHA_SHARE_FUNC void QuatRotate(AxVector3& dst, Quat quat, AxVector3 pt)
{
	Quat tmp;
	AxVector4 q4 = MakeVector4(pt.x, pt.y, pt.z, 0.0f);

	Quat pt_quat;
	pt_quat.mm[0] = q4.x;
	pt_quat.mm[1] = q4.y;
	pt_quat.mm[2] = q4.z;
	pt_quat.mm[3] = q4.w;

	QuatMultiply((AxVector4*)&tmp, quat, &pt_quat);

	Quat mult_tmp = (Quat)(-quat->xyz, quat->w);

	Quat ret;
	QuatMultiply((AxVector4*)&ret, tmp, mult_tmp);
	dst = QuatXYZ(ret);
}


ALPHA_SHARE_FUNC Quat QuatConjugate(Quat q)
{
	return (quat)(-q.xyz, q.w);
}


ALPHA_SHARE_FUNC Quat qmultiply(Quat q0, Quat q1)
{
	AxVector3 v1 = QuatXYZ(q0);
	AxVector3 v2 = QuatXYZ(q1);
	AxFp32 s1 = q0.mm[3];
	AxFp32 s2 = q1.mm[3];
	AxFp32 w = s1 * s2 - Dot(v1, v2);
	AxVector3 v3 = s1 * v2 + s2 * v1 + Cross(v1, v2);
	return (quat)(v3, w);
}

ALPHA_SHARE_FUNC AxVector3 QuatRotate(Quat q, AxVector3 v)
{
	return qmultiply(qmultiply(q, (quat)(v, 0)), qconjugate(q)).xyz;
}

/*
ALPHA_SHARE_FUNC Quat QuatCloser(Quat q0, Quat q1)
{
	Quat q0plus = q0 + q1;
	q0 -= q1;
	exint4 pluscloser = (exint4)-(dot(q0, q0) > dot(q0plus, q0plus));
	return select(q0, q0plus, pluscloser);
}
*/

ALPHA_SHARE_FUNC void PrintInfo(const AxMat3x3F&  m, const char* head, bool oneline = false)
{
	if (oneline)
	{
		printf("%s {%f,%f,%f,%f,%f,%f,%f,%f,%f}\n", head, m.mm[0], m.mm[1], m.mm[2], m.mm[3], m.mm[4], m.mm[5], m.mm[6], m.mm[7], m.mm[8]);
	}
	else {
		printf("%s\t\n\t     [%f,%f,%f]\n", head, m.mm[0], m.mm[1], m.mm[2]);
		printf("\t     [%f,%f,%f]\n", m.mm[3], m.mm[4], m.mm[5]);
		printf("\t     [%f,%f,%f]\n", m.mm[6], m.mm[7], m.mm[8]);
	}
}



ALPHA_SHARE_FUNC void PrintInfo(const char* head, const AxMat3x3F&  m, bool oneline = false)
{
	PrintInfo(m, head, oneline);
}


namespace AlphaCore
{
	namespace Math
	{
		namespace ShareCode
		{
			template<typename T>
			ALPHA_SHARE_FUNC  AxMatT<3, T> Outerproduct(const AxVector3T<T>& v1, const AxVector3T<T>& v2)
			{
				AxMatT<3, T> t = MakeMat3x3T(v1.x * v2.x, v1.x * v2.y, v1.x * v2.z,
					v1.y * v2.x, v1.y * v2.y, v1.y * v2.z,
					v1.z * v2.x, v1.z * v2.y, v1.z * v2.z);
				return t;
			}
			ALPHA_SHARE_FUNC void Pivot(AxUInt32 i, AxUInt32 j, AxMat3x3F& S, AxVector3& D, AxMat3x3F& Q)
			{
				const int& n = 3;  // should be 3
				float temp;
				/// scratch variables used in pivoting
				float cotan_of_2_theta;
				float tan_of_theta;
				float cosin_of_theta;
				float sin_of_theta;
				float z;
				float Sij = At(S, i, j);
				float Sjj_minus_Sii = At(D, j) - At(D, i);

				if (fabs(Sjj_minus_Sii) * (10 * 1e-8f) > fabs(Sij)) {
					tan_of_theta = Sij / Sjj_minus_Sii;
				}
				else {
					/// pivot on Sij
					cotan_of_2_theta = 0.5f*Sjj_minus_Sii / Sij;

					if (cotan_of_2_theta < 0.) {
						tan_of_theta =
							-1. / (sqrt(1. + cotan_of_2_theta * cotan_of_2_theta) - cotan_of_2_theta);
					}
					else {
						tan_of_theta =
							1. / (sqrt(1. + cotan_of_2_theta * cotan_of_2_theta) + cotan_of_2_theta);
					}
				}

				cosin_of_theta = 1. / sqrt(1. + tan_of_theta * tan_of_theta);
				sin_of_theta = cosin_of_theta * tan_of_theta;
				z = tan_of_theta * Sij;
				At(S, i, j) = 0;
				At(D, i) -= z;
				At(D, j) += z;
				for (AxUInt32 k = 0; k < i; ++k) {
					temp = At(S, k, i);
					At(S, k, i) = cosin_of_theta * temp - sin_of_theta * At(S, k, j);
					At(S, k, j) = sin_of_theta * temp + cosin_of_theta * At(S, k, j);
				}
				for (AxUInt32 k = i + 1; k < j; ++k) {
					temp = At(S, i, k);
					At(S, i, k) = cosin_of_theta * temp - sin_of_theta * At(S, k, j);
					At(S, k, j) = sin_of_theta * temp + cosin_of_theta * At(S, k, j);
				}
				for (int k = j + 1; k < n; ++k) {
					temp = At(S, i, k);
					At(S, i, k) = cosin_of_theta * temp - sin_of_theta * At(S, j, k);
					At(S, j, k) = sin_of_theta * temp + cosin_of_theta * At(S, j, k);
				}
				for (int k = 0; k < n; ++k)
				{
					temp = At(Q, k, i);
					At(Q, k, i) = cosin_of_theta * temp - sin_of_theta * At(Q, k, j);
					At(Q, k, j) = sin_of_theta * temp + cosin_of_theta * At(Q, k, j);
				}
			}

			//template<typename T>
			ALPHA_SHARE_FUNC bool DiagonalizeSymmetricMatrix(
				const AxMatT<3, AxFp32>& input, 
				AxMatT<3, AxFp32>& Q, 
				AxVector3T<AxFp32>& D,
				unsigned int MAX_ITERATIONS = 250)
			{
				/// use Givens rotation matrix to eliminate off-diagonal entries.
				/// initialize the rotation matrix as idenity
				Q = MakeMat3x3Identity();
				int n = 3;  // should be 3
				/// temp matrix.  Assumed to be symmetric
				AxMatT<3, AxFp32> S = input;

				for (int i = 0; i < n; ++i) {
					At(D, i) = At(S, i, i);
				}
				AxVector3T<AxFp32> OriginD = D;
				//PrintInfo("D:", D);
				unsigned int iterations = 0;
				/// Just iterate over all the non-diagonal enteries
				/// using the largest as a pivot.
				do {
					/// check for absolute convergence
					/// are symmetric off diagonals all zero
					float er = 0;
					for (int i = 0; i < n; ++i) {
						for (int j = i + 1; j < n; ++j) {
							er += fabs(At(S, i, j));
						}
					}
					if (fabs(er) < 1e-6) {
						return true;
					}
					iterations++;
					float max_element = 0.0f;
					int ip = 0;
					int jp = 0;
					/// loop over all the off-diagonals above the diagonal
					for (int i = 0; i < n; ++i) {
						for (int j = i + 1; j < n; ++j) {

							if (fabs(At(D, i)) * (10 * 1e-6) > fabs(At(S, i, j))) {
								/// value too small to pivot on
								At(S, i, j) = 0;
							}
							if (fabs(At(S, i, j)) > max_element) {
								max_element = fabs(At(S, i, j));
								ip = i;
								jp = j;
							}
						}
					}
					AlphaCore::Math::ShareCode::Pivot(ip, jp, S, D, Q);
				} while (iterations < MAX_ITERATIONS);

				Q = MakeMat3x3Identity();
				D = OriginD;
				//printf("DiagonalizeSymmetricMatrix ERROR SP");
				return false;
			}

		}
	}
}

#endif