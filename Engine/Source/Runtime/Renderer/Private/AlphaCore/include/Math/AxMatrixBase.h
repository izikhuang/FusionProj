#ifndef __ALPHA_CORE_MATRIX_BASE_H__
#define __ALPHA_CORE_MATRIX_BASE_H__

#include <AxMacro.h>
#include <Math/AxVectorBase.h>
#include <Math/AxVectorHelper.h>
#include <math.h>
#include <stdio.h>

template<class T>
struct AxMatrix2x2T
{
	T m[4];
};

template<class T>
struct AxMatrix3x3T
{
	T m[9];
};

template<class T>
struct AxMatrix4x4T
{
	T m[16];
	ALPHA_KERNEL_FUNC AxMatrix4x4T<T> operator[](int idx) {
		return m[idx];
	}
};

template<class T>
struct AxMatrix9x9T
{
	T m[81];
};



typedef AxMatrix3x3T<float>		AxMatrix3x3;
typedef AxMatrix3x3T<double>	AxMatrix3x3D;

typedef AxMatrix4x4T<float> AxMatrix4x4;
typedef AxMatrix4x4T<double> AxMatrix4x4D;

ALPHA_KERNEL_FUNC  AxMatrix3x3 MakeMat3x3(
	float m11, float m12, float m13,
	float m21, float m22, float m23,
	float m31, float m32, float m33)
{
	AxMatrix3x3 t;
	t.m[0] = m11; t.m[1] = m12; t.m[2] = m13;
	t.m[3] = m21; t.m[4] = m22; t.m[5] = m23;
	t.m[6] = m31; t.m[7] = m32; t.m[8] = m33;
	return t;
}



ALPHA_KERNEL_FUNC AxMatrix3x3 MakeMat3x3()
{
	AxMatrix3x3 t;
	t.m[0] = 0; t.m[1] = 0; t.m[2] = 0;
	t.m[3] = 0; t.m[4] = 0; t.m[5] = 0;
	t.m[6] = 0; t.m[7] = 0; t.m[8] = 0;
	return t;
}

ALPHA_KERNEL_FUNC  AxMatrix3x3 Make3x3Identity()
{
	AxMatrix3x3 t;
	t.m[0] = 1; t.m[1] = 0; t.m[2] = 0;
	t.m[3] = 0; t.m[4] = 1; t.m[5] = 0;
	t.m[6] = 0; t.m[7] = 0; t.m[8] = 1;
	return t;
}

ALPHA_KERNEL_FUNC AxMatrix3x3 Make3x3ByDiagonal(float m00, float m11, float m22)
{
	AxMatrix3x3 t;
	t.m[0] = m00; t.m[1] = 0; t.m[2] = 0;
	t.m[3] = 0; t.m[4] = m11; t.m[5] = 0;
	t.m[6] = 0; t.m[7] = 0; t.m[8] = m22;
	return t;
}

ALPHA_KERNEL_FUNC AxMatrix3x3 operator*(AxMatrix3x3& a, float b)
{
	return MakeMat3x3(a.m[0] * b, a.m[1] * b, a.m[2] * b,
					  a.m[3] * b, a.m[4] * b, a.m[5] * b,
					  a.m[6] * b, a.m[7] * b, a.m[8] * b);

}

ALPHA_KERNEL_FUNC AxMatrix3x3 operator*(const AxMatrix3x3& a,const AxMatrix3x3& b)
{
	return MakeMat3x3(a.m[0] * b.m[0] + a.m[1] * b.m[3] + a.m[2] * b.m[6],
		a.m[0] * b.m[1] + a.m[1] * b.m[4] + a.m[2] * b.m[7],
		a.m[0] * b.m[2] + a.m[1] * b.m[5] + a.m[2] * b.m[8],
		a.m[3] * b.m[0] + a.m[4] * b.m[3] + a.m[5] * b.m[6],
		a.m[3] * b.m[1] + a.m[4] * b.m[4] + a.m[5] * b.m[7],
		a.m[3] * b.m[2] + a.m[4] * b.m[5] + a.m[5] * b.m[8],
		a.m[6] * b.m[0] + a.m[7] * b.m[3] + a.m[8] * b.m[6],
		a.m[6] * b.m[1] + a.m[7] * b.m[4] + a.m[8] * b.m[7],
		a.m[6] * b.m[2] + a.m[7] * b.m[5] + a.m[8] * b.m[8]);
}


ALPHA_KERNEL_FUNC bool Inverse(const float m[16], float invOut[16])
{
	int inv[16], det;
	unsigned i = 0;

	inv[0]  =  m[5] * m[10] * m[15] -
		       m[5] * m[11] * m[14] -
		       m[9] * m[6] * m[15] +
		       m[9] * m[7] * m[14] +
		       m[13] * m[6] * m[11] -
		       m[13] * m[7] * m[10];

	inv[4]  = -m[4] * m[10] * m[15] +
		       m[4] * m[11] * m[14] +
		       m[8] * m[6] * m[15] -
		       m[8] * m[7] * m[14] -
		       m[12] * m[6] * m[11] +
		       m[12] * m[7] * m[10];

	inv[8]  =  m[4] * m[9] * m[15] -
		       m[4] * m[11] * m[13] -
		       m[8] * m[5] * m[15] +
		       m[8] * m[7] * m[13] +
		       m[12] * m[5] * m[11] -
		       m[12] * m[7] * m[9];

	inv[12] = -m[4] * m[9] * m[14] +
		       m[4] * m[10] * m[13] +
		       m[8] * m[5] * m[14] -
		       m[8] * m[6] * m[13] -
		       m[12] * m[5] * m[10] +
		       m[12] * m[6] * m[9];

	inv[1] = - m[1] * m[10] * m[15] +
		       m[1] * m[11] * m[14] +
		       m[9] * m[2] * m[15] -
		       m[9] * m[3] * m[14] -
		       m[13] * m[2] * m[11] +
		       m[13] * m[3] * m[10];

	inv[5] =   m[0] * m[10] * m[15] -
			   m[0] * m[11] * m[14] -
			   m[8] * m[2] * m[15] +
			   m[8] * m[3] * m[14] +
			   m[12] * m[2] * m[11] -
			   m[12] * m[3] * m[10];

	inv[9] =  -m[0] * m[9] * m[15] +
			   m[0] * m[11] * m[13] +
			   m[8] * m[1] * m[15] -
			   m[8] * m[3] * m[13] -
			   m[12] * m[1] * m[11] +
			   m[12] * m[3] * m[9];

	inv[13] =  m[0] * m[9] * m[14] -
			   m[0] * m[10] * m[13] -
			   m[8] * m[1] * m[14] +
			   m[8] * m[2] * m[13] +
			   m[12] * m[1] * m[10] -
			   m[12] * m[2] * m[9];

	inv[2] =   m[1] * m[6] * m[15] -
			   m[1] * m[7] * m[14] -
			   m[5] * m[2] * m[15] +
			   m[5] * m[3] * m[14] +
			   m[13] * m[2] * m[7] -
			   m[13] * m[3] * m[6];

	inv[6] = - m[0] * m[6] * m[15] +
			   m[0] * m[7] * m[14] +
			   m[4] * m[2] * m[15] -
			   m[4] * m[3] * m[14] -
			   m[12] * m[2] * m[7] +
			   m[12] * m[3] * m[6];

	inv[10] =  m[0] * m[5] * m[15] -
			   m[0] * m[7] * m[13] -
			   m[4] * m[1] * m[15] +
			   m[4] * m[3] * m[13] +
			   m[12] * m[1] * m[7] -
			   m[12] * m[3] * m[5];

	inv[14] = -m[0] * m[5] * m[14] +
			   m[0] * m[6] * m[13] +
			   m[4] * m[1] * m[14] -
			   m[4] * m[2] * m[13] -
			   m[12] * m[1] * m[6] +
			   m[12] * m[2] * m[5];

	inv[3]  = -m[1] * m[6] * m[11] +
			   m[1] * m[7] * m[10] +
			   m[5] * m[2] * m[11] -
			   m[5] * m[3] * m[10] -
			   m[9] * m[2] * m[7] +
			   m[9] * m[3] * m[6];

	inv[7]  =  m[0] * m[6] * m[11] -
			   m[0] * m[7] * m[10] -
			   m[4] * m[2] * m[11] +
			   m[4] * m[3] * m[10] +
			   m[8] * m[2] * m[7] -
			   m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] +
		       m[0] * m[7] * m[9] +
		       m[4] * m[1] * m[11] -
		       m[4] * m[3] * m[9] -
		       m[8] * m[1] * m[7] +
		       m[8] * m[3] * m[5];

	inv[15] =  m[0] * m[5] * m[10] -
		       m[0] * m[6] * m[9] -
		       m[4] * m[1] * m[10] +
		       m[4] * m[2] * m[9] +
		       m[8] * m[1] * m[6] -
		       m[8] * m[2] * m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
	if (det == 0)
		return false;

	det = 1.0 / det;

	for (i = 0; i < 16; i++)
		invOut[i] = inv[i] * det;

	return true;
}

ALPHA_KERNEL_FUNC AxMatrix4x4 MakeMat4x4(
	float m00, float m01, float m02, float m03,
	float m10, float m11, float m12, float m13,
	float m20, float m21, float m22, float m23,
	float m30, float m31, float m32, float m33)
{
	AxMatrix4x4 t;
	t.m[0] = m00;  t.m[1] = m01;  t.m[2] = m02;  t.m[3] = m03;
	t.m[4] = m10;  t.m[5] = m11;  t.m[6] = m12;  t.m[7] = m13;
	t.m[8] = m20;  t.m[9] = m21;  t.m[10] = m22; t.m[11] = m23;
	t.m[12] = m30; t.m[13] = m31; t.m[14] = m32; t.m[15] = m33;
	return t;
}

ALPHA_KERNEL_FUNC bool Inverse(AxMatrix4x4& m4x4)
{
	return Inverse(m4x4.m, m4x4.m);
}

ALPHA_KERNEL_FUNC AxMatrix4x4 MakeMat4x4Idenity()
{
	return MakeMat4x4(1, 0, 0, 0,
					  0, 1, 0, 0,
					  0, 0, 1, 0,
					  0, 0, 0, 1);
}

ALPHA_KERNEL_FUNC AxMatrix4x4 MakeMat4x4(AxMatrix3x3& rot)
{
	AxMatrix4x4 t;
	t.m[0] = rot.m[0];  t.m[1] = rot.m[1];  t.m[2]  = rot.m[2];  t.m[3]  = 0;
	t.m[4] = rot.m[3];  t.m[5] = rot.m[4];  t.m[6]  = rot.m[5];  t.m[7]  = 0;
	t.m[8] = rot.m[6];  t.m[9] = rot.m[7];  t.m[10] = rot.m[8];  t.m[11] = 0;
	t.m[12] = 0; t.m[13] = 0; t.m[14] = 0; t.m[15] = 1;
	return t;
}

ALPHA_KERNEL_FUNC void Transpose(AxMatrix4x4& matx)
{
	float tmp;
	tmp = matx.m[1];  matx.m[1] = matx.m[4];  matx.m[4] = tmp;
	tmp = matx.m[2];  matx.m[2] = matx.m[8];  matx.m[8] = tmp;
	tmp = matx.m[3];  matx.m[3] = matx.m[12];  matx.m[12] = tmp;
	tmp = matx.m[6];  matx.m[6] = matx.m[9];  matx.m[9] = tmp;
	tmp = matx.m[7];  matx.m[7] = matx.m[13];  matx.m[13] = tmp;
	tmp = matx.m[11]; matx.m[11] = matx.m[14];  matx.m[14] = tmp;
}														   

ALPHA_KERNEL_FUNC AxVector3 operator*=(AxVector3& vec,AxMatrix4x4 mat)
{
	AxVector3 tmp= vec;
	vec.x = mat.m[0] * tmp.x + mat.m[4] * tmp.y + mat.m[8] * tmp.z + mat.m[12];
 	vec.y = mat.m[1] * tmp.x + mat.m[5] * tmp.y + mat.m[9] * tmp.z + mat.m[13];
 	vec.z = mat.m[2] * tmp.x + mat.m[6] * tmp.y + mat.m[10]* tmp.z + mat.m[14];
	return vec;
}
ALPHA_KERNEL_FUNC AxVector3 operator*(AxVector3 vec, AxMatrix4x4 mat)
{
	AxVector3 tmp = vec;
	vec.x = mat.m[0] * tmp.x + mat.m[4] * tmp.y + mat.m[8] * tmp.z + mat.m[12];
	vec.y = mat.m[1] * tmp.x + mat.m[5] * tmp.y + mat.m[9] * tmp.z + mat.m[13];
	vec.z = mat.m[2] * tmp.x + mat.m[6] * tmp.y + mat.m[10] * tmp.z + mat.m[14];
	return vec;
}

ALPHA_KERNEL_FUNC void SetTransform4x4(AxMatrix4x4& m4x4,const AxVector3& ts)
{
	m4x4.m[12] = ts.x; m4x4.m[13] = ts.y; m4x4.m[14] = ts.z;
}

ALPHA_KERNEL_FUNC AxMatrix3x3 EulerToMatrix3x3_XYZ(AxVector3 eulerXYZ)
{
	eulerXYZ *= ALPHA_DEGREE_TO_RADIUS;
	AxMatrix3x3 rotateX = MakeMat3x3(
		1, 0, 0,
		0, cos(eulerXYZ.x), sin(eulerXYZ.x),
		0, -sin(eulerXYZ.x), cos(eulerXYZ.x));

	AxMatrix3x3 rotateY = MakeMat3x3(
		cos(eulerXYZ.y), 0, -sin(eulerXYZ.y),
		0, 1, 0,
		sin(eulerXYZ.y), 0, cos(eulerXYZ.y));

	AxMatrix3x3 rotateZ = MakeMat3x3(
		cosf(eulerXYZ.z), sinf(eulerXYZ.z), 0,
		-sinf(eulerXYZ.z), cosf(eulerXYZ.z), 0,
		0, 0, 1);
	AxMatrix3x3 rotate3x3 = rotateX * rotateY  * rotateZ;
	return rotate3x3;
}

template<typename T>
ALPHA_KERNEL_FUNC AxMatrix4x4T<T> MakeMat4x4() {
	AxMatrix4x4T<T> m;
	for (int i = 0; i < 16; ++i) m[i] = 0;
	return m;
}

namespace AlphaCore
{
	namespace Math
	{
		ALPHA_KERNEL_FUNC AxMatrix4x4 MakeTransform(AxVector3 transform, AxVector3 Euler, AxVector3 scaler)
		{
			AxMatrix3x3 rot = EulerToMatrix3x3_XYZ(Euler);
			AxMatrix4x4 t=  MakeMat4x4(rot);
			SetTransform4x4(t, transform);
			return t;
		}

		ALPHA_KERNEL_FUNC AxMatrix4x4 ToLeftOrRightHand(const AxMatrix4x4& m)
		{
			AxMatrix4x4 ret;

			ret.m[0] = m.m[0]; 	 ret.m[1] = m.m[1]; 	ret.m[2] = m.m[2];    ret.m[3]  =  m.m[3] ;
			ret.m[4] = m.m[8]; 	 ret.m[5] = m.m[9]; 	ret.m[6] = m.m[10];	  ret.m[7]  =  m.m[7] ;
			ret.m[8] = m.m[4]; 	 ret.m[9] = m.m[5]; 	ret.m[10] = m.m[6];	  ret.m[11] =  m.m[11];
			ret.m[12] = m.m[12]; ret.m[13] = m.m[13]; 	ret.m[14] = m.m[14];  ret.m[15] =  m.m[15];
			
			return ret;
		}


		ALPHA_KERNEL_FUNC void LookAtMatrix_LHC(AxVector3 eye ,AxVector3 lookAt,AxVector3 up, AxMatrix4x4& mat)
		{
			/*
			AxVector3 zaxis = Normalize(lookAt - eye);
			AxVector3 xaxis = Normalize(Cross(up, zaxis));
			AxVector3 yaxis = Cross(zaxis, xaxis);
 			mat.m[0] = xaxis.x;			 mat.m[0] = yaxis.x;		  mat.m[0] = zaxis.x;			mat.m[0] = 0;
			mat.m[0] = xaxis.y;			 mat.m[0] = yaxis.y;		  mat.m[0] = zaxis.y;			mat.m[0] = 0;
			mat.m[0] = xaxis.z;			 mat.m[0] = yaxis.z;		  mat.m[0] = zaxis.z;			mat.m[0] = 0;
			mat.m[0] = -Dot(xaxis, eye); mat.m[0] = -Dot(yaxis, eye); mat.m[0] = -Dot(zaxis, eye);  mat.m[0] = 1;
			*/
		}
	}
}

ALPHA_KERNEL_FUNC void PrintInfo(const char* Head,const AxMatrix4x4& m4)
{
	printf("%s : [%f,%f,%f,%f] \n", Head, m4.m[0], m4.m[1], m4.m[2], m4.m[3]);
	printf("     [%f,%f,%f,%f] \n", m4.m[4], m4.m[5], m4.m[6], m4.m[7]);
	printf("     [%f,%f,%f,%f] \n", m4.m[8], m4.m[9], m4.m[10], m4.m[11]);
	printf("     [%f,%f,%f,%f] \n", m4.m[12], m4.m[13], m4.m[14], m4.m[15]);
}

#endif
