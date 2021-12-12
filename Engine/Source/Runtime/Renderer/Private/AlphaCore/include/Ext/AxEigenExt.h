#ifndef __ALPHA_CORE_EIGEN_EXT_H__
#define __ALPHA_CORE_EIGEN_EXT_H__

#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::Matrix<float, 3, 3>				EigenMatrix3x3F;
typedef Eigen::Matrix<float, 2, 2>				EigenMatrix2x2F;
typedef Eigen::SparseMatrix<float>				EigenSparseMatrixF32;
typedef Eigen::Matrix<float, 3, 1, 0, 3, 1>		EigenVector3F;
typedef Eigen::Matrix<float, 2, 1, 0, 2, 1>		EigenVector2F;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> EigenVectorXF32;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> EigenDataBufferF32;

class SparseMatrixTripletF32 : public Eigen::Triplet<float, int>
{
public:
	SparseMatrixTripletF32()
	{
		m_row = 0; m_col = 0; m_value = 0;
	}

	SparseMatrixTripletF32(const int& i, const int& j, const float& v = 0)
	{
		m_row = i; m_col = j; m_value = v;
	}

	void Assign(int c, int r, float val)
	{
		this->m_col = c;
		this->m_row = r;
		this->m_value = val;
	}

	void SetValue(float val)
	{
		this->m_value = val;
	}

};

class SparseMatrixTripletF64 : public Eigen::Triplet<double, int>
{
public:
	SparseMatrixTripletF64()
	{
		m_row = 0; m_col = 0; m_value = 0;
	}

	SparseMatrixTripletF64(const int& i, const int& j, const double& v = 0)
	{
		m_row = i; m_col = j; m_value = v;
	}

	void Assign(int c, int r, double val)
	{
		this->m_col = c;
		this->m_row = r;
		this->m_value = val;
	}

	void SetValue(double val)
	{
		this->m_value = val;
	}

};

#endif 
