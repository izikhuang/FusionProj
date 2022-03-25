#ifndef  __ALPHA_CORE_EIGEN_EXT_PCG_SOLVER_H__
#define  __ALPHA_CORE_EIGEN_EXT_PCG_SOLVER_H__

#include <Grid/AxFieldBase3D.h>
#include <Ext/AxEigenExt.h>

class AxEigenPCGPressureSolver
{
public:


	void BuildLaplacian3D(AxVector3UI res,float dx,bool neumannBC)
	{
 		std::vector< SparseMatrixTripletF32> triplets;

		int nx = res.x;
		int ny = res.y;
		int nz = res.z;

		float h = dx;
		float h2 = h*h;

 		int n = nx * ny * nz;
 		A.resize(n, n);
		A.setZero();

		std::cout << "dx:" << dx << std::endl;
		std::cout << "voxelNum:" << n << std::endl;

		int slice = nx * ny;
 		for (int index = 0; index < n; index++)
		{
			int k = index / slice;
			int j = (index%slice) / nx;
			int i = index % nx;

			if (i >= 0 && j >= 0 && i < nx&&j < ny)
			{
				// - X Face Boundary
				if (i == 0)
				{
					if (!neumannBC)
						triplets.push_back(SparseMatrixTripletF32(index, index, 1 / h2));
				}
				else
				{
					triplets.push_back(SparseMatrixTripletF32(index, index - 1, -1 / h2));
					triplets.push_back(SparseMatrixTripletF32(index, index, 1 / h2));
				}

				// + X Face Boundary
				if (i == nx - 1) 
				{
					if (!neumannBC)
						triplets.push_back(SparseMatrixTripletF32(index, index, 1 / h2));
				}
				else
				{
					triplets.push_back(SparseMatrixTripletF32(index, index + 1, -1 / h2));
					triplets.push_back(SparseMatrixTripletF32(index, index, 1 / h2));
				}

				// - Y Face Boundary
				if (j == 0) 
				{
					if (!neumannBC)
						triplets.push_back(SparseMatrixTripletF32(index, index, 1 / h2));
				}
				else
				{
					triplets.push_back(SparseMatrixTripletF32(index, index - ny, -1 / h2));
					triplets.push_back(SparseMatrixTripletF32(index, index, 1 / h2));
				}

				// + Y Face Boundary
				if (j == ny-1) 
				{
					if (!neumannBC)
						triplets.push_back(SparseMatrixTripletF32(index, index, 1 / h2));
				}
				else
				{
					triplets.push_back(SparseMatrixTripletF32(index, index + nx, -1 / h2));
					triplets.push_back(SparseMatrixTripletF32(index, index, 1 / h2));
				}

				// - Z Face Boundary
				if (k == 0)
				{
					if (!neumannBC)
						triplets.push_back(SparseMatrixTripletF32(index, index, 1 / h2));
				}
				else
				{
					triplets.push_back(SparseMatrixTripletF32(index, index - slice, -1 / h2));
					triplets.push_back(SparseMatrixTripletF32(index, index, 1 / h2));
				}

				// + Z Face Boundary
				if (k == nz - 1) 
				{
					if (!neumannBC)
						triplets.push_back(SparseMatrixTripletF32(index, index, 1 / h2));
				}
				else
				{
					triplets.push_back(SparseMatrixTripletF32(index, index + slice, -1 / h2));
					triplets.push_back(SparseMatrixTripletF32(index, index, 1 / h2));
				}

			}
		}
		//std::cout << "ret:" << triplets.size() << std::endl;

		A.setFromTriplets(triplets.begin(), triplets.end());
		//std::cout << "PCG Matrix:\n" << A << std::endl;

 		m_RHS.resize(n);
		m_Pressure.resize(n);
		
	}

 
 
	void Solve(float* divergence, float* pressure)
	{
		memcpy(m_RHS.data(), divergence, m_RHS.size() * sizeof(float));
		memcpy(m_Pressure.data(), pressure, m_Pressure.size() * sizeof(float));
 
		//
		//	m_RHS *= -1;
		//
		//Eigen::ConjugateGradient<EigenSparseMatrixF32, Eigen::Lower, Eigen::IncompleteCholesky<float>> pcg;
		Eigen::ConjugateGradient<EigenSparseMatrixF32, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> pcg;

		pcg.setMaxIterations(13000);
		float residual = 0;
		pcg.compute(A);
		m_Pressure = pcg.solve(m_RHS);

		memcpy(pressure, m_Pressure.data(), m_Pressure.size() * sizeof(float));

		auto r = A*m_Pressure - m_RHS;
		float e = r.norm();
		std::cout << "Ax-b e:" << e << std::endl;
		AX_INFO("        - estimated error: {0}", pcg.error());
		AX_INFO("        - #iterations:     {0}", pcg.iterations());
	}

private:
	EigenSparseMatrixF32 A;
	EigenVectorXF32 m_RHS;
	EigenVectorXF32 m_Pressure;

};


#endif 
