#ifndef __AX_CONJUGATE_GRADIENT_H__
#define __AX_CONJUGATE_GRADIENT_H__

#include "GridDense/AxGridDense.h"

//1
class AxPossionCGSolver
{
public:
	AxPossionCGSolver();
	~AxPossionCGSolver();


	void Solve(AxScalarFieldF32* pressure,AxScalarFieldF32* divergence);
	//without Perconditioner
	void SolveDevice(AxScalarFieldF32* pressure, 
					 AxScalarFieldF32* divergence, 
					 AxScalarFieldF32* wTemp,
					 AxScalarFieldF32* apTemp,
					 AxScalarFieldF32* rTemp,
					 AxInt32 iterations = 1000,
					 AxFp32 residual  = 1e-6);

	//
	void SolverMGP();

	AxInt32 GetIterations() { return m_iIterations; };
	void SetIterations(AxInt32 iters) { m_iIterations = iters; }

	//apply Perconditioner

private:

	
	AxInt32 m_iIterations;
};


#endif 
