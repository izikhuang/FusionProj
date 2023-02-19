#ifndef __AX_MULTI_GRID_SOLVER_H__
#define __AX_MULTI_GRID_SOLVER_H__

#include "GridDense/AxFieldBase3D.h"

class AxMultiGridSolver
{
public:
	AxMultiGridSolver(
		AxInt32 maxLevel = 4,
		AxUInt32 smooth = 2,
		AxUInt32 bottomSmooth = 8)
	{
		SetMultiGridParams(maxLevel, smooth, bottomSmooth);
	}

	void SolveMGVCycle(AxScalarFieldF32* rhs,
		AxScalarFieldF32* x,
		AxScalarFieldF32* temp);

	void SolveMGVCycleDevice(AxScalarFieldF32* rhs,
		AxScalarFieldF32* x,
		AxScalarFieldF32* temp);

    // CPU, t = r - Az
	void Residual(AxScalarFieldF32* zCurrent, AxScalarFieldF32* rCurrent, AxScalarFieldF32* tCurrent);
	
	void SetMultiGridParams(
		AxInt32 maxLevel,
		AxUInt32 smooth,
		AxUInt32 bottomSmooth
	)
	{
		SetMaxLevel(maxLevel);
		SetSmooth(smooth);
		SetBottomSmooth(bottomSmooth);
	}
	
	AxInt32 GetMaxLevel() { return m_iMaxLevel; }
	void SetMaxLevel(AxInt32 maxLevel) { m_iMaxLevel = maxLevel; }
	AxUInt32 GetSmooth() { return m_iSmooth; }
	void SetSmooth(AxUInt32 smooth) { m_iSmooth = smooth; }
	AxUInt32 GetBottomSmooth() { return m_iBottomSmooth; }
	void SetBottomSmooth(AxUInt32 bottomSmooth) { m_iBottomSmooth = bottomSmooth; }

	void SetAccelerateMark(bool active)
	{
		m_bUseFastKernel = active;
	}

private:
	// id = -1				=> original field
	// id = 0 ~ maxLevel-1	=> subfield
	AxScalarFieldF32* _getSubField(AxScalarFieldF32* fieldMG, AxInt32 id);

protected:
	AxInt32 m_iMaxLevel;
	AxUInt32 m_iSmooth;
	AxUInt32 m_iBottomSmooth;
	bool m_bUseFastKernel;
};

#endif