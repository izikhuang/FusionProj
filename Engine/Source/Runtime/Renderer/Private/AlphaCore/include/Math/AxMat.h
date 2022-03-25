
#ifndef __AX_MAT_H__
#define __AX_MAT_H__

#include <AxDataType.h>

template<unsigned SIZE, typename T>
struct AxMatT
{	
	T mm[SIZE*SIZE];
};

typedef AxMatT<2, AxFp32> AxMat2x2F;
typedef AxMatT<3, AxFp32> AxMat3x3F;
typedef AxMatT<4, AxFp32> AxMat4x4F;

typedef AxMatT<2, AxFp64> AxMat2x2D;
typedef AxMatT<3, AxFp64> AxMat3x3D;
typedef AxMatT<4, AxFp64> AxMat4x4D;

struct Quat
{
	AxFp32 mm[4];
};

#include <ostream>
inline std::ostream& operator <<(std::ostream& os, Quat& q)
{
	os << "Quat:" << q.mm[0] << "," << q.mm[1] << "," << q.mm[2] << "," << q.mm[3];
	return os;
}



#endif //
