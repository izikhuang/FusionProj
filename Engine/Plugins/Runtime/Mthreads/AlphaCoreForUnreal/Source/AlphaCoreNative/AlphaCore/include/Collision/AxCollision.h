#ifndef __AXCOLLISION__H__
#define __AXCOLLISION__H__

#if __FSA_ARCH_PROTOCOL_AXCOLLISION__H__ 1
namespace AlphaCore
{
namespace Collision  
{
 ALPHA_SPMD_FUNC void MarkStaticContactMass (
AxContact contact,
AxBufferF* mass,
AxBufferF* mass2,
AxFp32 coeff) ;

  }

namespace Collision  
{
  namespace CUDA 
{
 ALPHA_SPMD_FUNC void MarkStaticContactMass (
AxContact contact,
AxBufferF* mass,
AxBufferF* mass2,
AxFp32 coeff,
AxUInt32 blockSize = 512) ;

  
}
 }



namespace Collision  
{
  namespace DX 
{
 ALPHA_SPMD_FUNC void MarkStaticContactMass (
AxContact contact,
AxBufferF* mass,
AxBufferF* mass2,
AxFp32 coeff,
AxUInt32 blockSize = 512) ;

  
}
 }



}
#endif //@FSA:[TOKEN]  __FSA_ARCH_PROTOCOL_AXCOLLISION__H__
#endif