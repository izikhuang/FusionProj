#ifndef __ALPHA_CORE_MACRO_H__
#define __ALPHA_CORE_MACRO_H__

/*
 * Avoid using uint64.
 * The extra bit of precision is NOT worth the cost in pain and suffering
 * induced by use of unsigned.
 */
#if defined(WIN32)

	typedef __int64				Int64;
	typedef unsigned __int64	uInt64;

#elif defined(MBSD)

	// On MBSD, int64/uint64 are also defined in the system headers so we must
	// declare these in the same way or else we get conflicts.
	#include <stdint.h>
	typedef int64_t				Int64;
	typedef uint64_t			uInt64;

#elif defined(AMD64)

	typedef long				Int64;
	typedef unsigned long		uInt64;

#else

	typedef long long			Int64;
	typedef unsigned long long	uInt64;

#endif

/*
 * The problem with int64 is that it implies that it is a fixed 64-bit quantity
 * that is saved to disk. Therefore, we need another integral type for
 * indexing our arrays.
 */


typedef Int64	exInt;
typedef float	axfpreal;
typedef float	Float32;
typedef double	Float64;
typedef unsigned int  uInt32;
typedef unsigned char Byte;


#ifdef ALPHA_CUDA
#include <cuda_runtime.h>
#define ALPHA_KERNEL_FUNC inline __device__ __host__ 
#define ALPHA_CUDA_THREAD_ID  __umul24(blockIdx.x, blockDim.x) + threadIdx.x
#define ALPHA_SPMD 

#else
#define ALPHA_KERNEL_FUNC inline 
#define ALPHA_SPMD 

#endif


#define ALPHA_EMPTY_FUNCTION_LOG	//std::cout << __FUNCTION__ << "  Empty " << std::endl
#define ALPHA_DEGREE_TO_RADIUS 0.017453292519444f
//
// AlphaCore use the 'dynamic properties' architecture design 
//
//
namespace AlphaProperty
{
	static const char* EngineInfo = 
		" -----------------------------------------------------------------------------------------\n"
		" 																							\n"
		" 																							\n"
		"     //\\   ||     ||======|| ||      ||   //\\    =======  ========  ||=====\\  ========	\n"
		"    //==\\  ||     ||======|| ||======||  //==\\   ||       ||    ||  ||=====//  ||====	\n"
		"   //    \\ ====== ||         ||      || //    \\  =======  ========  ||     \\  =========	\n"
		"																							\n"
		"   Data Orient Process & Dyanmic Properties architeture for multi-physics simultion    	\n"
		" \n"
		"                       °¢¶û·¨ÄÚºË  ¥¢¥ë¥Õ¥¡¥³¥¢  [ version alpha 0.0.1 ]						\n"
		"  \n"
		"                             Get started breaking the row ...								\n"
		" 																							\n"
		" -----------------------------------------------------------------------------------------\n"
		"\n";
		
	static const char* Position			 = "P";
	static const char* PtVel			 = "v";
	static const char* Accelerate		 = "accel";
 	static const char* Stiffness		 = "stiffness";
	static const char* RestLength		 = "restlength";
	static const char* Mass				 = "mass";
										 
	static const char* AdvectTmp		 = "advectTmp";
	static const char* AdvectTmp2		 = "advectTmp2";

	static const char* DensityField  	 = "density";
	static const char* DensityField2	 = "density2";
  	static const char* VelField			 = "vel";
	static const char* VelField2		 = "vel2";
 	static const char* TempratureField	 = "temperature";
	static const char* TempratureField2  = "temperature2";
 	static const char* DivregenceField   = "divergence";
	static const char* VelDivField		 = "velDiv";

	static const char* DivregenceField2  = "divergence2";
 	static const char* CurlField		 = "curl";
	static const char* CurlField2		 = "curl2";
 	static const char* PressureField	 = "presure";
	static const char* PressureField2	 = "presure2";
	static const char* GradientField     = "gradient";
	static const char* GradientField2    = "gradient2";
										 
	static const char* BurnField		 = "burn";
	static const char* HeatField		 = "heat";
	static const char* HeatField2		 = "heat2";
	static const char* TempDivField		 = "tempDiv";
	static const char* ExplosiveDivField = "expDiv";
	static const char* FuelField	     = "fuel";
 

}



#endif