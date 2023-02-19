#ifndef __AXBLAS__H__
#define __AXBLAS__H__

#include <cmath>
#include <AxLog.h>
#include <AxTimeTick.h>
#include <AxVectorHelper.h>

#define __FSA_ARCH_PROTOCOL_AXBLAS__H__ 1
#if __FSA_ARCH_PROTOCOL_AXBLAS__H__ 1
namespace AlphaCore
{
    namespace BLAS
    {
        ALPHA_SPMD_FUNC void axpy(
            AxUInt32 n,
            const AxFp32 alpha,
            const AxFp32 *x,
            const AxInt32 incx,
            AxFp32 *y,
            const AxInt32 incy);

        ALPHA_SPMD_FUNC void scal(
            AxUInt32 n,
            const AxFp32 alpha,
            AxFp32 *x,
            const AxInt32 incx);

        ALPHA_SPMD_FUNC void copy(
            AxUInt32 n,
            const AxFp32 *x,
            const AxInt32 incx,
            AxFp32 *y,
            const AxInt32 incy);

        ALPHA_SPMD_FUNC void dot(
            AxUInt32 n,
            const AxFp32 *x,
            const AxInt32 incx,
            const AxFp32 *y,
            const AxInt32 incy,
            AxFp32 *result);

        ALPHA_SPMD_FUNC void nrm2(
            AxUInt32 n,
            const AxFp32 *x,
            const AxInt32 incx,
            AxFp32 *result);

        ALPHA_SPMD_FUNC void sum(
            AxUInt32 n,
            const AxFp32 *x,
            const AxInt32 incx,
            AxFp32 *result);

        ALPHA_SPMD_FUNC void add(
            AxUInt32 n,
            const AxFp32 alpha,
            AxFp32 *x,
            const AxInt32 incx);

    }

    namespace BLAS
    {
        namespace CUDA
        {
            ALPHA_SPMD_FUNC void axpy(
                AxUInt32 n,
                const AxFp32 alpha,
                const AxFp32 *x,
                const AxInt32 incx,
                AxFp32 *y,
                const AxInt32 incy,
                AxUInt32 blockSize = 512);

            ALPHA_SPMD_FUNC void scal(
                AxUInt32 n,
                const AxFp32 alpha,
                AxFp32 *x,
                const AxInt32 incx,
                AxUInt32 blockSize = 512);

            ALPHA_SPMD_FUNC void copy(
                AxUInt32 n,
                const AxFp32 *x,
                const AxInt32 incx,
                AxFp32 *y,
                const AxInt32 incy,
                AxUInt32 blockSize = 512);

            ALPHA_SPMD_FUNC void dot(
                AxUInt32 n,
                const AxFp32 *x,
                const AxInt32 incx,
                const AxFp32 *y,
                const AxInt32 incy,
                AxFp32 *result,
                AxUInt32 blockSize = 128);

            ALPHA_SPMD_FUNC void nrm2(
                AxUInt32 n,
                const AxFp32 *x,
                const AxInt32 incx,
                AxFp32 *result,
                AxUInt32 blockSize = 128);

            ALPHA_SPMD_FUNC void sum(
                AxUInt32 n,
                const AxFp32 *x,
                const AxInt32 incx,
                AxFp32 *result,
                AxUInt32 blockSize = 128);

            ALPHA_SPMD_FUNC void add(
                AxUInt32 n,
                const AxFp32 alpha,
                AxFp32 *x,
                const AxInt32 incx,
                AxUInt32 blockSize = 512);

        }
    }

    namespace BLAS
    {
        namespace DX
        {
            ALPHA_SPMD_FUNC void axpy(
                AxUInt32 n,
                const AxFp32 alpha,
                const AxFp32 *x,
                const AxInt32 incx,
                AxFp32 *y,
                const AxInt32 incy,
                AxUInt32 blockSize = 512);

            ALPHA_SPMD_FUNC void scal(
                AxUInt32 n,
                const AxFp32 alpha,
                AxFp32 *x,
                const AxInt32 incx,
                AxUInt32 blockSize = 512);

            ALPHA_SPMD_FUNC void copy(
                AxUInt32 n,
                const AxFp32 *x,
                const AxInt32 incx,
                AxFp32 *y,
                const AxInt32 incy,
                AxUInt32 blockSize = 512);

            ALPHA_SPMD_FUNC void dot(
                AxUInt32 n,
                const AxFp32 *x,
                const AxInt32 incx,
                const AxFp32 *y,
                const AxInt32 incy,
                AxFp32 *result,
                AxUInt32 blockSize = 512);

            ALPHA_SPMD_FUNC void nrm2(
                AxUInt32 n,
                const AxFp32 *x,
                const AxInt32 incx,
                AxFp32 *result,
                AxUInt32 blockSize = 512);

            ALPHA_SPMD_FUNC void sum(
                AxUInt32 n,
                const AxFp32 *x,
                const AxInt32 incx,
                AxFp32 *result,
                AxUInt32 blockSize = 512);

            ALPHA_SPMD_FUNC void add(
                AxUInt32 n,
                const AxFp32 alpha,
                AxFp32 *x,
                const AxInt32 incx,
                AxUInt32 blockSize = 512);

        }
    }

}

#endif //@FSA:[TOKEN]  __FSA_ARCH_PROTOCOL_AXBLAS__H__
#endif