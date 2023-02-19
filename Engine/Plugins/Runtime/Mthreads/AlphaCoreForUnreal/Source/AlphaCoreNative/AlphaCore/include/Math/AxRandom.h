#ifndef __AX_RANDOM_H__
#define __AX_RANDOM_H__

#include "AxDataType.h"
#include "AxMacro.h"
#include "AxDataType.ShareCode.h"

namespace AlphaCore
{
    namespace Math
    {
        ALPHA_SHARE_FUNC AxFp32 FloorIL(AxFp32 val)
        {
            AxUInt32 tmp;
            AxUInt32 shift;
            AxFp32 tmp_f = 0.0f;
            AxFp32 inVal = val;
            tmp = AlphaCore::ToInt32(&inVal);
            shift = (tmp >> 23) & 0xff;

            if (shift < 0x7f)
            {
                tmp_f = (tmp > 0x80000000) ? -1.0F : 0.0F;
            }
            else if (shift < 0x96)
            {
                AxUInt32 mask = 0xffffffff << (0x96 - shift);
                if (tmp & 0x80000000)
                {
                    if ((tmp & ~mask) & 0x7fffff)
                    {
                        tmp &= mask;
                        tmp_f = AlphaCore::ToFp32(&tmp) - 1;
                    }
                }
                else
                {
                    AxInt32 t = tmp & mask;
                    tmp_f = AlphaCore::ToFp32(&t);
                }
            }
            return tmp_f;
        }

        /// 
        /// Consistent integer hash with the HDK.
        ///  + SYSwang_inthash
        ///  + 
        /// 
        ALPHA_SHARE_FUNC AxUInt32 Int32Hash(AxUInt32 key)
        {
            key += ~(key << 16);
            key ^= (key >> 5);
            key += (key << 3);
            key ^= (key >> 13);
            key += ~(key << 9);
            key ^= (key >> 17);
            return key;
        }

        /// Generates a uniform random number in the 0-1 range from the given seed and
        /// updates the seed.
        ALPHA_SHARE_FUNC AxFp32 FastRandom(AxInt32* seed)
        {
            AxInt32 temp;
            *seed = (*seed) * 1664525 + 1013904223;
            temp = 0x3f800000 | (0x007fffff & (*seed));
            AxFp32* fp = (AxFp32*)&temp;
            return (*fp) - 1.0f;
        }

        ALPHA_SHARE_FUNC AxVector3 FastRandomVector3(AxInt32* seed)
        {
            *seed = (*seed) * (*seed) + (*seed) * 3;
            AxVector3 pos;
            pos.x = Math::FastRandom(seed);
            pos.y = Math::FastRandom(seed);
            pos.z = Math::FastRandom(seed);
            return pos;
        }

    }
}

#define HASH4I(x,y,z,w)        \
    ((x^0xffff3ce3) * (y^0xffff7ba5) * (z^0xffffd169) * (w^0xffff0397))

#define HASH4(x, y, z, w)      \
    HASH4I((int)AlphaCore::Math::FloorIL(x), \
           (int)AlphaCore::Math::FloorIL(y), \
           (int)AlphaCore::Math::FloorIL(z), \
           (int)AlphaCore::Math::FloorIL(w))


#endif //
