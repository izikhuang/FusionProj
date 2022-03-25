#ifndef __AX_STORAGEHELPER_SHARECODE_H__
#define __AX_STORAGEHELPER_SHARECODE_H__

#include <AxMacro.h>
#include <AxDataType.h>

namespace AlphaCore 
{
	namespace StorageHelper 
	{
		namespace ShareCode 
		{
			template<typename T>
			ALPHA_SHARE_FUNC void MakeIdenityBuffer(AxUInt32 idx,T* idenityInputRaw)
			{
				idenityInputRaw[idx] = idx;
			}
		}
	}//@namespace end of : SolidUtility
}
#endif