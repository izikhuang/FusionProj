#ifndef __ALPHA_CORE_UTILITY_IMAGE_H__
#define __ALPHA_CORE_UTILITY_IMAGE_H__

#include <string>
#include <Grid/AxFieldBase2D.h>
#include <Math/AxVectorHelper.h>

namespace AlphaCore
{
	namespace Image
	{
		void SaveAsTga(std::string path,AxImageRGBA8* img);
		void GenerateTestImage(AxImageRGBA8* img);
	}
}

namespace AlphaCore
{
	namespace Image
	{
		namespace Internal
		{
			ALPHA_KERNEL_FUNC void SetColor(const uInt32& x, const uInt32& y, AxColorRGBA8 color, AxColorRGBA8* colorRaw,const AlphaCore::Desc::AxField2DInfo& info)
			{
				AxVector2UI res = info.Resolution;
				colorRaw[res.x * y + x] = color;
			}
			ALPHA_KERNEL_FUNC void SetColor(const uInt32& x, const uInt32& y, AxColorRGBA8 color, AxColorRGBA8* colorRaw, const AxVector2UI& res)
			{
 				colorRaw[res.x * y + x] = color;
			}

		}


		void SetColor(AxColorRGBA8 color, AxColorRGBA8* colorRaw, const AlphaCore::Desc::AxField2DInfo& info);
	}
}

#endif // !ALPHA_CORE_IMAGE_H__
