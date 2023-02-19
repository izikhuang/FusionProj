#ifndef __AX_GEO_HELPER_H__
#define __AX_GEO_HELPER_H__

#include "AxGeo.h"


namespace AlphaCore
{
	namespace GeometryHelper
	{
		void AxGeometryToOpenVDB(AxGeometry* geo, std::string outputPath);
		void AxGeometryToOpenVDB(AxGeometry* geo, std::string outputPath, std::string fieldName);
		void AxGeometryToOpenVDB(AxGeometry* geo, std::string outputPath, std::vector<std::string> fieldNames);
		void OpenVdbToAxGeometry(AxGeometry* geo, std::string vdbFilePath);
	}
}

#endif
