#ifndef __ALPHA_CORE_HOUDINI_JSON_IO_H__
#define __ALPHA_CORE_HOUDINI_JSON_IO_H__

#include <string>
#include <Utility/AxDescrition.h>
#include <Grid/AxFieldBase3D.h>


#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
 

namespace AlphaUtility
{
	namespace Hou
	{

		bool LoadGeo(std::string filePath);

		bool LoadVolumeFromJSON(std::string filePath,std::vector<AxScalarFieldF32*>& ret);
		bool UpdateVolumeFromJSON(std::string filePath, std::vector<AxScalarFieldF32*>& ret);
		bool UpdateVolumeFromJSON(std::string filePath, AxScalarFieldF32* dstField);

		bool WriteVolumeToJSON(std::string filePath, std::vector<AxScalarFieldF32*>& ret);
		bool WriteVolumeToJSON(std::string filePath, AxScalarFieldF32* field);
		bool WriteVolumeToJSON(std::string filePath, AxVecFieldF32* field);


		rapidjson::Value ToJsonVolume(AxScalarFieldF32* field, rapidjson::Document::AllocatorType& allocator);

	}
}

#endif