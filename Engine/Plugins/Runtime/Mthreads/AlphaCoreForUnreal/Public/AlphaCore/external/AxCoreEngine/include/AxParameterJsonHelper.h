
#ifndef __AX_PARAMETER_JSON_HELPER_H__
#define __AX_PARAMETER_JSON_HELPER_H__

#include "Utility/AxParameter.h"
#include "thirdParty/rapidjson/rapidjson.h"
#include "thirdParty/rapidjson/document.h"
#include "thirdParty/rapidjson/stringbuffer.h"
#include "thirdParty/rapidjson/writer.h"

#include <map>
namespace AlphaCore
{
	namespace JsonHelper
	{
		void AxFloatParamDeserilization(AxFloatParam& parm ,const rapidjson::Value& val);
		void AxVector3FParamDeserilization(AxVector3FParam& parm, const rapidjson::Value& val);
		void AxVector4FParamDeserilization(AxVector4FParam& parm, const rapidjson::Value& val);
		void AxStringParamDeserilization(AxStringParam& parm, const rapidjson::Value& val);
		void AxIntParamDeserilization(AxIntParam& parm, const rapidjson::Value& val);
		void AxVector2IParamDeserilization(AxVector2IParam& parm, const rapidjson::Value& val);
		void AxVector2FParamDeserilization(AxVector2FParam& parm, const rapidjson::Value& val);

		void AxToggleParamDeserilization(AxToggleParam& parm, const rapidjson::Value& val);

		//void Deserilization();
		void AxRampCurve32RAWDataDeserilization(AxRampCurve32RAWData& ramp, const rapidjson::Value& val);

		std::string RapidJsonToString(const rapidjson::Value& val);
	}
}



#endif // !__AX_PARAMETER_JSON_HELPER_H__
