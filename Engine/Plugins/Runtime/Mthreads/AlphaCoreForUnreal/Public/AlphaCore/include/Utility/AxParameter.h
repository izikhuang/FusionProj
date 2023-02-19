#ifndef __AX_PARAMETER_H__
#define __AX_PARAMETER_H__

#include "AxStorage.h"
#include "AxDataType.h"
#include <array>
#include <map>

typedef AxInt32 AxKeyFrame;



class AxIParameterBase
{
public:
	AxIParameterBase(){};
	~AxIParameterBase(){};

	static AxIParameterBase* CreateParameterByTypeToken(std::string token);
	static std::string JsonDataTypeTokenToParamTypeName(std::string token);
	static std::string JsonDataTypeTokenToRawDataTypeName(std::string token);

	virtual void JsonDeserialization(std::string jsonRaw) {};
	virtual void JsonSerialization(std::string jsonRaw) {};
	virtual void PrintParamInfo() {};

	virtual std::string GetParamName() const { return ""; };
};




template<class T>
class AxRamp
{
public:

	enum KeyType
	{
		kConstant,
		kLinear,
		kCatmullRom,
		kMonotoneCubic,
		kBezier,
		kBSpline,
		kHermite
	};

	void RemoveKey();
	void AddKey();

	template<class BakeType, AxUInt32 BakeSize>
	struct RAWDataConstant
	{
		AxFp32 InputMin;
		AxFp32 InputMax;
		AxUInt32 Size = BakeSize;
		BakeType LookUpTableRaw[BakeSize];
	};

	struct RAWData
	{
		AxFp32 InputMin;
		AxFp32 InputMax;
		AxUInt32 BakeSize;
		T* RampDatas;
	};

private:

	bool m_bIsColorRamp;
	AxStorage<T>* m_Datas;
	AxStorage<T>* m_BakedData;

};

typedef AxRamp<AxFp32>::RAWDataConstant<AxFp32, 64>				 AxRampCurveRawData64;
typedef AxRamp<AxFp32>::RAWDataConstant<AxFp32, 128>			 AxRampCurveRawData128;
typedef AxRamp<AxColorRGBA8>::RAWDataConstant<AxColorRGBA8, 64>	 AxRampColorRawData64;
typedef AxRamp<AxColorRGBA8>::RAWDataConstant<AxColorRGBA8, 128> AxRampColorRawData128;
typedef AxRamp<AxColorRGBA8>::RAWDataConstant<AxColorRGBA8, 32>	 AxRampColor32RAWData;
typedef AxRamp<AxFp32>::RAWDataConstant<AxFp32, 32>				 AxRampCurve32RAWData;

static inline AxRamp<AxFp32>::RAWDataConstant<AxFp32, 32> MakeDefaultRampCurve32(
	AxFp32* bakeData32 = nullptr,
	AxFp32 min = 0.0f,
	AxFp32 max = 1.0f)
{
	AxRampCurve32RAWData ret;
	ret.InputMin = min;
	ret.InputMax = max;
	AX_FOR_I(32)
	{
		if (bakeData32 == nullptr)
		{
			ret.LookUpTableRaw[i] = (AxFp32)i / 31.0f;
		}else {
			ret.LookUpTableRaw[i] = bakeData32[i];
		}
	}
	return ret; 
}
template <typename T>
class AxParameterT: public AxIParameterBase, public AxStorage<T>
{
public:
	AxParameterT(const AxParameterT<T>& other)
	{
		this->Resize(other.Size());

	}
	AxParameterT()
	{
		this->Resize(1);
		m_BakeStartFrame = 1;
	}
	~AxParameterT() {};
	struct RawData
	{
		T*		 ParamRaw;
		AxInt32	 BakedStartFrame;
		AxUInt32 ParamSize;
	};

	RawData GetBakedParamRawData()
	{
		RawData raw;
		raw.ParamRaw = this->GetDataRaw();
		raw.BakedStartFrame = m_BakeStartFrame;
		return raw;
	}

	RawData GetBakedParamRawDataDevice()
	{
		RawData raw;
		raw.ParamRaw = this->GetDataRawDevice();
		raw.BakedStartFrame = m_BakeStartFrame;
		return raw;
	}

	void SetBakeStartFrame(AxInt32 bakeStartFrame) { m_BakeStartFrame = bakeStartFrame; };

	virtual void JsonDeserialization(std::string jsonRaw);
	virtual void JsonSerialization(std::string jsonRaw);

	virtual void PrintParamInfo()
	{
		//AX_INFO("AxParameterT Name : {}",m_sName);
	}


	AxUInt32 GetNumKeyFrames() { return m_KeyFrameList.size(); }
	AxUInt32 GetNumBakedFrames() { return this->Size(); }

	bool IsDefault();
	T GetDefaultValue() { return m_DefaultValue; };
	virtual std::string GetParamName() const{ 
		return this->m_sName;
	};
	 
	T GetParamValueByFloatFrame(AxFp32 floatFrame)
	{
		return m_DefaultValue;
		/*
		AxInt32 off = frame - m_BakeStartFrame;
		AxInt32 size = this->Size();
		off = std::min(size, std::max(off, 0));
 		return this->Get(off);
		*/
	}

	T GetParamValueByFrame(AxInt32 frame)
	{
		AxInt32 off = frame - m_BakeStartFrame;
		AxInt32 size = this->Size() - 1;
		off = std::min(size, std::max(off, 0));
		return this->Get(off);
	}

	T GetParamValue()
	{
		return GetParamValueByFrame(0);
	}

	std::string ToString() const
	{
		std::stringstream sstr;
		AX_FOR_I(this->m_Data.size())
		{
			sstr << this->m_Data[i];
			if (i != this->m_Data.size() - 1)
				sstr << " , ";
		}
		return sstr.str();
	}

private:

	T m_DefaultValue;
	T m_Min;
	T m_Max;
	AxInt32 m_BakeStartFrame;
	std::string m_sLabel;
	std::vector<AxKeyFrame> m_KeyFrameList;

};


typedef AxParameterT<AxFp32>			AxFloatParam;
typedef AxParameterT<AxVector2>			AxVector2FParam;
typedef AxParameterT<AxVector3>			AxVector3FParam;
typedef AxParameterT<AxVector4>			AxVector4FParam;
typedef AxParameterT<AxInt32>			AxIntParam;
typedef AxParameterT<AxVector2I>		AxVector2IParam;
typedef AxParameterT<AxUChar>			AxToggleParam;
typedef AxParameterT<std::string>		AxStringParam;
typedef AxRamp<AxFp32>					AxRampParam;

template<typename T>
inline std::ostream& operator <<(std::ostream& out, const AxParameterT<T>& other)
{
	std::cout << other.GetParamName()<<" : "<< other.ToString();
	return out;
}


inline std::ostream& operator <<(std::ostream& out, const AxRampCurve32RAWData& other)
{
	out << "Ramp@AsRaw:";
	AX_FOR_I(int(other.Size)) {
		out << other.LookUpTableRaw[i];
		if (i != other.Size - 1)
			out << ",";
	}
	out << "|" << "Range:" << other.InputMin << "," << other.InputMax;
	out << std::endl;
	return out;
}


class AxSimParameterMap
{
public:
	AxFp32 GetFloat(std::string name);
	std::string GetString(std::string name);
	AxInt32 GetInt(std::string name);
	AxVector2 GetVector2(std::string name);
	AxVector3 GetVector3(std::string name);
	AxVector4 GetVector4(std::string name);
	void GetRamp(std::string name);
	bool GetToggle(std::string name);

private:

	std::map<std::string, AxFloatParam*> m_FloatParameters;
	std::map<std::string, AxStringParam*> m_StrParameters;
	std::map<std::string, AxIntParam*> m_IntParameters;
	std::map<std::string, AxVector2FParam*> m_V2FParameters;
	std::map<std::string, AxVector3FParam*> m_V3FParameters;
	std::map<std::string, AxVector4FParam*> m_V4FParameters;
	std::map<std::string, AxToggleParam*> m_ToggleParam;
};


#endif
