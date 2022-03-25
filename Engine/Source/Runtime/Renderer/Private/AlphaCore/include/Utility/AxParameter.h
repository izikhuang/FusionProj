#ifndef __AX_PARAMETER_H__
#define __AX_PARAMETER_H__

#include "AxStorage.h"
#include <AxDataType.h>



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

	virtual std::string GetParamName() { return ""; };
};



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

	template<class BakeType,bool isColor,int BakeSize>
	struct RawData
	{
		bool IsColor = isColor;
		BakeType LookUpTableRaw[BakeSize];
	};
private:

	bool m_bIsColorRamp;
	AxBufferF* m_Datas;
};

typedef AxRamp::RawData<AxFp32,	   false, 64>	AxRampCurveRawData64;
typedef AxRamp::RawData<AxFp32,	   false, 128>	AxRampCurveRawData128;
typedef AxRamp::RawData<AxColorRGBA8, true,  64>	AxRampColorRawData64;
typedef AxRamp::RawData<AxColorRGBA8, true,  128>	AxRampColorRawData128;

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
	virtual std::string GetParamName() const { return "";/* return m_sName;*/ };
	 
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

	std::string ToString() const { return "h"; };
	/*
	{
		std::stringstream sstr;
		AX_FOR_I(m_Data.size())
		{
			sstr << m_Data[i];
			if (i != m_Data.size() - 1)
				sstr << " , ";
		}
		return sstr.str();
	}
	//*/
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

template<typename T>
inline std::ostream& operator <<(std::ostream& out, const AxParameterT<T>& other)
{
	std::cout << other.GetParamName()<<" : "<< other.ToString();
	return out;
}


#endif
