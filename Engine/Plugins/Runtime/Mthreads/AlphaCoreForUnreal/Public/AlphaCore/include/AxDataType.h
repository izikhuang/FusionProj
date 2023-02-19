#ifndef __AX_DATA_TYPE_H__
#define __AX_DATA_TYPE_H__

#include <string>
#include "AxMacro.h"

typedef char				AxInt8;
typedef int					AxInt32;
typedef long long			AxInt64;
typedef float				AxReal;
typedef float				AxFp32;
//typedef short				AxFp16;
typedef double				AxFp64;
typedef short				AxInt16;
typedef unsigned char		AxUInt8;
typedef unsigned short		AxUInt16;
typedef unsigned int		AxUInt32;
typedef unsigned long long  AxUInt64;
typedef unsigned char		Byte;
typedef unsigned char		AxUChar;
typedef unsigned short		AxFixed16;
typedef unsigned int		AxFixed32;
typedef unsigned long		AxFixed64;
typedef std::string			AxString;


///////////////////////////////////////////
/// AxFp16 101
///////////////////////////////////////////
struct AxFp16;

ALPHA_SHARE_FUNC float InternalToAxFp32(const unsigned short& h);

ALPHA_SHARE_FUNC unsigned short InternalToAxFp16(const float& f, unsigned int& sign, unsigned int& remainder);



/* fp32/fp16 convert */
ALPHA_SHARE_FUNC AxFp16 ToAxFp16(const float& f);

ALPHA_SHARE_FUNC float ToAxFp32(const AxFp16& v);
/* Make AxFp16 */
ALPHA_SHARE_FUNC AxFp16 MakeAxFp16(const float& f);


/* Some basic arithmetic operations */
/* add */
ALPHA_SHARE_FUNC AxFp16 operator+(const AxFp16& lh, const AxFp16& rh);

ALPHA_SHARE_FUNC AxFp16 operator+(const AxFp16& lh, const float& rh);

ALPHA_SHARE_FUNC AxFp16& operator+=(AxFp16& lh, const AxFp16& rh);

ALPHA_SHARE_FUNC AxFp16& operator+=(AxFp16& lh, const float& rh);

/* subtraction */
ALPHA_SHARE_FUNC AxFp16 operator-(const AxFp16& lh, const AxFp16& rh);

ALPHA_SHARE_FUNC AxFp16 operator-(const AxFp16& lh, const float& rh);

ALPHA_SHARE_FUNC AxFp16& operator-=(AxFp16& lh, const AxFp16& rh);

ALPHA_SHARE_FUNC AxFp16& operator-=(AxFp16& lh, const float& rh);

/* multiplication */
ALPHA_SHARE_FUNC AxFp16 operator*(const AxFp16& lh, const AxFp16& rh);

ALPHA_SHARE_FUNC AxFp16 operator*(const AxFp16& lh, const float& rh);

ALPHA_SHARE_FUNC AxFp16& operator*=(AxFp16& lh, const AxFp16& rh);

ALPHA_SHARE_FUNC AxFp16& operator*=(AxFp16& lh, const float& rh);

/* division */
ALPHA_SHARE_FUNC AxFp16 operator/(const AxFp16& lh, const AxFp16& rh);

ALPHA_SHARE_FUNC AxFp16 operator/(const AxFp16& lh, const float& rh);

ALPHA_SHARE_FUNC AxFp16& operator/=(AxFp16& lh, const AxFp16& rh);

ALPHA_SHARE_FUNC AxFp16& operator/=(AxFp16& lh, const float& rh);


struct AxContext
{
	AxContext(AxFp32 dt = 0.0416f)
	{
		Dt = dt;
		Time = 0.0f;
		Frame = 0.0f;
		FFrame = 0.0f;
	}

	AxFp32 Dt;
	AxFp32 Time;
	AxInt32 Frame;
	AxFp32 FFrame;

};

struct AxStartNum2I
{
	AxInt32 Start;
	AxInt32 Num;
};

ALPHA_SHARE_FUNC AxStartNum2I MakeStartNum2I(AxInt32 start = -1, AxInt32 num = -1)
{
	AxStartNum2I e;
	e.Start = start;
	e.Num = num;
	return e;
}

inline std::ostream& operator <<(std::ostream& os, const AxStartNum2I& rhs)
{
	os << "StartNum Info : Start " << rhs.Start << "  +  " << rhs.Num;
	return os;
}

namespace AlphaCore
{
	enum AxDataType
	{
		kFP16,
		kFP32,
		kFP64,
		kInt8,
		kInt16,
		kInt32,
		kInt64,
		kUInt8,
		kUInt16,
		kUInt32,
		kUInt64,
		kArrayMapDesc,
		kString,
		kInvalidDataType
	};

	enum AxGeoNodeType
	{
		kPoint,
		kPrimitive,
		kVertex,
		kGeoDetail,
		kGeoIndices,
		kInvalidPropertyType
	};

	enum AxIntegrator
	{
		FirstOrderEuler,
		BDF2,
		MidPoint
	};

	enum AxBackendAPI
	{
		CUDA,
		PlayStation,
		DirectX,
		OpenGL,
		OpenCL, // Sudi 3.0
		Vulkan, // 1.2
		iOSMetal,
		CPUx86,
		InvalidDevice
	};

	namespace AxSPMDBackendName
	{
		static const char* CUDA = "cuda";
		static const char* DirectX = "dx";
		static const char* Vulkan = "vulkan";
		static const char* x86 = "x86";
		static const char* OpenCL = "ocl";
	}

	enum AxPrimitiveType
	{
		kPrimVolume,	// 1vertex
		kOpenVDB,		// 1vertex
		kPrimPolyon,	// nVertexs
		kPrimParticle,	// Particle System Point Block
		kSphere,		// 1Vertex
		kTetrahedra,
		kBox,
		kConstraint,
		kPolyLineMark,
		kPrInvalidGeo
	};

	enum AxExecuteType
	{
		kExcPoints,
		kExcPrimitives,
		kExcVertices,
		kExcVoxels,
		kExcTaskGroup,
		kExcSBPoint2PointLink,
		kExcSBPrim2PrimLink,
		kNonExc
	};

	static std::string ExecuteTypeToString(AlphaCore::AxExecuteType type)
	{
		switch (type)
		{
		case AlphaCore::kExcPoints:
			return "kExcPoints";
			break;
		case AlphaCore::kExcPrimitives:
			return "kExcPrimitives";
			break;
		case AlphaCore::kExcVertices:
			return "kExcVertices";
			break;
		case AlphaCore::kExcVoxels:
			return "kExcVoxels";
			break;
		case AlphaCore::kExcTaskGroup:
			return "kExcTaskGroup";
			break;
		case AlphaCore::kExcSBPoint2PointLink:
			return "kExcSBPointLink";
			break;
		case AlphaCore::kExcSBPrim2PrimLink:
			return "kExcSBPrimitiveLink";
			break;
		default:
			return "kNonExc";
			break;
		}
		return "Undefined Execute Type";
	}

	static AlphaCore::AxExecuteType ExecuteTypeFromString(std::string str)
	{
		if (str == "kExcPoints")
			return AlphaCore::kExcPoints;
		if (str == "kExcPrimitives")
			return AlphaCore::kExcPrimitives;
		if (str == "kExcVertices")
			return AlphaCore::kExcVertices;
		if (str == "kExcVoxels")
			return AlphaCore::kExcVoxels;
		if (str == "kExcTaskGroup")
			return AlphaCore::kExcTaskGroup;
		if (str == "kExcSBPoint2PointLink")
			return AlphaCore::kExcSBPoint2PointLink;
		if (str == "kExcSBPrim2PrimLink")
			return AlphaCore::kExcSBPrim2PrimLink;
		return  AlphaCore::AxExecuteType::kNonExc;
	}

	template<class T>
	inline AxUInt32 TypeVecSize() { return 1; }

	template<>
	inline AxUInt32 TypeVecSize<int>() { return 1; }

	template<>
	inline AxUInt32 TypeVecSize<AxUChar>() { return 1; }

	template<>
	inline AxUInt32 TypeVecSize<float>() { return 1; }

	template<>
	inline AxUInt32 TypeVecSize<double>() { return 1; }

	template<>
	inline AxUInt32 TypeVecSize<AxUInt32>() { return 1; }

	template<>
	inline AxUInt32 TypeVecSize<AxPrimitiveType>() { return 1; }

	template<>
	inline AxUInt32 TypeVecSize<std::string>() { return 1; }


	template<class T>
	inline AxDataType TypeID() { return AxDataType::kInvalidDataType; }

	template<>
	inline AxDataType TypeID<int>() { return AxDataType::kInt32; }

	template<>
	inline AxDataType TypeID<float>() { return AxDataType::kFP32; }

	template<>
	inline AxDataType TypeID<double>() { return AxDataType::kFP64; }

	template<>
	inline AxDataType TypeID<AxUInt32>() { return (AxDataType::kUInt32); }

	template<>
	inline AxDataType TypeID<AxPrimitiveType>() { return AxDataType::kInt32; }

	template<>
	inline AxDataType TypeID<AxUChar>() { return AxDataType::kUInt8; }

	template<>
	inline AxDataType TypeID<std::string>() { return AxDataType::kString; }

	template<class T>
	inline AxString TypeName() { return "INVALID_TYPE"; }

	template<>
	inline AxString TypeName<int>() { return "AxInt32"; }

	template<>
	inline AxString TypeName<bool>() { return "AxBool"; }

	template<>
	inline AxString TypeName<float>() { return "AxFp32"; }

	template<>
	inline AxString TypeName<double>() { return "AxFp64"; }

	template<>
	inline AxString TypeName<AxUInt32>() { return "AxUInt32"; }

	template<>
	inline AxString TypeName<AxUChar>() { return "AxUInt8"; }

	template<>
	inline AxString TypeName<AxPrimitiveType>() { return "AxPrimitive"; }

	template<>
	inline AxString TypeName<std::string>() { return "std_string"; }

	inline bool IsIntDataType(AxDataType type)
	{
		if (type == kInt8 || type == kInt16 || type == kInt32 || type == kInt64 ||
			type == kUInt8 || type == kUInt16 || type == kUInt32 || type == kUInt64)
			return true;
		return false;
	}

	inline bool IsFloatDataType(AxDataType type)
	{
		if (type == kFP16 || type == kFP32 || type == kFP64)
			return true;
		return false;
	}

	inline bool IsStringDataType(AxDataType type)
	{
		if (type == kString)
			return true;
		return false;
	}

	static const char* DataTypeToString(AxDataType dataType)
	{
		switch (dataType)
		{
		case AlphaCore::kFP16:
			return "Float16";
		case AlphaCore::kFP32:
			return "Float32";
		case AlphaCore::kFP64:
			return "Float64";
		case AlphaCore::kInt8:
			return "Int8";
		case AlphaCore::kInt16:
			return "Int16";
		case AlphaCore::kInt32:
			return "Int32";
		case AlphaCore::kInt64:
			return "Int64";
		case AlphaCore::kUInt8:
			return "UInt8";
		case AlphaCore::kUInt16:
			return "UInt16";
		case AlphaCore::kUInt32:
			return "UInt32";
		case AlphaCore::kUInt64:
			return "UInt64";
		case AlphaCore::kString:
			return "String";
		case AlphaCore::kArrayMapDesc:
			return "ArrayDesc";
		default:
			break;
		}
		return "InvalidDataType";
	}

	static const char* PrimitiveTypeToString(AxPrimitiveType primType)
	{
		switch (primType)
		{
		case AlphaCore::kPrimVolume:
			return "PrimVolume";
		case AlphaCore::kOpenVDB:
			return "OpenVDB";
		case AlphaCore::kPrimPolyon:
			return "PrimPology";
		case AlphaCore::kPrimParticle:
			return "PrimParticle";
		case AlphaCore::kSphere:
			return "Sphere";
		case AlphaCore::kTetrahedra:
			return "Tetrahedra";
		case AlphaCore::kBox:
			return "Box";
		case AlphaCore::kPolyLineMark:
			return "CurveLine";
		default:
			break;
		}
		return "InvalidPrimitive";
	}

	static const char* GeoNodeTypeToString(AxGeoNodeType dataType)
	{
		switch (dataType)
		{
		case AlphaCore::kPoint:
			return "Point";
			break;
		case AlphaCore::kPrimitive:
			return "Primitive";
			break;
		case AlphaCore::kVertex:
			return "Vertex";
			break;
		case AlphaCore::kGeoDetail:
			return "GeoDetail";
			break;
		case AlphaCore::kGeoIndices:
			return "GeoIndices";
			break;
		default:
			break;
		}
		return "InvalidGeoNodeType";
	}

	static const char* AxBackendAPIToString(AlphaCore::AxBackendAPI backendType)
	{
		if (backendType == AlphaCore::AxBackendAPI::CUDA)
			return "CUDA";
		if (backendType == AlphaCore::AxBackendAPI::CPUx86)
			return "CPUx86";
		if (backendType == AlphaCore::AxBackendAPI::Vulkan)
			return "VulKan";
		return "InvalidBackendAPI";
	}

}

#endif // 
