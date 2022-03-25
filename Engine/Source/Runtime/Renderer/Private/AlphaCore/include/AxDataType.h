#ifndef __AX_DATA_TYPE_H__
#define __AX_DATA_TYPE_H__

#include <string>

typedef int					AxInt32;
typedef long long			AxInt64;
typedef float				AxReal;
typedef float				AxFp32;
typedef double				AxFp64;
typedef unsigned char		AxUInt8;
typedef unsigned short		AxUInt16;
typedef unsigned int		AxUInt32;
typedef unsigned long long  AxUInt64;
typedef unsigned char		Byte;
typedef unsigned char		AxUChar;

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

	enum AxSPMDBackend
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


	template<class T>
	inline AxUInt32 TypeVecSize() { return 1; }

	template<>
	inline AxUInt32 TypeVecSize<int>() { return 1; }

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
	inline AxDataType TypeID<std::string>() { return AxDataType::kString; }

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

	static const char* AxSPMDBackendToString(AlphaCore::AxSPMDBackend backendType)
	{
		if (backendType == AlphaCore::AxSPMDBackend::CUDA)
			return "CUDA";
		if (backendType == AlphaCore::AxSPMDBackend::CPUx86)
			return "CPUx86";
		if (backendType == AlphaCore::AxSPMDBackend::Vulkan)
			return "VulKan";
		return "InvalidBackendAPI";
	}

}

#endif // 
