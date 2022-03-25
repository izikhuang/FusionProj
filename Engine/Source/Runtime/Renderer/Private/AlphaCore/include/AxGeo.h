#ifndef __AX_GEO_H__
#define __AX_GEO_H__

#include <Utility/AxStorage.h>
#include <map>
#include <unordered_set>
#include <iostream>
#include <tuple>
#include <Collision/AxCollision.DataType.h>

namespace AlphaCore
{
	static const char AXC_FileFormathead[AX_CACHE_HEAD_TOKEN] = "ALPHA_CORE_RAW_BUFFER_FORMAT|V1.0";
}

class AxGeometry;
class AxField;

typedef std::tuple<std::string, AlphaCore::AxDataType> AxPropertyKey;
typedef std::map<AxPropertyKey, AxStorageBase*> AxPropertyDict;

struct AxTopologyRawData
{
	AxUInt32 NumPoints;		// points 
	AxUInt32 NumPrims;			// primList
	AxUInt32 NumVertices;		// indices
	AxVector3* Position;
	AxVector2UI* Primitives;
	AxUInt32* TopologyIndices;
	AlphaCore::AxPrimitiveType* PrimType;
};

struct AxPropertyDictInfo
{
	//point , primitive , vertex, geoMetaData , indices
	AxPropertyDict* PropertieMaps[5];
	AlphaCore::AxGeoNodeType Type[5];
};

//
// DO NOT USE THIS MACRO ERASE PROPERTY !!!!!
//
// see : google std::map error:
//	
//	cannot increment value-initialized map/set iterator
//
//
#define AX_FOR_ALL_PROPERTIES_ITR(dict,storage) \
	for(AxPropertyDict::iterator itr = dict->begin();\
		itr != dict->end() && (storage = itr->second);\
		++itr)

struct AxTopolgyDesc
{
	AxBufferUInt32* Indices;
	AxBuffer2UI* Primitives;
	AxBufferV3* Positions;
	AxPrimTypeBuffer * PrimitiveType;
};

template <typename T>
struct AxIdxMap
{
	AxIdxMap()
	{
		Map = nullptr;
		Values = nullptr;
	}
	AxIdxMap(AxBuffer2UI* map, AxStorage<T>* indices)
	{
		Map = map;
		Values = indices;
	}

	bool IsValid()
	{
		return (Map != nullptr && Values != nullptr);
	}

	AxBuffer2UI* Map;
	AxStorage<T>* Values;

	void DeviceMalloc()
	{
		Map->DeviceMalloc();
		Values->DeviceMalloc();
	}

	void PrintData(const char* item = "item: ")
	{
		if (Map == nullptr)
			return;
		AX_FOR_I(Map->Size())
		{
			std::cout << item << i << ": " << Map->Get(i).y << " [ ";
			AX_FOR_J(Map->Get(i).y) {
				if (j < (Map->Get(i).y - 1)) { 
					std::cout << Values->Get(Map->Get(i).x + j) << ","; 
				}
				else {
					std::cout << Values->Get(Map->Get(i).x + j) << " ]" << std::endl;
				}
			}
		}
	}
};

typedef AxIdxMap<AxUInt32>	AxIdxMapUI32;
typedef AxIdxMap<int>		AxIdxMapI32;
typedef AxIdxMap<float>		AxIdxMapFp32;
typedef AxIdxMap<double>	AxIdxMapFp64;
typedef AxIdxMapUI32		AxIndicesMap;

struct AxCachePropHead
{
	// "P" | point | VecSize | DataPrecision 8 / 16
	char PropertyName[64];
	AlphaCore::AxGeoNodeType	PropertyClass;
	AlphaCore::AxDataType		DataType;
 
	AxUInt32 BankWidth;
	AxUInt32 VecSize;
	AxUInt64 Offset;
	AxUInt32 Items; //for indices or detail attribute ,lenght not coherence
};

 class AxCacheFormatGeoHead
 {
 public:
	 AxUInt32 NumPoints;		// points 
	 AxUInt32 NumPrims;			// primList
	 AxUInt32 NumVertices;		// indices
	 AxUInt32 NumProperties;
	 AxUInt32 PropHeadSize;
	 std::vector<AxCachePropHead> PropertyHeadList;

	 template<typename T>
	 AxInt64 GetOff(std::string name,AlphaCore::AxGeoNodeType nodeType)
 	 {
		 if (nodeType == AlphaCore::AxGeoNodeType::kPoint)
		 {
			 AxPropertyKey key = std::make_tuple(name, AlphaCore::TypeID<T>());
			 if (m_PointOffsetMap.count(key) == 0)
				 return -1;
			return  m_PointOffsetMap.at(key);
		 }
		 if (nodeType == AlphaCore::AxGeoNodeType::kPrimitive)
		 {
			 AxPropertyKey key = std::make_tuple(name, AlphaCore::TypeID<T>());
			 if (m_PrimitiveOffsetMap.count(key) == 0)
				 return -1;
			 return  m_PrimitiveOffsetMap.at(key);
		 }
		 return -1;
	 }

	 AxInt32 EvalPropHeadSize()
	 {
		 return -1;
	 }

	 void Save(std::ofstream& ofs);
	 void Read(std::ifstream& ifs);

	 void operator =(const AxCacheFormatGeoHead& that);

 private:

	 friend class AxGeometry;
	 void _updateMaps();

	 std::map< AxPropertyKey, AxInt64> m_PointOffsetMap;
	 std::map< AxPropertyKey, AxInt64> m_PrimitiveOffsetMap;
	 std::map< AxPropertyKey, AxInt64> m_VerticesOffsetMap;
	 std::map< AxPropertyKey, AxInt64> m_GeometryOffsetMap;
	 std::map< AxPropertyKey, AxInt64> m_IndicesOffsetMap;

 };


 inline bool operator==(AxCachePropHead& a, AxCachePropHead& b)
 {
	 return std::memcmp(&a, &b, sizeof(AxCachePropHead)) == 0;
 }

 inline bool operator!=(AxCachePropHead& a, AxCachePropHead& b)
 {
	 return !(a == b);
 }


 inline bool operator==(AxCacheFormatGeoHead& a, AxCacheFormatGeoHead& b)
 {
	 if (a.NumPoints != b.NumPoints)
		 return false;
	 if (a.NumPrims != b.NumPrims)
		 return false;
	 if (a.NumVertices != b.NumVertices)
		 return false;
	 if (a.NumProperties != b.NumProperties)
		 return false;
	 AX_FOR_I(a.PropertyHeadList.size())
	 {
		 if (a.PropertyHeadList[i] != b.PropertyHeadList[i])
			 return false;
	 }
	 return true;
 }

 inline bool operator!=(AxCacheFormatGeoHead& a, AxCacheFormatGeoHead& b)
 {
	 return !(a == b);
 }


 inline std::ostream& operator<<(std::ostream& out, AxCachePropHead& c)
 {
	 out << "  " <<"Name : \"" << c.PropertyName <<"\" <";
	 out << "  " << AlphaCore::GeoNodeTypeToString(c.PropertyClass) << " >";
	 out << "  " << AlphaCore::DataTypeToString(c.DataType) << std::endl;
	 out << "  " << "BankWidth : " << c.BankWidth << " |";
	 out << "  " << "File Offset : " << c.Offset << " |";
	 out << "  " << "Items : " << c.Items << std::endl;
	 return out;
 }

 inline std::ostream& operator<<(std::ostream& out, AxCacheFormatGeoHead& c)
 {
	 out << c.NumPoints		  << "  Points" << std::endl;
	 out << c.NumPrims		  << "  Primitives" << std::endl;
	 out << c.NumVertices	  << "  Vertex" << std::endl;
	 out << c.NumProperties	  << "  Properties" << std::endl;
	 AX_FOR_I(c.PropertyHeadList.size())
		 out << c.PropertyHeadList[i] << std::endl;
	 return out;
 }


 //
 // @Problem
 //
struct AxPrimitive
{
	bool Valid;
	AxUInt32 NumVertices;
	AxUInt32* Points;
	AxUInt32 VtxStartPos;
	AlphaCore::AxPrimitiveType Type;
	int VolumeIdx;
};

class ALPHA_CLASS AxGeometry
{
public:
	AxGeometry();
	virtual ~AxGeometry();

	AxTopolgyDesc GetTopolgyDesc();

	void SetName(std::string n) { m_sName = n; };
	std::string GetName() { return m_sName; };

	void UpdatePointPropertyFromFile(std::string path, const std::vector<std::string>& propList, bool loadToDevice = false);
 
	static AxGeometry* Build();
	static AxGeometry* Load(std::string path);
	static AxUInt64 EvalEdgeTokenID(AxUInt32 id0,AxUInt32 id1,AxUInt32 npts);
	virtual bool Save(std::string path);
	virtual bool Read(std::string path);
	virtual void DrawVisualization() {};
	virtual void DrawVisualizationDevice() {};
	template<typename T>
	AxStorage<T>* GetProperty(std::string name, AlphaCore::AxGeoNodeType geoNodeType)
	{
		if (AlphaCore::TypeID<T>() == AlphaCore::AxDataType::kString)//TODO:STL string Notgood,char8,char16,char32
			return nullptr;

		auto key = std::make_tuple(name, AlphaCore::TypeID<T>());
		//std::cout << "Get:" << name<<"AlphaCore::TypeID<T>()"<< AlphaCore::TypeID<T>() << std::endl;
		switch (geoNodeType)
		{
		case AlphaCore::kPoint:
		{
			if (m_PointProperties.find(key) == m_PointProperties.end())
				return nullptr;
			return (AxStorage<T>*)m_PointProperties[key];//TODO : not safe
		}
		break;
		case AlphaCore::kPrimitive:
		{
			if (m_PrimitiveProperties.find(key) == m_PrimitiveProperties.end())
				return nullptr;
			return (AxStorage<T>*)m_PrimitiveProperties[key];//TODO : not safe
		}
		break;
		case AlphaCore::kVertex:
		{
			if (m_VerticesProperties.find(key) == m_VerticesProperties.end())
				return nullptr;
			return (AxStorage<T>*)m_VerticesProperties[key];//TODO : not safe
		}
		break;
		case AlphaCore::kGeoDetail:
		{
			if (m_GeometryProperties.find(key) == m_GeometryProperties.end())
				return nullptr;
			return (AxStorage<T>*)m_GeometryProperties[key];//TODO : not safe
		}
		break;
		case AlphaCore::kGeoIndices:
		{
			if (m_IndicesProperties.find(key) == m_IndicesProperties.end())
				return nullptr;
			return (AxStorage<T>*)m_IndicesProperties[key];//TODO : not safe
		}
		break;
		default:
			break;
		}
		return nullptr;
	}

	AxBufferS* AddPropertyS(std::string name, AlphaCore::AxGeoNodeType geoNodeType);
	AxBufferS* GetPropertyS(std::string name, AlphaCore::AxGeoNodeType geoNodeType);

	bool AddProperty(AxStorageBase* storage, AlphaCore::AxDataType dataType, AlphaCore::AxGeoNodeType geoNodeType);

	template<typename T>
	AxIdxMap<T> AddArrayProperty(std::string name, std::string rawBlockName, AlphaCore::AxGeoNodeType geoNodeType)
	{
		AxIdxMap<T> ret;
		ret.Map = AddProperty<AxVector2UI>(name,geoNodeType);
		ret.Values = AddProperty<T>(rawBlockName, AlphaCore::AxGeoNodeType::kGeoIndices);
		return ret;
	}

	template<typename T>
	AxIdxMap<T> AddArrayProperty(std::string name, AlphaCore::AxGeoNodeType geoNodeType)
	{
		return AddArrayProperty<T>(name, name + "__VALUES", geoNodeType);
	}

	template<typename T>
	AxIdxMap<T> AddPointArrayProperty(std::string name)
	{
		return AddArrayProperty<T>(name, name + "__VALUES", AlphaCore::AxGeoNodeType::kPoint);
	}

	template<typename T>
	AxIdxMap<T> GetPointArrayProperty(std::string name)
	{
		AxIdxMap<T> ret;
		ret.Map = GetProperty<AxVector2UI>(name, AlphaCore::AxGeoNodeType::kPoint);
		ret.Values = GetProperty<T>(name + "__VALUES", AlphaCore::AxGeoNodeType::kGeoIndices);
		return ret;
	}

	template<typename T>
	AxIdxMap<T> GetArrayProperty(std::string name, AlphaCore::AxGeoNodeType geoNodeType)
	{
		AxIdxMap<T> ret;
		ret.Map = GetProperty<AxVector2UI>(name, geoNodeType);
		ret.Values = GetProperty<T>(name + "__VALUES", AlphaCore::AxGeoNodeType::kGeoIndices);
		return ret;
	}

	template<typename T>
	AxStorage<T>* AddProperty(std::string name, AlphaCore::AxGeoNodeType geoNodeType)
	{
		if (AlphaCore::TypeID<T>() == AlphaCore::AxDataType::kString)//TODO:STL string Notgood,char8,char16,char32
			return nullptr;
		auto currProp = this->GetProperty<T>(name, geoNodeType);
		if (currProp != nullptr)
			return currProp;
		AxStorage<T>* ret = nullptr;
		ret = new AxStorage<T>(name, numGeoNodes(geoNodeType));
		auto key = std::make_tuple(name, AlphaCore::TypeID<T>());
		switch (geoNodeType)
		{
		case AlphaCore::kPoint:
			m_PointProperties[key] = (AxStorageBase*)ret;
			break;
		case AlphaCore::kPrimitive:
			m_PrimitiveProperties[key] = (AxStorageBase*)ret;
			break;
		case AlphaCore::kVertex:
			m_VerticesProperties[key] = (AxStorageBase*)ret;
			break;
		case AlphaCore::kGeoDetail:
			m_GeometryProperties[key] = (AxStorageBase*)ret;
			break;
		case AlphaCore::kGeoIndices:
			m_IndicesProperties[key] = (AxStorageBase*)ret;
			break;
		default:
			break;
		}
		return ret;
	}

	//------------------ Add ------------------ 
	template<typename T>
	AxStorage<T>* AddPointProperty(std::string name){
		return AddProperty<T>(name,AlphaCore::AxGeoNodeType::kPoint);
	}

	template<typename T>
	AxStorage<T>* AddPrimitiveProperty(std::string name){
		return AddProperty<T>(name, AlphaCore::AxGeoNodeType::kPrimitive);
	}

	template<typename T>
	AxStorage<T>* AddVertexProperty(std::string name){
		return AddProperty<T>(name, AlphaCore::AxGeoNodeType::kVertex);
	}

	template<typename T>
	AxStorage<T>* AddGeoMetaDataProperty(std::string name,AxUInt32 size=0){
		auto r = AddProperty<T>(name, AlphaCore::AxGeoNodeType::kGeoDetail);
		r->Resize(size);
		return r;
	}

	template<typename T>
	AxStorage<T>* AddIndicesProperty(std::string name, AxUInt32 size = 0) {
		auto r = AddProperty<T>(name, AlphaCore::AxGeoNodeType::kGeoIndices);
		r->Resize(size);
		return r;
	}

	//------------------ Get ------------------ 
	template<typename T>
	AxStorage<T>* GetPointProperty(std::string name){
		return GetProperty<T>(name, AlphaCore::AxGeoNodeType::kPoint);
	}

	AxStorageBase* GetPointProperty(std::string name, AlphaCore::AxDataType dataType);

	template<typename T>
	AxStorage<T>* GetPrimitiveProperty(std::string name){
		return GetProperty<T>(name, AlphaCore::AxGeoNodeType::kPrimitive);
	}

	template<typename T>
	AxStorage<T>* GetVertexProperty(std::string name){
		return GetProperty<T>(name, AlphaCore::AxGeoNodeType::kVertex);
	}

	template<typename T>
	AxStorage<T>* GetGeoMetaDataProperty(std::string name){
		return GetProperty<T>(name, AlphaCore::AxGeoNodeType::kGeoDetail);
	}

	void AddSphere(AxUInt32 p0, AxFp32 radius);
	void AddTetrahedron(AxUInt32 p0, AxUInt32 p1, AxUInt32 p2, AxUInt32 p3);
	void AddTriangle(AxUInt32 a, AxUInt32 b, AxUInt32 c);


	AxIdxMapUI32 BuildPoint2PointMap();
	AxIdxMapUI32 BuildPoint2PrimitiveMap();
	AxIdxMapUI32 GetPoint2PointMap()const;

	//AxBufferCollisionTask* BuildVFEECollisionTask();

	AxPrimitive AddPrimitive(AxUInt32 nVtx,AlphaCore::AxPrimitiveType type, int* indicesLocal);
	AxPrimitive GetPrimitive(AxUInt32 primId);

	void AddLine(AxUInt32 a, AxUInt32 b);
	void AddPointBlock(AxUInt32 numPoints);
	void ResizePoints(AxUInt32 numPoints);
	void PrintPropertyData(AxUInt64 start = 0, int end = -1/* -1 means print all data*/);

	void PrintTopology();
	void PrintPrimitiveTypes();
	void EvalGeoHead(AxCacheFormatGeoHead& head);

	AxUInt32 GetNumVertex() { return m_IndicesBuffer->Size(); };
	AxUInt32 GetNumPoints() { return m_PositionBuffer->Size(); };
	AxUInt32 GetNumPrimitives() { return m_PrimStartNumBuffer->Size(); };

	AxBufferV3* GetPositionProperty() { return m_PositionBuffer; };
	AxPropertyDictInfo GetAllPropertyInfo();
	void ClearAndDestory();

	void SetPointPosition(AxUInt32 ptId, AxFp32 x, AxFp32 y, AxFp32 z);

	AxBufferUInt32* BuildEdgeList();
 
	AxField* GetField(AxUInt32 fieldIdx);
	AxField* FindField(std::string name);
 
	AxBufferUChar* GetRTriangleInfo(bool autoBuild = true);


	void DeviceMallocAllProperties(bool loadToDevice=true);
	void DeviceMallocPointProperty(bool loadToDevice = true);
	void DeviceMallocPrimitiveProperty(bool loadToDevice = true);

	AxVector3 GetPointPosition(AxUInt32 ptIdx);
	AxBufferV3* AddPointNormalProperty();
	AxBufferV3* GetPointNormalProperty();

	AxBufferV3* AddPointVelocityProperty();
	AxBufferV3* GetPointVelocityProperty();

protected:

	AxUInt32 numGeoNodes(AlphaCore::AxGeoNodeType type);
	AxCacheFormatGeoHead loadHead(std::ifstream& ifs);
	AxUInt32 totalProperties();

	void saveAxcFormatHead(std::ofstream& ofs, AxCacheFormatGeoHead& head);
	void resizePrimitives(AxUInt32 numPrimitives);
	std::string m_sName;

private:
	
	AxBufferV3*			m_PositionBuffer;		//numPoints
	AxBuffer2UI*		m_PrimStartNumBuffer;	//numPrimitive
	AxBufferUInt32*		m_IndicesBuffer;		//numVertices
 	AxPrimTypeBuffer*	m_PrimTypeBuffer;
	std::unordered_set<std::string> m_PropertyLoaderMask;

	AxPropertyDict m_PointProperties;
	AxPropertyDict m_PrimitiveProperties;
	AxPropertyDict m_VerticesProperties;
 	AxPropertyDict m_GeometryProperties;
	AxPropertyDict m_IndicesProperties;


	std::map<std::string, int> m_FieldMap2Idx;
	std::vector<AxField*> m_FieldList;

	AxBufferUChar* m_RTriangleToken;
	AxBufferV3*	   m_NormalProperty;		//nromalProp
	AxIdxMapUI32 m_Point2PointMap;
	void _pointsBufferChangeCallback();
	void _primitiveBufferChangeCallback();
	void _vertexBufferChangeCallback();

	
};

#endif
