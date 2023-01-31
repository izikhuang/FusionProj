#ifndef __AX_BVH_TREE_H__
#define __AX_BVH_TREE_H__

#include "AxDataType.h"
#include "Math/AxVectorBase.h"
#include "AccelTree/AxAccelTree.DataType.h"
#include "Collision/AxCollision.DataType.h"

class AxGeometry;
class AxBVHTree
{
public:

	struct RAWDesc
	{
		bool		 Valid;
		AxUChar*	 RTriangleTokens;
		AxVector3*	 Pos0Buffer;
		AxVector3*	 Pos1Buffer;
		AxFp32*		 MaxEta;
		AxFp32*		 PointRadius;
		AxBVHNode*   AllNodes;
		AxAABB*	     AllBVs;
		AxAABB*		 PrimBVs;
		AxVector2UI* PrimList;
		AxVector3UI* IndicesBuffer;
		AxUInt32*	 SortedPrimId;
		AxInt32		 NumPoints;
		AxInt32		 NumPrimitives;
		AlphaCore::AxCollisionTaskDesc* Tasks;
	};

	static AxBVHTree::RAWDesc GetRawDesc(AxBVHTree* bvh, AlphaCore::AxBackendAPI deviceMode, bool useSortedData = false);
 
	AxBVHTree();
	~AxBVHTree();

	void Build(AxGeometry* data, const char* posBuf0, const char* posBuf1,bool initDeviceData=false);
	void DeviceMalloc();

	void UpdateBV(bool useSortedIndex = false);
	void UpdateBVDevice(bool useSortedIndex = false);

	void UpdateTree(bool updateHierarchy = true, bool sortedIndices = false);
	void UpdateTreeDevice(bool updateHierarchy = true, bool sortedIndices = false);

	void SaveAsVisualizationAsPointCloud(std::string path);

 	void PrintData();
	void PrintDataDevice();

	void BuildNodeDepthInfoDFS();

	AxBufferAABB*	 GetAllBVBuffer() { return m_AABB_Triangle_Buffer; };
	AxUInt32 GetLeafStartOffset()	  { return m_iNnumPrims - 1; };
	AxUInt32 GetNumPrimitives()		  { return m_iNnumPrims; };
	AxUInt32 GetNumNodes()			  { return m_iNumNodes; };
	AxUInt32 GetNumPoints();		  
									  
	AxAABB* GetLeafAABBRaw()		  { return m_AABB_Triangle_Buffer->GetDataRaw<AxAABB>() + (m_iNnumPrims - 1); }
	AxAABB* GetLeafAABBRawDevice()	  { return m_AABB_Triangle_Buffer->GetDataRawDevice<AxAABB>() + (m_iNnumPrims - 1); }
	AxBufferUInt32* GetMortonCode()	  { return m_MortonCode; };
	AxBufferBVHNode* GetAllNodes()	  { return  m_BVNodeBuffer; }
	AxGeometry*	GetOwnGeometry()	  { return  m_OwnGeoData; }

public:

	void updateMortonCode(bool sortedIndices = false);
	void updateMortonCodeDevice(bool sortedIndices = false);

	void updateHierarchyStructure(bool sortedIndices = false);
	void updateHierarchyStructureDevice(bool sortedIndices = false);

	void buildNodeDepthInfoDFS(int idx, int leafStart, AxBVHNode *nodes, AxAABB *AllBVs, int depth = 0);

	//
	AxBufferBVHNode* m_BVNodeBuffer;
	AxBufferAABB*	 m_AABB_Triangle_Buffer;
	AxBufferAABB*	 m_AABB_Edge_Buffer;	// nullptr
	AxBufferAABB*	 m_AABB_Point_Buffer;	// nullptr
	AxGeometry*		 m_OwnGeoData;

private:
	AxBufferV3*		 m_Pos0Buffer;
	AxBufferV3*		 m_Pos1Buffer;
	AxBufferUInt32*	 m_TopologyIndicesBuffer;
	AxBufferUInt32*	 m_PrimIdIdenityBuffer;
	AxBufferUInt32*	 m_SortedTopologyIndicesBuffer;
	AxBufferF*		 m_PrimDepthBuffer;
	AxBufferF*		 m_PrimMaxThicknessBuffer;
	AxBufferF*		 m_PointRadiusBuffer;
 	AxBufferUChar*   m_RTriangleToken;

	AxBufferUChar*   m_SortedRTriangleToken;
	AxBufferF*		 m_SortedPrimMaxThicknessBuffer;

	//TODO 只需要一份就可以
	AxBufferUInt32* m_MortonCode;
	AxBufferUInt64* m_FinalMortonCode;

	AxBuffer2UI*	m_PrimList2I;
	AxAABB			m_RootAABB;
	AxBufferI*		m_BVHMergeFlagsBuffer;
	AxUInt32		m_iNnumPrims;
	AxUInt32		m_iNumNodes;

	AxBufferCollisionTask* m_CollisionTask;
	AxBufferCollisionTask* m_SortedCollisionTask;

};

inline std::ostream& operator<<(std::ostream& out, const AxBVHNode& node)
{
	out << " AsRaw@RxBVHNode:" << node.Left << " , " << node.Right << " Parent: " << node.Parent << " BVID: " << node.BVId;
	return out;
}

namespace AlphaCore 
{
	namespace AccelTree 
	{
		ALPHA_SPMD_FUNC void UpdateTrianglesAABB(
			AxBufferV3* startPosBuf,
			AxBufferV3* endPosBuf,
			AxBufferUInt32* topologyIndicesBuf,
 			AxBufferAABB* triangleBvBuf,
			AxUInt32 startAABBIdx,
			AxBufferF* triangleDepthOffsetBuf,
			AxBufferF* triangleMaxEtaBuf,
			AxBufferUInt32* sotedTriangleIdBuf);

		ALPHA_SPMD_FUNC AxAABB ReduceAABB(AxBufferAABB* triangleBvBuf,
			AxUInt32 start,
			AxUInt32 numPrims);

		ALPHA_SPMD_FUNC void UpdateMortonCode3D(AxBufferUInt32* mortonCodeBuf,
			AxBufferUInt64* finalMortonCodeBuf,
			AxBufferAABB* aabbRawBuf,
			AxUInt32 start,
			AxAABB root);

		ALPHA_SPMD_FUNC void UpdateHierarchyTree(AxBufferUInt64* mortonCodeSortedBuf,
 			AxBufferBVHNode* allNodesBuf,
			AxUInt32 numPrims,
			AxUInt32 numInterval,
			AxUInt32 numNodes);

		ALPHA_SPMD_FUNC void PostUpdateIntervalBV(AxBufferBVHNode* allNodesBuf,
			AxBufferAABB* allAABBBuf,
			AxBufferI* childReadyFlagsBuf,
			AxUInt32 leafStart);

		ALPHA_SPMD_FUNC void InitAABBBuffer(AxBufferAABB* allAABBBuf,
			AxUInt32 startIdx,
			AxUInt32 numAABB);

		namespace CUDA {
			ALPHA_SPMD_FUNC void UpdateTrianglesAABB(
				AxBufferV3* startPosBuf,
				AxBufferV3* endPosBuf,
				AxBufferUInt32* topologyIndicesBuf,
				AxBufferAABB* triangleBvBuf,
				AxUInt32 startAABBIdx,
				AxBufferF* triangleDepthOffsetBuf,
				AxBufferF* triangleMaxEtaBuf,
				AxBufferUInt32* sotedTriangleIdBuf,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC AxAABB ReduceAABB(AxBufferAABB* triangleBvBuf,
				AxUInt32 start,
				AxUInt32 numPrims);

			ALPHA_SPMD_FUNC void UpdateMortonCode3D(AxBufferUInt32* mortonCodeBuf,
				AxBufferUInt64* finalMortonCodeBuf,
				AxBufferAABB* aabbRawBuf,
				AxUInt32 start,
				AxAABB root,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void UpdateHierarchyTree(AxBufferUInt64* mortonCodeSortedBuf,
 				AxBufferBVHNode* allNodesBuf,
				AxUInt32 numPrims,
				AxUInt32 numInterval,
				AxUInt32 numNodes,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void PostUpdateIntervalBV(AxBufferBVHNode* allNodesBuf,
				AxBufferAABB* allAABBBuf,
				AxBufferI* childReadyFlagsBuf,
				AxUInt32 leafStart,
				AxUInt32 blockSize = 512);

			ALPHA_SPMD_FUNC void InitAABBBuffer(AxBufferAABB* aabbBuf,
				AxUInt32 startIdx,
				AxUInt32 numAABB,
				AxUInt32 blockSize = 512);

		}
	}
}//@namespace end of : AccelTree

#endif