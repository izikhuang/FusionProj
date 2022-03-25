#ifndef __AX_BVH_TREE_DOT_SHARECODE_H__
#define __AX_BVH_TREE_DOT_SHARECODE_H__

#include "AxAccelTree.DataType.h"

namespace AlphaCore {
	namespace AccelTree {
		namespace ShareCode {

			ALPHA_SHARE_FUNC AxAABB MergeAABB(const AxAABB &box0, const AxAABB &box1) 
			{
				AxAABB ret;
				ret.Min.x = fminf(box0.Min.x, box1.Min.x);
				ret.Min.y = fminf(box0.Min.y, box1.Min.y);
				ret.Min.z = fminf(box0.Min.z, box1.Min.z);
				ret.Max.x = fmaxf(box0.Max.x, box1.Max.x);
				ret.Max.y = fmaxf(box0.Max.y, box1.Max.y);
				ret.Max.z = fmaxf(box0.Max.z, box1.Max.z);
				return ret;
			}


			ALPHA_SHARE_FUNC void ExtBV(AxVector3& max, AxVector3& min, AxVector3 currA, AxVector3 currB)
			{
				max.x = fmaxf(fmaxf(currA.x, currB.x), max.x);
				max.y = fmaxf(fmaxf(currA.y, currB.y), max.y);
				max.z = fmaxf(fmaxf(currA.z, currB.z), max.z);

				min.x = fminf(fminf(currA.x, currB.x), min.x);
				min.y = fminf(fminf(currA.y, currB.y), min.y);
				min.z = fminf(fminf(currA.z, currB.z), min.z);
			}

			/*
			ALPHA_SHARE_FUNC bool BoundingBoxOverlap(
				const AxVector3& maxA, const AxVector3& minA,
				const AxVector3& maxB, const AxVector3& minB)
			{
				if (maxB.x < minA.x || minB.x > maxA.x)
					return false;
				if (maxB.y < minA.y || minB.y > maxA.y)
					return false;
				if (maxB.z < minA.z || minB.z > maxA.z)
					return false;
				return true;
			}
			*/

			ALPHA_SHARE_FUNC bool BoundingBoxOverlap(
				const AxVector3& maxA, const AxVector3& minA,
				const AxVector3& maxB, const AxVector3& minB)
			{
				if (maxB.x < minA.x || minB.x > maxA.x || maxB.y < minA.y || minB.y > maxA.y || maxB.z < minA.z || minB.z > maxA.z)
					return false;
				return true;
			}
			/*
			ALPHA_SHARE_FUNC bool Intersect(const AxAABB &a, const AxAABB &b)
			{
				if (a.Min.x > b.Max.x || a.Max.x < b.Min.x) return false;
				if (a.Min.y > b.Max.y || a.Max.y < b.Min.y) return false;
				if (a.Min.z > b.Max.z || a.Max.z < b.Min.z) return false;
				return true;
			}
			*/
			ALPHA_SHARE_FUNC bool Intersect(const AxAABB &a, const AxAABB &b)
			{
				if (a.Min.x > b.Max.x || a.Max.x < b.Min.x || a.Min.y > b.Max.y || a.Max.y < b.Min.y || a.Min.z > b.Max.z || a.Max.z < b.Min.z)
					return false;
				return true;
			}

			ALPHA_SHARE_FUNC void InitAABBBuffer(AxUInt32 idx, AxAABB* allAABBRaw)
			{
				allAABBRaw[idx] = AlphaCore::AccelTree::MakeAABB();
			}

			ALPHA_SHARE_FUNC int __AxCLZ64(AxUInt64 v)
			{

#ifdef __CUDA_ARCH__
				return __clzll(v);
#else 
				unsigned long index0;
				unsigned char isNonezero = _BitScanReverse64(&index0, v);
				if (isNonezero) {
					return 63 - index0;
				}
				else {
					return 64;
				}
#endif
			}

			ALPHA_SHARE_FUNC bool Intersect(const AxAABB &a, const AxAABB &b, AxFp32 thickness)
			{
				if (a.Min.x > b.Max.x + thickness || a.Max.x + thickness < b.Min.x) return false;
				if (a.Min.y > b.Max.y + thickness || a.Max.y + thickness < b.Min.y) return false;
				if (a.Min.z > b.Max.z + thickness || a.Max.z + thickness < b.Min.z) return false;
				return true;
			}

			ALPHA_SHARE_FUNC int CommonUpperBits(const AxUInt64 &code1, const AxUInt64 &code2) 
			{
				return __AxCLZ64(code1^code2);
			}

			ALPHA_SHARE_FUNC AxVector2I DetermineRange(AxUInt64 *code, int numPrims, int idx) 
			{
				// determin direction of the range
				AxVector2I range;
				if (idx == 0) {
					range.x = 0;
					range.y = numPrims - 1;
					return range;
				}

				const int L_delta = CommonUpperBits(code[idx], code[idx - 1]);
				const int R_delta = CommonUpperBits(code[idx], code[idx + 1]);
				const int d = (R_delta > L_delta) ? 1 : -1;

				// compute upper bound for the length of the range
				int delta_min = L_delta < R_delta ? L_delta : R_delta;
				int l_max = 2;
				int delta = -1;
				int i_tmp = idx + d * l_max;
				if (0 <= i_tmp && i_tmp < numPrims) {
					delta = CommonUpperBits(code[idx], code[i_tmp]);
				}

				while (delta > delta_min) {
					l_max <<= 1;
					i_tmp = idx + d * l_max;
					delta = -1;
					if (0 <= i_tmp && i_tmp < numPrims) {
						delta = CommonUpperBits(code[idx], code[i_tmp]);
					}
				}

				// find the other end using binary search
				int l = 0;
				int t = l_max >> 1;
				while (t > 0) {
					i_tmp = idx + (l + t)*d;
					delta = -1;
					if (0 <= i_tmp && i_tmp < numPrims) {
						delta = CommonUpperBits(code[idx], code[i_tmp]);
					}
					if (delta > delta_min) {
						l += t;
					}
					t >>= 1;
				}

				int jdx = idx + l * d;
				if (d < 0) {
					range.x = jdx;
					range.y = idx;
					return range;
				}
				range.x = idx;
				range.y = jdx;
				return range;
			}

			ALPHA_SHARE_FUNC int FindSplit(AxUInt64 * code, int first, int last) 
			{
				if (first == last) {
					return -1;
				}
				AxUInt64 first_code = code[first];
				AxUInt64 last_code = code[last];
				if (first_code == last_code) {
					return (first + last) >> 1;
				}
				int delta_node = CommonUpperBits(first_code, last_code);
				// binary search
				int split = first;
				int stride = last - first;
				do {
					stride = (stride + 1) >> 1;
					const int middle = split + stride;
					if (middle < last) {
						const int delta = CommonUpperBits(first_code, code[middle]);
						if (delta > delta_node) {
							split = middle;
						}
					}
				} while (stride > 1);
				return split;
			}

			ALPHA_SHARE_FUNC void UpdateTrianglesAABB(AxUInt32 idx,
				AxVector3* startPosRaw,
				AxVector3* endPosRaw,
				AxUInt32* topologyIndicesRaw,
 				AxFp32* triangleDepthOffsetRaw,
				AxFp32* triangleMaxEtaRaw,
				AxAABB* triangleBvRaw,
				AxUInt32* sortedIdRaw = nullptr)
			{
				AxFp32 minV = -AX_BV_MAX_SIZE;
				AxFp32 maxV =  AX_BV_MAX_SIZE;
				AxFp32 maxThickness = triangleMaxEtaRaw == nullptr ? 0.001f : triangleMaxEtaRaw[idx];
				AxVector3 maxPrim, minPrim;		// todo:remove this
				AxVector3 maxB, minB;			// todo:remove this
 
				AxVector2UI edgeVtxIds[3];
				edgeVtxIds[0].x = 2;	edgeVtxIds[0].y = 1;
				edgeVtxIds[1].x = 1;	edgeVtxIds[1].y = 0;
				edgeVtxIds[2].x = 0;	edgeVtxIds[2].y = 2;

				AxAABB bv = MakeAABB();
				AxAABB bvPrim = MakeAABB();
				AxFp32 triangleDepth = triangleDepthOffsetRaw == nullptr ? 0.0f : triangleDepthOffsetRaw[idx];

				AxUInt32 ptIds[3];
				ptIds[0] = topologyIndicesRaw[idx * 3 + 0];
				ptIds[1] = topologyIndicesRaw[idx * 3 + 1];
				ptIds[2] = topologyIndicesRaw[idx * 3 + 2];

				AxVector3 pts0[3];
				pts0[0] = startPosRaw[ptIds[0]];
				pts0[1] = startPosRaw[ptIds[1]];
				pts0[2] = startPosRaw[ptIds[2]];
				AxVector3 pts1[3];
				pts1[0] = endPosRaw[ptIds[0]];
				pts1[1] = endPosRaw[ptIds[1]];
				pts1[2] = endPosRaw[ptIds[2]];

				AxVector3 n = Normalize(Cross(pts1[2] - pts1[1], pts1[1] - pts1[0]))*triangleDepth;
				AssignV3(maxPrim, minV);
				AssignV3(minPrim, maxV);

				bool vertexToken[3];
				bool edgeToken[3];

				AX_FOR_I(3)
				{
					AxVector3 p0 = pts0[i];
					AxVector3 p1 = pts1[i];
					//trace : p1 -n  -->   p1 --> p0
					AlphaCore::AccelTree::ShareCode::ExtBV(maxPrim, minPrim, p1 - n, p1);
					AlphaCore::AccelTree::ShareCode::ExtBV(maxPrim, minPrim, p0, p1);
				}
				maxPrim += maxThickness;
				minPrim -= maxThickness;
				bvPrim.Max = maxPrim;
				bvPrim.Min = minPrim;

				if (sortedIdRaw)
				{
					triangleBvRaw[sortedIdRaw[idx]] = bvPrim;
				}
				else {
					triangleBvRaw[idx] = bvPrim;
				}
			}

			ALPHA_SHARE_FUNC AxUInt32 ExpandBits(AxUInt32 v)
			{
				v = (v * 0x00010001u) & 0xFF0000FFu;
				v = (v * 0x00000101u) & 0x0F00F00Fu;
				v = (v * 0x00000011u) & 0xC30C30C3u;
				v = (v * 0x00000005u) & 0x49249249u;
				return v;
			}

			ALPHA_SHARE_FUNC AxUInt32 MortonCode3D(AxVector3 xyz, float resolution)
			{
				xyz.x = fminf(fmaxf(xyz.x*resolution, 0.0f), resolution - 1.0f);
				xyz.y = fminf(fmaxf(xyz.y*resolution, 0.0f), resolution - 1.0f);
				xyz.z = fminf(fmaxf(xyz.z*resolution, 0.0f), resolution - 1.0f);
				AxUInt32 xx = ExpandBits(static_cast<AxUInt32>(xyz.x));
				AxUInt32 yy = ExpandBits(static_cast<AxUInt32>(xyz.y));
				AxUInt32 zz = ExpandBits(static_cast<AxUInt32>(xyz.z));
				return xx * 4 + yy * 2 + zz;
			}


			ALPHA_SHARE_FUNC AxVector3 ResizeInUint(const AxVector3 &p, const AxAABB &rootBox)
			{
				AxVector3 pm;
				pm.x = p.x - rootBox.Min.x;
				pm.y = p.y - rootBox.Min.y;
				pm.z = p.z - rootBox.Min.z;
				pm.x /= (rootBox.Max.x - rootBox.Min.x);
				pm.y /= (rootBox.Max.y - rootBox.Min.y);
				pm.z /= (rootBox.Max.z - rootBox.Min.z);
				return pm;
			}

			ALPHA_SHARE_FUNC AxVector3 Centroid(const AxAABB &box)
			{
				AxVector3 sum;
				sum.x = (box.Min.x + box.Max.x) * 0.5f;
				sum.y = (box.Min.y + box.Max.y) * 0.5f;
				sum.z = (box.Min.z + box.Max.z) * 0.5f;
				return sum;
			}

			ALPHA_SHARE_FUNC void UpdateMortonCode3D(AxUInt32 idx,
				AxUInt32* mortonCodeRaw,
				AxUInt64* finalMortonCodeRaw,
				AxAABB* aabbRawRaw,
				AxUInt32 start,
				AxAABB root)
			{
				AxUInt32 _morton = MortonCode3D(ResizeInUint(Centroid(aabbRawRaw[start + idx]), root), 512.0f);
				AxUInt64 _temp = _morton;
				AxUInt64 ret = (_temp << 32) | idx;
				finalMortonCodeRaw[idx] = ret;
			}


			ALPHA_SHARE_FUNC void UpdateHierarchyTree(AxUInt32 idx,
				AxUInt64* mortonCodeSortedRaw,
				AxBVHNode* leafNodesRaw,
				AxBVHNode* allNodesRaw,
				AxUInt32 numPrims,
				AxUInt32 numInterval,
				AxUInt32 numNodes)
			{
				leafNodesRaw[idx].BVId = AxUInt32(mortonCodeSortedRaw[idx] & 0xFFFFFFFF) + numPrims - 1;
				if (idx == numInterval - 1)
					leafNodesRaw[idx + 1].BVId = AxUInt32(mortonCodeSortedRaw[idx + 1] & 0xFFFFFFFF) + numPrims - 1;

				AxVector2I range = DetermineRange(mortonCodeSortedRaw, numPrims, idx);
				int split = FindSplit(mortonCodeSortedRaw, range.x, range.y);
				// set child and parent
				if (split == -1)
					return;
				int childA, childB;
				if (split == range.x) {
					childA = split + numInterval;
				}
				else {
					childA = split;
				}

				if (split + 1 == range.y) {
					childB = split + 1 + numInterval;
				}
				else {
					childB = split + 1;
				}

				allNodesRaw[idx].Left = childA;
				allNodesRaw[idx].Right = childB;
				allNodesRaw[childA].Parent = idx;
				allNodesRaw[childB].Parent = idx;
				allNodesRaw[idx].BVId = idx;
			}

			ALPHA_SHARE_FUNC void PostUpdateIntervalBV(AxUInt32 idx,
				AxBVHNode* allNodesRaw,
				AxAABB* allAABBRaw,
				AxInt32* childReadyFlagsRaw,
				AxUInt32 leafStart)
			{
				AxBVHNode nodeI = allNodesRaw[idx + leafStart];
				int parentId = nodeI.Parent;
 				while (parentId != -1)
				{
#ifdef __CUDA_ARCH__
					int old = atomicCAS(childReadyFlagsRaw + parentId, 0, 1);
#else
					int old = childReadyFlagsRaw[parentId];
					childReadyFlagsRaw[parentId] = 1;
#endif
					if (old == 0)
						return;
					AxBVHNode parentNode = allNodesRaw[parentId];
					AxAABB left = allAABBRaw[parentNode.Left];
					AxAABB right = allAABBRaw[parentNode.Right];
					AxAABB aabb = AlphaCore::AccelTree::ShareCode::MergeAABB(left, right);
					allAABBRaw[parentId] = aabb;
					parentId = parentNode.Parent;
				}
			}


		}
	}//@namespace end of : AccelTree
}

#endif // !__AX_BVH_TREE_DOT_SHARECODE_H__
