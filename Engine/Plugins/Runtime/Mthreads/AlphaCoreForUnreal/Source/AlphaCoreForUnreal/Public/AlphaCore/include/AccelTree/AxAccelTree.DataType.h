#ifndef __AX_ACCEL_TREE_DOT_DATA_TYPE_H__
#define __AX_ACCEL_TREE_DOT_DATA_TYPE_H__

#include <iostream>
#include "Math/AxVectorBase.h"
#include "Math/AxVectorHelper.h"
#include "Utility/AxStorage.h"

#define AX_BV_MAX_SIZE 1000000000.0
#define AX_BVH_TREE_DEBUG 0
struct AxAABB
{
	AxVector3 Min;
	AxVector3 Max;
#if AX_BVH_TREE_DEBUG == 1
	AxInt32 bvID;
	AxInt32 depth;
#endif
};


namespace AlphaCore
{
	template<>
	inline AxString TypeName<AxAABB>() { return "AxAABB"; }
}

namespace AlphaCore
{
	ALPHA_SHARE_FUNC AxVector3 AABBCenter(const AxAABB& aabb)
	{
		return  (aabb.Min + aabb.Max) * 0.5f;
	}

	ALPHA_SHARE_FUNC AxVector3 AABBSize(const AxAABB& aabb)
	{
		return  aabb.Max - aabb.Min;
	}

	namespace AccelTree
	{
		ALPHA_SHARE_FUNC AxAABB MakeAABB()
		{
			AxAABB t;
			//printf("SO");
#if AX_BVH_TREE_DEBUG == 1 
			t.bvID = -1;
			t.depth = -1;
#endif
			t.Min.x =  AX_BV_MAX_SIZE;
			t.Min.y =  AX_BV_MAX_SIZE;
			t.Min.z =  AX_BV_MAX_SIZE;
			t.Max.x = -AX_BV_MAX_SIZE;
			t.Max.y = -AX_BV_MAX_SIZE;
			t.Max.z = -AX_BV_MAX_SIZE;
			return t;
		}

		ALPHA_SHARE_FUNC void ExtAABB(AxAABB& old,const AxVector3& pos)
		{
			old.Min.x = fminf(old.Min.x, pos.x);
			old.Min.y = fminf(old.Min.y, pos.y);
			old.Min.z = fminf(old.Min.z, pos.z);
			old.Max.x = fmaxf(old.Max.x, pos.x);
			old.Max.y = fmaxf(old.Max.y, pos.y);
			old.Max.z = fmaxf(old.Max.z, pos.z);
		}


		ALPHA_SHARE_FUNC AxAABB Merge(AxAABB a, AxAABB b)
		{
			AxAABB ret;
			ret.Min.x = fminf(a.Min.x, b.Min.x);
			ret.Min.y = fminf(a.Min.y, b.Min.y);
			ret.Min.z = fminf(a.Min.z, b.Min.z);
			ret.Max.x = fmaxf(a.Max.x, b.Max.x);
			ret.Max.y = fmaxf(a.Max.y, b.Max.y);
			ret.Max.z = fmaxf(a.Max.z, b.Max.z);
			return ret;
		}

		ALPHA_SHARE_FUNC AxAABB MakeAABB(const AxVector3& a, const AxVector3& b, AxFp32 eta = 0.0f)
		{
			AxAABB t;
#if AX_BVH_TREE_DEBUG == 1
			t.bvID = -1;
			t.depth = -1;
#endif
			t.Min.x = fminf(a.x, b.x);
			t.Min.y = fminf(a.y, b.y);
			t.Min.z = fminf(a.z, b.z);
			t.Max.x = fmaxf(a.x, b.x);
			t.Max.y = fmaxf(a.y, b.y);
			t.Max.z = fmaxf(a.z, b.z);

			t.Min.x -= eta;
			t.Min.y -= eta;
			t.Min.z -= eta;
			t.Max.x += eta;
			t.Max.y += eta;
			t.Max.z += eta;
			return t;
		}
	}
}

inline std::ostream& operator<<(std::ostream& out, const AxAABB& aabb)
{
	//out << " Max:" << aabb.Max.x << " , " << aabb.Max.y << " , " << aabb.Max.z;
	//out << " Min:" << aabb.Min.x << " , " << aabb.Min.y << " , " << aabb.Min.z << std::endl;
	out << " AsRaw@RxAABB:" << aabb.Max.x << " , " << aabb.Max.y << " , " << aabb.Max.z << " | "
		<< aabb.Min.x << " , " << aabb.Min.y << " , " << aabb.Min.z;
#if AX_BVH_TREE_DEBUG == 1
	out << " | " << aabb.depth;
#endif
	//out << std::endl;
	return out;
}

ALPHA_SHARE_FUNC void PrintInfo(const char* head, const AxAABB& aabb)
{
	printf("%s  Max: %f , %f ,%f  Min: %f ,%f ,%f\n", head, aabb.Max.x, aabb.Max.y, aabb.Max.z, aabb.Min.x, aabb.Max.y, aabb.Max.z);
	printf(" AsRaw@RxAABB: %f,%f,%f|%f,%f,%f\n", aabb.Max.x, aabb.Max.y, aabb.Max.z, aabb.Min.x, aabb.Max.y, aabb.Max.z);

}



typedef AxStorage<AxAABB> AxBufferAABB;

struct AxBVHNode
{
	AxInt32 Left;
	AxInt32 Right;
	AxInt32 Parent;
	AxInt32 BVId;
};

ALPHA_SHARE_FUNC AxBVHNode MakeBVHnode()
{
	AxBVHNode t;
	t.Left		= -1;
	t.Right		= -1;
	t.Parent	= -1;
	t.BVId		= -1;
	return t;
}

typedef AxStorage<AxBVHNode> AxBufferBVHNode;

#endif