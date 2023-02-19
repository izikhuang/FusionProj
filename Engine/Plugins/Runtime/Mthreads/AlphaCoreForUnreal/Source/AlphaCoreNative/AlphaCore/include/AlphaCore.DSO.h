#ifndef __ALPHA_CORE_DSO_H__
#define __ALPHA_CORE_DSO_H__


#include "AxLog.h"
#include "AlphaCore.h"
#include "AxCollision.DataType.h"
#include "AxSolidCollisionDetection.ShareCode.h"
#include "AxGeo.h"
#include "GridDense/AxGridDense.ShareCode.h"

using namespace AlphaCore;
using namespace AlphaCore::Collision;
using namespace AlphaCore::GridDense::ShareCode;

namespace AlphaCore
{
	ALPHA_SHARE_FUNC void Info(const AxBufferArrayHandleI32& handleI)
	{

	}


	ALPHA_SHARE_FUNC void Info(const AxArrayDesc<AxInt32>& handleI)
	{

	}

	ALPHA_SHARE_FUNC void RTriangleTo6B(AxUChar rtriangleToken, bool* b6)
	{

	}

	template<typename T, typename S>
	ALPHA_SHARE_FUNC void Info(const AxBufferHandle<T, S>& handle)
	{
		AxLogBlock log;
		log.Push("--------------------\n");
		log.Push(" Buffer Name : \"");
		log.Push(handle.Name);
		log.Push("\"   ");

		log.Push(" Is Valid : ");
		log.Push(handle.IsValid ? "True" : "False");
		log.Push("   ");

		AxUInt64 size = handle.Size();
		log.LogInt(" Buffer Size : ", (int)size);
		log.LogInt(" Storage Size : ", (int)handle.SizeStorage());
		log.Push(" Is Array : ");
		log.Push(handle.IsArray ? "True" : "False");
		log.Push("\n--------------------\n");
		log.Trace();
	}


	ALPHA_SHARE_FUNC void Info(const char* head)
	{
		printf("%s \n", head);
	}

	template<typename T>
	ALPHA_SHARE_FUNC void Info(const char* head,const ScalarFieldRAWDesc<T>& field )
	{
		printf("%s | %s\n", head, field.IsValid ? "Valid":"InValid");
	}


	ALPHA_SHARE_FUNC void Info(const char* head, const AxUInt64& ui64)
	{
		printf("%s %lld \n", head, ui64);
	}

	ALPHA_SHARE_FUNC void Info(const char* head, const AxUInt32& ui32)
	{
		printf("%s %d \n", head, ui32);
	}

	ALPHA_SHARE_FUNC void Info(const char* head, const AxFp32& f32)
	{
		printf("%s %f \n", head, f32);
	}
	
	ALPHA_SHARE_FUNC void Info(const char* head ,const AxVector3I& v)
	{
		printf("%s [%d,%d,%d] \n", head, v.x, v.y, v.z);
	}

	ALPHA_SHARE_FUNC void InfoPoint(const AxVector3& v)
	{
		printf("v %f %f %f\n", v.x, v.y, v.z);
	}

	ALPHA_SHARE_FUNC void Info(const AxVector3I& v)
	{
		Info("", v);
 	}

	//AxPrimitiveHandle
	ALPHA_SHARE_FUNC void Info(const char* head, const AxPrimitiveHandle& prim)
	{
		AxLogBlock log;
		log.Push(head);
		log.LogInt("Primitive NPoints : ", prim.Points);
		log.Push(" | Indices : ");
		AX_FOR_I(prim.Points)
		{
			char zz[16];
			if (i != 0)
				log.Push(",");
			log.Push(log.__itoa(prim.Indices[i], zz));
		}
		log.Push("\n");
		log.Trace();
	}
	//
	ALPHA_SHARE_FUNC void Info(const char* head, const AxVector3& v3)
	{
		AxLogBlock log;
		log.LogFloat3(head, v3.x, v3.y, v3.z);
	}

	ALPHA_SHARE_FUNC void Info(const AxVector3& v3)
	{
		AxLogBlock log;
		log.LogFloat3("AxVector3 : ", v3.x, v3.y, v3.z);
	}

	ALPHA_SHARE_FUNC void Info(const AxPrimitiveHandle& prim)
	{
		Info("", prim);
	}

	ALPHA_SHARE_FUNC void InfoRTriangle(const char* head, const AxUChar& rtToken)
	{
		AlphaCore::Collision::ShareCode::PrintRTriangleInfo(head, rtToken);
	}

	ALPHA_SHARE_FUNC void InfoRTriangle(const AxUChar& rtToken)
	{
		AlphaCore::Collision::ShareCode::PrintRTriangleInfo("", rtToken);
	}
}




namespace AlphaCore
{
	namespace Geometric
	{
		ALPHA_SHARE_FUNC AxUInt32 AddPoint(AxGeoDesc $geo0, AxVector3 pos)
		{

			return 0;
		}

		ALPHA_SHARE_FUNC void AddTriangle(AxGeoDesc $geo0, AxUInt32 i0, AxUInt32 i1, AxUInt32 i2)
		{


		}


		ALPHA_SHARE_FUNC void AddPrimitivePolyLine(
			AxGeoDesc $geo0,
			AxInt32 i0,
			AxInt32 i1,
			AxInt32 i2 = -1,
			AxInt32 i3 = -1,
			AxInt32 i4 = -1)
		{


		}

		ALPHA_SHARE_FUNC AxVector3 AddContact(
			AxGeoDesc $geo0,
			AxUInt32 i0,
			AxUInt32 i1,
			AxUInt32 i2,
			AxUInt32 i3)
		{


		}
	}

	namespace Grid
	{
		template<typename T>
		ALPHA_SHARE_FUNC AxVector3I IndexToIndex3(ScalarFieldRAWDesc<T> scalarField,AxUInt32 threadID)
		{
			AxVector3UI res = scalarField.FieldInfo.Resolution;
			AxUInt32 nvSlice = res.x * res.y;
			AxUInt32 x = threadID % res.x;
			AxUInt32 y = (threadID % nvSlice) / res.x;
			AxUInt32 z = threadID / nvSlice;
			return MakeVector3I(x, y, z);
		}

		template<typename T>
		ALPHA_SHARE_FUNC AxVector3I IndexToIndex3(VectorFieldRawDesc<T> vecField, AxUInt32 threadID)
		{
			return MakeVector3I(1, 1, 1);
		}

		template<typename T>
		ALPHA_SHARE_FUNC AxUInt32 Index3ToIndex(ScalarFieldRAWDesc<T> scalarField,AxVector3I id3)
		{
			auto info = scalarField.FieldInfo;
			return id3.z * info.Resolution.x * info.Resolution.y + id3.y * info.Resolution.x + id3.x;
		}

		template<typename T>
		ALPHA_SHARE_FUNC AxVector3 Index3ToPosition(
			ScalarFieldRAWDesc<T> scalarField, 
			AxVector3I id3,
			bool useRotation = true)
		{
			auto info = scalarField.FieldInfo;
			AxVector3 origin = info.Pivot - info.FieldSize * 0.5f;
			AxVector3 localPos = MakeVector3(
				((AxFp32)id3.x + 0.5f) * info.VoxelSize.x,
				((AxFp32)id3.y + 0.5f) * info.VoxelSize.y,
				((AxFp32)id3.z + 0.5f) * info.VoxelSize.z);
			if (!useRotation)
				return localPos + origin;
			AxVector3 relPos = (localPos + origin) - info.Pivot;
			return relPos * info.RotationMatrix + info.Pivot;
		}

		template<typename T>
		ALPHA_SHARE_FUNC AxVector3 Index3ToPosition(VectorFieldRawDesc<T> vecField, AxVector3I id3)
		{
			return MakeVector3T(0.0f, 0.0f, 0.0f);
		}
		

		template<typename T>
		ALPHA_SHARE_FUNC AxVector3T<T> SampleFieldValue(VectorFieldRawDesc<T> vecField, AxVector3 pos)
		{
		}

		template<typename T>
		ALPHA_SHARE_FUNC T SampleFieldValue(ScalarFieldRAWDesc<T> scalarField, AxVector3 pos)
		{
			/*
			if (useRotation)
			{
				pos -= info.Pivot;
				pos *= info.InverseRotMatrix;
				pos += info.Pivot;
			}
			*/
			auto info = scalarField.FieldInfo;
			AxVector3 size = info.FieldSize;
			AxVector3 origin = info.Pivot - size * 0.5f;
			AxVector3 bboxSpace = MakeVector3(
				((pos.x - origin.x) / size.x),
				((pos.y - origin.y) / size.y),
				((pos.z - origin.z) / size.z));

			AxVector3 voxelCoord = MakeVector3(
				bboxSpace.x * (AxFp32)info.Resolution.x,
				bboxSpace.y * (AxFp32)info.Resolution.y,
				bboxSpace.z * (AxFp32)info.Resolution.z);

			AxVector3 coordB = voxelCoord - 0.5f;
			AxVector3I lowerLeft = MakeVector3I(
				floor(coordB.x),
				floor(coordB.y),
				floor(coordB.z));

			float cx = coordB.x - (float)lowerLeft.x;
			float cy = coordB.y - (float)lowerLeft.y;
			float cz = coordB.z - (float)lowerLeft.z;
			auto rawData = scalarField.RawDataPtr;
			//TODO Integrate VDB
			AxFp32 v0 = GetValue(lowerLeft.x, lowerLeft.y, lowerLeft.z, rawData, info);
			AxFp32 v1 = GetValue(lowerLeft.x + 1, lowerLeft.y, lowerLeft.z, rawData, info);
			AxFp32 v2 = GetValue(lowerLeft.x, lowerLeft.y + 1, lowerLeft.z, rawData, info);
			AxFp32 v3 = GetValue(lowerLeft.x + 1, lowerLeft.y + 1, lowerLeft.z, rawData, info);
			AxFp32 v4 = GetValue(lowerLeft.x, lowerLeft.y, lowerLeft.z + 1, rawData, info);
			AxFp32 v5 = GetValue(lowerLeft.x + 1, lowerLeft.y, lowerLeft.z + 1, rawData, info);
			AxFp32 v6 = GetValue(lowerLeft.x, lowerLeft.y + 1, lowerLeft.z + 1, rawData, info);
			AxFp32 v7 = GetValue(lowerLeft.x + 1, lowerLeft.y + 1, lowerLeft.z + 1, rawData, info);

			float iv1 = AlphaCore::Math::LerpF32(AlphaCore::Math::LerpF32(v0, v1, cx), AlphaCore::Math::LerpF32(v2, v3, cx), cy);
			float iv2 = AlphaCore::Math::LerpF32(AlphaCore::Math::LerpF32(v4, v5, cx), AlphaCore::Math::LerpF32(v6, v7, cx), cy);
			return AlphaCore::Math::LerpF32(iv1, iv2, cz);
		}

		ALPHA_SHARE_FUNC AxVector3 RampColor(AxFp32 value, AxRampColor32RAWData rampColor)
		{

		}

		ALPHA_SHARE_FUNC AxVector3 RampValue(AxFp32 value, AxRampCurve32RAWData rampColor)
		{

		}

		template<typename T>
		ALPHA_SHARE_FUNC void SetFieldValue(const ScalarFieldRAWDesc<T> field, AxVector3I id3, AxFp32 value)
		{
			//AxContact
			if (field.IsValid == false)
				return;
			field.RawDataPtr[Index3ToIndex(field,id3)] = value;

		}

		template<typename T>
		ALPHA_SHARE_FUNC void SetFieldValue(const VectorFieldRawDesc<T> field, AxVector3I id3, AxVector3T<T> value)
		{
			//AxContact
			if (field.Active == false)
				return;
			SetValue<T>(id3.x, id3.y, id3.z, value.x, field.RawDataPtrX, field.FieldInfoX);
			SetValue<T>(id3.x, id3.y, id3.z, value.y, field.RawDataPtrY, field.FieldInfoY);
			SetValue<T>(id3.x, id3.y, id3.z, value.z, field.RawDataPtrZ, field.FieldInfoZ);
		}


		template<typename T>
		ALPHA_SHARE_FUNC AxVector3T<T> GetFieldValue(const VectorFieldRawDesc<T> field, AxVector3I id3)
		{
			if (field.Active == false)
				return MakeVector3T<T>(0, 0, 0);

			T x = GetValue<T>(id3.x, id3.y, id3.z, field.RawDataPtrX, field.FieldInfoX);
			T y = GetValue<T>(id3.x, id3.y, id3.z, field.RawDataPtrY, field.FieldInfoY);
			T z = GetValue<T>(id3.x, id3.y, id3.z, field.RawDataPtrZ, field.FieldInfoZ);
			return MakeVector3T<T>(x, y, z);
		}

		template<typename T>
		ALPHA_SHARE_FUNC T GetFieldValue(const VectorFieldRawDesc<T>& field, AxInt32 x, AxInt32 y, AxInt32 z, AxUInt32 off)
		{
			if (field.Active == false)
				return T(0);
			if (off == 0)
				return GetValue<T>(x, y, z, field.RawDataPtrX, field.FieldInfoX);
			if (off == 1)
				return GetValue<T>(x, y, z, field.RawDataPtrY, field.FieldInfoY);
			if (off == 2)
				return GetValue<T>(x, y, z, field.RawDataPtrZ, field.FieldInfoZ);
			return T(0);
		}


		template<typename T>
		ALPHA_SHARE_FUNC T GetFieldValue(const ScalarFieldRAWDesc<T> field, AxVector3I id3)
		{
			if (field.IsValid == false)
				return 0.0f;
			return GetValue(id3.x, id3.y, id3.z, field.RawDataPtr, field.FieldInfo);
 		}
	}

	namespace Collision
	{

		/*
		ALPHA_SHARE_FUNC AxUInt32 FlipCCDTrianglePair(
				AxVector3* selfPosRaw,			AxVector3* otherPosRaw,
				AxVector3* selfPrdPRaw,			AxVector3* otherPrdPRaw,
				AxUInt32 selfTriangleID,		AxUInt32 otherTriangleID,
				AxUChar selfRTriangleInfo,		AxUChar otherRTriangleInfo,
				AxFp32 selfMaxEta,				AxFp32 otherMaxEta,
				AxVector3UI selfTriaglePtID3,   AxVector3UI otherTriaglePtID3,
				const AxAABB& selfTriangleAABB,
				const AxAABB& otherTriangleAABB,
				AxContact* ctxBuffer,
				bool processSelf,
				AxSPMDTick& lock,
		*/
		ALPHA_SHARE_FUNC void CCDTrianglePair(
			AxTriangleElementHandle triangleA,
			AxTriangleElementHandle triangleB,
			AxBufferContactHandle& contactPool)
		{
			if (triangleA.Valid == false || triangleB.Valid == false)
			{
				Info("Invalid Triangle");
				return;
			}
			if (contactPool.IsValid == false)
			{
				Info("Non-Contact Buffer");
				return;
			}
			AxVector3* posRawThis = triangleA.PosHandle.data;
			AxVector3* posRawOther = triangleB.PosHandle.data;
			AxVector3* prdPRawThis = triangleA.PrdPHandle.data;
			AxVector3* prdPOther = triangleB.PrdPHandle.data;

			AxUInt32 contacts = AlphaCore::Collision::ShareCode::FlipCCDTrianglePair(
				posRawThis, posRawOther,
				prdPRawThis, prdPOther,
				triangleA.Index, triangleB.Index,
				triangleA.RTriangleToken, triangleB.RTriangleToken,
				triangleA.Thickness, triangleB.Thickness,
				triangleA.ID3, triangleB.ID3,
				triangleA.BoundingBox, triangleB.BoundingBox,
				contactPool.data,
				true, contactPool.tick);
		}
	}
}

using namespace AlphaCore::Grid;

#endif