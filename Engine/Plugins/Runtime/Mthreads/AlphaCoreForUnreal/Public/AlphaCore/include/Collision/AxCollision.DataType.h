#ifndef __AX_COLLISION_DOT_DATA_TYPE_H__
#define __AX_COLLISION_DOT_DATA_TYPE_H__

#include "AxMacro.h"
#include "AxDataType.h"
#include "Utility/AxStorage.h"

class AxGeometry;
typedef AxUChar AxCtxToken;


namespace AlphaCore
{
	enum AxContactType
	{
		NonContact = 0b0000000, // ??
		kVSDF_Contact = 0b0000001,
		kVF = 0b0000010,
		kEE = 0b0000100,
		kVE = 0b0001000,
		kVV = 0b0010000,
		kWithStatic = 0b0100000,
		kSwapPivot = 0b1000000
	};

	struct AxCollisionTaskDesc
	{
		AxUChar numVertexTasks;
		AxUChar numEdgeTasks;
		AxUChar TaskToken[6];
	};

	struct AxCapsuleCollider
	{
		AxFp32    Radius;
		AxFp32    HalfHeight;
		AxVector3 Pivot;
		AxVector3 Direction;
	};

	struct AxOBBCollider
	{
		AxVector3 Size;  //width,height,length
		AxVector3 Pivot;
		AxVector3 Forward;
		AxVector3 Up;
	};

	inline AxCollisionTaskDesc MakeDefaultCollisionTaskDesc()
	{
		AxCollisionTaskDesc desc;
		desc.TaskToken[0] = 0;
		desc.TaskToken[1] = 0;
		desc.TaskToken[2] = 0;
		desc.TaskToken[3] = 0;
		desc.TaskToken[4] = 0;
		desc.TaskToken[5] = 0;
		desc.numVertexTasks = 0;
		desc.numEdgeTasks = 0;
		return desc;
	}

	struct AxContact
	{
		bool NormalDir;	//remove this
		AxCtxToken Token; //Token VF / EE 
		AxUInt32 Points[4];
	};

	ALPHA_SHARE_FUNC void PrintInfo(const AxContact& ctx)
	{
		printf("Contact: [%d,%d,%d,%d]\n", ctx.Points[0], ctx.Points[1], ctx.Points[2], ctx.Points[3]);
	}

	namespace Collision
	{
		namespace SimData
		{
			ALPHA_SPMD_CLASS class AxPBDCollisionResolveData
			{
			public:
				AxPBDCollisionResolveData()
				{
					ContactFixProp = nullptr;
					CollisionNProp = nullptr;
					Contact2PtIndicesProp = nullptr;
					ContactIdenityProp = nullptr;
					Point2ContactVertexStartProp = nullptr;
					Point2ContactVertexEndProp = nullptr;
					CollisionMoveProp = nullptr;
				}
				~AxPBDCollisionResolveData()
				{
					std::cout << "collision resolve data deconstruction" << std::endl;
					if (m_OwnGeometry != nullptr)
					{
						delete ContactFixProp;
						delete Contact2PtIndicesProp;
						delete ContactIdenityProp;
						delete Point2ContactVertexStartProp;
						delete Point2ContactVertexEndProp;
					}
				}

				struct RawData
				{
					bool Valid;
					AxUInt32 ContactNum;
					AxVector4* ContactFixRaw; // ContactVertex FixDirection + Static or Self ContactNum*4
					AxVector3* CollisionNRaw;//for Debug
					AxUInt32* Contact2PtIndicesRaw;
					AxUInt32* ContactIdenityRaw;
					AxUInt32* Point2ContactVertexStartRaw;
					AxUInt32* Point2ContactVertexEndRaw;
					AxVector3* CollisionMoveRaw;

				};

				AxBufferV4* ContactFixProp;
				AxBufferV3* CollisionNProp;
				AxBufferUInt32* Contact2PtIndicesProp;
				AxBufferUInt32* ContactIdenityProp;
				AxBufferUInt32* Point2ContactVertexStartProp;
				AxBufferUInt32* Point2ContactVertexEndProp;
				AxBufferV3* CollisionMoveProp;

				RawData GetRAWDesc(AlphaCore::AxBackendAPI deviceMode = AlphaCore::AxBackendAPI::CPUx86) const;

				void Build(AxGeometry* geo, AxUInt32 ctxNum, bool initDeviceData = false);
				void LoadToDevice();

				AxUInt32 SPMDPivotSize();
				AxGeometry* GetOwnGeometry() { return m_OwnGeometry; };

			private:
				AxGeometry* m_OwnGeometry;

			};
		}
		//----------------------------------
		// Vertex-Edge Contact Token
		//----------------------------------
		ALPHA_SHARE_FUNC AxCtxToken VESelfToken()
		{
			return AxContactType::kVE;
		}

		ALPHA_SHARE_FUNC int VEStaticToken()
		{
			return AxContactType::kVE | AxContactType::kWithStatic;
		}

		ALPHA_SHARE_FUNC int VStaticEToken()
		{
			return AxContactType::kVE | AxContactType::kWithStatic | AxContactType::kSwapPivot;
		}

		//----------------------------------
		// Vertex-Vertex Contact Token
		//----------------------------------
		ALPHA_SHARE_FUNC int VVSelfToken()
		{
			return AxContactType::kVV;
		}

		ALPHA_SHARE_FUNC int VVStaticToken()
		{
			return AxContactType::kVV | AxContactType::kWithStatic;
		}

		ALPHA_SHARE_FUNC int VStaticVToken()
		{
			return AxContactType::kVV | AxContactType::kWithStatic | AxContactType::kSwapPivot;
		}
		//----------------------------------
		// Vertex-Face Contact Token
		//----------------------------------
		ALPHA_SHARE_FUNC int VFSelfToken()
		{
			return AxContactType::kVF;
		}

		ALPHA_SHARE_FUNC int VFStaticToken()
		{
			return (AxContactType::kVF | AxContactType::kWithStatic);
		}

		ALPHA_SHARE_FUNC int VStaticFToken()
		{
			return AxContactType::kVF | AxContactType::kWithStatic | AxContactType::kSwapPivot;
		}
		//----------------------------------
		// Edge-Edge Contact Token
		//----------------------------------
		ALPHA_SHARE_FUNC int EESelfToken()
		{
			return AxContactType::kEE;
		}

		ALPHA_SHARE_FUNC int EEStaticToken()
		{
			return (AxContactType::kEE | AxContactType::kWithStatic);
		}

		//-----------------
		// RxContact Real
		//-----------------
		ALPHA_SHARE_FUNC bool IsEE(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kEE) != AxContactType::kEE)
				return false;
			return true;
		}

		ALPHA_SHARE_FUNC bool IsEEStatic(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kEE) != AxContactType::kEE)
				return false;
			if ((ctx & AxContactType::kWithStatic) != AxContactType::kWithStatic)
				return false;
			return true;
		}
		ALPHA_SHARE_FUNC bool IsEESelf(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kEE) != AxContactType::kEE)
				return false;
			if ((ctx & AxContactType::kWithStatic) == AxContactType::kWithStatic)
				return false;
			return true;
		}

		ALPHA_SHARE_FUNC bool ContactInvolvesStatic(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kWithStatic) != AxContactType::kWithStatic)
				return false;
			return true;
		}
		//----------------------------------
		// Vertex-Edge Contact Token
		//----------------------------------
		ALPHA_SHARE_FUNC bool IsVEStatic(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kVE) != AxContactType::kVE)
				return false;
			if ((ctx & AxContactType::kWithStatic) != AxContactType::kWithStatic)
				return false;
			if ((ctx & AxContactType::kSwapPivot) == AxContactType::kSwapPivot)
				return false;
			return true;
		}

		ALPHA_SHARE_FUNC bool IsVStaticE(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kVE) != AxContactType::kVE)
				return false;
			if ((ctx & AxContactType::kWithStatic) != AxContactType::kWithStatic)
				return false;
			if ((ctx & AxContactType::kSwapPivot) != AxContactType::kSwapPivot)
				return false;
			return true;
		}

		ALPHA_SHARE_FUNC bool IsVESelf(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kVE) != AxContactType::kVE)
				return false;
			if ((ctx & AxContactType::kWithStatic) == AxContactType::kWithStatic)
				return false;
			return true;
		}
		ALPHA_SHARE_FUNC bool IsVE(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kVE) != AxContactType::kVE)
				return false;
			return true;
		}
		//----------------------------------
		// Vertex-Vertex Contact Token
		//----------------------------------
		ALPHA_SHARE_FUNC bool IsVVStatic(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kVV) != AxContactType::kVV)
				return false;
			if ((ctx & AxContactType::kWithStatic) != AxContactType::kWithStatic)
				return false;
			if ((ctx & AxContactType::kSwapPivot) == AxContactType::kSwapPivot)
				return false;
			return true;
		}

		ALPHA_SHARE_FUNC bool IsVStaticV(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kVV) != AxContactType::kVV)
				return false;
			if ((ctx & AxContactType::kWithStatic) != AxContactType::kWithStatic)
				return false;
			if ((ctx & AxContactType::kSwapPivot) != AxContactType::kSwapPivot)
				return false;
			return true;
		}

		ALPHA_SHARE_FUNC bool IsVVSelf(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kVV) != AxContactType::kVV)
				return false;
			if ((ctx & AxContactType::kWithStatic) == AxContactType::kWithStatic)
				return false;
			return true;
		}

		ALPHA_SHARE_FUNC bool IsVV(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kVV) != AxContactType::kVV)
				return false;
			return true;
		}
		//----------------------------------
		// Vertex-Face Contact Token
		//----------------------------------
		ALPHA_SHARE_FUNC bool IsVF(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kVF) != AxContactType::kVF)
				return false;
			return true;
		}

		ALPHA_SHARE_FUNC bool IsVFStatic(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kVF) != AxContactType::kVF)
				return false;
			if ((ctx & AxContactType::kWithStatic) != AxContactType::kWithStatic)
				return false;
			if ((ctx & AxContactType::kSwapPivot) == AxContactType::kSwapPivot)
				return false;
			return true;
		}

		ALPHA_SHARE_FUNC bool IsVFSelf(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kVF) != AxContactType::kVF)
				return false;
			if ((ctx & AxContactType::kWithStatic) == AxContactType::kWithStatic)
				return false;
			return true;
		}

		ALPHA_SHARE_FUNC bool IsVStaticF(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kVF) != AxContactType::kVF)
				return false;
			if ((ctx & AxContactType::kWithStatic) != AxContactType::kWithStatic)
				return false;
			if ((ctx & AxContactType::kSwapPivot) != AxContactType::kSwapPivot)
				return false;
			return true;
		}

		//----------------------------------
		// Edge-Edge Contact Token
		//----------------------------------
		ALPHA_SHARE_FUNC bool IsContactWithStatic(const AxCtxToken& ctx)
		{
			if ((ctx & AxContactType::kWithStatic) == AxContactType::kWithStatic)
				return true;
			return false;
		}
		ALPHA_SHARE_FUNC bool IsContactWithSelf(const AxCtxToken& ctx)
		{
			if (IsVFSelf(ctx) || IsVVSelf(ctx) || IsVESelf(ctx) || IsEESelf(ctx))
				return true;
			return false;
		}

		ALPHA_SHARE_FUNC bool UseSelfBuf(const AxCtxToken& ctx, AxUChar id)
		{
			if (IsVF(ctx))
			{
				if (IsVFSelf(ctx))
					return true;
				if (IsVFStatic(ctx))
					return id == 0 ? true : false;
				if (IsVStaticF(ctx))
					return id == 0 ? false : true;
			}
			if (IsEE(ctx))
			{
				if (IsEESelf(ctx))
					return true;
				if (IsEEStatic(ctx))
					return id <= 1 ? true : false;
			}
			if (IsVE(ctx))
			{
				if (IsVESelf(ctx))
					return true;
				if (IsVEStatic(ctx))
					return (id == 0) ? true : false;
				if (IsVStaticE(ctx))
					return (id == 0) ? false : true;
			}

			if (IsVV(ctx))
			{
				if (IsVVSelf(ctx))
					return true;
				if (IsVVStatic(ctx))
					return  id == 1 ? false : true;
				if (IsVStaticV(ctx))
					return  id == 1 ? true : false;
			}
			return false;
		}
	}
}


inline std::ostream& operator << (std::ostream& out, const AlphaCore::AxCollisionTaskDesc& c)
{
	out << "AsRaw@CollisionTaskDesc:" << c.numVertexTasks << "," << c.numEdgeTasks << ",";
	out << c.TaskToken[0] << "," << c.TaskToken[1] << "," << c.TaskToken[2] << ",";
	out << c.TaskToken[3] << "," << c.TaskToken[4] << "," << c.TaskToken[5];
	return out;
}


inline std::ostream& operator << (std::ostream& out, const AlphaCore::AxCapsuleCollider& capsule)
{
	out << "AsRaw@Capsule:" << capsule.Pivot << "," << capsule.Direction << ",";
	out << capsule.Radius << "," << capsule.HalfHeight << std::endl;
	return out;
}


inline std::ostream& operator << (std::ostream& out, const AlphaCore::AxContact& c)
{
	if (AlphaCore::Collision::IsVFSelf(c.Token))
		out << "type: VF \t| ";
	if (AlphaCore::Collision::IsVFStatic(c.Token))
		out << "type: VF Static \t| ";
	if (AlphaCore::Collision::IsVVSelf(c.Token))
		out << "type: VV \t| ";
	if (AlphaCore::Collision::IsVVStatic(c.Token))
		out << "type: VV Static \t| ";
	if (AlphaCore::Collision::IsVStaticV(c.Token))
		out << "type: V Static V \t| ";
	if (AlphaCore::Collision::IsVESelf(c.Token))
		out << "type: VE \t| ";
	if (AlphaCore::Collision::IsVEStatic(c.Token))
		out << "type: VE Static \t| ";
	if (AlphaCore::Collision::IsVStaticE(c.Token))
		out << "type: V Static E\t| ";
	if (AlphaCore::Collision::IsEESelf(c.Token))
		out << "type: EE \t| ";
	if (AlphaCore::Collision::IsEEStatic(c.Token))
		out << "type: EE  Static \t| ";
	if (AlphaCore::Collision::IsVStaticF(c.Token))
		out << "type: V Static  F \t| ";

	out << "  Normal Direction : " << (c.NormalDir ? "True" : "False") << " | ";
	out << "  Id : " << c.Points[0] << ", Id : " << c.Points[1] << " ," << c.Points[2] << " ," << c.Points[3] << " | \n";

	int tokenID = -1;
	if (AlphaCore::Collision::IsEESelf(c.Token))
		tokenID = 0;
	if (AlphaCore::Collision::IsEEStatic(c.Token))
		tokenID = 1;
	if (AlphaCore::Collision::IsVFSelf(c.Token))
		tokenID = 2;
	if (AlphaCore::Collision::IsVFStatic(c.Token))
		tokenID = 3;
	if (AlphaCore::Collision::IsVStaticF(c.Token))
		tokenID = 4;
	out << "AsRaw@Contact:" << tokenID << "|" << c.Points[0] << "," << c.Points[1] << "," << c.Points[2] << "," << c.Points[3];

	return out;
}

typedef AxStorage<AlphaCore::AxContact>  AxBufferContact;
typedef AxStorage<AlphaCore::AxCollisionTaskDesc>  AxBufferCollisionTask;
typedef AxBufferHandle<AlphaCore::AxContact, void> AxBufferContactHandle;

#endif