#ifndef __AX_PBD_H__
#define __AX_PBD_H__

#include "AxPBD.ShareCode.h"
#include <Utility/AxStorage.h>
#include <Collision/AxContact.h>

#include <AxGeo.h>

namespace AlphaCore {
	namespace PBD {
		namespace SimData
		{
			class AxPBDConstraintData
			{
			public:
				AxPBDConstraintData();
				~AxPBDConstraintData();

				void Build(AxGeometry* constraintGeo, AxGeometry* attachGeo = nullptr);
				void DeviceMalloc();
				void LoadToDevice();

				AxBufferF*	 StiffnessBuffer;
				AxBufferF*	 CompressStiffBuffer;
				AxBufferF*	 DampingRatioBuffer;
				AxBufferF*	 RestBuffer;
				AxBufferV3*	 LambdaBuffer;
				AxBufferV4*	 RestVectorBuffer;
				AxBufferI*	 ConstraintTypeId;
				AxBufferV3*	 AttachUVWBuffer;
				AxBufferV3*	 AttachPrevPBuffer;
				AxBufferV3*	 AttachPrdPBuffer;
				AxBufferV3*	 AttachPosBuffer;
				AxBuffer3UI* AttachPtId3Buffer; //@Problem Houdini export as Int3
				AxBufferV4*	 DmInvBuffer;
				AxBuffer2UI* SPMDGroupSNBuffer;
				//Hair Object
				AxBufferF*	Inertia;
				AxBufferV3*	Omega;
				AxBufferV3*	PrevOmega;
				AxBufferF*	Orient;
				AxBufferF*	PrevOrient;
				AxGeometry* m_OwnGeometry;
  			};

			class AxPBDCollisionData
			{
			public:
				AxPBDCollisionData();
				~AxPBDCollisionData();

				void Build(AxGeometry* geo);
				void DeviceMalloc();
				void LoadToDevice();

				AxBufferUChar* RTriangleInfo;
				
			private:
			};
		}
	}
}

namespace AlphaCore {
	namespace PBD {
		ALPHA_SPMD_FUNC void UpdteXPBDGaussSeidel(AxBuffer2UI* primList2IBuf,
			AxBufferUInt32* prim2PtIndicesBuf,
			AxBufferI* primTypeBuf,
			AxBufferV3* posBuf,
			AxBufferV3* pprevBuf,
			AxBufferF* restLengthBuf,
			AxBufferV4* restVectorBuf,
			AxBufferF* massBuf,
			AxBufferF* stiffnessBuf,
			AxBufferF* dampingRatioBuf,
			AxBufferV3* lambdaBuf,
			AxBufferV3* attachUVWBuf,
			AxBufferV3* attachPosBuf,
			AxBuffer3UI* attachPt3IBuf,
			AxBufferF*	orientBuf,//Raw 纯数据
			AxBufferF*	orientprevBuf,
			AxBufferF*	inertiasBuf,
			AxBuffer2UI* SPMDGroupSNBuf,
			float dt);

		namespace CUDA {
			ALPHA_SPMD_FUNC void UpdteXPBDGaussSeidel(AxBuffer2UI* primList2IBuf,
				AxBufferUInt32* prim2PtIndicesBuf,
				AxBufferI* primTypeBuf,
				AxBufferV3* posBuf,
				AxBufferV3* pprevBuf,
				AxBufferF* restLengthBuf,
				AxBufferV4* restVectorBuf,
				AxBufferF* massBuf,
				AxBufferF* stiffnessBuf,
				AxBufferF* dampingRatioBuf,
				AxBufferV3* lambdaBuf,
				AxBufferV3* attachUVWBuf,
				AxBufferV3* attachPosBuf,
				AxBuffer3UI* attachPt3IBuf,
				AxBufferF* orientBuf,//Raw 纯数据
				AxBufferF* orientprevBuf,
				AxBufferF* inertiasBuf,
				AxBuffer2UI* SPMDGroupSNBuf,
				float dt,
				AxUInt32 blockSize = 512);

		}
	}
}//@namespace end of : PBD
#endif