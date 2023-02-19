#ifndef __AX_PBD_H__
#define __AX_PBD_H__

#include "AxPBD.ShareCode.h"
#include "Utility/AxStorage.h"
#include "Collision/AxContact.h"

#include "AxGeo.h"

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
				void SetTopolGeo(AxGeometry* topolGeo);
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
				// Volume for pressure constraint
				AxBufferF*    PtVolume;    // point attribute volume
				AxIdxMapUI32* VolumePts;   // triangle indices list for calculating each point volume
				AxBufferV3*   PressureGradient;
				AxGeometry*   topolGeo;
				//Hair Object
				AxBufferF*	Inertia;
				AxBufferV3*	Omega;
				AxBufferV3*	PrevOmega;
				AxBufferF*	Orient;
				AxBufferF*	PrevOrient;
				AxGeometry* m_OwnGeometry;

				void SetEdgeColoringGroupData(AxUInt32* data,AxUInt32 size);
				void PrintEdgeColoringGroupData();

			private:
				void buildVolumePts();
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

		ALPHA_SPMD_FUNC void ComputePressureVal(
			AxBufferV3* posBuf,
			AxBufferF* ptVolumeBuf,
			AxIdxMapUI32* pt2VolPtsMap,
			AxBufferV3* outPressureGradBuf);

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
			AxBufferF* ptVolumeBuf,
 			AxBufferV3* pressureGradBuf,
			AxBuffer2UI* SPMDGroupSNBuf,
			float dt);

		namespace CUDA {

			ALPHA_SPMD_FUNC void ComputePressureVal(
				AxBufferV3* posBuf,
				AxBufferF* ptVolumeBuf,
				AxIdxMapUI32* pt2VolPtsMap,
				AxBufferV3* outPressureGradBuf,
				AxUInt32 blockSize = 512);

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
				AxBufferF* ptVolumeBuf,
				AxBufferV3* pressureGradBuf,
				AxBuffer2UI* SPMDGroupSNBuf,
				float dt,
				AxUInt32 blockSize = 512);
		}
	}
}//@namespace end of : PBD
#endif