#ifndef __AX_SOLIDUTILITY_DOT_DATATYPE_H__
#define __AX_SOLIDUTILITY_DOT_DATATYPE_H__

#include "Utility/AxStorage.h"
#include "Collision/AxCollision.DataType.h"
class AxGeometry;
namespace AlphaCore
{
	namespace SolidUtility
	{
		static const char* ModuleName = "SolidUtility";
		namespace SimData
		{
			ALPHA_SPMD_CLASS class AxSolidData
			{
			public:
				AxSolidData();
				~AxSolidData();

				struct RAWDesc
				{
					bool Valid;
					AxVector3*  PosRaw;
					AxVector3*  PrdPositionRaw;
					AxVector3*  PrevPosRaw;
					AxVector3*  LastPosRaw;
					AxVector3*  VelRaw;
					AxVector3*  VelLastRaw;
					AxVector3*  VelPrevRaw;
					AxVector3*  AccelRaw;

					AxUChar* RTriangleTokenRaw;
					AxFp32*	MassRaw;
					AxFp32*	ThicknessRaw;
					AxFp32*	PrimDepthRaw;
					AxFp32*	PointMaxEtaRaw;
					AxFp32*	PrimMaxEtaRaw;
					AxVector2UI* EdgeListRaw;
					AxVector3UI* IndicesBuffer;
					AxUInt32 NumPrimitives;
					AxUInt32 NumPoints;
				};

				static RAWDesc GetRAWDesc(AxSolidData* target,AlphaCore::AxBackendAPI deviceMode);

				void Build(AxGeometry* geo, bool initDeviceData = false);
				void DeviceMalloc();
				void LoadToDevice();

				AxUInt32 SPMDPivotSize();


				AxBufferV3*    RestPosBuffer;
				AxBufferV3*    PosBuffer;
				AxBufferV3*    PrdPositionBuffer;
				AxBufferV3*    PrevPosBuffer;
				AxBufferV3*    LastPosBuffer;
							   
				AxBufferV3*    VelBuffer;
				AxBufferV3*    VelLastBuffer;
				AxBufferV3*    VelPrevBuffer;
							   
				AxBufferV3*    AccelBuffer;
				AxBufferF*	   MassBuffer;
				AxBufferF*	   ThicknessBuffer;
				AxBuffer2UI*   EdgeList;
				AxBufferF*	   PrimDepthBuffer;
				AxBufferF*	   PointMaxEtaBuffer;
				AxBufferF*	   PrimMaxEtaBuffer;
				AxBufferUChar* RTriangleBuffer;

				AxGeometry* GetOwnGeometry() { return m_OwnGeometry; };
			private:
				AxGeometry* m_OwnGeometry;
			};
		}
	}
}

#endif
