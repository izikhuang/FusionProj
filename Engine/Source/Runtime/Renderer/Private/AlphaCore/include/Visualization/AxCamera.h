#ifndef __ALPHA_CORE_CAMERA_3D_H__
#define __ALPHA_CORE_CAMERA_3D_H__

#include <AxMacro.h>
#include <Math/AxVectorBase.h>
#include <Math/AxMatrixBase.h>
#include <Utility/AxDescrition.h>

namespace AlphaCore
{
	namespace Visualization
	{
		ALPHA_KERNEL_FUNC AxVector3	  PixelToRay(float u,float v);
		ALPHA_KERNEL_FUNC AxMatrix4x4 ExactCamera(AlphaCore::Desc::AxCameraInfo cam);

	}
}

#endif