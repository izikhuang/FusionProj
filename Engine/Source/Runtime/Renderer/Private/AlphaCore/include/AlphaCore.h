#ifndef __ALPHA_CORE_ENGINE_H__
#define __ALPHA_CORE_ENGINE_H__

/*-------------------------------------------------------------------------------------------
 *
 *    //\\   ||     ||======|| ||      ||   //\\    =======  ========  ||=====\\  =======
 *   //==\\  ||     ||======|| ||======||  //==\\   ||       ||    ||  ||=====//  ||====
 *	//    \\ ====== ||         ||      || //    \\  =======  ========  ||     \\  ========
 *
 *  Get started breaking the row ... �������ں� [����ե�����]
 *
 *-------------------------------------------------------------------------------------------
*/


#include <AxMacro.h>
#include <Math/AxVectorBase.h>
#include <Math/AxVectorHelper.h>
#include <Math/AxMatrixBase.h>

#include <Grid/AxFieldBase3D.h>
#include <Grid/AxFieldBase2D.h>

#include <Grid/AxFluid3DOperator.h>

#include <Utility/AxStorage.h>
#ifdef ALPHA_GLUT
#include <Utility/AxGlut.h>
#endif
#include <Utility/AxIO.h>
#include <Utility/AxDescrition.h>
#include <Utility/AxImage.h>
#include <Math/AxMath101.h>

#include <Ext/AxHoudiniJsonIO.h>
#include <Visualization/AxVolumeRender.h>
#include <Visualization/AxCamera.h>

#ifdef ALPHA_CUDA
#include <Grid/AxFluid3DOperatorDevice.h>
#endif

#include <Test/AxTestSet.h>

#include "AxSimObject.h"
#include "AxSimWorld.h"
#include "AxLog.h"


#endif //
