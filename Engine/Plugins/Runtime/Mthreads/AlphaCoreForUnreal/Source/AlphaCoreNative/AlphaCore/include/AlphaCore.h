#ifndef __ALPHA_CORE_ENGINE_H__
#define __ALPHA_CORE_ENGINE_H__

/*-------------------------------------------------------------------------------------------
 *
 *    //\\   ||     ||======|| ||      ||   //\\    =======  ========  ||=====\\  =======
 *   //==\\  ||     ||======|| ||======||  //==\\   ||       ||    ||  ||=====//  ||====
 *	//    \\ ====== ||         ||      || //    \\  =======  ========  ||     \\  ========
 *
 *  Get started breaking the row ... °¢¶û·¨ÄÚºË [¥¢¥ë¥Õ¥¡¥³¥¢]
 *
 *-------------------------------------------------------------------------------------------
*/


#include "AxMacro.h"
#include "Math/AxVectorBase.h"
#include "Math/AxVectorHelper.h"
#include "Math/AxMatrixBase.h"

#include "GridDense/AxFieldBase3D.h"
#include "GridDense/AxFieldBase2D.h"

//#include <GridDense/AxFluid3DOperator.h>

#include "Utility/AxStorage.h"
#ifdef ALPHA_GLUT
#include "Utility/AxGlut.h"
#endif
#include "Utility/AxIO.h"
#include "Utility/AxDescrition.h"
#include "Utility/AxImage.h"
#include "Math/AxMath101.h"
#include "VolumeRender/AxVolumeRender.h"
#include "GridDense/AxGridDense.h"
#include "AxSimObject.h"
#include "AxSimWorld.h"
#include "AxLog.h"
#include "AxGeo.h"

#include "MicroSolver/AxMicroSolverFactory.h"
#include "AxTimeTick.h"

#ifdef ALPHA_GL
#include "GL/glew.h"
#include "GL/freeglut.h"
#include "GL/helper_gl.h"
#include "GL/freeglut_std.h"
#endif

#include "Catalyst/AxCatalystObject.h"
#include "StormSystem/AxStormSystem.h"

#endif //
