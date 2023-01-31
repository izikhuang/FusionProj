#ifndef __AX_VOLUME_RENDER_DATATYPE_H__
#define __AX_VOLUME_RENDER_DATATYPE_H__

#include "AxDataType.h"
#include "Math/AxMatrixBase.h"
#include "GridDense/AxFieldBase2D.h"
#include "GridDense/AxFieldBase3D.h"

typedef AxField2DBase<AxColorRGBA8> AxTextureRGBA8;
typedef AxField2DBase<AxColorRGBA> AxTextureRGBA;
typedef AxField2DBase<AxFp32> AxTextureR32;
typedef AxField2DBase<AxFp64> AxTextureR64;

struct AxGasVolumeMaterial
{
    bool needUpdate = true;
    bool usePhase = true;
    float densityScale = 1.0f;
    float shadowScale = 1.0f;
    float stepSize = 0.5f;
    float phase = 0.5f;

    AxVector2 minMaxInputDensity;
    AxVector2 minMaxInputTemperature; // 0~2
    // AxVector2 minMaxOuputTemperature;// 0~1
    AxVector2 minMaxInputHeat;

    AxUChar lookUpTableDensity[128];
    AxUChar LookUpTableTemperature[128];
    AxColorRGBA8 lookUpTableDensityColor[128];
    AxColorRGBA8 LookUpTableHeat[128];
};

struct AxVolumeRenderObject
{
    AxScalarFieldF32::RAWDesc density;
    AxScalarFieldF32::RAWDesc heat;
    AxScalarFieldF32::RAWDesc temperature;
    AxGasVolumeMaterial material;
};

static AxVolumeRenderObject MakeDefaultVolumeRenderObject()
{
    AxVolumeRenderObject t;
    return t;
}

struct AxSceneRenderDesc
{
    AxUChar lightNum;
    AlphaCore::Desc::AxCameraInfo camInfo;
    AlphaCore::Desc::AxPointLightInfo lightInfo[3];
};

#endif