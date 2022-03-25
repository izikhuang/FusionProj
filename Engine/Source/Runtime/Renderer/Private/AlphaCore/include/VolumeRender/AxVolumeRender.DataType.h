#ifndef __AX_VOLUME_RENDER_DATATYPE_H__
#define __AX_VOLUME_RENDER_DATATYPE_H__

#include <AxDataType.h>
#include <Math/AxMatrixBase.h>
#include <Grid/AxFieldBase2D.h>

typedef AxField2DBase<AxColorRGBA8>		AxTextureRGBA8;
typedef AxField2DBase<AxColorRGBA>      AxTextureRGBA;
typedef AxField2DBase<AxFp32>			AxTextureR32;
typedef AxField2DBase<AxFp64>		    AxTextureR64;



struct AxGasVolumeMaterial
{
    AxVector2 minMaxInputDensity;
    float densityScale;
    float shadowScale;
    float lookUpTableDensity[128];
    AxColorRGBA8 lookUpTableDensityColor[128];
    
    AxVector2 minMaxInputHeat;
    float LookUpTableHeat[128];

    AxVector2 minMaxInputTemperature;// 0~2
    AxVector2 minMaxOuputTemperature;// 0~1
    AxColorRGBA8 LookUpTableTemperature[128];

};

struct AxVolumeRenderObject
{
    AxFp32* density;
    AxFp32* heat;
    AxFp32* temp;
    AxGasVolumeMaterial material;

    AlphaCore::Desc::AxField3DInfo densityInfo;
    AlphaCore::Desc::AxField3DInfo heatInfo;
    AlphaCore::Desc::AxField3DInfo tempInfo;
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