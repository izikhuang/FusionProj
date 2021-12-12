
#include <AxSimObject.h>
#include <Grid/AxFieldBase3D.h>
#include <Grid/AxFieldBase2D.h>

#include <unordered_map>

namespace AlphaCore
{
	namespace Param
	{
		struct AxCatalystSolverParam
		{
			AxCombustionParam CombustionParam;
			uInt32 GaussSeidelIterations;
			uInt32 Substeps;
 			AxAdvectInfo AdvectInfo;
			AxVector3    BuoyancyDir;
 			float		 Dissipation;
			float		 Vorticity;
  		};

		inline AxCatalystSolverParam MakeDefault_CatalystParam()
		{
			AxCatalystSolverParam parm;
			parm.Substeps = 1;
			parm.Vorticity = 10;
			parm.Dissipation = 0.1;
			parm.GaussSeidelIterations = 20;
			parm.BuoyancyDir = MakeVector3(0.0f, 1.0f, 0.0f);
			parm.AdvectInfo.AdvtType = AlphaCore::Flow::AdvectType::MacCormack;
			return parm;
		}
	}
}

class AxCatalystObject : public AxSimObject
{
public:
	AxCatalystObject();
	~AxCatalystObject();

	AlphaCore::Param::AxCatalystSolverParam SolverParam;

	AxScalarFieldF32* GetDensityField()		{ return m_DensityField; }
	AxScalarFieldF32* GetHeatField()		{ return m_HeatField;}
	AxScalarFieldF32* GetTempratureField()	{ return m_TempratureField;}

	void SetEmitterCachePath(std::string emitterCache) { m_sEmitterPath = emitterCache; };
	virtual int Save(std::string path);
	virtual int Read(std::string path);

	void SetDx(float dx);
	void SetShape(AxVector3 pivot, AxVector3 size);
	void SetShape(AxVector3 pivot, AxVector3 size,float dx);

	AxVector3 GetPivot() { return  m_Pivot; };
	AxVector3 GetSize() { return m_Size;};
	
	void SetRenderingParameter(uInt32 sizeX, uInt32 sizeY);
	void DoRendering(AlphaCore::Desc::AxCameraInfo& info);
	void SaveLastRenderRet(std::string path);

	void SetRenderSavePath(std::string path) { m_sRenderImageSavePath = path; };

protected:

	virtual void OnInit();
	virtual void OnReset();
 	virtual void OnPreSim(float dt);
 	virtual void OnUpdateSim(float dt);
 	virtual void OnPostSim(float dt);
	
private:
	AxScalarFieldF32* m_DensityField;
	AxScalarFieldF32* m_FuelField;	
	AxScalarFieldF32* m_TempratureField;	
	AxScalarFieldF32* m_DivField;		
	AxScalarFieldF32* m_VelDivField;	
	AxScalarFieldF32* m_PressureField;		
	AxScalarFieldF32* m_PressTmpField;		
	AxScalarFieldF32* m_BurnField;	
	AxScalarFieldF32* m_HeatField;	
	AxVecFieldF32* m_VelField;	
	AxVecFieldF32* m_VelFieldNew;
	AxScalarFieldF32* m_AdvectTmp;
	AxScalarFieldF32* m_AdvectTmp2;
	AxVecFieldF32* m_CurlField;	
	AxVecFieldF32* m_VortexDir;
	AxScalarFieldF32* m_CurlMag;

	std::string m_sLastEmitterPath;
	std::string m_sEmitterPath;

	AxScalarFieldF32* m_TmpEmitter;
	AxScalarFieldF32* m_FuelEmitter;

	void _addScalarField(AxScalarFieldF32* field);
	void _updateShape();
	std::unordered_map<std::string, AxScalarFieldF32*> m_Fields;

	AxVector2UI m_CurrRenderSize;
	AxVector2UI m_LastRenderSize;
	AxVector3 m_Pivot;
	AxVector3 m_Size;
	float m_fDx;
	std::string m_sRenderImageSavePath;

	AxImageRGBA8 m_RenderRet;
};

