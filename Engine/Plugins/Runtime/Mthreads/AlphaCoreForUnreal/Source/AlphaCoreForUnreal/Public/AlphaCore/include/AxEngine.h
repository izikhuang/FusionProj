#ifndef __AX_ENGINE_H__
#define __AX_ENGINE_H__

#include <string>
#include "AxDataType.h"




class AxSimObject;
class ALPHA_CLASS AlphaCoreEngine
{
public:
	AlphaCoreEngine();
	~AlphaCoreEngine();
	static AlphaCoreEngine* GetInstance();
	std::string GetLanchTimeAsString();
	std::string EvalCustomDebugFile(std::string path);

	void LaunchEngine();
	void RunDevSample(std::string sampleName);

	std::string GetDevAssetRootPath() const;
	void SetDevAssetRootPath(std::string path);
	AxSimObject* GetAxcAsset(std::string moduleName,std::string assetName);

	void SetPrintDataFStreamPath(std::string path);
	std::string GetPrintDataFStreamPath();

private:
	static AlphaCoreEngine* m_Instance;
	std::string m_sLanchTime;
	std::string m_sAssetPath;
	std::string m_sPrintDataFStreamPath;
	bool m_bInited;
};


#endif