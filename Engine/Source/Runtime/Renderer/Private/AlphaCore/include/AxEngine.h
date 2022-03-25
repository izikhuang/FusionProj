#ifndef __AX_ENGINE_H__
#define __AX_ENGINE_H__

#include <string>
class AxSimObject;
class AlphaCoreEngine
{
public:
	AlphaCoreEngine();
	~AlphaCoreEngine();
	static AlphaCoreEngine* GetInstance();
	std::string GetLanchTimeAsString();
	std::string EvalCustomDebugFile(std::string path);

	void LaunchEngine();
	std::string GetDevAssetRootPath() const;
	void SetDevAssetRootPath(std::string path);
	AxSimObject* GetAxcAsset(std::string moduleName,std::string assetName);
private:
	static AlphaCoreEngine* m_Instance;
	std::string m_sLanchTime;
	std::string m_sAssetPath;
	bool m_bInited;
};


#endif