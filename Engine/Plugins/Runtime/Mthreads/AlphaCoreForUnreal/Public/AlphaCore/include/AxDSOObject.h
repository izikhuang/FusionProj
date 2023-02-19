#ifndef __AX_DSO_OBJECT_H__
#define __AX_DSO_OBJECT_H__

#include <string>
#include "AxDataType.h"

class AxSimCallbackData;
class AxDSOObject
{
public:
	static AxDSOObject* Load(std::string dsoPath);
	void DoCallback(AxSimCallbackData* data, AxContext context, AlphaCore::AxBackendAPI api);
	AxDSOObject()
	{
		DSOInstance = nullptr;
	}
	~AxDSOObject()
	{

	}
	bool IsValid()
	{
		return DSOInstance != nullptr;
	}

	void LoadDSO(std::string dsoPath);
	void CloseDSO();
private:
	void* DSOInstance;
};

#endif