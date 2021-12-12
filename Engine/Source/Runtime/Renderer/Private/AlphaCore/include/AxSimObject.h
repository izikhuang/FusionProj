#ifndef __ALPHA_CORE_SIMULATION_OBJECT_H__
#define __ALPHA_CORE_SIMULATION_OBJECT_H__

#include <AxMacro.h>
#include <string>
#include <map>

namespace AlphaCore
{
	enum PropertyType
	{
		AxPointProperty,
		AxPrimitiveProperty,
		AxObjectProperty,
		Ax
	};
}

class AxPropertyBase
{
public:
	AxPropertyBase();
	~AxPropertyBase();

	void AddProperty();
private:

	// map



};


class AxSimObject
{
public:
	AxSimObject()
	{
		Init();
	};
	virtual ~AxSimObject()
	{

	}

	void SetName(std::string n) { m_sName = n; };
	std::string GetName() { return m_sName; };

	virtual int Save(std::string path);
	virtual int Read(std::string path);
	virtual int Save(std::ifstream& ifs);
	virtual int Read(std::ofstream& ofs);
	
	void Init();
 	void SetCacheWritePath(std::string cacheFrameCode) { m_sCacheSavePath = cacheFrameCode; };

	void SetDx(float dx);
protected:

	void preSim(float dt);
	void sim(float dt);
	void postSim(float dt);

	virtual void OnInit() {};
	virtual void OnReset() {};

	virtual void OnPreSim(float dt)
	{

	}

	virtual void OnUpdateSim(float dt)
	{

	}

	virtual void OnPostSim(float dt)
	{

	}	

	std::string m_sName;
	std::string m_sCacheSavePath;

	friend class AxSimWorld;
	int m_iCookTimes;
};

#endif