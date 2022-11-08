#ifndef __AX_TIME_TICK_H__
#define __AX_TIME_TICK_H__

#include <chrono>
#include <string>
#include <map>
#include "AxDataType.h"

struct AxTimer
{
	std::chrono::time_point<std::chrono::high_resolution_clock> Start;
	AxFp64 Total;
};

class AxTimeTick
{
public:
	static AxTimeTick* GetInstance();
	void StartTick(std::string name);
	AxFp64 EndTick(std::string name, bool print = false);
	void GetCost(std::string name)
	{


	}
	void RestartTick(std::string name);
	void RestartAll();
private:
	AxTimeTick();
	~AxTimeTick();
	static AxTimeTick* m_Instance;
	//ScenePath 
	std::map<std::string, AxTimer> m_TickMap;
};



#define ACTIVE_DEV_TICK 1
#if ACTIVE_DEV_TICK == 1

#define AX_DEV_TICK_START(key)	AxTimeTick::GetInstance()->StartTick(key);
#define AX_DEV_TICK_END(key)	AxTimeTick::GetInstance()->EndTick(key,true);

#else

#define AX_DEV_TICK_START(key)	
#define AX_DEV_TICK_END(key)	

#endif

#endif // !__AX_TIME_TICK_H__