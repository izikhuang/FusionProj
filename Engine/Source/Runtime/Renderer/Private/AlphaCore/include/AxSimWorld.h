#ifndef __ALPHA_CORE_SIM_WORLD_H__
#define __ALPHA_CORE_SIM_WORLD_H__


#include <vector>

class AxSimObject;
class AxSimWorld
{
public:
	AxSimWorld();
	~AxSimWorld();

	void Init();
	void Step(float dt);
	
	void AddObject(AxSimObject* obj) { m_Objs.push_back(obj); };

private:

	std::vector<AxSimObject*> m_Objs;

};

#endif