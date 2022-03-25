#include <chrono>
#include <string>
#include <map>

class AxTimeTick
{
public:
	static AxTimeTick* GetInstance();
	void StartTick(std::string name);
	void EndTick(std::string name);
	void GetCost(std::string name)
	{
		//chrono::time_point<system_clock>
		
		std::chrono::system_clock::now();
		auto start = system_clock::now();

	}
	void RestartTick(std::string name);
	void RestartAll();
private:
	AxTimeTick();
	~AxTimeTick();
	static AxTimeTick* m_Instance;
	//ScenePath 
	std::map<std::string, > m_TickMap;
};