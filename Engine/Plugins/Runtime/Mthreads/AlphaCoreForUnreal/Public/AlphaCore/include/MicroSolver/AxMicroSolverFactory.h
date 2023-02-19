#ifndef __AX_MICRO_SOLVER_FACTORY_H__
#define __AX_MICRO_SOLVER_FACTORY_H__

#include <string>
#include <map>
#include <vector>

class AxMicroSolverBase;
typedef	AxMicroSolverBase*(*AxSolverConstructor)();

class AxMicroSolverFactory
{
public:
	static AxMicroSolverFactory* GetInstance();
	void ClearAndDestory();
	bool RegisterProduct(std::string product_name, AxSolverConstructor constructor);
	AxMicroSolverBase* CreateSolver(std::string product_name);
	std::vector<AxMicroSolverBase*> CreateSolverStackFromJsonContent(std::string jsonContent);
	void CreateSolverStackFromJsonContent(std::vector<AxMicroSolverBase*>& retList,std::string jsonContent);
private:
	void _addProductEXT();
	AxMicroSolverFactory();
	~AxMicroSolverFactory();
	static AxMicroSolverFactory* m_Instance;
	//ScenePath 
	std::map<std::string, AxSolverConstructor> m_CreatorMap;
};
#endif


