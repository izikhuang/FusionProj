#ifndef __AX_STORAGE_POOL_H__
#define __AX_STORAGE_POOL_H__

class AxStoragePool
{
public:

	static AxStoragePool* GetInstance();

	void PrintData();
private:

	static AxStoragePool* m_Instance;

	AxStoragePool();
 	~AxStoragePool();

};

#endif