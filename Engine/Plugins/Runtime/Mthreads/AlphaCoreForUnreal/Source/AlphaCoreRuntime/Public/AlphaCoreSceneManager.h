// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include <vector>
#include <memory>
#include <mutex>
#include "AlphaCoreRuntimeModule.h"

#include <AlphaCore.h>

class ALPHACORERUNTIME_API AlphaCoreSceneManager
{
public:
	static AlphaCoreSceneManager* GetInstance();
	static void ClearAndDestory(); 
	AxSimWorld* GetWorld();
	
	void SetSimWorldNotInit();
	void SetSimWorldInited();
	void SetSimWorldRendering();
	void SetSimWorldRenderFinished();
	int GetSimWorldStatus();
private:
	int m_SimWorldStatus = 0;
	AlphaCoreSceneManager();
	~AlphaCoreSceneManager();

	AxSimWorld* m_World;
	static AlphaCoreSceneManager* m_Instance;
	static std::mutex m_Mutex;
	std::mutex m_MutexWorld;
};
