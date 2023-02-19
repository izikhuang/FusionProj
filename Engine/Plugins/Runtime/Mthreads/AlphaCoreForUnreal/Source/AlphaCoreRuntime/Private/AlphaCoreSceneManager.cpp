// Fill out your copyright notice in the Description page of Project Settings.
#include "AlphaCoreSceneManager.h"
#include "AlphaCoreRuntimeModule.h"
#include "Render/RenderAlphaCore.h"


DEFINE_LOG_CATEGORY_STATIC(ALCoreSceneManager, Log, All);

AlphaCoreSceneManager* AlphaCoreSceneManager::m_Instance = nullptr;
std::mutex AlphaCoreSceneManager::m_Mutex;
AlphaCoreSceneManager::AlphaCoreSceneManager()
{
	//UE_LOG(ALCoreSceneManager, Warning, TEXT("Create AxSimWorld Object Begin | Address: %p"), m_World);
	m_World = new AxSimWorld();
	m_World->SetSPMDBackend(AlphaCore::AxBackendAPI::CUDA);

	// AxSimWorld Init
	m_World->SetFPS(24);
	m_World->SetSubstep(1);
	m_World->SetFrame(0);

	this->SetSimWorldNotInit();
	//UE_LOG(ALCoreSceneManager, Warning, TEXT("Create AxSimWorld Object End | Address: %p"), m_World);
}

AlphaCoreSceneManager::~AlphaCoreSceneManager() {
	if (GetSimWorldStatus() == 0) {
		delete m_World;
		return;
	}
	while (true) {
		m_MutexWorld.lock();
		if (this->m_SimWorldStatus == 3) {
			delete m_World;
			this->m_SimWorldStatus = 0;
			m_MutexWorld.unlock();
			break;
		}
		m_MutexWorld.unlock();
	}
}

int AlphaCoreSceneManager::GetSimWorldStatus() {
	return m_SimWorldStatus;
}

void AlphaCoreSceneManager::SetSimWorldNotInit()
{
	m_MutexWorld.lock();
	m_SimWorldStatus = 0;
	m_MutexWorld.unlock();
}

void AlphaCoreSceneManager::SetSimWorldInited()
{
	m_MutexWorld.lock();
	m_SimWorldStatus = 1;
	m_MutexWorld.unlock();
}

void AlphaCoreSceneManager::SetSimWorldRendering()
{
	m_MutexWorld.lock();
	m_SimWorldStatus = 2;
	m_MutexWorld.unlock();
}

void AlphaCoreSceneManager::SetSimWorldRenderFinished()
{
	m_MutexWorld.lock();
	m_SimWorldStatus = 3;
	m_MutexWorld.unlock();
}
AlphaCoreSceneManager* AlphaCoreSceneManager::GetInstance()
{
	if (m_Instance == nullptr) {
		std::unique_lock<std::mutex> lock(m_Mutex);
		if (m_Instance == nullptr) {
			//UE_LOG(LogTemp, Warning, TEXT("Create AlphaCoreSceneManager Object Start| Address: %p"), m_Instance);
			m_Instance = new (std::nothrow) AlphaCoreSceneManager;
			//UE_LOG(LogTemp, Warning, TEXT("Create AlphaCoreSceneManager Object End| Address: %p"), m_Instance);
		}
	}
	return m_Instance;
}

AxSimWorld* AlphaCoreSceneManager::GetWorld()
{
	return m_World;
}

void AlphaCoreSceneManager::ClearAndDestory()
{
	if (m_Instance)
	{
		std::unique_lock<std::mutex> lock(m_Mutex);

		delete m_Instance;
		m_Instance = nullptr;
	}
}