// Fill out your copyright notice in the Description page of Project Settings.
#include "AxUESceneManager.h"
#include "AlphaCoreForUnreal.h"
#include "Render/RenderAlphaCore.h"



AxSceneManager* AxSceneManager::m_Instance = nullptr;
std::mutex AxSceneManager::m_Mutex;
AxSceneManager::AxSceneManager()
{
	//UE_LOG(LogTemp, Warning, TEXT("Create AxSimWorld Object Begin | Address: %p"), m_World);
	m_World = new AxSimWorld();
	this->SetSimWorldNotStarted();
	//UE_LOG(LogTemp, Warning, TEXT("Create AxSimWorld Object End | Address: %p"), m_World);
}
AxSceneManager::~AxSceneManager() {
	if (GetSimWorldStatus() == 0) {
		delete m_World;
		return;
	}
	while (true) {
		m_MutexWorld.lock();
		if (this->m_SimWorldStatus == 2) {
			delete m_World;
			this->m_SimWorldStatus = 0;
			m_MutexWorld.unlock();
			break;
		}
		m_MutexWorld.unlock();
	}
}

int AxSceneManager::GetSimWorldStatus() {
	return m_SimWorldStatus;
}

void AxSceneManager::SetSimWorldNotStarted()
{
	m_MutexWorld.lock();
	m_SimWorldStatus = 0;
	m_MutexWorld.unlock();
}
void AxSceneManager::SetSimWorldStepStarted()
{
	m_MutexWorld.lock();
	m_SimWorldStatus = 1;
	m_MutexWorld.unlock();
}
void AxSceneManager::SetSimWorldStepFinished()
{
	m_MutexWorld.lock();
	m_SimWorldStatus = 2;
	m_MutexWorld.unlock();
}

AxSceneManager* AxSceneManager::GetInstance()
{
	if (m_Instance == nullptr) {
		std::unique_lock<std::mutex> lock(m_Mutex);
		if (m_Instance == nullptr) {
			//UE_LOG(LogTemp, Warning, TEXT("Create AxSceneManager Object Start| Address: %p"), m_Instance);
			m_Instance = new (std::nothrow) AxSceneManager;
			//UE_LOG(LogTemp, Warning, TEXT("Create AxSceneManager Object End| Address: %p"), m_Instance);
		}
	}
	return m_Instance;
}

AxSimWorld* AxSceneManager::GetWorld()
{
	return m_World;
}

void AxSceneManager::ClearAndDestory()
{
	if (m_Instance)
	{
		std::unique_lock<std::mutex> lock(m_Mutex);

		delete m_Instance;
		m_Instance = nullptr;
	}
}

