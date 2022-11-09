// Fill out your copyright notice in the Description page of Project Settings.

#include "UAxSceneManager.h"
#include "AlphaCoreForUnreal.h"


UUAxSceneManager::UUAxSceneManager(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	world = new AxSimWorld();
	//auto scn = new AxSceneObject();
	//world->AddSceneObject(scn);
}


UUAxSceneManager* UUAxSceneManager::m_Instance = nullptr;
UUAxSceneManager* UUAxSceneManager::GetInstance()
{

	if (m_Instance == nullptr)
	{
		m_Instance = NewObject<UUAxSceneManager>();
		//UE_LOG(LogTemp, Warning, TEXT("Create UUAxSceneManager Object | Address: %p"), m_Instance);
		AX_WARN("Create UUAxSceneManager Object| ");
	}
	
	return m_Instance;
}

void UUAxSceneManager::ClearAndDestory()
{
	if (m_Instance != nullptr) {
		//AxVolumeRenderObjects().swap(m_AxVolumeRenderDatas);

		delete world;
		world = nullptr;
		m_Instance = nullptr;
	}
}

UUAxSceneManager::~UUAxSceneManager() {
	//AxVolumeRenderObjects().swap(m_AxVolumeRenderDatas);
	m_Instance = nullptr;
	AX_WARN("UUAxSceneManager::~~~UUAxSceneManager ");
}


//void UUAxSceneManager::AddAxVolumeRenderData(AxVolumeRenderObject* VolumeRenderData)
//{
//	m_AxVolumeRenderDatas.push_back(VolumeRenderData);
//	AX_WARN("m_AxVolumeRenderDatas size : {} ", m_AxVolumeRenderDatas.size());
//}
