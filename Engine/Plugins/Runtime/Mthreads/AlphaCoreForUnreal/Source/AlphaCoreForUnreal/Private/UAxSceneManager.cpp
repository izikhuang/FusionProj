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
		UE_LOG(LogTemp, Warning, TEXT("Create UUAxSceneManager Object | Address: %p"), m_Instance);
	}
	
	return m_Instance;
}

void UUAxSceneManager::ClearAndDestory()
{
	if (m_Instance != nullptr) {
		//AxVolumeRenderObjects().swap(m_AxVolumeRenderDatas);
		//world->~AxSimWorld();
		m_Instance = nullptr;
	}
}

UUAxSceneManager::~UUAxSceneManager() {
	// 销毁，由于UE4完善的垃圾回收功能，所以将全部引用指针置为nullptr即可由UE4自动回收掉
	//AxVolumeRenderObjects().swap(m_AxVolumeRenderDatas);
	m_Instance = nullptr;
	AX_WARN("UUAxSceneManager::~~~UUAxSceneManager ");
}


//void UUAxSceneManager::AddAxVolumeRenderData(AxVolumeRenderObject* VolumeRenderData)
//{
//	m_AxVolumeRenderDatas.push_back(VolumeRenderData);
//	AX_WARN("m_AxVolumeRenderDatas size : {} ", m_AxVolumeRenderDatas.size());
//}
