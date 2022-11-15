// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include <vector>
#include <memory>
#include <mutex>
//#include "CoreMinimal.h"
#include "AlphaCoreForUnreal.h"
//#include "UObject/NoExportTypes.h"
//#include "Tickable.h"
//#include "UAxSceneManager.generated.h"


//UCLASS(Blueprintable, BlueprintType)
class AxSceneManager/* : public UObject*//*, public FTickableGameObject*/
{
	//GENERATED_BODY()

public:

	static AxSceneManager* GetInstance();
	static void ClearAndDestory(); 
	AxSimWorld* GetWorld();
	
	void SetSimWorldNotStarted();
	void SetSimWorldStepStarted();
	void SetSimWorldStepFinished();
	int GetSimWorldStatus();
private:
	int m_SimWorldStatus = 0;
	AxSceneManager();
	~AxSceneManager();

	AxSimWorld* m_World;
	static AxSceneManager* m_Instance;
	static std::mutex m_Mutex;
	std::mutex m_MutexWorld;
};
