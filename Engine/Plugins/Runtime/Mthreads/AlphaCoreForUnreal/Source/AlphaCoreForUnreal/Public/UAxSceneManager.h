// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include <vector>
#include "CoreMinimal.h"
#include "AlphaCoreForUnreal.h"
//#include "AxUCatalystActor.h"
#include "UObject/NoExportTypes.h"
#include "Tickable.h"
#include "UAxSceneManager.generated.h"

/**
 * 
 */

//typedef std::map<int, string> mapStudent
class AAxUCatalystActor;

typedef std::vector<AxVolumeRenderObject*> AxVolumeRenderObjects;

UCLASS(Blueprintable, BlueprintType)
class ALPHACOREFORUNREAL_API UUAxSceneManager : public UObject/*, public FTickableGameObject*/
{
	GENERATED_BODY()

public:
	virtual ~UUAxSceneManager();

	void ClearAndDestory(); 

	static UUAxSceneManager* GetInstance();

	//void AddAxVolumeRenderData(AxVolumeRenderObject* VolumeRenderData);

	//AxVolumeRenderObjects GetAxVolumeRenderDatas() { return m_AxVolumeRenderDatas; };
	//AxVolumeRenderObjects m_AxVolumeRenderDatas;
	//AxVolumeRenderObject* m_AxVolumeRenderDatas[32];

	AxSimWorld* world;
private:

	UUAxSceneManager();
	UUAxSceneManager(const FObjectInitializer& ObjectInitializer);
	static UUAxSceneManager* m_Instance;
	
};
