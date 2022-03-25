#ifndef __AX_UE_LOG_H__
#define __AX_UE_LOG_H__

#include <string>
#include <functional>
#include <AxLog.h>



#define AX_UE_LOGINFO_CALLBACK		std::bind(AlphaCore::logInfoDelegateUE4,std::placeholders::_1)
#define AX_UE_LOGWARRNING_CALLBACK	std::bind(AlphaCore::logWarnningDelegateUE4,std::placeholders::_1)

DECLARE_LOG_CATEGORY_EXTERN(AlphaCoreLOG, Log, All);
DEFINE_LOG_CATEGORY(AlphaCoreLOG);

namespace AlphaCore
{
	void logInfoDelegateUE4(const std::string& v)
	{
		FString msg(v.c_str());
		UE_LOG(AlphaCoreLOG, Log, TEXT("%s"), *msg);
	}

	void logWarnningDelegateUE4(const std::string& v)
	{
		FString msg(v.c_str());
		UE_LOG(AlphaCoreLOG, Warning, TEXT("%s"), *msg);
	}

	void ActiveUELog()
	{
		AlphaCore::Logger::GetInstance()->AddLogInfoCallback(AX_UE_LOGINFO_CALLBACK);
		AlphaCore::Logger::GetInstance()->AddWarnInfoCallback(AX_UE_LOGWARRNING_CALLBACK);
	}

	namespace Unreal
	{
		inline void PrintInfo(const char* head, FVector vec)
		{
			UE_LOG(LogTemp, Log, TEXT("%s : [%f,%f,%f] \n"), *FString(head), vec.X, vec.Y, vec.Z);
		}

		inline void PrintInfo(const char* head)
		{
			UE_LOG(LogTemp, Log, TEXT("%s \n"), *FString(head));
		}

		inline void PrintInfo(const char* head, AxVector3 vec)
		{
			UE_LOG(LogTemp, Log, TEXT("%s : [%f,%f,%f] \n"), *FString(head), vec.x, vec.y, vec.z);
		}

		inline void PrintInfo(const char* head, float f)
		{
			UE_LOG(LogTemp, Log, TEXT("%s : [%f] \n"), *FString(head),f);
		}

		inline void PrintAsCPPDebugInfo(const char* head, FVector vec)
		{
			UE_LOG(LogTemp, Log, TEXT("AxVector3 %s = MakeVector3(%f,%f,%f); \n"), *FString(head), vec.X, vec.Y, vec.Z);
		}

		inline void PrintAsCPPDebugInfo(const char* head, float f)
		{
			UE_LOG(LogTemp, Log, TEXT("float %s = %f; \n"), *FString(head), f);
		}

		inline void PrintAsCPPDebugInfo(const char* head, FVector2D size)
		{
			UE_LOG(LogTemp, Log, TEXT("AxVector2 %s = MakeVector2(%f,%f); \n"), *FString(head), size.X, size.Y);
		}

		inline AxVector3 FormFVector(FVector vec)
		{
			return MakeVector3(vec.X, vec.Y, vec.Z);
		}
	}
}

#endif // !__RX_UE_LOG_H__
