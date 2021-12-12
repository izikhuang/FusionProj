#ifndef __ALPHA_CORE_LOG_H__
#define __ALPHA_CORE_LOG_H__

#include <string>
#include <cstdlib>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#define SPDLOG_TRACE_ON
											 
#define LOG_RESET	     "\033[0m"			 
#define LOG_BLACK	     "\033[30m"			 /* Black */
#define LOG_RED		     "\033[31m"			 /* Red */
#define LOG_GREEN	     "\033[32m"			 /* Green */
#define LOG_YELLOW	     "\033[33m"			 /* Yellow */
#define LOG_BLUE	     "\033[34m"			 /* Blue */
#define LOG_MAGENTA	     "\033[35m"			 /* Magenta */
#define LOG_CYAN	     "\033[36m"			 /* Cyan */
#define LOG_WHITE	     "\033[37m"			 /* White */
#define LOG_BOLD_BLACK   "\033[1m\033[30m"   /* Bold Black */
#define LOG_BOLD_RED     "\033[1m\033[31m"   /* Bold Red */
#define LOG_BOLD_GREEN   "\033[1m\033[32m"   /* Bold Green */
#define LOG_BOLD_YELLOW  "\033[1m\033[33m"   /* Bold Yellow */
#define LOG_BOLD_BLUE    "\033[1m\033[34m"   /* Bold Blue */
#define LOG_BOLD_MAGENTA "\033[1m\033[35m"   /* Bold Magenta */
#define LOG_BOLD_CYAN    "\033[1m\033[36m"   /* Bold Cyan */
#define LOG_BOLD_WHITE   "\033[1m\033[37m"   /* Bold White */

// Windows
#if defined(_WIN64)
	#define AX_PLATFORM_WINDOWS
#endif

#if defined(_WIN32) && !defined(_WIN64)
	static_assert(false, "32-bit Windows systems are not supported")
#endif

// Linux
#if defined(__linux__)
	#define AX_PLATFORM_LINUX
#endif

// OSX
#if defined(__APPLE__)
	#define AX_PLATFORM_OSX
#endif

#if (defined(AX_PLATFORM_LINUX) || defined(AX_PLATFORM_OSX))
	#define AX_PLATFORM_UNIX
#endif
#if defined(AX_PLATFORM_WINDOWS)
	#define AX_UNREACHABLE __assume(0);
#else
	#define AX_UNREACHABLE __builtin_unreachable();
#endif

namespace spdlog {
	class logger;
}

//******************************************************************************
//                               Logging
//******************************************************************************
namespace AlphaCore
{
	class Logger {
	private:
		Logger();
		std::shared_ptr<spdlog::logger> m_Console;
		std::shared_ptr<spdlog::logger> m_FileStreaming;

		int m_iLevel;
		static Logger* m_Instance;
		bool m_bTraceOutputCMD;
	public:
		static Logger* GetInstance();
		void SetLogPath(std::string filename);
		void Trace(const std::string &s);
		void Debug(const std::string &s);
		void Info(const std::string &s);
		void Warn(const std::string &s);
		void Error(const std::string &s, bool raise_exception = true);
		void Critical(const std::string &s);
		void Flush();
		void SetLevel(const std::string &level);
		//bool IsLevelEffective(const std::string &level_name);

		std::string GetSystemTime(const char* format = "[%Y-%m-%d %H:%M:%S] ");

		void SetTraceActiveCMDMark(bool e) { m_bTraceOutputCMD = e; };
		int GetLevel();
		static int LevelEnumFromString(const std::string &level);
		//void SetLevelDefault();

		void AddLogInfoCallback(std::function<void(const std::string & msg)> CALLBACK_FUN);

		void AddWarnInfoCallback(std::function<void(const std::string & msg)> CALLBACK_FUN);

	private:

		bool m_bInfoCallback;
		bool m_bWarnCallback;
		std::function<void(const std::string & msg)> m_LogInfoCallback;
		std::function<void(const std::string & msg)> m_LogWarnCallback;

	};
};

//******************************************************************************
//                         Log Macro default Utils
//******************************************************************************

#ifdef _WIN64
	#define __FILENAME__  (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
	#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define TRACE_HEAD			"[ TRACE ]"
#define INFO_HEAD			"[ INFO ]"
#define ERROR_HEAD			"[ ERROR ]"
#define WARRNING_HEAD		"[ WARRN ]"
#define DEBUG_HEAD			"[ DEBUG ]"

#define SPD_AUGMENTED_LOG(X,HEAD, ...)          \
  AlphaCore::Logger::GetInstance()->X(     \
      fmt::format(HEAD,"[{}] ", __FUNCTION__) + \
      fmt::format(__VA_ARGS__))


#define AX_TRACE_DATA(...)	SPD_AUGMENTED_LOG(Info,TRACE_HEAD, __VA_ARGS__)

#define BUILD_FOR_UNREAL
#ifndef BUILD_FOR_UNREAL
#define AX_TRACE(...)	SPD_AUGMENTED_LOG(Trace,TRACE_HEAD, __VA_ARGS__)
#define AX_DEBUG(...)	SPD_AUGMENTED_LOG(Debug,DEBUG_HEAD, __VA_ARGS__)
#define AX_INFO(...)	SPD_AUGMENTED_LOG(Info,INFO_HEAD, __VA_ARGS__)
#define AX_WARN(...)	SPD_AUGMENTED_LOG(Warn,WARRNING_HEAD, __VA_ARGS__)
#define AX_ERROR(...)	SPD_AUGMENTED_LOG(Error,ERROR_HEAD, __VA_ARGS__);   AX_UNREACHABLE;
#define AX_CRITICAL(...) SPD_AUGMENTED_LOG(Critical,"", __VA_ARGS__);AX_UNREACHABLE;

#define AX_TRACE_IF(condition, ...)		   if (condition)    { AX_TRACE(__VA_ARGS__);}
#define AX_TRACE_UNLESS(condition, ...)    if (!(condition)) { AX_TRACE(__VA_ARGS__);}
#define AX_DEBUG_IF(condition, ...)		   if (condition)	 { AX_DEBUG(__VA_ARGS__);}
#define AX_DEBUG_UNLESS(condition, ...)    if (!(condition)) { AX_DEBUG(__VA_ARGS__);}
#define AX_INFO_IF(condition, ...)		   if (condition)	 { AX_INFO(__VA_ARGS__);}
#define AX_INFO_UNLESS(condition, ...)     if (!(condition)) { AX_INFO(__VA_ARGS__); }
#define AX_WARN_IF(condition, ...)		   if (condition)	 { AX_WARN(__VA_ARGS__);}
#define AX_WARN_UNLESS(condition, ...)	   if (!(condition)) { AX_WARN(__VA_ARGS__);}
#define AX_ERROR_IF(condition, ...)		   if (condition)	 { AX_ERROR(__VA_ARGS__);}
#define AX_ERROR_UNLESS(condition, ...)    if (!(condition)) { AX_ERROR(__VA_ARGS__);}
#define AX_CRITICAL_IF(condition, ...)	   if (condition)	 { AX_CRITICAL(__VA_ARGS__); }
#define AX_CRITICAL_UNLESS(condition, ...) if (!(condition)) { AX_CRITICAL(__VA_ARGS__);}
#define AX_LOG_SET_PATTERN(x)			   spdlog::set_pattern(x);
#else
#define AX_TRACE(...)
#define AX_DEBUG(...)
#define AX_INFO(...)
#define AX_WARN(...)
#define AX_ERROR(...)
#define AX_CRITICAL(...)
#define AX_TRACE_IF(condition, ...)
#define AX_TRACE_UNLESS(condition, ...)
#define AX_DEBUG_IF(condition, ...)	
#define AX_DEBUG_UNLESS(condition, ...) 
#define AX_INFO_IF(condition, ...)
#define AX_INFO_UNLESS(condition, ...)
#define AX_WARN_IF(condition, ...)
#define AX_WARN_UNLESS(condition, ...)
#define AX_ERROR_IF(condition, ...)
#define AX_ERROR_UNLESS(condition, ...)
#define AX_CRITICAL_IF(condition, ...)
#define AX_CRITICAL_UNLESS(condition, ...)
#define AX_LOG_SET_PATTERN(x)
#endif

#define AX_LOG_FILEPATH(PATH) AlphaCore::Logger::GetInstance()->SetLogPath(PATH)

#ifdef ALPHA_CUDA

#define AX_GET_CUDA_LAST_ERROR {\
		cudaError_t cudaStatus = cudaGetLastError();	\
		if (cudaStatus != cudaSuccess) {				\
			AX_ERROR("Kernel launch failed {} Code: {} @ {}", cudaGetErrorString(cudaStatus),__FILE__,__LINE__);\
		}else{\
			AX_INFO("Kernel launch succ Code: {} @ {} ",__FILE__,__LINE__);\
		}\
	}
#endif 


#define TRACE_HDA_RAY(origin,dir) AX_TRACE("{},{},{}>{},{},{}",origin.x,origin.y,origin.z,dir.x,dir.y,dir.z)

#endif // !__ALPHA_CORE_LOG_H__
