#ifndef __ALPHA_CORE_LOG_H__
#define __ALPHA_CORE_LOG_H__

#include <string>
#include <cstdlib>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include "AxMacro.h"
#include "AxDataType.h"


#ifdef USE_AX_LOG
#ifndef SPDLOG_TRACE_ON
#define SPDLOG_TRACE_ON
#endif
// Must include "spdlog/common.h" to define SPDLOG_HEADER_ONLY
// before including "spdlog/fmt/fmt.h"
#include "thirdparty/spdlog/include/spdlog/common.h"
#include "thirdparty/spdlog/include/spdlog/fmt/fmt.h"
#endif

#define LOG_COLOR_END    "\033[0m"			 
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


enum class AxLogColor
{
	kLogRed,
	kLogGreen,
	kLogYellow,
	kLogBlue,
	kLogNonColor,
	kLogColorEnd
};

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
	class ALPHA_CLASS Logger 
	{
	private:
		Logger();
		std::shared_ptr<spdlog::logger> m_Console;
		std::shared_ptr<spdlog::logger> m_FileStreaming;

		int m_iLevel;
		static Logger* m_Instance;
		bool m_bTraceOutputCMD;
	public:
		static Logger* GetInstance();
		void SetLogPath(std::string filename,std::string loggerName = "test");
		void Trace(const std::string &s);
		void Debug(const std::string &s);
		void Info(const std::string &s);
		void Dev(const std::string& s);
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

		void EnableDevInfo();
		void DisableDevInfo();

		void SetLogColor(AxLogColor color);
	private:
		std::string getSysTimeMilliseconds();
		bool m_bInfoCallback;
		bool m_bWarnCallback;
		bool m_bActiveDevInfo;
		std::function<void(const std::string & msg)> m_LogInfoCallback;
		std::function<void(const std::string & msg)> m_LogWarnCallback;
		AxLogColor m_eLogColor;
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

#define TRACE_HEAD			"[ TRACE ] "
#define INFO_HEAD			"[ INFO ] "
#define ERROR_HEAD			"[ ERROR ] "
#define WARRNING_HEAD		"[ WARRN ] "
#define DEBUG_HEAD			"[ DEBUG ] "
#define DEV_HEAD			"[ DEV_TRACE  ] "

#define SPD_AUGMENTED_LOG(X,HEAD, ...)          \
  AlphaCore::Logger::GetInstance()->X(     \
      fmt::format(HEAD,"[{}] ", __FUNCTION__) + \
      fmt::format(__VA_ARGS__))


#define AX_TRACE_DATA(...)	SPD_AUGMENTED_LOG(Info,TRACE_HEAD, __VA_ARGS__)


#ifdef USE_AX_LOG
#define AX_TRACE(...)	SPD_AUGMENTED_LOG(Trace,TRACE_HEAD, __VA_ARGS__)
#define AX_DEBUG(...)	SPD_AUGMENTED_LOG(Debug,DEBUG_HEAD, __VA_ARGS__)
#define AX_INFO(...)	SPD_AUGMENTED_LOG(Info, (std::string(INFO_HEAD)+__FUNCTION__+" | ").c_str(), __VA_ARGS__)

#define AX_INFO_BLUE(...)	AlphaCore::Logger::GetInstance()->SetLogColor(AxLogColor::kLogBlue);\
SPD_AUGMENTED_LOG(Info, (std::string(INFO_HEAD)+__FUNCTION__+" | ").c_str(), __VA_ARGS__);\
AlphaCore::Logger::GetInstance()->SetLogColor(AxLogColor::kLogNonColor)

#define AX_INFO_GREEN(...)	AlphaCore::Logger::GetInstance()->SetLogColor(AxLogColor::kLogGreen);\
SPD_AUGMENTED_LOG(Info, (std::string(INFO_HEAD)+__FUNCTION__+" | ").c_str(), __VA_ARGS__);\
AlphaCore::Logger::GetInstance()->SetLogColor(AxLogColor::kLogNonColor)



#define AX_WARN(...)	SPD_AUGMENTED_LOG(Warn, (std::string(WARRNING_HEAD)+__FUNCTION__+" | ").c_str(), __VA_ARGS__)
#define AX_ERROR(...)	SPD_AUGMENTED_LOG(Error,(std::string(ERROR_HEAD)+__FUNCTION__+" | ").c_str(), __VA_ARGS__);   AX_UNREACHABLE;
#define AX_CRITICAL(...)SPD_AUGMENTED_LOG(Critical,"", __VA_ARGS__);AX_UNREACHABLE;
	
#define AX_DEV_INFO(...)	SPD_AUGMENTED_LOG(Dev, (std::string(DEV_HEAD)+__FUNCTION__+" | ").c_str(), __VA_ARGS__)

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
#define AX_DEV_INFO(...)
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

#define AX_INFO_BLUE(...)
#define AX_INFO_GREEN(...)

#endif

#define AX_LOG_FILEPATH(PATH) AlphaCore::Logger::GetInstance()->SetLogPath(PATH)

#ifdef ALPHA_CUDA

#define AX_GET_DEVICE_LAST_ERROR {\
		cudaError_t cudaStatus = cudaGetLastError();	\
		if (cudaStatus != cudaSuccess) {				\
			AX_ERROR("Kernel launch failed {} Code: {} @ {}", cudaGetErrorString(cudaStatus),__FILE__,__LINE__);\
		}else{\
			AX_DEV_INFO("Kernel launch succ Code: {} @ {} ",__FILE__,__LINE__);\
		}\
	}
#endif 


#define TRACE_HDA_RAY(origin,dir) AX_TRACE("{},{},{}>{},{},{}",origin.x,origin.y,origin.z,dir.x,dir.y,dir.z)


#ifdef AX_FILE_IO_DEBUG
	#define POS_READ_INFO(head,stream)  AX_INFO(" {} {} {}"," [ Read  POS ] ",head,stream.tellg());
	#define POS_WRITE_INFO(head,stream) AX_INFO(" {} {} {}"," [ Write POS ] ",head,stream.tellp());
#else
	#define POS_READ_INFO(head,stream)
	#define POS_WRITE_INFO(head,stream)
#endif

template<AxUInt32 SIZE>
struct AxLogInfoBlock
{
	ALPHA_SHARE_FUNC AxLogInfoBlock()
	{
		AX_FOR_I(SIZE)
			m_charPool[i] = 0;
	}
	char m_charPool[SIZE];
	ALPHA_SHARE_FUNC char* __strcpy(char* dest, const char* src)
	{
		int i = 0;
		do {
			dest[i] = src[i];
		} while (src[i++] != 0);
		return dest;
	}

	ALPHA_SHARE_FUNC void __reverse(char* str, int len)
	{
		int i = 0, j = len - 1, temp;
		while (i < j) {
			temp = str[i];
			str[i] = str[j];
			str[j] = temp;
			i++;
			j--;
		}
	}

	ALPHA_SHARE_FUNC int __itoaEXT(int x, char str[], int d)
	{
		int i = 0;
		while (x) {
			str[i++] = (x % 10) + '0';
			x = x / 10;
		}
		while (i < d)
			str[i++] = '0';
		__reverse(str, i);
		str[i] = '\0';
		return i;
	}

	ALPHA_SHARE_FUNC void __ftoa(float n, char* res, int afterpoint)
	{
		if (n < 0.0f)
		{
			res[0] = '-';
			n = fabsf(n);
			res += 1;
		}
		if (n < 1.0f)
		{
			res[0] = '0';
			res += 1;
		}
		int ipart = (int)n;
		float fpart = n - (float)ipart;
		int i = __itoaEXT(ipart, res, 0);
		if (afterpoint != 0) {
			res[i] = '.'; // add dot
			fpart = fpart * powf(10, afterpoint);
			__itoaEXT((int)fpart, res + i + 1, afterpoint);
		}
	}

	//push string
	ALPHA_SHARE_FUNC char* __strcat(char* dest, const char* src) {
		int i = 0;
		while (dest[i] != 0) {
			i++;
			if (i >= SIZE)
				return dest;
		}
		__strcpy(dest + i, src);
		return dest;
	}

	ALPHA_SHARE_FUNC char* Push(const char* src)
	{
		return __strcat(m_charPool, src);
	}

	ALPHA_SHARE_FUNC void LogInfo(const char* info)
	{
		__strcat(m_charPool, info);
	}

	ALPHA_SHARE_FUNC char* __itoa(int num, char* str, int radix = 10)
	{
		/* index table */
		char index[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
		unsigned unum; /* */
		int i = 0, j, k;
		if (radix == 10 && num < 0) /* neg */
		{
			unum = (unsigned)-num;
			str[i++] = '-';
		}
		else unum = (unsigned)num; /* other */
		/* inverse */
		do
		{
			str[i++] = index[unum % (unsigned)radix];
			unum /= radix;
		} while (unum);
		str[i] = '\0';
		/* transformation */
		if (str[0] == '-') k = 1; /* neg */
		else k = 0;
		char temp;
		for (j = k; j <= (i - k - 1) / 2.0; j++)
		{
			temp = str[j];
			str[j] = str[i - j - 1];
			str[i - j - 1] = temp;
		}
		return str;
	}

	ALPHA_SHARE_FUNC void LogInt(const char* head, int val, const char* endMark = "\n")
	{
		char n[32];
		__itoa(val, n);
		__strcat(m_charPool, head);
		__strcat(m_charPool, n);
 		__strcat(m_charPool, endMark);
	}

	ALPHA_SHARE_FUNC void LogFloat3(
		const char* head,
		float x,
		float y,
		float z,
		bool withEndLine = true,
		bool traceData = false)
	{
		char xx[16];
		char yy[16];
		char zz[16];
		__ftoa(x, xx, 5);
		__ftoa(y, yy, 5);
		__ftoa(z, zz, 5);
		__strcat(m_charPool, head);
		__strcat(m_charPool, xx);
		__strcat(m_charPool, ",");
		__strcat(m_charPool, yy);
		__strcat(m_charPool, ",");
		__strcat(m_charPool, zz);
		if (withEndLine)
			__strcat(m_charPool, "\n");
		if (traceData)
			Trace();
	}

	ALPHA_SHARE_FUNC void Log_Float(char* info)
	{
		__strcat(m_charPool, info);
	}

	ALPHA_SHARE_FUNC void Trace()
	{
		printf("%s", m_charPool);
	}


};

typedef AxLogInfoBlock<8192> AxLogBlock;

inline std::ostream& operator<<(std::ostream& out, AxLogBlock& c)
{
	out << c.m_charPool;
	return out;
}


#endif // !__ALPHA_CORE_LOG_H__
