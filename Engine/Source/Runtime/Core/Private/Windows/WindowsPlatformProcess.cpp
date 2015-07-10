// Copyright 1998-2015 Epic Games, Inc. All Rights Reserved.

#include "CorePrivatePCH.h"
#include "EngineVersion.h"
#include "Resources/Windows/ModuleVersionResource.h"

#include "AllowWindowsPlatformTypes.h"
	#include <shellapi.h>
	#include <shlobj.h>
	#include <LM.h>
	#include <Psapi.h>

	namespace ProcessConstants
	{
		uint32 WIN_STD_INPUT_HANDLE = STD_INPUT_HANDLE;
		uint32 WIN_STD_OUTPUT_HANDLE = STD_OUTPUT_HANDLE;
		uint32 WIN_ATTACH_PARENT_PROCESS = ATTACH_PARENT_PROCESS;		
		uint32 WIN_STILL_ACTIVE = STILL_ACTIVE;
	}

#include "HideWindowsPlatformTypes.h"
#include "WindowsPlatformMisc.h"
#include "EventPool.h"

#pragma comment(lib, "psapi.lib")


// static variables
TArray<FString> FWindowsPlatformProcess::DllDirectoryStack;


void FWindowsPlatformProcess::AddDllDirectory(const TCHAR* Directory)
{
	// Normalize the input directory
	FString NormalizedDirectory = Directory;
	FPaths::NormalizeDirectoryName(NormalizedDirectory);
	FPaths::MakePlatformFilename(NormalizedDirectory);

	// Get the size of the PATH variable
	TArray<TCHAR> PathVariable;
	PathVariable.AddUninitialized(::GetEnvironmentVariable(TEXT("PATH"), NULL, 0));

	// Get the actual value of variable.
	if (::GetEnvironmentVariable(TEXT("PATH"), PathVariable.GetData(), PathVariable.Num()) == 0)
	{
		// Log a warning if reading value fails, but continue anyway.
		UE_LOG(LogWindows, Warning, TEXT("Failed to load PATH environment variable. It either doesn't exist, or is too long."));
		PathVariable.Add(TEXT(';'));
	}

	// Set the new path variable with the input directory at the start. Skip over any existing instances of the input directory.
	FString NewPathVariable = NormalizedDirectory;
	for(const TCHAR* PathPos = PathVariable.GetData(); PathPos < PathVariable.GetData() + PathVariable.Num(); )
	{
		// Scan to the end of this directory
		const TCHAR* PathEnd = PathPos;
		while(*PathEnd != ';' && *PathEnd != 0) PathEnd++;

		// Add it to the new path variable if it doesn't match the input directory
		if(PathEnd - PathPos != NormalizedDirectory.Len() || FCString::Strnicmp(*NormalizedDirectory, PathPos, PathEnd - PathPos) != 0)
		{
			NewPathVariable.AppendChar(TEXT(';'));
			NewPathVariable.AppendChars(PathPos, PathEnd - PathPos);
		}

		// Move to the next string
		PathPos = PathEnd + 1;
	}
	SetEnvironmentVariable(TEXT("PATH"), *NewPathVariable);
}

void* FWindowsPlatformProcess::GetDllHandle( const TCHAR* Filename )
{
	check(Filename);
	::SetErrorMode(SEM_NOOPENFILEERRORBOX);
	return ::LoadLibraryW(Filename);
}

void FWindowsPlatformProcess::FreeDllHandle( void* DllHandle )
{
	// It is okay to call FreeLibrary on 0
	::FreeLibrary((HMODULE)DllHandle);
}

FString FWindowsPlatformProcess::GenerateApplicationPath( const FString& AppName, EBuildConfigurations::Type BuildConfiguration)
{
	FString PlatformName = GetBinariesSubdirectory();
	FString ExecutablePath = FString::Printf(TEXT("..\\%s\\%s"), *PlatformName, *AppName);

	if (BuildConfiguration != EBuildConfigurations::Development && BuildConfiguration != EBuildConfigurations::DebugGame)
	{
		ExecutablePath += FString::Printf(TEXT("-%s-%s"), *PlatformName, EBuildConfigurations::ToString(BuildConfiguration));
	}

	ExecutablePath += TEXT(".exe");

	return ExecutablePath;
}

void* FWindowsPlatformProcess::GetDllExport( void* DllHandle, const TCHAR* ProcName )
{
	check(DllHandle);
	check(ProcName);
	return (void*)::GetProcAddress( (HMODULE)DllHandle, TCHAR_TO_ANSI(ProcName) );
}

int32 FWindowsPlatformProcess::GetDllApiVersion( const TCHAR* Filename )
{
	int32 Result = -1;

	// Retrieves the embedded API version from a DLL
	check(Filename);
	HMODULE hModule = LoadLibraryEx(Filename, NULL, LOAD_LIBRARY_AS_DATAFILE);
	if(hModule != NULL)
	{
		HRSRC hResInfo = FindResource(hModule, MAKEINTRESOURCE(ID_MODULE_API_VERSION_RESOURCE), RT_RCDATA);
		if(hResInfo != NULL)
		{
			HGLOBAL hResGlobal = LoadResource(hModule, hResInfo);
			if(hResGlobal != NULL)
			{
				void *pResData = LockResource(hResGlobal);
				if(pResData != NULL)
				{
					::DWORD Length = SizeofResource(hModule, hResInfo);
					if(Length > 0)
					{
						char *Str = (char*)pResData;
						if(Str[Length - 1] == 0)
						{
							char *End = Str;
							uint64 Value = FCStringAnsi::Strtoui64(Str, &End, 10);
							if(*End == 0 && Value <= INT_MAX)
							{
								Result = (int32)Value;
							}
						}
					}
					UnlockResource(pResData);
				}
			}
		}
		FreeLibrary(hModule);
	}

	return Result;
}

void FWindowsPlatformProcess::PushDllDirectory(const TCHAR* Directory)
{
	// set the directory in windows
	::SetDllDirectory(Directory);
	// remember it
	DllDirectoryStack.Push(Directory);
}

void FWindowsPlatformProcess::PopDllDirectory(const TCHAR* Directory)
{
	// don't allow too many pops (indicates bad code that should be fixed, but won't kill anything, so using ensure)
	ensureMsgf(DllDirectoryStack.Num() > 0, TEXT("Tried to PopDllDirectory too many times"));
	// verify we are popping the top
	checkf(DllDirectoryStack.Top() == Directory, TEXT("There was a PushDllDirectory/PopDllDirectory mismatch (Popped %s, which didn't match %s)"), *DllDirectoryStack.Top(), Directory);
	// pop it off
	DllDirectoryStack.Pop();

	// and now set the new DllDirectory to the old value
	if (DllDirectoryStack.Num() > 0)
	{
		::SetDllDirectory(*DllDirectoryStack.Top());
	}
	else
	{
		::SetDllDirectory(TEXT(""));
	}
}

static void LaunchWebURL( const FString& URLParams, FString* Error )
{
	UE_LOG(LogWindows, Log, TEXT("LaunchURL %s"), *URLParams);

	FString BrowserOpenCommand;

	// First lookup the program Id for the default browser.
	FString ProgId;
	if (FWindowsPlatformMisc::QueryRegKey(HKEY_CURRENT_USER, TEXT("Software\\Microsoft\\Windows\\Shell\\Associations\\UrlAssociations\\http\\UserChoice"), TEXT("Progid"), ProgId))
	{
		// If we found it, then lookup it's open shell command in the classes registry.
		FString BrowserRegPath = ProgId + TEXT("\\shell\\open\\command");
		FWindowsPlatformMisc::QueryRegKey(HKEY_CLASSES_ROOT, *BrowserRegPath, NULL, BrowserOpenCommand);
	}

	// If we failed to find a default browser using the newer location, revert to using shell open command for the HTTP file association.
	if (BrowserOpenCommand.IsEmpty())
	{
		FWindowsPlatformMisc::QueryRegKey(HKEY_CLASSES_ROOT, TEXT("http\\shell\\open\\command"), NULL, BrowserOpenCommand);
	}

	// If we have successfully looked up the correct shell command, then we can create a new process using that command
	// we do this instead of shell execute due to security concerns.  By starting the browser directly we avoid most issues.
	if (!BrowserOpenCommand.IsEmpty())
	{
		FString ExePath, ExeArgs;

		// If everything has gone to plan, the shell command should be something like this:
		// "C:\Program Files (x86)\Mozilla Firefox\firefox.exe" -osint -url "%1"
		// We need to extract out the executable portion, and the arguments portion and expand any %1's with the URL,
		// then start the browser process.

		// Extract the exe and any arguments to the executable.
		const int32 FirstQuote = BrowserOpenCommand.Find(TEXT("\""));
		if (FirstQuote != INDEX_NONE)
		{
			const int32 SecondQuote = BrowserOpenCommand.Find(TEXT("\""), ESearchCase::IgnoreCase, ESearchDir::FromStart, FirstQuote + 1);
			if (SecondQuote != INDEX_NONE)
			{
				ExePath = BrowserOpenCommand.Mid(FirstQuote + 1, (SecondQuote - 1) - FirstQuote);
				ExeArgs = BrowserOpenCommand.Mid(SecondQuote + 1);
			}
		}

		// If anything failed to parse right, don't continue down this path, just use shell execute.
		if (!ExePath.IsEmpty())
		{
			ExeArgs = ExeArgs.Replace(TEXT("%1"), *URLParams);

			// Now that we have the shell open command to use, run the shell command in the open process with any and all parameters.
			if (FPlatformProcess::CreateProc(*ExePath, *ExeArgs, true, false, false, NULL, 0, NULL, NULL).IsValid())
			{
				// Success!
				return;
			}
			else
			{
				if (Error)
				{
					*Error = NSLOCTEXT("Core", "UrlFailed", "Failed launching URL").ToString();
				}
			}
		}
	}

	// If all else fails just do a shell execute and let windows sort it out.  But only do it if it's an
	// HTTP or HTTPS address.  A malicious address could be problematic if just passed directly to shell execute.
	if (URLParams.StartsWith(TEXT("http://")) || URLParams.StartsWith(TEXT("https://")))
	{
		const HINSTANCE Code = ::ShellExecuteW(NULL, TEXT("open"), *URLParams, NULL, NULL, SW_SHOWNORMAL);
		if (Error)
		{
			*Error = ((PTRINT)Code <= 32) ? NSLOCTEXT("Core", "UrlFailed", "Failed launching URL").ToString() : TEXT("");
		}
	}
}

static void LaunchDefaultHandlerForURL( const TCHAR* URL, FString* Error )
{
	// ShellExecute will open the default handler for a URL
	const HINSTANCE Code = ::ShellExecuteW(NULL, TEXT("open"), URL, NULL, NULL, SW_SHOWNORMAL);
	if (Error)
	{
		*Error = ((PTRINT)Code <= 32) ? NSLOCTEXT("Core", "UrlFailed", "Failed launching URL").ToString() : TEXT("");
	}
}

void FWindowsPlatformProcess::LaunchURL( const TCHAR* URL, const TCHAR* Parms, FString* Error )
{
	check(URL);

	// Initialize the error to empty string.
	if (Error)
	{
		*Error = TEXT("");
	}

	// Use the default handler if we have a URI scheme name that doesn't look like a Windows path, and is not http: or https:
	FString SchemeName;
	if(FParse::SchemeNameFromURI(URL, SchemeName) && SchemeName.Len() > 1 && SchemeName != TEXT("http") && SchemeName != TEXT("https"))
	{
		LaunchDefaultHandlerForURL(URL, Error);
	}
	else
	{
		FString URLParams = FString::Printf(TEXT("%s %s"), URL, Parms ? Parms : TEXT("")).TrimTrailing();
		LaunchWebURL( URLParams, Error );
	}

}

FProcHandle FWindowsPlatformProcess::CreateProc( const TCHAR* URL, const TCHAR* Parms, bool bLaunchDetached, bool bLaunchHidden, bool bLaunchReallyHidden, uint32* OutProcessID, int32 PriorityModifier, const TCHAR* OptionalWorkingDirectory, void* PipeWrite )
{
	//UE_LOG(LogWindows, Log,  TEXT("CreateProc %s %s"), URL, Parms );

	// initialize process attributes
	SECURITY_ATTRIBUTES Attr;
	Attr.nLength = sizeof(SECURITY_ATTRIBUTES);
	Attr.lpSecurityDescriptor = NULL;
	Attr.bInheritHandle = true;

	// initialize process creation flags
	uint32 CreateFlags = NORMAL_PRIORITY_CLASS;
	if (PriorityModifier < 0)
	{
		CreateFlags = (PriorityModifier == -1) ? BELOW_NORMAL_PRIORITY_CLASS : IDLE_PRIORITY_CLASS;
	}
	else if (PriorityModifier > 0)
	{
		CreateFlags = (PriorityModifier == 1) ? ABOVE_NORMAL_PRIORITY_CLASS : HIGH_PRIORITY_CLASS;
	}

	if (bLaunchDetached)
	{
		CreateFlags |= DETACHED_PROCESS;
	}

	// initialize window flags
	uint32 dwFlags = 0;
	uint16 ShowWindowFlags = SW_HIDE;
	if (bLaunchReallyHidden)
	{
		dwFlags = STARTF_USESHOWWINDOW;
	}
	else if (bLaunchHidden)
	{
		dwFlags = STARTF_USESHOWWINDOW;
		ShowWindowFlags = SW_SHOWMINNOACTIVE;
	}

	if (PipeWrite != nullptr)
	{
		dwFlags |= STARTF_USESTDHANDLES;
	}

	// initialize startup info
	STARTUPINFO StartupInfo = {
		sizeof(STARTUPINFO),
		NULL, NULL, NULL,
		(::DWORD)CW_USEDEFAULT,
		(::DWORD)CW_USEDEFAULT,
		(::DWORD)CW_USEDEFAULT,
		(::DWORD)CW_USEDEFAULT,
		(::DWORD)0, (::DWORD)0, (::DWORD)0,
		(::DWORD)dwFlags,
		ShowWindowFlags,
		0, NULL,
		::GetStdHandle(ProcessConstants::WIN_STD_INPUT_HANDLE),
		HANDLE(PipeWrite),
		HANDLE(PipeWrite)
	};

	// create the child process
	FString CommandLine = FString::Printf(TEXT("\"%s\" %s"), URL, Parms);
	PROCESS_INFORMATION ProcInfo;

	if (!CreateProcess(NULL, CommandLine.GetCharArray().GetData(), &Attr, &Attr, true, (::DWORD)CreateFlags, NULL, OptionalWorkingDirectory, &StartupInfo, &ProcInfo))
	{
		UE_LOG(LogWindows, Warning, TEXT("CreateProc failed (%u) %s %s"), ::GetLastError(), URL, Parms);
		if (OutProcessID != nullptr)
		{
			*OutProcessID = 0;
		}

		return FProcHandle();
	}

	if (OutProcessID != nullptr)
	{
		*OutProcessID = ProcInfo.dwProcessId;
	}

	::CloseHandle( ProcInfo.hThread );

	return FProcHandle(ProcInfo.hProcess);
}

bool FWindowsPlatformProcess::IsProcRunning( FProcHandle & ProcessHandle )
{
	bool bApplicationRunning = true;
	uint32 WaitResult = ::WaitForSingleObject(ProcessHandle.Get(), 0);
	if (WaitResult != WAIT_TIMEOUT)
	{
		bApplicationRunning = false;
	}
	return bApplicationRunning;
}

void FWindowsPlatformProcess::WaitForProc( FProcHandle & ProcessHandle )
{
	::WaitForSingleObject(ProcessHandle.Get(), INFINITE);
}

void FWindowsPlatformProcess::CloseProc(FProcHandle & ProcessHandle)
{
	if (ProcessHandle.IsValid())
	{
		::CloseHandle(ProcessHandle.Get());
		ProcessHandle.Reset();
	}
}

void FWindowsPlatformProcess::TerminateProc( FProcHandle & ProcessHandle, bool KillTree )
{
	if (KillTree)
	{
		HANDLE SnapShot = ::CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);

		if (SnapShot != INVALID_HANDLE_VALUE)
		{
			::DWORD ProcessId = ::GetProcessId(ProcessHandle.Get());

			PROCESSENTRY32 Entry;
			Entry.dwSize = sizeof(PROCESSENTRY32);

			if (::Process32First(SnapShot, &Entry))
			{
				do
				{
					if (Entry.th32ParentProcessID == ProcessId)
					{
						HANDLE ChildProcHandle = ::OpenProcess(PROCESS_ALL_ACCESS, 0, Entry.th32ProcessID);

						if (ChildProcHandle)
						{
							FProcHandle ChildHandle(ChildProcHandle);
							TerminateProc(ChildHandle, KillTree);
//							::TerminateProcess(ChildProcHandle, 1);
						}
					}
				}
				while(::Process32Next(SnapShot, &Entry));
			}
		}
	}

	TerminateProcess(ProcessHandle.Get(),0);
}

uint32 FWindowsPlatformProcess::GetCurrentProcessId()
{
	return ::GetCurrentProcessId();
}

void FWindowsPlatformProcess::SetThreadAffinityMask( uint64 AffinityMask )
{
	if( AffinityMask != FPlatformAffinity::GetNoAffinityMask() )
	{
		::SetThreadAffinityMask( ::GetCurrentThread(), (DWORD_PTR)AffinityMask );
	}
}

bool FWindowsPlatformProcess::GetProcReturnCode( FProcHandle & ProcHandle, int32* ReturnCode )
{
	return ::GetExitCodeProcess( ProcHandle.Get(), (::DWORD *)ReturnCode ) && *((uint32*)ReturnCode) != ProcessConstants::WIN_STILL_ACTIVE;
}

bool FWindowsPlatformProcess::GetApplicationMemoryUsage(uint32 ProcessId, SIZE_T* OutMemoryUsage)
{
	bool bSuccess = false;
	HANDLE ProcessHandle = ::OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, false, ProcessId);

	if (ProcessHandle != NULL)
	{
		PROCESS_MEMORY_COUNTERS_EX MemoryInfo;

		if (GetProcessMemoryInfo(ProcessHandle, (PROCESS_MEMORY_COUNTERS*)&MemoryInfo, sizeof(MemoryInfo)))
		{
			*OutMemoryUsage = MemoryInfo.PrivateUsage;
			bSuccess = true;
		}

		::CloseHandle(ProcessHandle);
	}

	return bSuccess;
}

bool FWindowsPlatformProcess::IsApplicationRunning( uint32 ProcessId )
{
	bool bApplicationRunning = true;
	HANDLE ProcessHandle = ::OpenProcess(SYNCHRONIZE, false, ProcessId);
	if (ProcessHandle == NULL)
	{
		bApplicationRunning = false;
	}
	else
	{
		uint32 WaitResult = WaitForSingleObject(ProcessHandle, 0);
		if (WaitResult != WAIT_TIMEOUT)
		{
			bApplicationRunning = false;
		}
		::CloseHandle(ProcessHandle);
	}
	return bApplicationRunning;
}

bool FWindowsPlatformProcess::IsApplicationRunning( const TCHAR* ProcName )
{
	// append the extension

	FString ProcNameWithExtension = ProcName;
	if( ProcNameWithExtension.Find( TEXT(".exe"), ESearchCase::IgnoreCase, ESearchDir::FromEnd ) == INDEX_NONE )
	{
		ProcNameWithExtension += TEXT(".exe");
	}

	HANDLE SnapShot = ::CreateToolhelp32Snapshot( TH32CS_SNAPPROCESS, 0 );
	if( SnapShot != INVALID_HANDLE_VALUE )
	{
		PROCESSENTRY32 Entry;
		Entry.dwSize = sizeof( PROCESSENTRY32 );

		if( ::Process32First( SnapShot, &Entry ) )
		{
			do
			{
				if( FCString::Strcmp( *ProcNameWithExtension, Entry.szExeFile ) == 0 )
				{
					::CloseHandle( SnapShot );
					return true;
				}
			} while( ::Process32Next( SnapShot, &Entry ) );
		}
	}

	::CloseHandle( SnapShot );
	return false;
}

FString FWindowsPlatformProcess::GetApplicationName( uint32 ProcessId )
{
	FString Output = TEXT("");
	HANDLE ProcessHandle = ::OpenProcess(PROCESS_QUERY_INFORMATION, false, ProcessId);
	if (ProcessHandle != NULL)
	{
		const int32 ProcessNameBufferSize = 4096;
		TCHAR ProcessNameBuffer[ProcessNameBufferSize];
		
		int32 InOutSize = ProcessNameBufferSize;
		static_assert(sizeof(::DWORD) == sizeof(int32), "DWORD size doesn't match int32. Is it the future or the past?");

#if WINVER == 0x0502
		GetProcessImageFileName(ProcessHandle, ProcessNameBuffer, InOutSize);
#else
		QueryFullProcessImageName(ProcessHandle, 0, ProcessNameBuffer, (PDWORD)(&InOutSize));
#endif

		Output = ProcessNameBuffer;
	}

	return Output;
}


bool FWindowsPlatformProcess::IsThisApplicationForeground()
{
	uint32 ForegroundProcess;
	::GetWindowThreadProcessId(GetForegroundWindow(), (::DWORD *)&ForegroundProcess);
	return (ForegroundProcess == GetCurrentProcessId());
}

void FWindowsPlatformProcess::ReadFromPipes(FString* OutStrings[], HANDLE InPipes[], int32 PipeCount)
{
	for (int32 PipeIndex = 0; PipeIndex < PipeCount; ++PipeIndex)
	{
		if (InPipes[PipeIndex] && OutStrings[PipeIndex])
		{
			*OutStrings[PipeIndex] += ReadPipe(InPipes[PipeIndex]);
		}
	}
}

/**
 * Executes a process, returning the return code, stdout, and stderr. This
 * call blocks until the process has returned.
 */
bool FWindowsPlatformProcess::ExecProcess( const TCHAR* URL, const TCHAR* Params, int32* OutReturnCode, FString* OutStdOut, FString* OutStdErr )
{
	PROCESS_INFORMATION ProcInfo;
	SECURITY_ATTRIBUTES Attr;

	FString CommandLine = FString::Printf(TEXT("%s %s"), URL, Params);

	Attr.nLength = sizeof(SECURITY_ATTRIBUTES);
	Attr.lpSecurityDescriptor = NULL;
	Attr.bInheritHandle = true;

	uint32 CreateFlags = NORMAL_PRIORITY_CLASS;
	CreateFlags |= DETACHED_PROCESS;

	uint32 dwFlags = STARTF_USESHOWWINDOW;
	uint16 ShowWindowFlags = SW_SHOWMINNOACTIVE;

	const int32 MaxPipeCount = 2;
	HANDLE ReadablePipes[MaxPipeCount] = {0};
	HANDLE WritablePipes[MaxPipeCount] = {0};
	const bool bRedirectOutput = OutStdOut != NULL || OutStdErr != NULL;

	if (bRedirectOutput)
	{
		dwFlags |= STARTF_USESTDHANDLES;
		for (int32 PipeIndex = 0; PipeIndex < MaxPipeCount; ++PipeIndex)
		{
			verify(::CreatePipe(&ReadablePipes[PipeIndex], &WritablePipes[PipeIndex], &Attr, 0));
			verify(::SetHandleInformation(ReadablePipes[PipeIndex], /*dwMask=*/ HANDLE_FLAG_INHERIT, /*dwFlags=*/ 0));
		}
	}

	bool bSuccess = false;
	STARTUPINFO StartupInfo = { sizeof(STARTUPINFO), NULL, NULL, NULL,
		(::DWORD)CW_USEDEFAULT, (::DWORD)CW_USEDEFAULT, (::DWORD)CW_USEDEFAULT, (::DWORD)CW_USEDEFAULT,
		(::DWORD)0, (::DWORD)0, (::DWORD)0, (::DWORD)dwFlags, ShowWindowFlags, 0, NULL,
		::GetStdHandle((::DWORD)ProcessConstants::WIN_STD_INPUT_HANDLE), WritablePipes[0], WritablePipes[1] };
	if (CreateProcess(NULL, CommandLine.GetCharArray().GetData(), &Attr, &Attr, true, CreateFlags,
		NULL, NULL, &StartupInfo, &ProcInfo))
	{
		if (bRedirectOutput)
		{
			FString* OutStrings[MaxPipeCount] = { OutStdOut, OutStdErr };
			FProcHandle ProcHandle(ProcInfo.hProcess);
			do 
			{
				ReadFromPipes(OutStrings, ReadablePipes, MaxPipeCount);
				FPlatformProcess::Sleep(0);
			} while (IsProcRunning(ProcHandle));
			ReadFromPipes(OutStrings, ReadablePipes, MaxPipeCount);
		}
		else
		{
			::WaitForSingleObject(ProcInfo.hProcess, INFINITE);
		}		
		if (OutReturnCode)
		{
			verify(::GetExitCodeProcess(ProcInfo.hProcess, (::DWORD*)OutReturnCode));
		}
		::CloseHandle(ProcInfo.hProcess);
		::CloseHandle(ProcInfo.hThread);
		bSuccess = true;
	}
	else
	{
		// if CreateProcess failed, we should return a useful error code, which GetLastError will have
		if (OutReturnCode)
		{
			*OutReturnCode = GetLastError();
		}
		if (bRedirectOutput)
		{
			for (int32 PipeIndex = 0; PipeIndex < MaxPipeCount; ++PipeIndex)
			{
				verify(::CloseHandle(WritablePipes[PipeIndex]));
			}
		}
	}

	if (bRedirectOutput)
	{
		for (int32 PipeIndex = 0; PipeIndex < MaxPipeCount; ++PipeIndex)
		{
			verify(::CloseHandle(ReadablePipes[PipeIndex]));
		}
	}

	return bSuccess;
}

bool FWindowsPlatformProcess::ExecElevatedProcess(const TCHAR* URL, const TCHAR* Params, int32* OutReturnCode)
{
	SHELLEXECUTEINFO ShellExecuteInfo;
	ZeroMemory(&ShellExecuteInfo, sizeof(ShellExecuteInfo));
	ShellExecuteInfo.cbSize = sizeof(ShellExecuteInfo);
	ShellExecuteInfo.fMask = SEE_MASK_UNICODE | SEE_MASK_NOCLOSEPROCESS;
	ShellExecuteInfo.lpFile = URL;
	ShellExecuteInfo.lpVerb = TEXT("runas");
	ShellExecuteInfo.nShow = SW_SHOW;
	ShellExecuteInfo.lpParameters = Params;

	bool bSuccess = false;
	if (ShellExecuteEx(&ShellExecuteInfo))
	{
		::WaitForSingleObject(ShellExecuteInfo.hProcess, INFINITE);
		if (OutReturnCode != NULL)
		{
			verify(::GetExitCodeProcess(ShellExecuteInfo.hProcess, (::DWORD*)OutReturnCode));
		}
		verify(::CloseHandle(ShellExecuteInfo.hProcess));
		bSuccess = true;
	}
	return bSuccess;
}

void FWindowsPlatformProcess::CleanFileCache()
{
	bool bShouldCleanShaderWorkingDirectory = true;
#if !(UE_BUILD_SHIPPING && WITH_EDITOR)
	// Only clean the shader working directory if we are the first instance, to avoid deleting files in use by other instances
	//@todo - check if any other instances are running right now
	bShouldCleanShaderWorkingDirectory = GIsFirstInstance;
#endif

	if (bShouldCleanShaderWorkingDirectory && !FParse::Param( FCommandLine::Get(), TEXT("Multiprocess")))
	{
		// get shader path, and convert it to the userdirectory
		FString ShaderDir = FString(FPlatformProcess::BaseDir()) / FPlatformProcess::ShaderDir();
		FString UserShaderDir = IFileManager::Get().ConvertToAbsolutePathForExternalAppForWrite(*ShaderDir);
		FPaths::CollapseRelativeDirectories(ShaderDir);

		// make sure we don't delete from the source directory
		if (ShaderDir != UserShaderDir)
		{
			IFileManager::Get().DeleteDirectory(*UserShaderDir, false, true);
		}

		FPlatformProcess::CleanShaderWorkingDir();
	}
}

const TCHAR* FWindowsPlatformProcess::BaseDir()
{
	static TCHAR Result[512]=TEXT("");
	if( !Result[0] )
	{
		// Get directory this executable was launched from.
		GetModuleFileName( hInstance, Result, ARRAY_COUNT(Result) );
		FString TempResult(Result);
		TempResult = TempResult.Replace(TEXT("\\"), TEXT("/"));
		FCString::Strcpy(Result, *TempResult);
		int32 StringLength = FCString::Strlen(Result);
		if(StringLength > 0)
		{
			--StringLength;
			for(; StringLength > 0; StringLength-- )
			{
				if( Result[StringLength - 1] == TEXT('/') || Result[StringLength - 1] == TEXT('\\') )
				{
					break;
				}
			}
		}
		Result[StringLength] = 0;

		FString CollapseResult(Result);
		FPaths::CollapseRelativeDirectories(CollapseResult);
		FCString::Strcpy(Result, *CollapseResult);
	}
	return Result;
}

const TCHAR* FWindowsPlatformProcess::UserDir()
{
	static FString WindowsUserDir;
	if( !WindowsUserDir.Len() )
	{
		TCHAR UserPath[MAX_PATH];
		// get the My Documents directory
		HRESULT Ret = SHGetFolderPath(NULL, CSIDL_PERSONAL, NULL, SHGFP_TYPE_CURRENT, UserPath);

		// make the base user dir path
		WindowsUserDir = FString(UserPath).Replace(TEXT("\\"), TEXT("/")) + TEXT("/");
	}
	return *WindowsUserDir;
}

const TCHAR* FWindowsPlatformProcess::UserTempDir()
{
	static FString WindowsUserTempDir;
	if( !WindowsUserTempDir.Len() )
	{
		TCHAR TempPath[MAX_PATH];
		ZeroMemory(TempPath, sizeof(TCHAR) * MAX_PATH);

		::GetTempPath(MAX_PATH, TempPath);

		WindowsUserTempDir = FString(TempPath).Replace(TEXT("\\"), TEXT("/"));
	}
	return *WindowsUserTempDir;
}

const TCHAR* FWindowsPlatformProcess::UserSettingsDir()
{
	static FString WindowsUserSettingsDir;
	if (!WindowsUserSettingsDir.Len())
	{
		TCHAR UserPath[MAX_PATH];
		// get the My Documents directory
		HRESULT Ret = SHGetFolderPath(NULL, CSIDL_LOCAL_APPDATA, NULL, SHGFP_TYPE_CURRENT, UserPath);

		// make the base user dir path
		WindowsUserSettingsDir = FString(UserPath).Replace(TEXT("\\"), TEXT("/")) + TEXT("/");
	}
	return *WindowsUserSettingsDir;
}

const TCHAR* FWindowsPlatformProcess::ApplicationSettingsDir()
{
	static FString WindowsApplicationSettingsDir;
	if( !WindowsApplicationSettingsDir.Len() )
	{
		TCHAR ApplictionSettingsPath[MAX_PATH];
		// get the My Documents directory
		HRESULT Ret = SHGetFolderPath(NULL, CSIDL_COMMON_APPDATA, NULL, SHGFP_TYPE_CURRENT, ApplictionSettingsPath);

		// make the base user dir path
		// @todo rocket this folder should be based on your company name, not just be hard coded to /Epic/
		WindowsApplicationSettingsDir = FString(ApplictionSettingsPath) + TEXT("/Epic/");
	}
	return *WindowsApplicationSettingsDir;
}

const TCHAR* FWindowsPlatformProcess::ComputerName()
{
	static TCHAR Result[256]=TEXT("");
	if( !Result[0] )
	{
		uint32 Size=ARRAY_COUNT(Result);
		GetComputerName( Result, (::DWORD*)&Size );
	}
	return Result;
}

const TCHAR* FWindowsPlatformProcess::UserName(bool bOnlyAlphaNumeric/* = true*/)
{
	static TCHAR Result[256]=TEXT("");
	static TCHAR ResultAlpha[256]=TEXT("");
	if( bOnlyAlphaNumeric )
	{
		if( !ResultAlpha[0] )
		{
			uint32 Size=ARRAY_COUNT(ResultAlpha);
			GetUserName( ResultAlpha, (::DWORD*)&Size );
			TCHAR *c, *d;
			for( c=ResultAlpha, d=ResultAlpha; *c!=0; c++ )
				if( FChar::IsAlnum(*c) )
					*d++ = *c;
			*d++ = 0;
		}
		return ResultAlpha;
	}
	else
	{
		if( !Result[0] )
		{
			uint32 Size=ARRAY_COUNT(Result);
			GetUserName( Result, (::DWORD*)&Size );
		}
		return Result;
	}
}

void FWindowsPlatformProcess::SetCurrentWorkingDirectoryToBaseDir()
{
	FPlatformMisc::CacheLaunchDir();
	verify(SetCurrentDirectoryW(BaseDir()));
}

/** Get the current working directory (only really makes sense on desktop platforms) */
FString FWindowsPlatformProcess::GetCurrentWorkingDirectory()
{
	// get the current working directory (uncached)
	TCHAR CurrentDirectory[MAX_PATH];
	GetCurrentDirectoryW(MAX_PATH, CurrentDirectory);
	return CurrentDirectory;
}

const TCHAR* FWindowsPlatformProcess::ExecutableName(bool bRemoveExtension)
{
	static TCHAR Result[512]=TEXT("");
	static TCHAR ResultWithExt[512]=TEXT("");
	if( !Result[0] )
	{
		// Get complete path for the executable
		if ( GetModuleFileName( hInstance, Result, ARRAY_COUNT(Result) ) != 0 )
		{
			// Remove all of the path information by finding the base filename
			FString FileName = Result;
			FString FileNameWithExt = Result;
			FCString::Strncpy( Result, *( FPaths::GetBaseFilename(FileName) ), ARRAY_COUNT(Result) );
			FCString::Strncpy( ResultWithExt, *( FPaths::GetCleanFilename(FileNameWithExt) ), ARRAY_COUNT(ResultWithExt) );
		}
		// If the call failed, zero out the memory to be safe
		else
		{
			FMemory::Memzero( Result, sizeof( Result ) );
			FMemory::Memzero( ResultWithExt, sizeof( ResultWithExt ) );
		}
	}

	return (bRemoveExtension ? Result : ResultWithExt);
}

const TCHAR* FWindowsPlatformProcess::GetModuleExtension()
{
	return TEXT("dll");
}

const TCHAR* FWindowsPlatformProcess::GetBinariesSubdirectory()
{
	if (PLATFORM_64BITS)
	{
		return TEXT("Win64");
	}
	return TEXT("Win32");
}

void FWindowsPlatformProcess::LaunchFileInDefaultExternalApplication( const TCHAR* FileName, const TCHAR* Parms /*= NULL*/, ELaunchVerb::Type Verb /*= ELaunchVerb::Open*/ )
{
	const TCHAR* VerbString = Verb == ELaunchVerb::Edit ? TEXT("edit") : TEXT("open");

	// First attempt to open the file in its default application
	UE_LOG(LogWindows, Log,  TEXT("LaunchFileInExternalEditor %s %s"), FileName, Parms ? Parms : TEXT("") );
	HINSTANCE Code = ::ShellExecuteW( NULL, VerbString, FileName, Parms ? Parms : TEXT(""), TEXT(""), SW_SHOWNORMAL );
	
	UE_LOG(LogWindows, Log,  TEXT("Launch application code for %s %s: %d"), FileName, Parms ? Parms : TEXT(""), (PTRINT)Code );

	// If opening the file in the default application failed, check to see if it's because the file's extension does not have
	// a default application associated with it. If so, prompt the user with the Windows "Open With..." dialog to allow them to specify
	// an application to use.
	if ( (PTRINT)Code == SE_ERR_NOASSOC || (PTRINT)Code == SE_ERR_ASSOCINCOMPLETE )
	{
		::ShellExecuteW( NULL, VerbString, TEXT("RUNDLL32.EXE"), *FString::Printf( TEXT("shell32.dll,OpenAs_RunDLL %s"), FileName ), TEXT(""), SW_SHOWNORMAL );
	}
}

void FWindowsPlatformProcess::ExploreFolder( const TCHAR* FilePath )
{
	if (IFileManager::Get().DirectoryExists( FilePath ))
	{
		// Explore the folder
		::ShellExecuteW( NULL, TEXT("explore"), FilePath, NULL, NULL, SW_SHOWNORMAL );
	}
	else
	{
		// Explore the file
		FString NativeFilePath = FString(FilePath).Replace(TEXT("/"), TEXT("\\"));
		FString Parameters = FString::Printf( TEXT("/select,%s"), *NativeFilePath);
		::ShellExecuteW( NULL, TEXT("open"), TEXT("explorer.exe"), *Parameters, NULL, SW_SHOWNORMAL );
	}
}

/**
 * Resolves UNC path to local (full) path if possible.
 *
 * @param	InUNCPath		UNC path to resolve
 *
 * @param	OutPath		Resolved local path
 *
 * @return true if the path was resolved, false otherwise
 */
bool FWindowsPlatformProcess::ResolveNetworkPath( FString InUNCPath, FString& OutPath )
{
	// Get local machine name first and check if this UNC path points to local share
	// (if it's not UNC path it will also fail this check)
	uint32 ComputerNameSize = MAX_COMPUTERNAME_LENGTH;
	TCHAR ComputerName[MAX_COMPUTERNAME_LENGTH + 3] = { '\\', '\\', '\0', };

	if ( GetComputerName( ComputerName + 2, (::DWORD*)&ComputerNameSize ) )
	{
		// Check if the filename is pointing to local shared folder
		if ( InUNCPath.StartsWith( ComputerName ) )
		{
			// Get the share name (it's the first folder after the computer name)
			int32 ComputerNameLen = FCString::Strlen( ComputerName );
			int32 ShareNameLen = InUNCPath.Find( TEXT( "\\" ), ESearchCase::CaseSensitive, ESearchDir::FromStart, ComputerNameLen + 1 ) - ComputerNameLen - 1;
			FString ShareName = InUNCPath.Mid( ComputerNameLen + 1, ShareNameLen );

			// NetShareGetInfo doesn't accept const TCHAR* as the share name so copy to temp array
			SHARE_INFO_2* BufPtr = NULL;
			::NET_API_STATUS res;
			TCHAR ShareNamePtr[MAX_PATH];
			FCString::Strcpy(ShareNamePtr, ShareName.Len() + 1, *ShareName);

			// Call the NetShareGetInfo function, specifying level 2
			if ( ( res = NetShareGetInfo( NULL, ShareNamePtr, 2, (LPBYTE*)&BufPtr ) ) == ERROR_SUCCESS )
			{
				// Construct the local path
				OutPath = FString( BufPtr->shi2_path ) + InUNCPath.Mid( ComputerNameLen + 1 + ShareNameLen );

				// Free the buffer allocated by NetShareGetInfo
				NetApiBufferFree(BufPtr);
				
				return true;
			}
		}
	}

	// InUNCPath is not an UNC path or it's not pointing to local folder or something went wrong in NetShareGetInfo (insufficient privileges?)
	return false;
}

void FWindowsPlatformProcess::Sleep( float Seconds )
{
	SCOPE_CYCLE_COUNTER( STAT_Sleep );
	FThreadIdleStats::FScopeIdle Scope;
	SleepNoStats( Seconds );
}

void FWindowsPlatformProcess::SleepNoStats(float Seconds)
{
	::Sleep((uint32)(Seconds * 1000.0));
}

void FWindowsPlatformProcess::SleepInfinite()
{
	check(FPlatformProcess::SupportsMultithreading());
	::Sleep(INFINITE);
}

#include "WindowsEvent.h"

FEvent* FWindowsPlatformProcess::CreateSynchEvent(bool bIsManualReset)
{
	// Allocate the new object
	FEvent* Event = NULL;	
	if (FPlatformProcess::SupportsMultithreading())
	{
		Event = new FEventWin();
	}
	else
	{
		// Fake event object.
		Event = new FSingleThreadEvent();
	}
	// If the internal create fails, delete the instance and return NULL
	if (!Event->Create(bIsManualReset))
	{
		delete Event;
		Event = NULL;
	}
	return Event;
}

#include "AllowWindowsPlatformTypes.h"

bool FEventWin::Wait(uint32 WaitTime, const bool bIgnoreThreadIdleStats /*= false*/)
{
	WaitForStats();

	SCOPE_CYCLE_COUNTER( STAT_EventWait );
	check( Event );

	FThreadIdleStats::FScopeIdle Scope( bIgnoreThreadIdleStats );
	return (WaitForSingleObject( Event, WaitTime ) == WAIT_OBJECT_0);
}

void FEventWin::Trigger()
{
	TriggerForStats();
	check( Event );
	SetEvent( Event );
}

void FEventWin::Reset()
{
	ResetForStats();
	check( Event );
	ResetEvent( Event );
}

#include "HideWindowsPlatformTypes.h"

#include "WindowsRunnableThread.h"

FRunnableThread* FWindowsPlatformProcess::CreateRunnableThread()
{
	return new FRunnableThreadWin();
}

void FWindowsPlatformProcess::ClosePipe( void* ReadPipe, void* WritePipe )
{
	if ( ReadPipe != NULL && ReadPipe != INVALID_HANDLE_VALUE )
	{
		::CloseHandle(ReadPipe);
	}
	if ( WritePipe != NULL && WritePipe != INVALID_HANDLE_VALUE )
	{
		::CloseHandle(WritePipe);
	}
}

bool FWindowsPlatformProcess::CreatePipe( void*& ReadPipe, void*& WritePipe )
{
	SECURITY_ATTRIBUTES Attr = { sizeof(SECURITY_ATTRIBUTES), NULL, true };
	
	if (!::CreatePipe(&ReadPipe, &WritePipe, &Attr, 0))
	{
		return false;
	}

	if (!::SetHandleInformation(ReadPipe, HANDLE_FLAG_INHERIT, 0))
	{
		return false;
	}

	return true;
}

FString FWindowsPlatformProcess::ReadPipe( void* ReadPipe )
{
	FString Output;

	uint32 BytesAvailable = 0;
	if (::PeekNamedPipe(ReadPipe, NULL, 0, NULL, (::DWORD*)&BytesAvailable, NULL) && (BytesAvailable > 0))
	{
		UTF8CHAR* Buffer = new UTF8CHAR[BytesAvailable + 1];
		uint32 BytesRead = 0;
		if (::ReadFile(ReadPipe, Buffer, BytesAvailable, (::DWORD*)&BytesRead, NULL))
		{
			if (BytesRead > 0)
			{
				Buffer[BytesRead] = '\0';
				Output += FUTF8ToTCHAR((const ANSICHAR*)Buffer).Get();
			}
		}
		delete [] Buffer;
	}

	return Output;
}

bool FWindowsPlatformProcess::ReadPipeToArray(void* ReadPipe, TArray<uint8> & Output)
{
	uint32 BytesAvailable = 0;
	if (::PeekNamedPipe(ReadPipe, NULL, 0, NULL, (::DWORD*)&BytesAvailable, NULL) && (BytesAvailable > 0))
	{
		Output.SetNumUninitialized(BytesAvailable);
		uint32 BytesRead = 0;
		if (::ReadFile(ReadPipe, Output.GetData(), BytesAvailable, (::DWORD*)&BytesRead, NULL))
		{
			if (BytesRead < BytesAvailable)
			{
				Output.SetNum(BytesRead);
			}

			return true;
		}
		else
		{
			Output.Empty();
		}
	}

	return false;
}

bool FWindowsPlatformProcess::WritePipe(void* WritePipe, const FString& Message, FString* OutWritten)
{
	// If there is not a message or WritePipe is null
	if (Message.Len() == 0 || WritePipe == nullptr)
	{
		return false;
	}

	// Convert input to UTF8CHAR
	uint32 BytesAvailable = Message.Len();
	UTF8CHAR* Buffer = new UTF8CHAR[BytesAvailable + 1];

	if (!FString::ToBlob(Message, Buffer, BytesAvailable))
	{
		return false;
	}

	// Write to pipe
	uint32 BytesWritten = 0;
	bool bIsWritten = !!WriteFile(WritePipe, Buffer, BytesAvailable, (::DWORD*)&BytesWritten, nullptr);

	if (OutWritten)
	{
		OutWritten->FromBlob(Buffer, BytesWritten);
	}

	return bIsWritten;
}

#include "AllowWindowsPlatformTypes.h"

FWindowsPlatformProcess::FWindowsSemaphore::FWindowsSemaphore(const FString & InName, HANDLE InSemaphore)
	:	FSemaphore(InName)
	,	Semaphore(InSemaphore)
{
}

FWindowsPlatformProcess::FWindowsSemaphore::~FWindowsSemaphore()
{
	// actual cleanup should be done in DeleteInterprocessSynchObject() since it can return errors
}

void FWindowsPlatformProcess::FWindowsSemaphore::Lock()
{
	check(Semaphore);
	DWORD WaitResult = WaitForSingleObject(Semaphore, INFINITE);
	if (WaitResult != WAIT_OBJECT_0)
	{
		DWORD ErrNo = GetLastError();
		UE_LOG(LogHAL, Warning, TEXT("WaitForSingleObject(,INFINITE) for semaphore '%s' failed with return code 0x%08x and LastError = %d"),
			GetName(),
			WaitResult,
			ErrNo);
	}
}

bool FWindowsPlatformProcess::FWindowsSemaphore::TryLock(uint64 NanosecondsToWait)
{
	check(Semaphore);
	DWORD MillisecondsToWait = NanosecondsToWait / 1000000ULL;
	DWORD WaitResult = WaitForSingleObject(Semaphore, MillisecondsToWait);
	if (WaitResult != WAIT_OBJECT_0 && WaitResult != WAIT_TIMEOUT)	// timeout is not a warning
	{
		DWORD ErrNo = GetLastError();
		UE_LOG(LogHAL, Warning, TEXT("WaitForSingleObject(,INFINITE) for semaphore '%s' failed with return code 0x%08x and LastError = %d"),
			GetName(),
			WaitResult,
			ErrNo);
	}

	return WaitResult == WAIT_OBJECT_0;
}

void FWindowsPlatformProcess::FWindowsSemaphore::Unlock()
{
	check(Semaphore);
	if (!ReleaseSemaphore(Semaphore, 1, NULL))
	{
		DWORD ErrNo = GetLastError();
		UE_LOG(LogHAL, Warning, TEXT("ReleaseSemaphore(,ReleaseCount=1,) for semaphore '%s' failed with LastError = %d"),
			GetName(),
			ErrNo);
	}
}

FWindowsPlatformProcess::FSemaphore * FWindowsPlatformProcess::NewInterprocessSynchObject(const FString & Name, bool bCreate, uint32 MaxLocks)
{
	HANDLE Semaphore = NULL;
	
	if (bCreate)
	{
		Semaphore = CreateSemaphore(NULL, MaxLocks, MaxLocks, *Name);
		if (NULL == Semaphore)
		{
			DWORD ErrNo = GetLastError();
			UE_LOG(LogHAL, Warning, TEXT("CreateSemaphore(Attrs=NULL, InitialValue=%d, MaxValue=%d, Name='%s') failed with LastError = %d"),
				MaxLocks, MaxLocks,
				*Name,
				ErrNo);
			return NULL;
		}
	}
	else
	{
		DWORD AccessRights = SYNCHRONIZE | SEMAPHORE_MODIFY_STATE;
		Semaphore = OpenSemaphore(AccessRights, false, *Name);
		if (NULL == Semaphore)
		{
			DWORD ErrNo = GetLastError();
			UE_LOG(LogHAL, Warning, TEXT("OpenSemaphore(AccessRights=0x%08x, bInherit=false, Name='%s') failed with LastError = %d"),
				AccessRights,
				*Name,
				ErrNo);
			return NULL;
		}
	}
	check(Semaphore);

	return new FWindowsSemaphore(Name, Semaphore);
}

bool FWindowsPlatformProcess::DeleteInterprocessSynchObject(FSemaphore * Object)
{
	if (NULL == Object)
	{
		return false;
	}

	FWindowsSemaphore * WinSem = static_cast< FWindowsSemaphore * >(Object);
	check( WinSem );

	HANDLE Semaphore = WinSem->GetSemaphore();
	bool bSucceeded = false;
	if (Semaphore)
	{
		bSucceeded = (CloseHandle(Semaphore) == TRUE);
		if (!bSucceeded)
		{
			DWORD ErrNo = GetLastError();
			UE_LOG(LogHAL, Warning, TEXT("CloseHandle() for semaphore '%s' failed with LastError = %d"),
				Object->GetName(),
				ErrNo);
		}
	}

	// delete anyways
	delete WinSem;

	return bSucceeded;
}

bool FWindowsPlatformProcess::Daemonize()
{
	// TODO: implement
	return true;
}

FProcHandle FWindowsPlatformProcess::OpenProcess(uint32 ProcessID)
{
	return FProcHandle(::OpenProcess(PROCESS_ALL_ACCESS, 0, ProcessID));
}

FWindowsPlatformProcess::FProcEnumerator::FProcEnumerator()
{
	SnapshotHandle = ::CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	CurrentEntry.dwSize = 0;
}

FWindowsPlatformProcess::FProcEnumerator::~FProcEnumerator()
{
	::CloseHandle(SnapshotHandle);
}

FWindowsPlatformProcess::FProcEnumInfo::FProcEnumInfo(const PROCESSENTRY32& InInfo)
	: Info(InInfo)
{

}

bool FWindowsPlatformProcess::FProcEnumerator::MoveNext()
{
	if (CurrentEntry.dwSize == 0)
	{
		CurrentEntry.dwSize = sizeof(PROCESSENTRY32);

		return ::Process32First(SnapshotHandle, &CurrentEntry) == TRUE;
	}

	return ::Process32Next(SnapshotHandle, &CurrentEntry) == TRUE;
}

FWindowsPlatformProcess::FProcEnumInfo FWindowsPlatformProcess::FProcEnumerator::GetCurrent() const
{
	return FProcEnumInfo(CurrentEntry);
}

uint32 FWindowsPlatformProcess::FProcEnumInfo::GetPID() const
{
	return Info.th32ProcessID;
}

uint32 FWindowsPlatformProcess::FProcEnumInfo::GetParentPID() const
{
	return Info.th32ParentProcessID;
}

FString FWindowsPlatformProcess::FProcEnumInfo::GetName() const
{
	return Info.szExeFile;
}

FString FWindowsPlatformProcess::FProcEnumInfo::GetFullPath() const
{
	return GetApplicationName(GetPID());
}

#include "HideWindowsPlatformTypes.h"
