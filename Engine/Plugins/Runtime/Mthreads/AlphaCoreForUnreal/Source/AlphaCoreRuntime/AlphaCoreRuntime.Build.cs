// Copyright Epic Games, Inc. All Rights Reserved.
using System.IO;
using System;
using UnrealBuildTool;

public class AlphaCoreRuntime : ModuleRules
{
	public AlphaCoreRuntime(ReadOnlyTargetRules Target) : base(Target)
	{
		//OptimizeCode = CodeOptimization.Never;

		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		var EngineDir = Path.GetFullPath(Target.RelativeEnginePath);

		PublicIncludePaths.AddRange(
			new string[] {
				//"../../Shaders/Private",
				// ... add public include paths required here ...
			}
			);
		
		PrivateIncludePaths.AddRange(
			new string[] {
				Path.Combine(ModuleDirectory,"Private"),
				Path.Combine(EngineDir, "Source","Runtime","Renderer","Private"),
				// ... add other private include paths required here ...
			}
			);

		PublicAdditionalLibraries.AddRange(
			new string[] {

			}
			);

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"Projects",
				"Engine",
				"AlphaCoreNative",
				// ... add other public dependencies that you statically link with here ...
            }
			);
			
		
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
                "CoreUObject",
				"Engine",
				"RenderCore",
				"Renderer",
				"DeveloperSettings",
				"RHI",
				"AlphaCoreNative",
				// ... add private dependencies that you statically link with here ...	
			}
			);
		
		
		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
				// ... add any modules that your module loads dynamically here ...
			}
			);
	}
}
