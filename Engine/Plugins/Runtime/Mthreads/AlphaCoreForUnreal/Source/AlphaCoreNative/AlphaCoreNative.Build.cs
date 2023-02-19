// Copyright Epic Games, Inc. All Rights Reserved.
using System.IO;
using System;
using UnrealBuildTool;

public class AlphaCoreNative : ModuleRules
{
	public AlphaCoreNative(ReadOnlyTargetRules Target) : base(Target)
	{
		Type = ModuleType.External;
		// PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		var EngineDir = Path.GetFullPath(Target.RelativeEnginePath);

		string CUDAPath = Environment.GetEnvironmentVariable("CUDA_PATH");
		string AlphaCore = Path.Combine(ModuleDirectory, "AlphaCore");
		string AlphaCore3rd = Path.Combine(AlphaCore, "thirdParty");
		string AlphaCoreEngine = Path.Combine(AlphaCore, "external", "AxCoreEngine");

		PublicIncludePaths.Add(ModuleDirectory);
		PublicIncludePaths.AddRange(
			new string[] {
				AlphaCore,
				Path.Combine(AlphaCore,"include"),
				Path.Combine(AlphaCore,"include","AccelTree"),
				Path.Combine(AlphaCore,"include","Collision"),
				Path.Combine(AlphaCore,"include","FluidUtility"),
				Path.Combine(AlphaCore,"include","Geometric"),
				Path.Combine(AlphaCore,"include","GridDense"),
				Path.Combine(AlphaCore,"include","Math"),
				Path.Combine(AlphaCore,"include","MicroSolver"),
				Path.Combine(AlphaCore,"include","Particle"),
				Path.Combine(AlphaCore,"include","PBD"),
				Path.Combine(AlphaCore,"include","ProceduralContent"),
				Path.Combine(AlphaCore,"include","SolidUtility"),
				//Path.Combine(AlphaCore,"include","Test"),
				Path.Combine(AlphaCore,"include","Utility"),
				Path.Combine(AlphaCore,"include","VolumeRender"),
				Path.Combine(AlphaCore, "thirdParty"),
				Path.Combine(AlphaCore3rd,"spdlog","include"),
				Path.Combine(AlphaCoreEngine,"include"),
				Path.Combine(AlphaCoreEngine,"include","Catalyst"),
				Path.Combine(AlphaCoreEngine,"include","Vera"),
				Path.Combine(AlphaCoreEngine,"include","UE4"),
				Path.Combine(CUDAPath, "include"),
				//Path.Combine(ModuleDirectory, "Public"),
				//Path.Combine(ModuleDirectory, "Public", "Render"),
				//Path.Combine(ModuleDirectory, "Public", "Storm"),
				//"../../Shaders/Private"
				// ... add public include paths required here ...
				}
			);

		
		PrivateIncludePaths.AddRange(
			new string[] {
				// ... add other private include paths required here ...
			}
			);

		PublicDefinitions.Add("ALPHA_CUDA=1");
		//PublicDefinitions.Add("ALPHA_UNREAL");
		PublicDefinitions.Add("RENDER_USE_RGBA");
		PublicDefinitions.Add("ALPHA_UNREAL");

		PublicAdditionalLibraries.AddRange(
			new string[] {
				Path.Combine(AlphaCore, "AlphaCoreLib","Release", "AlphaCore.lib"),
				Path.Combine(CUDAPath, "lib","x64","cudart_static.lib")
			}
			);

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				// ... add other public dependencies that you statically link with here ...
            }
			);
			
		
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{

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
