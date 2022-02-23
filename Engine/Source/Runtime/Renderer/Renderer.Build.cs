// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System.IO;

public class Renderer : ModuleRules
{
	public Renderer(ReadOnlyTargetRules Target) : base(Target)
	{
		PrivateIncludePaths.AddRange(
			new string[] {
				"Runtime/Renderer/Private",
				"Runtime/Renderer/Private/CompositionLighting",
				"Runtime/Renderer/Private/PostProcess",
				"../Shaders/Shared",
				"Runtime/Renderer/Private/AlphaCore/include",
				}
			);

		PublicDependencyModuleNames.Add("Core");
        PublicDependencyModuleNames.Add("Engine");
        PublicDependencyModuleNames.Add("MaterialShaderQualitySettings");

        if (Target.bBuildEditor == true)
        {
            PrivateDependencyModuleNames.Add("TargetPlatform");
        }

        // Renderer module builds faster without unity
        // Non-unity also provides faster iteration
		// Not enabled by default as it might harm full rebuild times without XGE
        //bFasterWithoutUnity = true;

        MinFilesUsingPrecompiledHeaderOverride = 4;

		PrivateDependencyModuleNames.AddRange(
			new string[] {
				"CoreUObject", 
				"ApplicationCore",
				"RenderCore", 
				"ImageWriteQueue",
				"RHI"
            }
            );

        PrivateIncludePathModuleNames.AddRange(new string[] { "HeadMountedDisplay" });
        DynamicallyLoadedModuleNames.AddRange(new string[] { "HeadMountedDisplay" });

		var alphacore_inc_dir = "AlphaCore/include";
		var alphacore_lib_dir = "AlphaCore/lib/Release";
		PublicIncludePaths.Add(Path.Combine("Runtime/Renderer/Private", alphacore_inc_dir));
		PublicAdditionalLibraries.Add(Path.Combine("Runtime/Renderer/Private", alphacore_lib_dir, "AlphaCore.lib"));
		Definitions.Add("ALPHA_CUDA");
		var cuda_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6";
		var cuda_include = "include";
		var cuda_lib = "lib/x64";
		PublicIncludePaths.Add(Path.Combine(cuda_path, cuda_include));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "cudart_static.lib"));
	}
}
