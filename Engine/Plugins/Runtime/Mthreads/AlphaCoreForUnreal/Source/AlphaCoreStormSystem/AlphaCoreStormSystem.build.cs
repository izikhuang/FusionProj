// Copyright 2022 Eidos-Montreal / Eidos-Sherbrooke

using System.IO;
using System;
using UnrealBuildTool;

public class AlphaCoreStormSystem : ModuleRules
{
	public AlphaCoreStormSystem(ReadOnlyTargetRules Target) : base(Target)
	{
		//OptimizeCode = CodeOptimization.Never;

		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicIncludePaths.AddRange(
			new string[] {
                // ... add public include paths required here ...
            }
			);

		PrivateIncludePaths.AddRange(
			new string[] {

			}
			);

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"Projects",
				"Slate",
				"SlateCore",
				"UnrealEd",
				"PropertyEditor",
				"EditorWidgets",
				"InputCore",
				"AlphaCoreNative",
				"AlphaCoreRuntime",
                // ... add other public dependencies that you statically link with here ...
            }
			);
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore",
				"PropertyEditor",
				"EditorWidgets",
				"InputCore",
				"EditorStyle",
				"DetailCustomizations",
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