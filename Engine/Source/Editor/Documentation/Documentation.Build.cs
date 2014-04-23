// Copyright 1998-2014 Epic Games, Inc. All Rights Reserved.

namespace UnrealBuildTool.Rules
{
	public class Documentation : ModuleRules
	{
		public Documentation(TargetInfo Target)
		{
			PublicIncludePaths.AddRange(
				new string[] {
					// ... add public include paths required here ...
				}
			);

			PrivateIncludePaths.AddRange(
				new string[] {
					"Developer/Documentation/Private",
					// ... add other private include paths required here ...
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
                    "CoreUObject",
                    "Engine",
                    "InputCore",
                    "Slate",
                    "EditorStyle",
                    "UnrealEd",
					"Analytics",
					"SourceCodeAccess"
				}
			);

			DynamicallyLoadedModuleNames.AddRange(
				new string[]
				{
                    "MessageLog"
				}
			);
		}
	}
}