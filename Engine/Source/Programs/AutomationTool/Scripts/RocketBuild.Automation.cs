﻿// Copyright 1998-2015 Epic Games, Inc. All Rights Reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using AutomationTool;
using UnrealBuildTool;

namespace Rocket
{
	public class RocketBuild : GUBP.GUBPNodeAdder
	{
		static readonly string[] CurrentTemplates = 
		{
			"TP_FirstPerson",
			"TP_FirstPersonBP",
			"TP_Flying",
			"TP_FlyingBP",
			"TP_Rolling",
			"TP_RollingBP",
			"TP_SideScroller",
			"TP_SideScrollerBP",
			"TP_ThirdPerson",
			"TP_ThirdPersonBP",
			"TP_TopDown",
			"TP_TopDownBP",
			"TP_TwinStick",
			"TP_TwinStickBP",
			"TP_Vehicle",
			"TP_VehicleBP",
			"TP_Puzzle",
			"TP_PuzzleBP",
			"TP_2DSideScroller",
			"TP_2DSideScrollerBP",
			"TP_VehicleAdv",
			"TP_VehicleAdvBP",
		};

		static readonly string[] CurrentFeaturePacks = 
		{
			"FP_FirstPerson",
			"FP_FirstPersonBP",
			"TP_Flying",
			"TP_FlyingBP",
			"TP_Rolling",
			"TP_RollingBP",
			"TP_SideScroller",
			"TP_SideScrollerBP",
			"TP_ThirdPerson",
			"TP_ThirdPersonBP",
			"TP_TopDown",
			"TP_TopDownBP",
			"TP_TwinStick",
			"TP_TwinStickBP",
			"TP_Vehicle",
			"TP_VehicleBP",
			"TP_Puzzle",
			"TP_PuzzleBP",
			"TP_2DSideScroller",
			"TP_2DSideScrollerBP",
			"TP_VehicleAdv",
			"TP_VehicleAdvBP",
			"StarterContent",
			"MobileStarterContent"
		};

		public RocketBuild()
		{
		}

		public override void AddNodes(GUBP bp, UnrealTargetPlatform HostPlatform)
		{
			if(!bp.BranchOptions.bNoInstalledEngine)
			{
				// Find all the target platforms for this host platform.
				List<UnrealTargetPlatform> TargetPlatforms = GetTargetPlatforms(bp, HostPlatform);

				// Get a list of all the code target platforms
				List<UnrealTargetPlatform> CodeTargetPlatforms = new List<UnrealTargetPlatform>();
				foreach(UnrealTargetPlatform TargetPlatform in TargetPlatforms)
				{
					if(IsCodeTargetPlatform(HostPlatform, TargetPlatform))
					{
						CodeTargetPlatforms.Add(TargetPlatform);
					}
				}

				// Build the DDC
				bp.AddNode(new BuildDerivedDataCacheNode(HostPlatform, GetCookPlatforms(HostPlatform, TargetPlatforms), CurrentFeaturePacks));

				// Strip symbols for all the targets built on the host platform
				bp.AddNode(new StripRocketNode(bp, HostPlatform, TargetPlatforms, CodeTargetPlatforms));

				// Generate a list of files that needs to be copied for each target platform
				bp.AddNode(new FilterRocketNode(bp, HostPlatform, TargetPlatforms, CodeTargetPlatforms, CurrentFeaturePacks, CurrentTemplates));

				// Copy the install to the output directory. This node adds a dependency on triggering a shared promotable if it's copying to the network.
				if(ShouldDoSeriousThingsLikeP4CheckinAndPostToMCP())
				{
					string OutputDir = CommandUtils.CombinePaths(CommandUtils.RootSharedTempStorageDirectory(), "Rocket", "Automated", GetBuildLabel(), CommandUtils.GetGenericPlatformName(HostPlatform));
					bp.AddNode(new CopyRocketNode(HostPlatform, CodeTargetPlatforms, OutputDir, true));
					bp.AddNode(new CopyRocketSymbolsNode(bp, HostPlatform, CodeTargetPlatforms, OutputDir + "Symbols", true));
				}
				else
				{
					string OutputDir = CommandUtils.CombinePaths(CommandUtils.CmdEnv.LocalRoot, "LocalBuilds", "Rocket", CommandUtils.GetGenericPlatformName(HostPlatform));
					bp.AddNode(new CopyRocketNode(HostPlatform, CodeTargetPlatforms, OutputDir, false));
					bp.AddNode(new CopyRocketSymbolsNode(bp, HostPlatform, CodeTargetPlatforms, OutputDir + "Symbols", false));
				}

				// Add the aggregate node for the entire install
				GUBP.GUBPNode PromotableNode = bp.FindNode(GUBP.SharedAggregatePromotableNode.StaticGetFullName());
				PromotableNode.AddDependency(FilterRocketNode.StaticGetFullName(HostPlatform));
				PromotableNode.AddDependency(BuildDerivedDataCacheNode.StaticGetFullName(HostPlatform));

				// Add a node for GitHub promotions
				if(HostPlatform == UnrealTargetPlatform.Win64)
				{
					string GitConfigRelativePath = "Engine/Build/Git/UnrealBot.ini";
					if(CommandUtils.FileExists(CommandUtils.CombinePaths(CommandUtils.CmdEnv.LocalRoot, GitConfigRelativePath)))
					{
						bp.AddNode(new BuildGitPromotable(bp, HostPlatform, GitConfigRelativePath));
						PromotableNode.AddDependency(BuildGitPromotable.StaticGetFullName(HostPlatform));
					}
				}
			}
		}

		public static string GetBuildLabel()
		{
			return FEngineVersionSupport.FromVersionFile(CommandUtils.CombinePaths(CommandUtils.CmdEnv.LocalRoot, @"Engine\Source\Runtime\Launch\Resources\Version.h")).ToString();
		}

		public static List<UnrealTargetPlatform> GetTargetPlatforms(GUBP bp, UnrealTargetPlatform HostPlatform)
		{
			List<UnrealTargetPlatform> TargetPlatforms = new List<UnrealTargetPlatform>();
			if(!bp.ParseParam("NoTargetPlatforms"))
			{
				// Always support the host platform
				TargetPlatforms.Add(HostPlatform);

				// Add other target platforms for each host platform
				if(HostPlatform == UnrealTargetPlatform.Win64)
				{
					TargetPlatforms.Add(UnrealTargetPlatform.Win32);
				}
				if(HostPlatform == UnrealTargetPlatform.Win64 || HostPlatform == UnrealTargetPlatform.Mac)
				{
					TargetPlatforms.Add(UnrealTargetPlatform.Android);
				}
				if(HostPlatform == UnrealTargetPlatform.Win64 || HostPlatform == UnrealTargetPlatform.Mac)
				{
					TargetPlatforms.Add(UnrealTargetPlatform.IOS);
				}
				if(HostPlatform == UnrealTargetPlatform.Win64)
				{
					TargetPlatforms.Add(UnrealTargetPlatform.Linux);
				}
				if(HostPlatform == UnrealTargetPlatform.Win64 || HostPlatform == UnrealTargetPlatform.Mac )
				{
					TargetPlatforms.Add(UnrealTargetPlatform.HTML5);
				}

				// Remove any platforms that aren't available on this machine
				TargetPlatforms.RemoveAll(x => !bp.ActivePlatforms.Contains(x));

				// Remove any platforms that aren't enabled on the command line
				string TargetPlatformFilter = bp.ParseParamValue("TargetPlatforms", null);
				if(TargetPlatformFilter != null)
				{
					List<UnrealTargetPlatform> NewTargetPlatforms = new List<UnrealTargetPlatform>();
					foreach (string TargetPlatformName in TargetPlatformFilter.Split(new char[]{ '+' }, StringSplitOptions.RemoveEmptyEntries))
					{
						UnrealTargetPlatform TargetPlatform;
						if(!Enum.TryParse(TargetPlatformName, out TargetPlatform))
						{
							throw new AutomationException("Unknown target platform '{0}' specified on command line");
						}
						else if(TargetPlatforms.Contains(TargetPlatform))
						{
							NewTargetPlatforms.Add(TargetPlatform);
						}
					}
					TargetPlatforms = NewTargetPlatforms;
				}
			}
			return TargetPlatforms;
		}

		public static string GetCookPlatforms(UnrealTargetPlatform HostPlatform, IEnumerable<UnrealTargetPlatform> TargetPlatforms)
		{
			// Always include the editor platform for cooking
			List<string> CookPlatforms = new List<string>();
			CookPlatforms.Add(Platform.Platforms[HostPlatform].GetEditorCookPlatform());

			// Add all the target platforms
			foreach(UnrealTargetPlatform TargetPlatform in TargetPlatforms)
			{
				if(TargetPlatform == UnrealTargetPlatform.Android)
				{
					CookPlatforms.Add(Platform.Platforms[TargetPlatform].GetCookPlatform(false, false, "ATC"));
				}
				else
				{
					CookPlatforms.Add(Platform.Platforms[TargetPlatform].GetCookPlatform(false, false, ""));
				}
			}
			return CommandUtils.CombineCommandletParams(CookPlatforms.Distinct().ToArray());
		}

		public static bool ShouldDoSeriousThingsLikeP4CheckinAndPostToMCP()
		{
			return CommandUtils.P4Enabled && CommandUtils.AllowSubmit && !GUBP.bPreflightBuild; // we don't do serious things in a preflight
		}

		public static UnrealTargetPlatform GetSourceHostPlatform(GUBP bp, UnrealTargetPlatform HostPlatform, UnrealTargetPlatform TargetPlatform)
		{
            if (TargetPlatform == UnrealTargetPlatform.HTML5 && HostPlatform == UnrealTargetPlatform.Mac && bp.HostPlatforms.Contains(UnrealTargetPlatform.Win64))
            {
                return UnrealTargetPlatform.Win64;
            }
			if (TargetPlatform == UnrealTargetPlatform.Android && HostPlatform == UnrealTargetPlatform.Mac && bp.HostPlatforms.Contains(UnrealTargetPlatform.Win64))
			{
				return UnrealTargetPlatform.Win64;
			}
			if(TargetPlatform == UnrealTargetPlatform.IOS && HostPlatform == UnrealTargetPlatform.Win64 && bp.HostPlatforms.Contains(UnrealTargetPlatform.Mac))
			{
				return UnrealTargetPlatform.Mac;
			}
			return HostPlatform;
		}

		public static bool IsCodeTargetPlatform(UnrealTargetPlatform HostPlatform, UnrealTargetPlatform TargetPlatform)
		{
			if(TargetPlatform == UnrealTargetPlatform.Linux)
			{
				return false;
			}
			if(HostPlatform == UnrealTargetPlatform.Win64 && TargetPlatform == UnrealTargetPlatform.IOS)
			{
				return false;
			}
			return true;
		}
	}

	public class BuildGitPromotable : GUBP.HostPlatformNode
	{
		string ConfigRelativePath;

		public BuildGitPromotable(GUBP bp, UnrealTargetPlatform HostPlatform, string InConfigRelativePath) : base(HostPlatform)
		{
			ConfigRelativePath = InConfigRelativePath;

			foreach(UnrealTargetPlatform OtherHostPlatform in bp.HostPlatforms)
			{
				AddDependency(GUBP.RootEditorNode.StaticGetFullName(OtherHostPlatform));
				AddDependency(GUBP.ToolsNode.StaticGetFullName(OtherHostPlatform));
				AddDependency(GUBP.InternalToolsNode.StaticGetFullName(OtherHostPlatform));
			}
		}

		public static string StaticGetFullName(UnrealTargetPlatform HostPlatform)
		{
			return "BuildGitPromotable" + StaticGetHostPlatformSuffix(HostPlatform);
		}

		public override string GetFullName()
		{
			return StaticGetFullName(HostPlatform);
		}

		public override void DoBuild(GUBP bp)
		{
			// Create a filter for all the promoted binaries
			FileFilter PromotableFilter = new FileFilter();
			PromotableFilter.AddRuleForFiles(AllDependencyBuildProducts, CommandUtils.CmdEnv.LocalRoot, FileFilterType.Include);
			PromotableFilter.ReadRulesFromFile(CommandUtils.CombinePaths(CommandUtils.CmdEnv.LocalRoot, ConfigRelativePath), "promotable");
			PromotableFilter.ExcludeConfidentialFolders();
			
			// Copy everything that matches the filter to the promotion folder
			string PromotableFolder = CommandUtils.CombinePaths(CommandUtils.CmdEnv.LocalRoot, "Engine", "Saved", "GitPromotable");
			CommandUtils.DeleteDirectoryContents(PromotableFolder);
			string[] PromotableFiles = CommandUtils.ThreadedCopyFiles(CommandUtils.CmdEnv.LocalRoot, PromotableFolder, PromotableFilter, bIgnoreSymlinks: true);
			BuildProducts = new List<string>(PromotableFiles);
		}
	}
	
	public class StripRocketNode : GUBP.HostPlatformNode
	{
		List<UnrealTargetPlatform> Platforms = new List<UnrealTargetPlatform>();
		public readonly string StrippedDir;

		public StripRocketNode(GUBP bp, UnrealTargetPlatform InHostPlatform, IEnumerable<UnrealTargetPlatform> InTargetPlatforms, IEnumerable<UnrealTargetPlatform> InCodeTargetPlatforms) : base(InHostPlatform)
		{
			Platforms = new List<UnrealTargetPlatform>(InTargetPlatforms);
			if(!Platforms.Contains(HostPlatform))
			{
				Platforms.Add(HostPlatform);
			}
			StrippedDir = Path.GetFullPath(CommandUtils.CombinePaths(CommandUtils.CmdEnv.LocalRoot, "Engine", "Saved", "Rocket", HostPlatform.ToString()));

			// Add all the host nodes
			AddDependency(GUBP.ToolsForCompileNode.StaticGetFullName(HostPlatform));
			AddDependency(GUBP.RootEditorNode.StaticGetFullName(HostPlatform));
			AddDependency(GUBP.ToolsNode.StaticGetFullName(HostPlatform));

			// Add all the target platform nodes
			foreach(UnrealTargetPlatform TargetPlatform in InTargetPlatforms)
			{
				UnrealTargetPlatform SourceHostPlatform = RocketBuild.GetSourceHostPlatform(bp, HostPlatform, TargetPlatform);
				if(SourceHostPlatform == HostPlatform)
				{
					AddDependency(GUBP.GamePlatformMonolithicsNode.StaticGetFullName(HostPlatform, bp.Branch.BaseEngineProject, TargetPlatform, Precompiled: InCodeTargetPlatforms.Contains(TargetPlatform)));
				}
			}
		}

		public static string StaticGetFullName(UnrealTargetPlatform HostPlatform)
		{
			return "StripRocket" + StaticGetHostPlatformSuffix(HostPlatform);
		}

		public override string GetFullName()
		{
			return StaticGetFullName(HostPlatform);
		}

		public override void DoBuild(GUBP bp)
		{
			BuildProducts = new List<string>();

			string InputDir = Path.GetFullPath(CommandUtils.CmdEnv.LocalRoot);

			string RulesFileName = CommandUtils.CombinePaths(CommandUtils.CmdEnv.LocalRoot, "Engine", "Build", "InstalledEngineFilters.ini");
			foreach(UnrealTargetPlatform Platform in Platforms)
			{
				// Read the filter for files on this platform
				FileFilter StripFilter = new FileFilter();
				StripFilter.ReadRulesFromFile(RulesFileName, "StripSymbols." + Platform.ToString(), HostPlatform.ToString());

				// Apply the filter to the build products
				List<string> SourcePaths = new List<string>();
				List<string> TargetPaths = new List<string>();
				foreach(string DependencyBuildProduct in AllDependencyBuildProducts)
				{
					string RelativePath = CommandUtils.StripBaseDirectory(Path.GetFullPath(DependencyBuildProduct), InputDir);
					if(StripFilter.Matches(RelativePath))
					{
						SourcePaths.Add(CommandUtils.CombinePaths(InputDir, RelativePath));
						TargetPaths.Add(CommandUtils.CombinePaths(StrippedDir, RelativePath));
					}
				}

				// Strip the files and add them to the build products
				StripSymbols(Platform, SourcePaths.ToArray(), TargetPaths.ToArray());
				BuildProducts.AddRange(TargetPaths);
			}

			SaveRecordOfSuccessAndAddToBuildProducts();
		}

		public static void StripSymbols(UnrealTargetPlatform TargetPlatform, string[] SourceFileNames, string[] TargetFileNames)
		{
			if(TargetPlatform == UnrealTargetPlatform.Win32 || TargetPlatform == UnrealTargetPlatform.Win64)
			{
				string PDBCopyPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "MSBuild", "Microsoft", "VisualStudio", "v12.0", "AppxPackage", "PDBCopy.exe");
				for(int Idx = 0; Idx < SourceFileNames.Length; Idx++)
				{
					CommandUtils.CreateDirectory(Path.GetDirectoryName(TargetFileNames[Idx]));
					CommandUtils.Run(PDBCopyPath, String.Format("{0} {1} -p", CommandUtils.MakePathSafeToUseWithCommandLine(SourceFileNames[Idx]), CommandUtils.MakePathSafeToUseWithCommandLine(TargetFileNames[Idx])));
				}
			}
			else
			{
				for(int Idx = 0; Idx < SourceFileNames.Length; Idx++)
				{
					CommandUtils.LogWarning("Stripping has not yet been implemented for {0}. Skipping strip for {1}.", TargetPlatform, SourceFileNames[Idx]);
					CommandUtils.CopyFile(SourceFileNames[Idx], TargetFileNames[Idx]);
				}
			}
		}
	}

	public class FilterRocketNode : GUBP.HostPlatformNode
	{
		List<UnrealTargetPlatform> SourceHostPlatforms;
		List<UnrealTargetPlatform> TargetPlatforms;
		List<UnrealTargetPlatform> CodeTargetPlatforms;
		string[] CurrentFeaturePacks;
		string[] CurrentTemplates;
		public readonly string DepotManifestPath;
		public readonly string StrippedManifestPath;

		public FilterRocketNode(GUBP bp, UnrealTargetPlatform InHostPlatform, List<UnrealTargetPlatform> InTargetPlatforms, List<UnrealTargetPlatform> InCodeTargetPlatforms, string[] InCurrentFeaturePacks, string[] InCurrentTemplates)
			: base(InHostPlatform)
		{
			TargetPlatforms = new List<UnrealTargetPlatform>(InTargetPlatforms);
			CodeTargetPlatforms = new List<UnrealTargetPlatform>(InCodeTargetPlatforms);
			CurrentFeaturePacks = InCurrentFeaturePacks;
			CurrentTemplates = InCurrentTemplates;
			DepotManifestPath = CommandUtils.CombinePaths(CommandUtils.CmdEnv.LocalRoot, "Engine", "Saved", "Rocket", HostPlatform.ToString(), "DistillDepot.txt");
			StrippedManifestPath = CommandUtils.CombinePaths(CommandUtils.CmdEnv.LocalRoot, "Engine", "Saved", "Rocket", HostPlatform.ToString(), "DistillStripped.txt");

			// Add the editor
			AddDependency(GUBP.VersionFilesNode.StaticGetFullName());
			AddDependency(GUBP.ToolsForCompileNode.StaticGetFullName(HostPlatform));
			AddDependency(GUBP.RootEditorNode.StaticGetFullName(HostPlatform));
			AddDependency(GUBP.ToolsNode.StaticGetFullName(HostPlatform));

			// Get all the external file lists. Android/HTML5 on Mac uses the same files as Win64.
			foreach(UnrealTargetPlatform CodeTargetPlatform in CodeTargetPlatforms)
			{
				UnrealTargetPlatform SourceHostPlatform = RocketBuild.GetSourceHostPlatform(bp, HostPlatform, CodeTargetPlatform);
				AddDependency(GUBP.GamePlatformMonolithicsNode.StaticGetFullName(SourceHostPlatform, bp.Branch.BaseEngineProject, CodeTargetPlatform, Precompiled: true));
			}

			// Add win64 tools on Mac, to get the win64 build of UBT, UAT and IPP
			if (HostPlatform == UnrealTargetPlatform.Mac && bp.HostPlatforms.Contains(UnrealTargetPlatform.Win64))
			{
				AddDependency(GUBP.ToolsNode.StaticGetFullName(UnrealTargetPlatform.Win64));
				AddDependency(GUBP.ToolsForCompileNode.StaticGetFullName(UnrealTargetPlatform.Win64));
			}

			// Add all the feature packs
			AddDependency(GUBP.MakeFeaturePacksNode.StaticGetFullName(GUBP.MakeFeaturePacksNode.GetDefaultBuildPlatform(bp)));

			// Find all the host platforms we need
			SourceHostPlatforms = TargetPlatforms.Select(x => RocketBuild.GetSourceHostPlatform(bp, HostPlatform, x)).Distinct().ToList();
			if(!SourceHostPlatforms.Contains(HostPlatform))
			{
				SourceHostPlatforms.Add(HostPlatform);
			}

			// Add the stripped target platforms
			foreach(UnrealTargetPlatform SourceHostPlatform in SourceHostPlatforms)
			{
				AddDependency(StripRocketNode.StaticGetFullName(SourceHostPlatform));
			}
		}

		public override int CISFrequencyQuantumShift(GUBP bp)
		{
			return base.CISFrequencyQuantumShift(bp) + 2;
		}

		public static string StaticGetFullName(UnrealTargetPlatform InHostPlatform)
		{
			return "FilterRocket" + StaticGetHostPlatformSuffix(InHostPlatform);
		}

		public override string GetFullName()
		{
			return StaticGetFullName(HostPlatform);
		}

		public override void DoBuild(GUBP bp)
		{
			FileFilter Filter = new FileFilter();

			// Include all the editor products
			AddRuleForBuildProducts(Filter, bp, GUBP.ToolsForCompileNode.StaticGetFullName(HostPlatform), FileFilterType.Include);
			AddRuleForBuildProducts(Filter, bp, GUBP.RootEditorNode.StaticGetFullName(HostPlatform), FileFilterType.Include);
			AddRuleForBuildProducts(Filter, bp, GUBP.ToolsNode.StaticGetFullName(HostPlatform), FileFilterType.Include);

			// Include the editor headers
			UnzipAndAddRuleForHeaders(GUBP.RootEditorNode.StaticGetFullName(HostPlatform), Filter, FileFilterType.Include);

			// Include the build dependencies for every code platform
			foreach(UnrealTargetPlatform CodeTargetPlatform in CodeTargetPlatforms)
			{
				UnrealTargetPlatform SourceHostPlatform = RocketBuild.GetSourceHostPlatform(bp, HostPlatform, CodeTargetPlatform);
				string FileListPath = GUBP.GamePlatformMonolithicsNode.StaticGetBuildDependenciesPath(SourceHostPlatform, bp.Branch.BaseEngineProject, CodeTargetPlatform);
				Filter.AddRuleForFiles(UnrealBuildTool.Utils.ReadClass<UnrealBuildTool.ExternalFileList>(FileListPath).FileNames, CommandUtils.CmdEnv.LocalRoot, FileFilterType.Include);
				UnzipAndAddRuleForHeaders(GUBP.GamePlatformMonolithicsNode.StaticGetArchivedHeadersPath(HostPlatform, bp.Branch.BaseEngineProject, CodeTargetPlatform), Filter, FileFilterType.Include);
			}

			// Add the monolithic binaries
			foreach(UnrealTargetPlatform CodeTargetPlatform in CodeTargetPlatforms)
			{
				UnrealTargetPlatform SourceHostPlatform = RocketBuild.GetSourceHostPlatform(bp, HostPlatform, CodeTargetPlatform);
				AddRuleForBuildProducts(Filter, bp, GUBP.GamePlatformMonolithicsNode.StaticGetFullName(SourceHostPlatform, bp.Branch.BaseEngineProject, CodeTargetPlatform, Precompiled: true), FileFilterType.Include);
			}

			// Include the feature packs
			foreach(string CurrentFeaturePack in CurrentFeaturePacks)
			{
				BranchInfo.BranchUProject Project = bp.Branch.FindGameChecked(CurrentFeaturePack);
				Filter.AddRuleForFile(GUBP.MakeFeaturePacksNode.GetOutputFile(Project), CommandUtils.CmdEnv.LocalRoot, FileFilterType.Include);
			}

			// Include all the templates
			foreach (string Template in CurrentTemplates)
			{
				BranchInfo.BranchUProject Project = bp.Branch.FindGameChecked(Template);
				string TemplateRelativeDirectory = "/" + Utils.StripBaseDirectory(Path.GetDirectoryName(Project.FilePath), CommandUtils.CmdEnv.LocalRoot) + "/";
				Filter.Include(TemplateRelativeDirectory + "/...");
				Filter.Exclude(TemplateRelativeDirectory + "/.../Binaries/...");
				Filter.Exclude(TemplateRelativeDirectory + "/.../Intermediate/...");
			}

			// Include all the standard rules
			string RulesFileName = CommandUtils.CombinePaths(CommandUtils.CmdEnv.LocalRoot, "Engine", "Build", "InstalledEngineFilters.ini");
			Filter.ReadRulesFromFile(RulesFileName, "CopyEditor", HostPlatform.ToString());
			Filter.ReadRulesFromFile(RulesFileName, "CopyTargetPlatforms", HostPlatform.ToString());

			// Custom rules for each target platform
			foreach(UnrealTargetPlatform TargetPlaform in TargetPlatforms)
			{
				string SectionName = String.Format("CopyTargetPlatform.{0}", TargetPlaform.ToString());
				Filter.ReadRulesFromFile(RulesFileName, SectionName, HostPlatform.ToString());
			}

			// Add the final exclusions for legal reasons.
			Filter.ExcludeConfidentialPlatforms();
			Filter.ExcludeConfidentialFolders();

			// Run the filter on the stripped symbols, and remove those files from the copy filter
			List<string> StrippedFiles = new List<string>();
			foreach(UnrealTargetPlatform SourceHostPlatform in SourceHostPlatforms)
			{
				StripRocketNode StripNode = (StripRocketNode)bp.FindNode(StripRocketNode.StaticGetFullName(SourceHostPlatform));
				foreach(string BuildProduct in StripNode.BuildProducts)
				{
					if(Utils.IsFileUnderDirectory(BuildProduct, StripNode.StrippedDir))
					{
						string RelativePath = CommandUtils.StripBaseDirectory(Path.GetFullPath(BuildProduct), StripNode.StrippedDir);
						if(Filter.Matches(RelativePath))
						{
							StrippedFiles.Add(RelativePath);
							Filter.Exclude("/" + RelativePath);
						}
					}
				}
			}
			WriteManifest(StrippedManifestPath, StrippedFiles);

			// Write the filtered list of depot files to disk, removing any symlinks
			List<string> DepotFiles = Filter.ApplyToDirectory(CommandUtils.CmdEnv.LocalRoot, true).ToList();
			WriteManifest(DepotManifestPath, DepotFiles);

			// Sort the list of output files
			SortedDictionary<string, bool> SortedFiles = new SortedDictionary<string,bool>(StringComparer.InvariantCultureIgnoreCase);
			foreach(string DepotFile in DepotFiles)
			{
				SortedFiles.Add(DepotFile, false);
			}
			foreach(string StrippedFile in StrippedFiles)
			{
				SortedFiles.Add(StrippedFile, true);
			}

			// Write the list to the log
			CommandUtils.Log("Files to be included in Rocket build:");
			foreach(KeyValuePair<string, bool> SortedFile in SortedFiles)
			{
				CommandUtils.Log("  {0}{1}", SortedFile.Key, SortedFile.Value? " (stripped)" : "");
			}

			// Add the build products
			BuildProducts = new List<string>{ StrippedManifestPath, DepotManifestPath };
		}

		static void AddRuleForBuildProducts(FileFilter Filter, GUBP bp, string NodeName, FileFilterType Type)
		{
			GUBP.GUBPNode Node = bp.FindNode(NodeName);
			if(Node == null)
			{
				throw new AutomationException("Couldn't find node '{0}'", NodeName);
			}
			Filter.AddRuleForFiles(Node.BuildProducts, CommandUtils.CmdEnv.LocalRoot, Type);
		}

		static void UnzipAndAddRuleForHeaders(string ZipFileName, FileFilter Filter, FileFilterType Type)
		{
			IEnumerable<string> FileNames = CommandUtils.UnzipFiles(ZipFileName, CommandUtils.CmdEnv.LocalRoot);
			Filter.AddRuleForFiles(FileNames, CommandUtils.CmdEnv.LocalRoot, FileFilterType.Include);
		}

		public static void WriteManifest(string FileName, IEnumerable<string> Files)
		{
			CommandUtils.CreateDirectory(Path.GetDirectoryName(FileName));
			CommandUtils.WriteAllLines(FileName, Files.ToArray());
		}
	}

	public class CopyRocketNode : GUBP.HostPlatformNode
	{
		public readonly string OutputDir;
		public List<UnrealTargetPlatform> CodeTargetPlatforms;

		public CopyRocketNode(UnrealTargetPlatform HostPlatform, List<UnrealTargetPlatform> InCodeTargetPlatforms, string InOutputDir, bool bWaitForPromotion) : base(HostPlatform)
		{
			OutputDir = InOutputDir;
			CodeTargetPlatforms = new List<UnrealTargetPlatform>(InCodeTargetPlatforms);

			// If this is a CIS build, we need to wait for the promotable trigger before running. If not, make an agent sharing group so that the samples node can use our build products.
			if(bWaitForPromotion)
			{
				AddDependency(GUBP.WaitForSharedPromotionUserInput.StaticGetFullName(false));
			}
			else
			{
				AgentSharingGroup = "RocketGroup" + StaticGetHostPlatformSuffix(HostPlatform);
			}

			AddDependency(FilterRocketNode.StaticGetFullName(HostPlatform));
			AddDependency(StripRocketNode.StaticGetFullName(HostPlatform));
			AddDependency(BuildDerivedDataCacheNode.StaticGetFullName(HostPlatform));
		}

		public static string StaticGetFullName(UnrealTargetPlatform HostPlatform)
		{
			return "CopyRocket" + StaticGetHostPlatformSuffix(HostPlatform);
		}

		public override string GetFullName()
		{
			return StaticGetFullName(HostPlatform);
		}

		public override void DoBuild(GUBP bp)
		{
			CommandUtils.DeleteDirectoryContents(OutputDir);

			// Extract the editor headers
			CommandUtils.UnzipFiles(GUBP.RootEditorNode.StaticGetFullName(HostPlatform), CommandUtils.CmdEnv.LocalRoot);

			// Extract all the headers for code target platforms
			foreach(UnrealTargetPlatform CodeTargetPlatform in CodeTargetPlatforms)
			{
				UnrealTargetPlatform SourceHostPlatform = RocketBuild.GetSourceHostPlatform(bp, HostPlatform, CodeTargetPlatform);
				string ZipFileName = GUBP.GamePlatformMonolithicsNode.StaticGetArchivedHeadersPath(SourceHostPlatform, bp.Branch.BaseEngineProject, CodeTargetPlatform);
				CommandUtils.UnzipFiles(ZipFileName, CommandUtils.CmdEnv.LocalRoot);
			}

			// Copy the depot files to the output directory
			FilterRocketNode FilterNode = (FilterRocketNode)bp.FindNode(FilterRocketNode.StaticGetFullName(HostPlatform));
			CopyManifestFilesToOutput(FilterNode.DepotManifestPath, CommandUtils.CmdEnv.LocalRoot, OutputDir);

			// Copy the stripped symbols to the output directory
			StripRocketNode StripNode = (StripRocketNode)bp.FindNode(StripRocketNode.StaticGetFullName(HostPlatform));
			CopyManifestFilesToOutput(FilterNode.StrippedManifestPath, StripNode.StrippedDir, OutputDir);

			// Copy the DDC to the output directory
			BuildDerivedDataCacheNode DerivedDataCacheNode = (BuildDerivedDataCacheNode)bp.FindNode(BuildDerivedDataCacheNode.StaticGetFullName(HostPlatform));
			CopyManifestFilesToOutput(DerivedDataCacheNode.SavedManifestPath, DerivedDataCacheNode.SavedDir, OutputDir);

			// Write the UE4CommandLine.txt file with the 
			string CommandLineFile = CommandUtils.CombinePaths(OutputDir, "UE4CommandLine.txt");
			CommandUtils.WriteAllText(CommandLineFile, "-installedengine -rocket");

			// Create a dummy build product
			BuildProducts = new List<string>();
			SaveRecordOfSuccessAndAddToBuildProducts();
		}

		static void CopyManifestFilesToOutput(string ManifestPath, string InputDir, string OutputDir)
		{
			// Read the files from the manifest
			CommandUtils.Log("Reading manifest: '{0}'", ManifestPath);
			string[] Files = CommandUtils.ReadAllLines(ManifestPath).Select(x => x.Trim()).Where(x => x.Length > 0).ToArray();

			// Create lists of source and target files
			CommandUtils.Log("Preparing file lists...");
			string[] SourceFiles = Files.Select(x => CommandUtils.CombinePaths(InputDir, x)).ToArray();
			string[] TargetFiles = Files.Select(x => CommandUtils.CombinePaths(OutputDir, x)).ToArray();

			// Copy everything
			CommandUtils.ThreadedCopyFiles(SourceFiles, TargetFiles);
		}
	}

	public class CopyRocketSymbolsNode : GUBP.HostPlatformNode
	{
		string SymbolsOutputDir;

		public CopyRocketSymbolsNode(GUBP bp, UnrealTargetPlatform HostPlatform, IEnumerable<UnrealTargetPlatform> TargetPlatforms, string InSymbolsOutputDir, bool bWaitForPromotion) : base(HostPlatform)
		{
			SymbolsOutputDir = InSymbolsOutputDir;

			if(bWaitForPromotion)
			{
				AddDependency(GUBP.WaitForSharedPromotionUserInput.StaticGetFullName(false));
			}
			else
			{
				AgentSharingGroup = "RocketGroup" + StaticGetHostPlatformSuffix(HostPlatform);
			}

			AddDependency(GUBP.ToolsForCompileNode.StaticGetFullName(HostPlatform));
			AddDependency(GUBP.RootEditorNode.StaticGetFullName(HostPlatform));
			AddDependency(GUBP.ToolsNode.StaticGetFullName(HostPlatform));

			foreach(UnrealTargetPlatform TargetPlatform in TargetPlatforms)
			{
				UnrealTargetPlatform SourceHostPlatform = RocketBuild.GetSourceHostPlatform(bp, HostPlatform, TargetPlatform);
				AddDependency(GUBP.GamePlatformMonolithicsNode.StaticGetFullName(SourceHostPlatform, bp.Branch.BaseEngineProject, TargetPlatform, Precompiled: true));
			}
		}

		public static string StaticGetFullName(UnrealTargetPlatform HostPlatform)
		{
			return "CopyRocketSymbols" + StaticGetHostPlatformSuffix(HostPlatform);
		}

		public override string GetFullName()
		{
			return StaticGetFullName(HostPlatform);
		}

		public override void DoBuild(GUBP bp)
		{
			// Make a lookup for all the known debug extensions, and filter all the dependency build products against that
			HashSet<string> DebugExtensions = new HashSet<string>(Platform.Platforms.Values.SelectMany(x => x.GetDebugFileExtentions()).Distinct().ToArray(), StringComparer.InvariantCultureIgnoreCase);
			foreach(string InputFileName in AllDependencyBuildProducts)
			{
				string Extension = Path.GetExtension(InputFileName);
				if(DebugExtensions.Contains(Extension))
				{
					string OutputFileName = CommandUtils.MakeRerootedFilePath(InputFileName, CommandUtils.CmdEnv.LocalRoot, SymbolsOutputDir);
					CommandUtils.CopyFile(InputFileName, OutputFileName);
				}
			}

			// Add a dummy build product
			BuildProducts = new List<string>();
			SaveRecordOfSuccessAndAddToBuildProducts();
		}
	}

	public class BuildDerivedDataCacheNode : GUBP.HostPlatformNode
	{
		string TargetPlatforms;
		string[] ProjectNames;
		public readonly string SavedDir;
		public readonly string SavedManifestPath;

		public BuildDerivedDataCacheNode(UnrealTargetPlatform InHostPlatform, string InTargetPlatforms, string[] InProjectNames)
			: base(InHostPlatform)
		{
			TargetPlatforms = InTargetPlatforms;
			ProjectNames = InProjectNames;
			SavedDir = CommandUtils.CombinePaths(CommandUtils.CmdEnv.LocalRoot, "Engine", "Saved", "Rocket", HostPlatform.ToString());
			SavedManifestPath = CommandUtils.CombinePaths(SavedDir, "DerivedDataCacheManifest.txt");

			AddDependency(GUBP.RootEditorNode.StaticGetFullName(HostPlatform));
		}

		public static string StaticGetFullName(UnrealTargetPlatform InHostPlatform)
		{
            return "BuildDerivedDataCache" + StaticGetHostPlatformSuffix(InHostPlatform);
		}

		public override int CISFrequencyQuantumShift(GUBP bp)
		{
			return base.CISFrequencyQuantumShift(bp) + 2;
		}

		public override string GetFullName()
		{
			return StaticGetFullName(HostPlatform);
		}

		public override void DoBuild(GUBP bp)
		{
			CommandUtils.CreateDirectory(SavedDir);

			BuildProducts = new List<string>();

			List<string> ManifestFiles = new List<string>();
			if(!bp.ParseParam("NoDDC"))
			{
				string EditorExe = CommandUtils.GetEditorCommandletExe(CommandUtils.CmdEnv.LocalRoot, HostPlatform);
				string RelativePakPath = "Engine/DerivedDataCache/Compressed.ddp";

				// Delete the output file
				string OutputPakFile = CommandUtils.CombinePaths(CommandUtils.CmdEnv.LocalRoot, RelativePakPath);
				string OutputTxtFile = Path.ChangeExtension(OutputPakFile, ".txt");

				// Generate DDC for all the non-code projects. We don't necessarily have editor DLLs for the code projects, but they should be the same as their blueprint counterparts.
				List<string> ProjectPakFiles = new List<string>();
				foreach(string ProjectName in ProjectNames)
				{
					BranchInfo.BranchUProject Project = bp.Branch.FindGameChecked(ProjectName);
					if(!Project.Properties.bIsCodeBasedProject)
					{
						CommandUtils.Log("Generating DDC data for {0} on {1}", Project.GameName, TargetPlatforms);
						CommandUtils.DDCCommandlet(Project.FilePath, EditorExe, null, TargetPlatforms, "-fill -DDC=CreateInstalledEnginePak -ProjectOnly");

						string ProjectPakFile = CommandUtils.CombinePaths(Path.GetDirectoryName(OutputPakFile), String.Format("Compressed-{0}.ddp", ProjectName));
						CommandUtils.DeleteFile(ProjectPakFile);
						CommandUtils.RenameFile(OutputPakFile, ProjectPakFile);

						string ProjectTxtFile = Path.ChangeExtension(ProjectPakFile, ".txt");
						CommandUtils.DeleteFile(ProjectTxtFile);
						CommandUtils.RenameFile(OutputTxtFile, ProjectTxtFile);

						ProjectPakFiles.Add(Path.GetFileName(ProjectPakFile));
					}
				}

				// Generate DDC for the editor, and merge all the other PAK files in
				CommandUtils.Log("Generating DDC data for engine content on {0}", TargetPlatforms);
				CommandUtils.DDCCommandlet(null, EditorExe, null, TargetPlatforms, "-fill -DDC=CreateInstalledEnginePak " + CommandUtils.MakePathSafeToUseWithCommandLine("-MergePaks=" + String.Join("+", ProjectPakFiles)));

				// Copy the DDP file to the output path
				string SavedPakFile = CommandUtils.CombinePaths(SavedDir, RelativePakPath);
				CommandUtils.CopyFile(OutputPakFile, SavedPakFile);
				BuildProducts.Add(SavedPakFile);

				// Add the pak file to the list of files to copy
				ManifestFiles.Add(RelativePakPath);
			}
			CommandUtils.WriteAllLines(SavedManifestPath, ManifestFiles.ToArray());
			BuildProducts.Add(SavedManifestPath);

			SaveRecordOfSuccessAndAddToBuildProducts();
		}

		public override float Priority()
		{
			return base.Priority() + 55.0f;
		}
	}
}
