using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Gauntlet
{
	public class UnrealHealthSnapshot
	{
		public string Name;
		public float ProfileLength;

		public float CpuUsedMemory;
		public float CpuPeakMemory;
		public float PhysicalUsedMemory;
		public float PhysicalPeakMemory;
		public float MVP;
		public float AvgFps;
		public float Hitches;
		public float AvgHitches;
		public float GTTime;
		public float RTTime;
		public float GPUTime;
		public float FTTime;
		public float RHIT;
		public int DrawCalls;
		public int DrawnPrims;
		public int UnbuiltHLODs;

		public UnrealHealthSnapshot()
		{
			Name = "Unknown";
		}

		public override string ToString()
		{
			StringBuilder SB = new StringBuilder();

			if (ProfileLength > 0)
			{
				SB.AppendFormat("Snapshot {0}\n", Name);
				SB.AppendFormat("Duration {0}\n", ProfileLength);

				SB.AppendFormat("Memory:\t{0} MB\n", CpuUsedMemory);
				SB.AppendFormat("MVP:\t\t{0:0.00}\n", MVP);
				SB.AppendFormat("AvgFps:\t\t{0:0.00}\n", AvgFps);
				SB.AppendFormat("HPM:\t\t{0:0.00}\n", Hitches);
				SB.AppendFormat("AvgH:\t\t{0:0.00}ms\n", AvgHitches);
				SB.AppendFormat("GT:\t\t{0:0.00}ms\n", GTTime);
				SB.AppendFormat("RT:\t\t{0:0.00}ms\n", RTTime);
				SB.AppendFormat("GPU:\t\t{0:0.00}ms\n", GPUTime);
				SB.AppendFormat("FT:\t\t{0:0.00}ms\n", FTTime);
				SB.AppendFormat("RHIT:\t\t{0:0.00}ms\n", RHIT);
				SB.AppendFormat("DrawCalls:\t{0}\n", DrawCalls);
				SB.AppendFormat("DrawnPrims:\t{0}\n", DrawnPrims);
				SB.AppendFormat("UnbuiltHLODs:\t{0}\n", UnbuiltHLODs);
			}
			else
			{
				SB.Append("No info");
			}
			return SB.ToString();
		}

		public virtual void CreateFromString(string InContent)
		{
			RegexUtil.MatchAndApplyGroups(InContent, @".+Snapshot:\s+(.+?)\s+=", (Groups) =>
			{
				Name = Groups[1];
			});


			RegexUtil.MatchAndApplyGroups(InContent, @"CPU Memory:[\s\w]+?([\d\.]+)MB,[\s\w]+?([\d\.]+)MB", (Groups) =>
			{
				CpuUsedMemory += Convert.ToSingle(Groups[1]);
				CpuPeakMemory += Convert.ToSingle(Groups[2]);
			});

			RegexUtil.MatchAndApplyGroups(InContent, @"Physical Memory:[\s\w]+?([\d\.]+)MB,[\s\w:]+?([\d\.]+)MB", (Groups) =>
			{
				PhysicalUsedMemory += Convert.ToSingle(Groups[1]);
				PhysicalPeakMemory += Convert.ToSingle(Groups[2]);
			});

			RegexUtil.MatchAndApplyGroups(InContent, @"MeasuredPerfTime\s(\d.+) Secs", (Groups) =>
			{
				ProfileLength += Convert.ToSingle(Groups[1]);
			});

			RegexUtil.MatchAndApplyGroups(InContent, @"MVP:\s(\d.+)%,\s.*AvgFPS:(\d.+),\s.*HitchesPerMinute:\s(\d.+),\sAvg\sHitch\s(\d.+)ms", (Groups) =>
			{
				MVP = Convert.ToSingle(Groups[1]);
				AvgFps = Convert.ToSingle(Groups[2]);
				Hitches = Convert.ToSingle(Groups[3]);
				AvgHitches = Convert.ToSingle(Groups[4]);
			});

			RegexUtil.MatchAndApplyGroups(InContent, @"GT:.* Avg\s(\d.+)ms,", (Groups) =>
			{
				GTTime = Convert.ToSingle(Groups[1]);
			});

			RegexUtil.MatchAndApplyGroups(InContent, @"RT:.* Avg\s(\d.+)ms,", (Groups) =>
			{
				RTTime = Convert.ToSingle(Groups[1]);
			});

			RegexUtil.MatchAndApplyGroups(InContent, @"GPU:.* Avg\s(\d.+)ms,", (Groups) =>
			{
				GPUTime = Convert.ToSingle(Groups[1]);
			});

			RegexUtil.MatchAndApplyGroups(InContent, @"FT:.* Avg:\s([\d\.]+)ms,", (Groups) =>
			{
				FTTime = Convert.ToSingle(Groups[1]);
			});

			RegexUtil.MatchAndApplyGroups(InContent, @"RHIT:Avg\s([\d\.]+)ms,", (Groups) =>
			{
				RHIT = Convert.ToSingle(Groups[1]);
			});

			RegexUtil.MatchAndApplyGroups(InContent, @"DrawCalls:[\w\s:]+?([\d\.]+),[\w\s:]+?([\d\.]+),[\w\s:]+?([\d\.]+)", (Groups) =>
			{
				DrawCalls = Convert.ToInt32(Groups[1]);
			});

			RegexUtil.MatchAndApplyGroups(InContent, @"DrawnPrims:[\w\s:]+?([\d\.]+),[\w\s:]+?([\d\.]+),[\w\s:]+?([\d\.]+)", (Groups) =>
			{
				DrawnPrims = Convert.ToInt32(Groups[1]);
			});
			RegexUtil.MatchAndApplyGroups(InContent, @"UnbuiltHLODs:\s(\d.+)", (Groups) =>
			{
				UnbuiltHLODs = Convert.ToInt32(Groups[1]);
			});
		}
	}

	public class UnrealSnapshotSummary<TSnapshotClass>
		where TSnapshotClass : UnrealHealthSnapshot, new()
	{
		public int SampleCount;
		public float SessionTime;
		public float PeakMemory;

		public List<float> MVP;
		public List<float> AvgFps;
		public List<float> Hitches;
		public List<float> AvgHitches;
		public List<float> GTTime;
		public List<float> RTTime;
		public List<float> GPUTime;
		public List<float> FTTime;
		public List<float> RHIT;
		public List<int> DrawCalls;
		public List<int> DrawnPrims;
		public List<int> UnbuiltHLODs;

		public IEnumerable<TSnapshotClass> Snapshots;

		public UnrealSnapshotSummary(string LogContents, string SnapshotTitles="")
		{
			SampleCount = 0;
			SessionTime = 0.0f;
			PeakMemory = 0.0f;
			MVP = new List<float>();
			AvgFps = new List<float>();
			Hitches = new List<float>();
			AvgHitches = new List<float>();
			GTTime = new List<float>();
			RTTime = new List<float>();
			GPUTime = new List<float>();
			FTTime = new List<float>();
			RHIT = new List<float>();
			DrawCalls = new List<int>();
			DrawnPrims = new List<int>();
			UnbuiltHLODs = new List<int>();

			CreateFromLog(LogContents, SnapshotTitles);
		}
		protected virtual void CreateFromLog(string LogContents, string InTitle)
		{
			UnrealLogParser Parser = new UnrealLogParser(LogContents);

			if (string.IsNullOrEmpty(InTitle))
			{
				InTitle = "=== Snapshot: ";
			}

			try
			{
				// Find all end of match reports
				string[] SessionSnapshots = Parser.GetGroupsOfLinesStartingWith(InTitle, 16);

				SampleCount = SessionSnapshots.Length;
				SessionTime = 0;

				// convert these blocks into snapshot info
				Snapshots = SessionSnapshots.Select(S =>
				{
					var Snapshot = new TSnapshotClass();
					Snapshot.CreateFromString(S);

					return Snapshot;
				}).ToList();

				// get average MVP, GT, RT, GPU times
				foreach (UnrealHealthSnapshot Snap in Snapshots)
				{
					SessionTime += Snap.ProfileLength;

					MVP.Add(Snap.MVP);

					if (Snap.AvgFps > 0)
					{
						AvgFps.Add(Snap.AvgFps);
					}

					Hitches.Add(Snap.Hitches);
					AvgHitches.Add(Snap.AvgHitches);

					if (Snap.GTTime > 0)
					{
						GTTime.Add(Snap.GTTime);
					}

					if (Snap.RTTime > 0)
					{
						RTTime.Add(Snap.RTTime);
					}

					if (Snap.GPUTime > 0)
					{
						GPUTime.Add(Snap.GPUTime);
					}

					if (Snap.FTTime > 0)
					{
						FTTime.Add(Snap.FTTime);
					}

					if (Snap.RHIT > 0)
					{
						RHIT.Add(Snap.RHIT);
					}

					if (Snap.DrawCalls > 0)
					{
						DrawCalls.Add(Snap.DrawCalls);
						DrawnPrims.Add(Snap.DrawnPrims);
					}
					if (Snap.UnbuiltHLODs > 0)
					{
						UnbuiltHLODs.Add(Snap.UnbuiltHLODs);
					}
				}

				// Now get peak memory from the last health report
				if (Snapshots.Count() > 0)
				{
					var LastSnapshot = Snapshots.Last();

					float SnapshotPeakMemory = LastSnapshot.PhysicalPeakMemory;

					//if PeakMemory is reporting as 0, use Memory if it's higher than our last
					if (SnapshotPeakMemory == 0)
					{
						Log.Info("PeakMemory reported as 0mb");
					}
					else
					{
						PeakMemory = SnapshotPeakMemory;
					}
				}
			}
			catch (Exception Ex)
			{
				Log.Info("Failed parsing PerformanceSummary: " + Ex.ToString());
			}
		}

		public override string ToString()
		{
			StringBuilder SB = new StringBuilder();

			if (SampleCount > 0)
			{
				SB.AppendFormat("Performance report from {0} samples and {1} seconds\n", SampleCount, SessionTime);

				SB.AppendFormat("Peak Memory: {0} MB\n", PeakMemory);
				if (MVP.Count > 0)
				{
					SB.AppendFormat("MVP:\t{0:0.00} (Min: {1:0.00}, Max: {2:0.00})\n", MVP.Average(), MVP.Min(), MVP.Max());
				}

				if (AvgFps.Count > 0)
				{
					SB.AppendFormat("AvgFPS:\t{0:0.00} (Min: {1:0.00}, Max: {2:0.00})\n", AvgFps.Average(), AvgFps.Min(), AvgFps.Max());
				}

				if (Hitches.Count > 0)
				{
					SB.AppendFormat("HPM:\t{0:0.00} (Min: {1:0.00}, Max: {2:0.00})\n", Hitches.Average(), Hitches.Min(), Hitches.Max());
				}

				if (AvgHitches.Count > 0)
				{
					SB.AppendFormat("AvgH:\t{0:0.00}ms (Min: {1:0.00}ms, Max: {2:0.00}ms)\n", AvgHitches.Average(), AvgHitches.Min(), AvgHitches.Max());
				}
				if (GTTime.Count > 0)
				{
					SB.AppendFormat("GT:\t{0:0.00}ms (Min: {1:0.00}ms, Max: {2:0.00}ms)\n", GTTime.Average(), GTTime.Min(), GTTime.Max());
				}
				if (RTTime.Count > 0)
				{
					SB.AppendFormat("RT:\t{0:0.00}ms (Min: {1:0.00}ms, Max: {2:0.00}ms)\n", RTTime.Average(), RTTime.Min(), RTTime.Max());
				}
				if (GPUTime.Count > 0)
				{
					SB.AppendFormat("GPU:\t{0:0.00}ms (Min: {1:0.00}ms, Max: {2:0.00}ms)\n", GPUTime.Average(), GPUTime.Min(), GPUTime.Max());
				}
				if (RHIT.Count > 0)
				{
					SB.AppendFormat("RHIT:\t{0:0.00}ms (Min: {1:0.00}ms, Max: {2:0.00}ms)\n", RHIT.Average(), RHIT.Min(), RHIT.Max());
				}

				if (DrawCalls.Count > 0)
				{
					SB.AppendFormat("DrawCalls:\t{0:0} (Min: {1}, Max: {2})\n", DrawCalls.Average(), DrawCalls.Min(), DrawCalls.Max());
				}

				if (DrawnPrims.Count > 0)
				{
					SB.AppendFormat("DrawnPrims:\t{0:0} (Min: {1}, Max: {2})\n", DrawnPrims.Average(), DrawnPrims.Min(), DrawnPrims.Max());
				}
			}
			else
			{
				SB.Append("No session info");
			}
			return SB.ToString();
		}
	}

}
