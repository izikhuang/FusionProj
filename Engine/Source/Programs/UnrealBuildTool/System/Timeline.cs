using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Tools.DotNETCommon;

namespace UnrealBuildTool
{
	/// <summary>
	/// Public interface for a timeline scope. Should be disposed to exit the scope.
	/// </summary>
	interface ITimelineEvent : IDisposable
	{
		void Finish();
	}

	/// <summary>
	/// Tracks simple high-level timing data
	/// </summary>
	static class Timeline
	{
		/// <summary>
		/// A marker in the timeline
		/// </summary>
		[DebuggerDisplay("{Name}")]
		class Event : ITimelineEvent
		{
			/// <summary>
			/// Name of the marker
			/// </summary>
			public string Name;

			/// <summary>
			/// Time at which the event ocurred
			/// </summary>
			public TimeSpan StartTime;

			/// <summary>
			/// Time at which the event ended
			/// </summary>
			public TimeSpan? FinishTime;

			/// <summary>
			/// Constructor
			/// </summary>
			/// <param name="Name">Event name</param>
			/// <param name="StartTime">Time of the event</param>
			/// <param name="FinishTime">Finish time for the event. May be null.</param>
			public Event(string Name, TimeSpan StartTime, TimeSpan? FinishTime)
			{
				this.Name = Name;
				this.StartTime = StartTime;
				this.FinishTime = FinishTime;
			}

			/// <summary>
			/// Finishes the current event
			/// </summary>
			public void Finish()
			{
				if(!FinishTime.HasValue)
				{
					FinishTime = Stopwatch.Elapsed;
				}
			}

			/// <summary>
			/// Disposes of the current event
			/// </summary>
			public void Dispose()
			{
				Finish();
			}
		}

		/// <summary>
		/// The stopwatch used for timing
		/// </summary>
		static Stopwatch Stopwatch = new Stopwatch();

		/// <summary>
		/// The recorded events
		/// </summary>
		static List<Event> Events = new List<Event>();

		/// <summary>
		/// Start the stopwatch
		/// </summary>
		public static void Start()
		{
			Stopwatch.Restart();
		}

		/// <summary>
		/// Records a timeline marker with the given name
		/// </summary>
		/// <param name="Name">The marker name</param>
		public static void AddEvent(string Name)
		{
			TimeSpan Time = Stopwatch.Elapsed;
			Events.Add(new Event(Name, Time, Time));
		}

		/// <summary>
		/// Enters a scope event with the given name. Should be disposed to terminate the scope.
		/// </summary>
		/// <param name="Name">Name of the event</param>
		/// <returns>Event to track the length of the event</returns>
		public static ITimelineEvent ScopeEvent(string Name)
		{
			Event Event = new Event(Name, Stopwatch.Elapsed, null);
			Events.Add(Event);
			return Event;
		}

		/// <summary>
		/// Enters a scope event with the calling method name as the event name
		/// </summary>
		/// <returns>Event to track the lifetime of the method. Dispose it before returning.</returns>
		[MethodImplAttribute(MethodImplOptions.NoInlining)]
		public static ITimelineEvent ScopeMethod()
		{
			StackFrame Frame = new StackFrame(1);
			MethodBase Method = Frame.GetMethod();
			return ScopeEvent(String.Format("{0}.{1}", Method.DeclaringType.Name, Method.Name));
		}

		/// <summary>
		/// Prints this information to the log
		/// </summary>
		public static void Print(TimeSpan MaxUnknownTime, LogEventType Verbosity)
		{
			// Print the start time
			Log.WriteLine(Verbosity, "Timeline:");
			Log.WriteLine(Verbosity, "");
			Log.WriteLine(Verbosity, "[{0,6}]", FormatTime(TimeSpan.Zero));

			// Create the root event
			TimeSpan FinishTime = Stopwatch.Elapsed;

			List<Event> OuterEvents = new List<Event>();
			OuterEvents.Add(new Event("<Root>", TimeSpan.Zero, FinishTime));

			// Print out all the child events
			TimeSpan LastTime = TimeSpan.Zero;
			for(int EventIdx = 0; EventIdx < Events.Count; EventIdx++)
			{
				Event Event = Events[EventIdx];

				// Pop events off the stack
				for (; OuterEvents.Count > 0; OuterEvents.RemoveAt(OuterEvents.Count - 1))
				{
					Event OuterEvent = OuterEvents.Last();
					if (Event.StartTime < OuterEvent.FinishTime.Value)
					{
						break;
					}
					UpdateLastEventTime(ref LastTime, OuterEvent.FinishTime.Value, MaxUnknownTime, OuterEvents, Verbosity);
				}

				// If there's a gap since the last event, print an unknown marker
				UpdateLastEventTime(ref LastTime, Event.StartTime, MaxUnknownTime, OuterEvents, Verbosity);

				// Print this event
				Print(Event.StartTime, Event.FinishTime, Event.Name, OuterEvents, Verbosity);

				// Push it onto the stack
				if(Event.FinishTime.HasValue && EventIdx + 1 < Events.Count && Events[EventIdx + 1].StartTime < Event.FinishTime.Value)
				{
					OuterEvents.Add(Event);
				}
			}

			// Remove everything from the stack
			for(; OuterEvents.Count > 1; OuterEvents.RemoveAt(OuterEvents.Count - 1))
			{
				UpdateLastEventTime(ref LastTime, OuterEvents.Last().FinishTime.Value, MaxUnknownTime, OuterEvents, Verbosity);
			}

			// Print the finish time
			Log.WriteLine(Verbosity, "[{0,6}]", FormatTime(FinishTime));
		}

		/// <summary>
		/// Updates the last event time
		/// </summary>
		/// <param name="LastTime"></param>
		/// <param name="NewTime"></param>
		/// <param name="MaxUnknownTime"></param>
		/// <param name="OuterEvents"></param>
		/// <param name="Verbosity"></param>
		static void UpdateLastEventTime(ref TimeSpan LastTime, TimeSpan NewTime, TimeSpan MaxUnknownTime, List<Event> OuterEvents, LogEventType Verbosity)
		{
			const string UnknownEvent = "<unknown>";
			if (NewTime - LastTime > MaxUnknownTime)
			{
				Print(LastTime, NewTime, UnknownEvent, OuterEvents, Verbosity);
			}
			LastTime = NewTime;
		}

		/// <summary>
		/// Prints an individual event to the log
		/// </summary>
		/// <param name="StartTime">Start time for the event</param>
		/// <param name="FinishTime">Finish time for the event. May be null.</param>
		/// <param name="Label">Event name</param>
		/// <param name="OuterEvents">List of all the start times for parent events</param>
		/// <param name="Verbosity">Verbosity for the output</param>
		static void Print(TimeSpan StartTime, TimeSpan? FinishTime, string Label, List<Event> OuterEvents, LogEventType Verbosity)
		{
			StringBuilder Prefix = new StringBuilder();

			for(int Idx = 0; Idx < OuterEvents.Count - 1; Idx++)
			{
				Prefix.AppendFormat(" {0,6}          ", FormatTime(StartTime - OuterEvents[Idx].StartTime));
			}

			Prefix.AppendFormat("[{0,6}]", FormatTime(StartTime - OuterEvents[OuterEvents.Count - 1].StartTime));

			if (!FinishTime.HasValue)
			{
				Prefix.AppendFormat("({0,6})", "???");
			}
			else if(FinishTime.Value == StartTime)
			{
				Prefix.Append(" ------ ");
			}
			else
			{
				Prefix.AppendFormat("({0,6})", "+" + FormatTime(FinishTime.Value - StartTime));
			}

			Log.WriteLine(Verbosity, "{0} {1}", Prefix.ToString(), Label);
		}

		/// <summary>
		/// Formats a timespan in milliseconds
		/// </summary>
		/// <param name="Time">The time to format</param>
		/// <returns>Formatted timespan</returns>
		static string FormatTime(TimeSpan Time)
		{
			int TotalMilliseconds = (int)Time.TotalMilliseconds;
			return String.Format("{0}.{1:000}", TotalMilliseconds / 1000, TotalMilliseconds % 1000);
		}
	}
}
