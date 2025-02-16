import unittest
import json
import warnings
import networkx as nx
from parent_teacher_conference_sorting import (
    schedule_meetings_aggregator,
)


class TestParentTeacherConferenceSortingAggregator(unittest.TestCase):
    def setUp(self):
        # Load sample data from the JSON file.
        with open("parent_teacher_conference_sorting/sample_data.json", "r") as f:
            self.sample_data = json.load(f)
        self.meeting_requests = self.sample_data["parent_teacher_meetings"]
        self.teacher_slots = self.sample_data["teacher_timeslots"]
        self.global_timeslots = self.sample_data["global_timeslots"]
        # Penalty and bonus parameters.
        self.drop_penalty = 1000
        self.reschedule_penalty = 50
        self.parent_bonus = 20

    def test_sorting_and_no_double_bookings(self):

        schedule, nonpreferred, flowDict, G = schedule_meetings_aggregator(
            self.meeting_requests,
            self.teacher_slots,
            self.global_timeslots,
            drop_penalty=self.drop_penalty,
            reschedule_penalty=self.reschedule_penalty,
            parent_bonus=self.parent_bonus,
        )

        total_requests = len(self.meeting_requests)
        scheduled_count = sum(1 for m in schedule if m["timeslot"] is not None)
        dropped_count = total_requests - scheduled_count
        distinct_timeslots_used = len(
            set(m["timeslot"] for m in schedule if m["timeslot"] is not None)
        )
        total_cost = nx.cost_of_flow(G, flowDict)

        # Debug printing of each scheduled meeting.
        print("\n--- Detailed Meeting Schedule ---")
        for m in schedule:
            mark = (
                "✓"
                if (m["timeslot"] is not None and m["timeslot"] == m["preferred"])
                else "✗"
            )
            print(
                f"Request: {m.get('meeting_request', 'N/A')}, Parent: {m['parent']}, Teacher: {m['teacher']}, "
                f"Timeslot: {m['timeslot']} (Preferred: {m['preferred']}) {mark}"
            )

        # Debug printing of flow network summary.
        print("\n--- Flow Network Summary ---")
        print(f"Total nodes in graph: {len(G.nodes())}")
        print(f"Total edges in graph: {len(G.edges())}")
        print(f"Total cost of flow: {total_cost}")

        # Performance metrics summary.
        print("\n--- Performance Metrics ---")
        print(f"Total meeting requests: {total_requests}")
        print(f"Scheduled meetings: {scheduled_count}")
        print(f"Dropped meeting requests: {dropped_count}")
        print(f"Distinct timeslots used: {distinct_timeslots_used}")

        # Check for double bookings: parent or teacher in more than one meeting per timeslot.
        parent_bookings = {}
        teacher_bookings = {}
        for m in schedule:
            if m["timeslot"] is None:
                continue
            parent_key = (m["parent"], m["timeslot"])
            teacher_key = (m["teacher"], m["timeslot"])
            parent_bookings[parent_key] = parent_bookings.get(parent_key, 0) + 1
            teacher_bookings[teacher_key] = teacher_bookings.get(teacher_key, 0) + 1

        double_booked_parents = [
            key for key, count in parent_bookings.items() if count > 1
        ]
        double_booked_teachers = [
            key for key, count in teacher_bookings.items() if count > 1
        ]
        self.assertEqual(
            len(double_booked_parents),
            0,
            f"Double bookings found for parents in timeslots: {double_booked_parents}",
        )
        self.assertEqual(
            len(double_booked_teachers),
            0,
            f"Double bookings found for teachers in timeslots: {double_booked_teachers}",
        )

        # Basic assertion to ensure all meeting requests were either scheduled or dropped.
        self.assertEqual(
            scheduled_count + dropped_count,
            total_requests,
            "Sum of scheduled and dropped meetings should equal total meeting requests.",
        )


if __name__ == "__main__":
    unittest.main()
