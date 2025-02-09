import unittest
from parent_teacher_conference_sorting import schedule_meetings_optimal

# Import the scheduling function.
# For example, if the scheduling function is in a module named scheduler, use:
# from scheduler import schedule_meetings_optimal
#
# In this snippet, we assume schedule_meetings_optimal is defined in the same file or is otherwise importable.


class TestSchedulingSolution(unittest.TestCase):
    def setUp(self):
        # Define sample inputs:
        self.time_slots = ["9:00", "9:30", "10:00", "10:30"]
        self.teachers = ["Math", "Science", "History", "English"]
        self.parent_preferences = {
            "Parent1": {
                "teachers": ["Math", "Science", "English"],
                "preferred_slots": ["9:00", "9:30"],
            },
            "Parent2": {
                "teachers": ["Science", "History", "English"],
                "preferred_slots": ["10:00", "10:30"],
            },
            "Parent3": {
                "teachers": ["Math", "History", "English"],
                "preferred_slots": ["9:30", "10:00"],
            },
        }
        # Calculate total meeting requests.
        # For example, Parent1 has 3 wishes, Parent2 has 3, Parent3 has 3 → total = 9.
        self.total_requests = sum(
            len(info["teachers"]) for info in self.parent_preferences.values()
        )
        # We use a preferred reward of 10 (as in the scheduling function).
        self.preferred_reward = 10

    def test_solution_output(self):
        # Call the scheduling function.
        schedule, total_reward = schedule_meetings_optimal(
            self.time_slots,
            self.teachers,
            self.parent_preferences,
            preferred_reward=self.preferred_reward,
        )

        # Compute counts:
        wishes_met = len(schedule)
        preferred_count = sum(
            1
            for ((parent, teacher), slot) in schedule.items()
            if slot in self.parent_preferences[parent]["preferred_slots"]
        )
        distinct_timeslots_used = len(set(schedule.values()))

        # Print the solution and counts.
        print("\n--- Optimal Scheduling Solution ---")
        for (parent, teacher), slot in schedule.items():
            # Mark whether this meeting is in a preferred timeslot.
            pref_mark = (
                "✓"
                if slot in self.parent_preferences[parent]["preferred_slots"]
                else ""
            )
            print(
                f"Parent: {parent:<8} Teacher: {teacher:<8} Timeslot: {slot:<5} {pref_mark}"
            )
        print("\nSummary:")
        print(f"  Total meeting requests (wishes): {self.total_requests}")
        print(f"  Total meeting wishes met: {wishes_met}")
        print(f"  Preferred timeslot assignments: {preferred_count}")
        print(
            f"  Distinct timeslots used: {distinct_timeslots_used} out of {len(self.time_slots)}"
        )

        # 1. Check that every meeting request was scheduled.
        self.assertEqual(
            wishes_met, self.total_requests, "Not all meeting requests were scheduled."
        )

        # 2. Verify that the number of preferred assignments is as expected.
        # (For our example, we expect 6 preferred assignments in an optimal solution.)
        expected_preferred = 6
        self.assertEqual(
            preferred_count,
            expected_preferred,
            f"Expected {expected_preferred} preferred timeslot assignments, got {preferred_count}.",
        )

        # 3. Check that the total reward (which is negative cost) matches.
        # Each preferred meeting gives a reward of 10, so the total_reward (which was computed as -flowCost)
        # should equal (preferred_count * preferred_reward).
        self.assertEqual(
            total_reward,
            preferred_count * self.preferred_reward,
            "Total reward does not match expected value.",
        )

        # 4. Verify that no teacher is double–booked in any timeslot.
        teacher_timeslot_bookings = {teacher: {} for teacher in self.teachers}
        for (parent, teacher), slot in schedule.items():
            teacher_timeslot_bookings[teacher][slot] = (
                teacher_timeslot_bookings[teacher].get(slot, 0) + 1
            )
        for teacher, times in teacher_timeslot_bookings.items():
            for slot, count in times.items():
                self.assertLessEqual(
                    count, 1, f"Teacher {teacher} is double-booked at {slot}."
                )

        # 5. Verify that no parent is double–booked in any timeslot.
        parent_timeslot_bookings = {parent: {} for parent in self.parent_preferences}
        for (parent, teacher), slot in schedule.items():
            parent_timeslot_bookings[parent][slot] = (
                parent_timeslot_bookings[parent].get(slot, 0) + 1
            )
        for parent, times in parent_timeslot_bookings.items():
            for slot, count in times.items():
                self.assertLessEqual(
                    count, 1, f"Parent {parent} is double-booked at {slot}."
                )


if __name__ == "__main__":
    unittest.main()
