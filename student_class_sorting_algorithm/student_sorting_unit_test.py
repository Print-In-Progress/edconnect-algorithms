import unittest
import random
import time
from collections import Counter
from student_sorting_algo_constrained_programming import combined_cp_solver


class TestStudentSorting(unittest.TestCase):
    def generate_test_data(
        self,
        num_students=100,
        num_classes=4,
        avg_prefs=5,
        gender_dist=None,
        params=None,
    ):
        """Generate realistic test data with controlled properties"""
        if gender_dist is None:
            gender_dist = {"m": 0.5, "f": 0.45, "nb": 0.05}

        students_data = {}
        student_ids = [f"s{i}" for i in range(1, num_students + 1)]

        # Assign gender based on distribution
        genders = []
        for gender, ratio in gender_dist.items():
            genders.extend([gender] * int(num_students * ratio))
        # Fill any remaining spots
        while len(genders) < num_students:
            genders.append(random.choice(list(gender_dist.keys())))
        random.shuffle(genders)

        # Create basic student data
        for i, student_id in enumerate(student_ids):
            students_data[student_id] = {"sex": genders[i], "prefs": []}

            # Add parameters if provided
            if params:
                for param in params:
                    if param["type"] == "binary":
                        # 30% yes probability for binary parameters
                        students_data[student_id][param["name"]] = (
                            "yes" if random.random() < 0.3 else "no"
                        )
                    elif param["type"] == "categorical":
                        # Assign one of 5 possible categories
                        students_data[student_id][
                            param["name"]
                        ] = f"cat{random.randint(1, 5)}"

        # Generate preferences
        for student_id in student_ids:
            # Choose random friends
            num_prefs = min(random.randint(1, avg_prefs * 2), num_students - 1)
            potential_friends = [s for s in student_ids if s != student_id]
            prefs = random.sample(potential_friends, num_prefs)
            students_data[student_id]["prefs"] = prefs

            # Create some mutual friendships (70% chance)
            for pref in prefs:
                if random.random() < 0.7:
                    if student_id not in students_data[pref]["prefs"]:
                        students_data[pref]["prefs"].append(student_id)

        # Set class sizes
        class_sizes = {
            f"class{i}": num_students // num_classes
            + (1 if i < num_students % num_classes else 0)
            for i in range(1, num_classes + 1)
        }

        return students_data, class_sizes

    def analyze_results(
        self, students_data, class_assignments, gender_ratio, parameters=None
    ):
        """Analyze and print detailed statistics about assignment results"""
        print("\n=== ASSIGNMENT RESULTS ===")

        # Basic stats
        total_students = len(students_data)
        assigned_students = sum(
            len(students) for students in class_assignments.values()
        )
        print(f"Total students: {total_students}")
        print(f"Assigned students: {assigned_students}")
        print(f"Assignment rate: {assigned_students/total_students:.2%}")

        # Class size distribution
        print("\nCLASS SIZES:")
        for class_id, students in class_assignments.items():
            print(f"{class_id}: {len(students)} students")

        # Gender distribution
        if gender_ratio:
            print("\nGENDER DISTRIBUTION:")
            for class_id, students in class_assignments.items():
                gender_counts = Counter(students_data[s]["sex"] for s in students)
                print(
                    f"{class_id}: "
                    + ", ".join(
                        f"{g}: {c} ({c/len(students):.2%})"
                        for g, c in gender_counts.items()
                    )
                )

                # Check if constraints are satisfied
                for gender, ratio in gender_ratio.items():
                    count = gender_counts.get(gender, 0)
                    max_allowed = int(ratio * len(students)) + 1  # +1 for rounding
                    status = "✓" if count <= max_allowed else "✗"
                    print(
                        f"  {gender} constraint ({ratio:.0%}): {count}/{max_allowed} {status}"
                    )

        # Preference satisfaction
        print("\nFRIEND PREFERENCE SATISFACTION:")
        total_prefs = 0
        satisfied_prefs = 0
        mutual_satisfied = 0
        students_with_satisfied_prefs = (
            0  # New counter for students with at least one satisfied preference
        )

        # Track which students had at least one preference satisfied
        student_satisfaction = {student: False for student in students_data}

        for class_id, students in class_assignments.items():
            class_prefs = 0
            class_satisfied = 0
            class_mutual = 0
            class_students_satisfied = (
                0  # Students in this class with at least one preference satisfied
            )

            for student in students:
                student_has_satisfied = False
                for friend in students_data[student]["prefs"]:
                    total_prefs += 1
                    class_prefs += 1
                    if friend in students:
                        satisfied_prefs += 1
                        class_satisfied += 1
                        student_has_satisfied = True
                        student_satisfaction[student] = True
                        if student in students_data[friend]["prefs"]:
                            mutual_satisfied += 1
                            class_mutual += 1

                if student_has_satisfied:
                    class_students_satisfied += 1

            print(
                f"{class_id}: {class_satisfied}/{class_prefs} preferences satisfied ({class_satisfied/max(1,class_prefs):.2%})"
            )
            print(
                f"  Mutual preferences: {class_mutual} ({class_mutual/max(1,class_satisfied):.2%} of satisfied)"
            )
            print(
                f"  Students with at least one preference: {class_students_satisfied}/{len(students)} ({class_students_satisfied/max(1,len(students)):.2%})"
            )

        # Calculate overall students with at least one preference satisfied
        students_with_satisfied_prefs = sum(
            1 for satisfied in student_satisfaction.values() if satisfied
        )

        print(
            f"\nOverall preference satisfaction: {satisfied_prefs}/{total_prefs} ({satisfied_prefs/max(1,total_prefs):.2%})"
        )
        print(
            f"Overall mutual preference satisfaction: {mutual_satisfied}/{satisfied_prefs} ({mutual_satisfied/max(1,satisfied_prefs):.2%})"
        )
        print(
            f"Students with at least one preference satisfied: {students_with_satisfied_prefs}/{total_students} ({students_with_satisfied_prefs/total_students:.2%})"
        )

        # Parameter distribution
        if parameters:
            print("\nPARAMETER DISTRIBUTION:")
            for param in parameters:
                param_name = param["name"]
                print(
                    f"\nParameter: {param_name} (Strategy: {param['strategy']}, Priority: {param.get('priority', 'N/A')})"
                )

                if param["type"] == "binary":
                    yes_total = sum(
                        1
                        for s, data in students_data.items()
                        if data.get(param_name) == "yes"
                    )
                    expected_per_class = yes_total // len(class_assignments)

                    for class_id, students in class_assignments.items():
                        yes_count = sum(
                            1
                            for s in students
                            if students_data[s].get(param_name) == "yes"
                        )
                        total = len(students)
                        status = ""

                        if param["strategy"] == "distribute":
                            deviation = abs(yes_count - expected_per_class)
                            status = f"(deviation: {deviation}, expected: ~{expected_per_class})"

                        print(
                            f"{class_id}: {yes_count}/{total} 'yes' students ({yes_count/max(1,total):.2%}) {status}"
                        )

                elif param["type"] == "categorical":
                    for class_id, students in class_assignments.items():
                        category_counts = Counter(
                            students_data[s].get(param_name) for s in students
                        )
                        print(
                            f"{class_id}: "
                            + ", ".join(
                                f"{cat}: {count}"
                                for cat, count in category_counts.items()
                            )
                        )

        return {
            "total_students": total_students,
            "assigned_students": assigned_students,
            "assignment_rate": assigned_students / total_students,
            "preference_satisfaction": satisfied_prefs / max(1, total_prefs),
            "mutual_satisfaction": mutual_satisfied / max(1, satisfied_prefs),
            "students_with_preferences": students_with_satisfied_prefs / total_students,
        }

    def test_complex_scenario(self):
        """Test a complex scenario with multiple parameters and gender balancing"""
        print("\n=== TEST: COMPLEX SCENARIO ===")

        # Define gender ratio with sum > 100% to provide flexibility
        gender_ratio = {"m": 0.55, "f": 0.55, "nb": 0.1}

        # Define multiple parameters with different priorities
        parameters = [
            {
                "name": "special_needs",
                "type": "binary",
                "strategy": "distribute",
                "priority": 1,  # Highest priority
            },
            {
                "name": "elementary_school",
                "type": "categorical",
                "strategy": "concentrate",
                "priority": 2,  # Second priority
            },
            {
                "name": "english_learner",
                "type": "binary",
                "strategy": "concentrate",
                "priority": 3,  # Lowest priority
            },
        ]

        # Generate test data
        students_data, class_sizes = self.generate_test_data(
            num_students=100, num_classes=4, params=parameters
        )

        # Add english_learner parameter (15% of students)
        for student in students_data:
            if "english_learner" not in students_data[student]:
                students_data[student]["english_learner"] = (
                    "yes" if random.random() < 0.15 else "no"
                )

        # Print dataset summary
        print("Dataset summary:")
        gender_counts = Counter(data["sex"] for data in students_data.values())
        print(f"Gender distribution: {dict(gender_counts)}")

        for param in parameters:
            if param["type"] == "binary":
                yes_count = sum(
                    1
                    for data in students_data.values()
                    if data.get(param["name"]) == "yes"
                )
                print(
                    f"{param['name']}: {yes_count} students with 'yes' ({yes_count/len(students_data):.2%})"
                )

        # Run the algorithm with time tracking
        start_time = time.time()
        result = combined_cp_solver(
            students_data,
            class_sizes,
            gender_ratio=gender_ratio,
            time_limit=60,
            factor_gender=True,
            parameters=parameters,
        )
        end_time = time.time()

        print(f"Total algorithm runtime: {end_time - start_time:.2f} seconds")

        # Analyze and validate results
        stats = self.analyze_results(
            students_data, result, gender_ratio=gender_ratio, parameters=parameters
        )

        # Assert key requirements
        self.assertEqual(
            sum(len(students) for students in result.values()), len(students_data)
        )
        self.assertEqual(stats["assignment_rate"], 1.0)

        # Verify every student is in exactly one class
        student_class_count = Counter()
        for class_id, students in result.items():
            for student in students:
                student_class_count[student] += 1

        for student, count in student_class_count.items():
            self.assertEqual(
                count, 1, f"Student {student} is assigned to {count} classes"
            )


if __name__ == "__main__":
    unittest.main()
