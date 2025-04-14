from ortools.sat.python import cp_model
import time
import networkx as nx
from community import best_partition
import itertools


def preprocess(students_data):
    """Preprocess student data to find community clusters"""
    start_time = time.time()
    G = nx.Graph()
    for student, data in students_data.items():
        for pref in data["prefs"]:
            G.add_edge(student, pref)
    partition = best_partition(G)
    clusters = {
        c: [node for node, cluster in partition.items() if cluster == c]
        for c in set(partition.values())
    }
    end_time = time.time()
    print(f"Preprocessing time: {end_time - start_time} seconds")
    return clusters


def cp_assign_classes(
    students_data,
    class_sizes,
    gender_ratio,
    time_limit=60,
    factor_gender=False,
    parameters=None,
):
    """
    Assign students to classes using CP-SAT solver

    Args:
        students_data: Dictionary with student data including preferences
        class_sizes: Dictionary with class names and their maximum sizes
        gender_ratio: Dictionary with gender keys and their target ratios
        time_limit: Time limit in seconds for the solver
        factor_gender: Whether to enforce gender ratio constraints
        parameters: List of parameter dictionaries with:
            - 'name': Parameter name in student data
            - 'type': 'binary' (yes/no) or 'categorical' (text values)
            - 'strategy': 'concentrate' or 'distribute'
            - 'priority': Integer priority (lower = more important)
    """
    start_time = time.time()
    model = cp_model.CpModel()

    # Create decision variables
    # Binary variable for each student-class assignment
    assigned = {}
    for student in students_data:
        for class_id in class_sizes:
            assigned[(student, class_id)] = model.NewBoolVar(
                f"student_{student}_class_{class_id}"
            )

    # Each student assigned to exactly one class
    for student in students_data:
        model.Add(sum(assigned[(student, class_id)] for class_id in class_sizes) == 1)

    # Class size constraints
    for class_id, size in class_sizes.items():
        model.Add(
            sum(assigned[(student, class_id)] for student in students_data) <= size
        )

    # Gender constraints if enabled
    if factor_gender:
        for class_id, size in class_sizes.items():
            for gender, ratio in gender_ratio.items():
                max_count = int(ratio * size)
                gender_count = sum(
                    assigned[(student, class_id)]
                    for student in students_data
                    if students_data[student]["sex"] == gender
                )
                model.Add(gender_count <= max_count)

    # User-defined parameter constraints
    if parameters:
        # Sort parameters by priority
        sorted_params = sorted(parameters, key=lambda p: p.get("priority", 999))

        for param in sorted_params:
            param_name = param["name"]
            param_type = param.get("type", "binary")
            strategy = param.get("strategy", "distribute")

            if param_type == "binary":
                # Handle yes/no parameters
                if strategy == "concentrate":
                    # Find students with "yes" value for this parameter
                    yes_students = [
                        student
                        for student in students_data
                        if students_data[student].get(param_name) == "yes"
                    ]

                    # Target class for concentration (first class by default)
                    target_class = list(class_sizes.keys())[0]

                    # Maximize students with this parameter in the target class
                    model.Maximize(
                        sum(
                            assigned[(student, target_class)]
                            for student in yes_students
                        )
                    )
                else:  # 'distribute'
                    # Count students with "yes" value
                    yes_students = [
                        student
                        for student in students_data
                        if students_data[student].get(param_name) == "yes"
                    ]
                    yes_count = len(yes_students)

                    # Calculate even distribution
                    for class_id, size in class_sizes.items():
                        target_count = yes_count // len(class_sizes)

                        # Limit number of students with parameter in each class
                        param_count = sum(
                            assigned[(student, class_id)] for student in yes_students
                        )
                        model.Add(param_count <= target_count + 1)
                        model.Add(param_count >= target_count - 1)

            elif param_type == "categorical":
                # Handle categorical parameters (e.g., elementary school)
                # Get all unique values for this parameter
                value_counts = {}
                for student in students_data:
                    value = students_data[student].get(param_name)
                    if value:
                        if value not in value_counts:
                            value_counts[value] = []
                        value_counts[value].append(student)

                if strategy == "concentrate":
                    # Try to keep students with the same value together
                    for value, students_with_value in value_counts.items():
                        for class_id in class_sizes:
                            # Create helper variables for each value-class pair
                            value_in_class = model.NewBoolVar(
                                f"{param_name}_{value}_in_{class_id}"
                            )

                            # If any student with this value is in the class, set value_in_class to 1
                            model.Add(
                                sum(
                                    assigned[(student, class_id)]
                                    for student in students_with_value
                                )
                                > 0
                            ).OnlyEnforceIf(value_in_class)

                            model.Add(
                                sum(
                                    assigned[(student, class_id)]
                                    for student in students_with_value
                                )
                                == 0
                            ).OnlyEnforceIf(value_in_class.Not())

                    # Minimize the number of classes that have each value
                    model.Minimize(
                        sum(
                            model.NewBoolVar(f"{param_name}_{value}_in_{class_id}")
                            for value in value_counts
                            for class_id in class_sizes
                        )
                    )

                else:  # 'distribute'
                    # Distribute students with the same value across classes
                    for value, students_with_value in value_counts.items():
                        if len(students_with_value) >= len(class_sizes):
                            # Ensure each class has roughly equal number of students with this value
                            target = len(students_with_value) // len(class_sizes)
                            for class_id in class_sizes:
                                value_count = sum(
                                    assigned[(student, class_id)]
                                    for student in students_with_value
                                )
                                model.Add(value_count <= target + 1)
                                model.Add(value_count >= target - 1)

    # Objective: maximize friend preferences being in the same class
    preference_pairs = []

    for student1 in students_data:
        for friend in students_data[student1]["prefs"]:
            if friend in students_data:  # Ensure friend exists in student data
                for class_id in class_sizes:
                    # Create variable that's 1 if both student and friend are in the same class
                    pair_in_class = model.NewBoolVar(
                        f"pair_{student1}_{friend}_in_{class_id}"
                    )

                    # Link with student assignments
                    model.AddBoolAnd(
                        [assigned[(student1, class_id)], assigned[(friend, class_id)]]
                    ).OnlyEnforceIf(pair_in_class)

                    model.AddBoolOr(
                        [
                            assigned[(student1, class_id)].Not(),
                            assigned[(friend, class_id)].Not(),
                        ]
                    ).OnlyEnforceIf(pair_in_class.Not())

                    # Mutual preference gets higher weight
                    weight = 2 if student1 in students_data[friend]["prefs"] else 1
                    preference_pairs.append((pair_in_class, weight))

    # Set the objective - maximize preference satisfaction
    model.Maximize(sum(weight * pair for pair, weight in preference_pairs))

    # Create solver and set time limit
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit

    # Solve the model
    solve_start = time.time()
    status = solver.Solve(model)
    solve_end = time.time()
    print(f"Solve time: {solve_end - solve_start} seconds")

    # Extract solution
    class_assignments = {class_id: [] for class_id in class_sizes}

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        # Assign students based on solver solution
        for student in students_data:
            assigned_class = None
            for class_id in class_sizes:
                if solver.Value(assigned[(student, class_id)]) == 1:
                    class_assignments[class_id].append(student)
                    assigned_class = class_id
                    break

            # Handle any unassigned students (should not happen with our constraints)
            if assigned_class is None:
                # Find best class based on preferences
                best_class = None
                best_score = -1

                for class_id, students in class_assignments.items():
                    if len(students) < class_sizes[class_id]:
                        # Calculate how many friends are in this class
                        score = len(
                            set(students_data[student]["prefs"]) & set(students)
                        )
                        if score > best_score:
                            best_score = score
                            best_class = class_id

                if best_class:
                    class_assignments[best_class].append(student)
                else:
                    # If all classes are full, find the least full
                    min_class = min(
                        class_assignments, key=lambda c: len(class_assignments[c])
                    )
                    class_assignments[min_class].append(student)
    else:
        print(f"No solution found. Solver status: {solver.StatusName(status)}")
        # Implement a fallback greedy assignment
        fallback_assignments = greedy_fallback_assignment(students_data, class_sizes)
        return fallback_assignments

    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

    return class_assignments


def greedy_fallback_assignment(students_data, class_sizes):
    """Fallback greedy assignment when CP-SAT can't find a solution"""
    print("Using greedy fallback assignment")

    class_assignments = {class_id: [] for class_id in class_sizes}
    remaining_students = list(students_data.keys())

    # First pass: assign based on mutual preferences
    for student in list(remaining_students):
        best_class = None
        best_score = -1

        for class_id, assigned_students in class_assignments.items():
            if len(assigned_students) >= class_sizes[class_id]:
                continue

            # Calculate preference score (mutual preferences get higher weight)
            score = sum(
                (
                    2
                    if friend in assigned_students
                    and student in students_data[friend]["prefs"]
                    else 1 if friend in assigned_students else 0
                )
                for friend in students_data[student]["prefs"]
            )

            if score > best_score:
                best_score = score
                best_class = class_id

        if best_class:
            class_assignments[best_class].append(student)
            remaining_students.remove(student)

    # Second pass: assign remaining students to balance class sizes
    for student in remaining_students:
        # Find class with most room left
        best_class = min(
            class_assignments.keys(),
            key=lambda c: len(class_assignments[c]) / class_sizes[c],
        )
        class_assignments[best_class].append(student)

    return class_assignments


def detect_preference_type(dataset, threshold=0.75):
    """Detect if preferences form clusters or are random"""
    total_pairs = 0
    total_similarity = 0

    for preferences in dataset.values():
        for pair in itertools.combinations(preferences["prefs"], 2):
            total_pairs += 1
            if pair[0] in dataset.get(pair[1], {"prefs": []})["prefs"]:
                total_similarity += 1

    if total_pairs == 0:
        return 0
    else:
        similarity = total_similarity / total_pairs
        return (
            "Friend Groups (Clusters)"
            if similarity >= threshold
            else "Random Preferences"
        )


def combined_cp_solver(
    students_data,
    class_sizes,
    gender_ratio,
    time_limit=60,
    factor_gender=False,
    parameters=None,
):
    """Main solver function that combines preprocessing with CP-SAT"""

    # Detect preference type
    preference_type = detect_preference_type(students_data)
    print(f"Detected preference type: {preference_type}")

    # Use CP-SAT solver
    return cp_assign_classes(
        students_data, class_sizes, gender_ratio, time_limit, factor_gender, parameters
    )


# Example usage of the new function:
"""
To use this solver:

parameters = [
    {
        'name': 'special_needs',  # Parameter name in student data
        'type': 'binary',         # 'binary' for yes/no or 'categorical' for text
        'strategy': 'distribute', # 'distribute' or 'concentrate'
        'priority': 1             # Lower number = higher priority
    },
    {
        'name': 'elementary_school',
        'type': 'categorical',
        'strategy': 'distribute',
        'priority': 2
    }
]

result = combined_cp_solver(
    students_data,
    class_sizes,
    {'m': 0.5, 'f': 0.5, 'nb': 0.05},  # Gender ratio
    time_limit=120,                    # 2 minutes max
    factor_gender=True,
    parameters=parameters
)
"""
