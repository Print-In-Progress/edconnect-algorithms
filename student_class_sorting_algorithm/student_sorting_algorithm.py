import networkx as nx
import pulp
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import time
from community import best_partition
import itertools


def preprocess(students_data):
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


def ilp_assign_classes_with_preprocessing(
    students_data,
    class_sizes,
    gender_ratio,
    time_limit=None,
    factor_gender=False,
    optional_parameter=None,
    parameter_concentration=None,
    second_optional_parameter=None,
    second_parameter_concentration=None,
):
    start_time = time.time()
    clusters = preprocess(students_data)
    ilp_model = LpProblem("Class_Assignment", LpMaximize)
    assigned = {
        (student, clazz): LpVariable(f"assign_{student}_to_{clazz}", cat="Binary")
        for student in students_data.keys()
        for clazz in class_sizes.keys()
    }
    cluster_vars = {
        (student1, student2, clazz): LpVariable(
            f"pair_{student1}_{student2}_in_{clazz}", cat="Binary"
        )
        for cluster in clusters.values()
        for student1 in cluster
        for student2 in cluster
        for clazz in class_sizes.keys()
        if student1 != student2
    }

    ilp_model += lpSum(
        len(
            set(students_data[student1]["prefs"])
            & set(students_data[student2]["prefs"])
        )
        * cluster_vars[student1, student2, clazz]
        for cluster in clusters.values()
        for student1 in cluster
        for student2 in cluster
        for clazz in class_sizes.keys()
        if student1 != student2
    )

    for student in students_data.keys():
        ilp_model += (
            lpSum(assigned[student, clazz] for clazz in class_sizes.keys()) <= 1
        )

    for clazz, size in class_sizes.items():
        ilp_model += (
            lpSum(assigned[student, clazz] for student in students_data.keys()) <= size
        )

    for cluster in clusters.values():
        for student1 in cluster:
            for student2 in cluster:
                for clazz in class_sizes.keys():
                    if student1 != student2:
                        ilp_model += (
                            cluster_vars[student1, student2, clazz]
                            <= assigned[student1, clazz]
                        )
                        ilp_model += (
                            cluster_vars[student1, student2, clazz]
                            <= assigned[student2, clazz]
                        )
                        ilp_model += (
                            cluster_vars[student1, student2, clazz]
                            >= assigned[student1, clazz] + assigned[student2, clazz] - 1
                        )

    if factor_gender:
        for clazz in class_sizes.keys():
            ilp_model += (
                lpSum(
                    assigned[student, clazz]
                    for student in students_data.keys()
                    if students_data[student]["sex"] == "m"
                )
                <= gender_ratio["m"] * class_sizes[clazz]
            )
            ilp_model += (
                lpSum(
                    assigned[student, clazz]
                    for student in students_data.keys()
                    if students_data[student]["sex"] == "f"
                )
                <= gender_ratio["f"] * class_sizes[clazz]
            )

    # Handle additional parameter if provided
    if optional_parameter is not None:
        if parameter_concentration:
            # Concentrate students with optional_parameter in one class
            ilp_model += (
                lpSum(
                    assigned[student, clazz]
                    for student, data in students_data.items()
                    for clazz in class_sizes.keys()
                    if data.get(optional_parameter) == "yes"
                )
                <= class_sizes[list(class_sizes.keys())[0]]
            )
        else:
            # Spread students with optional_parameter evenly across classes
            for clazz in class_sizes.keys():
                ilp_model += lpSum(
                    assigned[student, clazz]
                    for student, data in students_data.items()
                    if data.get(optional_parameter) == "yes"
                ) <= class_sizes[clazz] / len(class_sizes)

    if second_optional_parameter is not None:
        if second_parameter_concentration:
            ilp_model += (
                lpSum(
                    assigned[student, clazz]
                    for student, data in students_data.items()
                    for clazz in class_sizes.keys()
                    if data.get(second_optional_parameter) == "yes"
                )
                <= class_sizes[list(class_sizes.keys())[0]]
            )
        else:
            for clazz in class_sizes.keys():
                ilp_model += lpSum(
                    assigned[student, clazz]
                    for student, data in students_data.items()
                    if data.get(second_optional_parameter) == "yes"
                ) <= class_sizes[clazz] / len(class_sizes)

    solve_start_time = time.time()
    ilp_model.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit))
    solve_end_time = time.time()
    print(f"Solve time: {solve_end_time - solve_start_time} seconds")

    class_assignments = {
        clazz: [
            student
            for student in students_data.keys()
            if assigned[student, clazz].value() == 1
        ]
        for clazz in class_sizes.keys()
    }

    unassigned_students = [
        student
        for student in students_data.keys()
        if not any(
            assigned[student, clazz].value() == 1 for clazz in class_sizes.keys()
        )
    ]
    for student in unassigned_students:
        max_pref = float("-inf")
        max_class = None
        for clazz, students in class_assignments.items():
            pref = len(set(students_data[student]["prefs"]) & set(students))
            if pref > max_pref and len(students) < class_sizes[clazz]:
                max_pref = pref
                max_class = clazz
        if max_class:
            class_assignments[max_class].append(student)

    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

    return class_assignments, ilp_model.sol_status


def ilp_assign_classes_less_constraints_with_preprocessing(
    students_data,
    class_sizes,
    gender_ratio,
    factor_gender=False,
    optional_parameter=None,
    parameter_concentration=None,
    second_optional_parameter=None,
    second_parameter_concentration=None,
):
    ilp_model = LpProblem(
        "Class_Assignment",
        LpMaximize,
    )

    assigned = {
        (student, clazz): LpVariable(f"assign_{student}_to_{clazz}", cat="Binary")
        for student in students_data.keys()
        for clazz in class_sizes.keys()
    }
    common_assigned = {
        (student1, student2, clazz): LpVariable(
            f"common_assign_{student1}_and_{student2}_to_{clazz}", cat="Binary"
        )
        for student1 in students_data.keys()
        for student2 in students_data.keys()
        for clazz in class_sizes.keys()
        if student1 != student2
    }

    ilp_model += lpSum(
        common_assigned[student1, student2, clazz]
        * len(
            set(students_data[student1]["prefs"])
            & set(students_data[student2]["prefs"])
        )
        for student1 in students_data.keys()
        for student2 in students_data.keys()
        for clazz in class_sizes.keys()
        if student1 != student2
    )

    for student in students_data.keys():
        ilp_model += (
            lpSum(assigned[student, clazz] for clazz in class_sizes.keys()) <= 1
        )

    for clazz, size in class_sizes.items():
        ilp_model += (
            lpSum(assigned[student, clazz] for student in students_data.keys()) <= size
        )

    for student1 in students_data.keys():
        for student2 in students_data[student1]["prefs"]:
            for clazz in class_sizes.keys():
                common_pref = len(
                    set(students_data[student1]["prefs"])
                    & set(students_data[student2]["prefs"])
                )
                ilp_model += (
                    common_assigned[student1, student2, clazz]
                    <= assigned[student1, clazz]
                )
                ilp_model += (
                    common_assigned[student1, student2, clazz]
                    <= assigned[student2, clazz]
                )
                ilp_model += (
                    common_assigned[student1, student2, clazz]
                    >= assigned[student1, clazz] + assigned[student2, clazz] - 1
                )
                ilp_model += common_assigned[student1, student2, clazz] * common_pref

    if factor_gender:
        for clazz in class_sizes.keys():
            ilp_model += (
                lpSum(
                    assigned[student, clazz]
                    for student in students_data.keys()
                    if students_data[student]["sex"] == "m"
                )
                <= gender_ratio["m"] * class_sizes[clazz]
            )
            ilp_model += (
                lpSum(
                    assigned[student, clazz]
                    for student in students_data.keys()
                    if students_data[student]["sex"] == "f"
                )
                <= gender_ratio["f"] * class_sizes[clazz]
            )

    # Handle additional parameter if provided
    if optional_parameter is not None:
        if parameter_concentration:
            # Concentrate students with optional_parameter in one class
            ilp_model += (
                lpSum(
                    assigned[student, clazz]
                    for student, data in students_data.items()
                    for clazz in class_sizes.keys()
                    if data.get(optional_parameter) == "yes"
                )
                <= class_sizes[list(class_sizes.keys())[0]]
            )
        else:
            # Spread students with optional_parameter evenly across classes
            for clazz in class_sizes.keys():
                ilp_model += lpSum(
                    assigned[student, clazz]
                    for student, data in students_data.items()
                    if data.get(optional_parameter) == "yes"
                ) <= class_sizes[clazz] / len(class_sizes)

    if second_optional_parameter is not None:
        if second_parameter_concentration:
            ilp_model += (
                lpSum(
                    assigned[student, clazz]
                    for student, data in students_data.items()
                    for clazz in class_sizes.keys()
                    if data.get(second_optional_parameter) == "yes"
                )
                <= class_sizes[list(class_sizes.keys())[0]]
            )
        else:
            for clazz in class_sizes.keys():
                ilp_model += lpSum(
                    assigned[student, clazz]
                    for student, data in students_data.items()
                    if data.get(second_optional_parameter) == "yes"
                ) <= class_sizes[clazz] / len(class_sizes)

    ilp_model.solve()

    class_assignments = {
        clazz: [
            student
            for student in students_data.keys()
            if assigned[student, clazz].value() == 1
        ]
        for clazz in class_sizes.keys()
    }

    for student in students_data.keys():
        assigned_class = False
        for clazz in class_sizes.keys():
            if assigned[student, clazz].value() == 1:
                assigned_class = True
                break
        if not assigned_class:
            max_pref = float("-inf")
            max_class = None
            for clazz, students in class_assignments.items():
                pref = len(set(students_data[student]["prefs"]) & set(students))
                if pref > max_pref and len(students) < class_sizes[clazz]:
                    max_pref = pref
                    max_class = clazz
            if max_class:
                class_assignments[max_class].append(student)

    return class_assignments


def average_similarity(dataset):
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
        return total_similarity / total_pairs


def detect_preference_type(dataset, threshold=0.75):
    avg_similarity_within_clusters = average_similarity(dataset)
    print(avg_similarity_within_clusters)
    if avg_similarity_within_clusters >= threshold:
        return "Friend Groups (Clusters)"
    else:
        return "Random Preferences"


def combined_ilp_solver_rand_detection(
    students,
    class_sizes,
    gender_ratio,
    time_limit,
    factor_gender=False,
    optional_parameter=None,
    parameter_concentration=None,
    second_optional_parameter=None,
    second_parameter_concentration=None,
):
    preference_type = detect_preference_type(students)
    print("\033[93m" + preference_type + "\033[0m")
    print(preference_type)
    if preference_type == "Friend Groups (Clusters)":
        initial_attempt = ilp_assign_classes_with_preprocessing(
            students,
            class_sizes,
            gender_ratio,
            time_limit,
            factor_gender,
            optional_parameter,
            parameter_concentration,
            second_optional_parameter,
            second_parameter_concentration,
        )
        if initial_attempt[1] == 1:
            return initial_attempt[0]
        if initial_attempt[1] == 2:
            print("\033[93m" + "Optimal Solution Not Found" + "\033[0m")
            print("Checking current solution")
            class_size_met = True
            for clazz, class_members in initial_attempt[0].items():
                if not class_sizes[clazz] >= len(class_members):
                    class_size_met = False
            if class_size_met:
                print("\033[93m" + "Current Solution is viable" + "\033[0m")
                return initial_attempt[0]
            else:
                print("Current Solution is not viable. Initializing Backup Solver")
                new_solution = ilp_assign_classes_less_constraints_with_preprocessing(
                    students,
                    class_sizes,
                    gender_ratio,
                    factor_gender,
                    optional_parameter,
                    parameter_concentration,
                    second_optional_parameter,
                    second_parameter_concentration,
                )
                return new_solution
    else:
        solution = ilp_assign_classes_less_constraints_with_preprocessing(
            students,
            class_sizes,
            gender_ratio,
            factor_gender,
            optional_parameter,
            parameter_concentration,
            second_optional_parameter,
            second_parameter_concentration,
        )
        return solution
