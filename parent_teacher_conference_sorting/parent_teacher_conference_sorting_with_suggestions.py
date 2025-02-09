import networkx as nx


def schedule_meetings_optimal(
    time_slots, teachers, parent_preferences, preferred_reward=10
):
    """
    Computes an optimal assignment of meeting requests to timeslots via a min–cost flow network.

    Each meeting request (a parent-teacher pair) is assigned a timeslot.
    A negative cost (i.e. reward) is given for assigning a meeting to a parent's preferred timeslot.

    Conflict constraints:
      - A teacher can only meet one parent per timeslot.
      - A parent can only have one meeting per timeslot.

    Returns:
      schedule: A dict mapping (parent, teacher) -> timeslot
      total_reward: The total reward (sum of rewards for preferred assignments)
    """
    G = nx.DiGraph()

    # Identify all meeting requests.
    meeting_requests = []  # Each element is a tuple (parent, teacher)
    for parent, prefs in parent_preferences.items():
        for teacher in prefs["teachers"]:
            meeting_requests.append((parent, teacher))
    total_requests = len(meeting_requests)

    # Set supply and demand: source supplies all meeting requests; sink demands them.
    source = "source"
    sink = "sink"
    G.add_node(source, demand=-total_requests)
    G.add_node(sink, demand=total_requests)

    # Build the network:
    # For each meeting request (p,t), create a meeting node M_{p,t} with an edge from the source.
    # Then, for every timeslot r, create a candidate edge (via auxiliary nodes) with cost -preferred_reward
    # if r is in the parent's preferred timeslots, else cost 0. Then route through parent– and teacher–timeslot gadgets.
    for p, t in meeting_requests:
        m_node = f"M_{p}_{t}"
        G.add_node(m_node, demand=0)
        G.add_edge(source, m_node, capacity=1, weight=0)

        for r in time_slots:
            # The cost on the candidate edge is negative if timeslot r is preferred.
            cost = (
                -preferred_reward
                if r in parent_preferences[p]["preferred_slots"]
                else 0
            )
            a_node = f"A_{p}_{t}_{r}"  # candidate node for meeting (p,t) in timeslot r
            b_node = f"B_{p}_{t}_{r}"
            G.add_node(a_node, demand=0)
            G.add_node(b_node, demand=0)

            # Edge from meeting node to candidate node.
            G.add_edge(m_node, a_node, capacity=1, weight=cost)

            # Edge from candidate node to parent's timeslot node.
            p_in = f"P_{p}_{r}_in"
            if p_in not in G:
                G.add_node(p_in, demand=0)
            G.add_edge(a_node, p_in, capacity=1, weight=0)

            # Parent gadget: split parent's timeslot into in and out nodes to enforce capacity 1.
            p_out = f"P_{p}_{r}_out"
            if p_out not in G:
                G.add_node(p_out, demand=0)
                G.add_edge(p_in, p_out, capacity=1, weight=0)

            # Edge from parent's gadget to meeting branch.
            G.add_edge(p_out, b_node, capacity=1, weight=0)

            # Teacher gadget: similar splitting for teacher timeslot.
            t_in = f"T_{t}_{r}_in"
            if t_in not in G:
                G.add_node(t_in, demand=0)
            G.add_edge(b_node, t_in, capacity=1, weight=0)

            t_out = f"T_{t}_{r}_out"
            if t_out not in G:
                G.add_node(t_out, demand=0)
                G.add_edge(t_in, t_out, capacity=1, weight=0)

            # Finally, connect teacher's timeslot out–node to the sink.
            G.add_edge(t_out, sink, capacity=1, weight=0)

    # Compute min–cost flow (which maximizes the total reward).
    flowCost, flowDict = nx.network_simplex(G)
    total_reward = -flowCost  # since costs for preferred timeslots are negative

    # Convert the flow into a schedule: for each meeting request, find the candidate edge that carried flow.
    schedule = {}
    for p, t in meeting_requests:
        m_node = f"M_{p}_{t}"
        for r in time_slots:
            a_node = f"A_{p}_{t}_{r}"
            if a_node in flowDict[m_node] and flowDict[m_node][a_node] > 0:
                schedule[(p, t)] = r
                break

    return schedule, total_reward


def suggest_alternative_timeslots_for_meeting(
    parent, teacher, schedule, time_slots, parent_preferences
):
    """
    For a given meeting (parent, teacher), provide a list of alternative timeslot suggestions
    that are free for both the teacher and the parent if we were to try rescheduling that meeting.

    When checking for available timeslots, we discount the current assignment (i.e. assume that
    meeting could be moved).

    Returns:
      A sorted list of timeslots (ordered by parent's preference if available).
    """
    # The timeslot already assigned to this meeting.
    current_slot = schedule.get((parent, teacher))

    # Determine timeslots already occupied by the teacher (except the current meeting).
    teacher_booked = {
        slot
        for (p, t), slot in schedule.items()
        if t == teacher and not (p == parent and t == teacher)
    }

    # Determine timeslots already occupied by the parent (except the current meeting).
    parent_booked = {
        slot
        for (p, t), slot in schedule.items()
        if p == parent and not (p == parent and t == teacher)
    }

    # Alternative suggestions: timeslots that are not already booked for teacher or parent.
    available = [
        slot
        for slot in time_slots
        if slot not in teacher_booked and slot not in parent_booked
    ]

    # Sort the suggestions so that parent's preferred timeslots come first.
    preferred = parent_preferences[parent]["preferred_slots"]
    suggestions_sorted = sorted(
        available, key=lambda slot: (slot not in preferred, slot)
    )

    return suggestions_sorted


def generate_schedule_with_suggestions(
    time_slots, teachers, parent_preferences, preferred_reward=10
):
    """
    Compute the optimal schedule and, for each meeting request, provide alternative timeslot suggestions
    if the meeting was not scheduled in one of the parent's preferred timeslots.

    Returns:
      schedule: a dict mapping (parent, teacher) -> assigned timeslot
      total_reward: the overall reward (from preferred assignments)
      suggestions: a dict mapping (parent, teacher) -> list of alternative timeslots that are free
                   for both teacher and parent (these suggestions apply if the assigned slot isn't preferred)
    """
    schedule, total_reward = schedule_meetings_optimal(
        time_slots, teachers, parent_preferences, preferred_reward
    )

    suggestions = {}
    for (parent, teacher), assigned_slot in schedule.items():
        # If the assigned slot is already one of parent's preferred slots, no suggestion is needed.
        if assigned_slot in parent_preferences[parent]["preferred_slots"]:
            suggestions[(parent, teacher)] = []
        else:
            suggestions[(parent, teacher)] = suggest_alternative_timeslots_for_meeting(
                parent, teacher, schedule, time_slots, parent_preferences
            )
    return schedule, total_reward, suggestions


# --- Example usage / demonstration ---
if __name__ == "__main__":
    # Define sample inputs.
    time_slots = ["9:00", "9:30", "10:00", "10:30"]
    teachers = ["Math", "Science", "History", "English"]
    parent_preferences = {
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

    # Compute the schedule and suggestions.
    schedule, total_reward, suggestions = generate_schedule_with_suggestions(
        time_slots, teachers, parent_preferences, preferred_reward=10
    )

    # Print the schedule.
    print("\n--- Optimal Scheduling Solution ---")
    for (parent, teacher), slot in schedule.items():
        pref_mark = "✓" if slot in parent_preferences[parent]["preferred_slots"] else ""
        print(
            f"Parent: {parent:<8} Teacher: {teacher:<8} Timeslot: {slot:<5} {pref_mark}"
        )

    print("\nSummary:")
    total_meetings = len(schedule)
    preferred_count = sum(
        1
        for (p, t), slot in schedule.items()
        if slot in parent_preferences[p]["preferred_slots"]
    )
    print(f"  Total meeting requests (wishes): {total_meetings}")
    print(f"  Preferred timeslot assignments: {preferred_count}")
    print(f"  Total reward: {total_reward}")

    # Print suggestions for meetings that did not get a preferred timeslot.
    print("\n--- Alternative Timeslot Suggestions ---")
    for (parent, teacher), sugg in suggestions.items():
        if sugg:  # there are suggestions available
            print(
                f"Meeting {parent} - {teacher} (assigned: {schedule[(parent, teacher)]})"
            )
            print(f"  Alternative suggestions: {', '.join(sugg)}")
