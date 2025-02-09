import networkx as nx


def schedule_meetings_optimal(
    time_slots, teachers, parent_preferences, preferred_reward=10, drop_penalty=1000
):
    """
    Computes an optimal assignment of meeting requests to timeslots via a min–cost flow network.

    Each meeting request (a parent-teacher pair) is assigned a timeslot if possible.
    A negative cost (i.e. reward) is given for assigning a meeting to a parent's preferred timeslot.

    If a meeting cannot be scheduled in any timeslot (due to conflicts), it will be dropped at a high penalty cost.

    Conflict constraints:
      - A teacher can only meet one parent per timeslot.
      - A parent can only have one meeting per timeslot.

    Returns:
      schedule: A dict mapping (parent, teacher) -> assigned timeslot for those meetings that are scheduled.
                (Dropped meetings will simply not appear in this dict.)
      total_reward: The total reward (taking into account drop penalties)
      unscheduled: A list of (parent, teacher) meeting requests that were dropped.
    """
    G = nx.DiGraph()

    # 1. Identify all meeting requests.
    meeting_requests = []  # Each element is a tuple (parent, teacher)
    for parent, prefs in parent_preferences.items():
        for teacher in prefs["teachers"]:
            meeting_requests.append((parent, teacher))
    total_requests = len(meeting_requests)

    # 2. Set up supply and demand.
    source = "source"
    sink = "sink"
    # The source supplies one unit for each meeting request.
    G.add_node(source, demand=-total_requests)
    # The sink must absorb total_requests units.
    G.add_node(sink, demand=total_requests)

    # 3. Build the network.
    for p, t in meeting_requests:
        m_node = f"M_{p}_{t}"
        G.add_node(m_node, demand=0)
        # Edge from source to meeting node.
        G.add_edge(source, m_node, capacity=1, weight=0)

        # For each timeslot, create a candidate route.
        for r in time_slots:
            # Cost: negative reward if timeslot is preferred, else 0.
            cost = (
                -preferred_reward
                if r in parent_preferences[p]["preferred_slots"]
                else 0
            )
            a_node = f"A_{p}_{t}_{r}"
            b_node = f"B_{p}_{t}_{r}"
            G.add_node(a_node, demand=0)
            G.add_node(b_node, demand=0)

            # Edge from meeting node to candidate node.
            G.add_edge(m_node, a_node, capacity=1, weight=cost)

            # Parent gadget: ensure parent p uses timeslot r at most once.
            p_in = f"P_{p}_{r}_in"
            if p_in not in G:
                G.add_node(p_in, demand=0)
            G.add_edge(a_node, p_in, capacity=1, weight=0)

            p_out = f"P_{p}_{r}_out"
            if p_out not in G:
                G.add_node(p_out, demand=0)
                G.add_edge(p_in, p_out, capacity=1, weight=0)

            G.add_edge(p_out, b_node, capacity=1, weight=0)

            # Teacher gadget: ensure teacher t uses timeslot r at most once.
            t_in = f"T_{t}_{r}_in"
            if t_in not in G:
                G.add_node(t_in, demand=0)
            G.add_edge(b_node, t_in, capacity=1, weight=0)

            t_out = f"T_{t}_{r}_out"
            if t_out not in G:
                G.add_node(t_out, demand=0)
                G.add_edge(t_in, t_out, capacity=1, weight=0)

            # Edge from teacher's gadget to sink.
            G.add_edge(t_out, sink, capacity=1, weight=0)

        # 4. Add a "drop" edge directly from the meeting node to the sink.
        # This edge allows the meeting to be dropped if it cannot be scheduled.
        G.add_edge(m_node, sink, capacity=1, weight=drop_penalty)

    # 5. Compute the min–cost flow.
    try:
        flowCost, flowDict = nx.network_simplex(G)
    except nx.NetworkXUnfeasible as e:
        raise Exception(
            "The flow problem is unfeasible. This may indicate that the conflict constraints are too tight or that the demands are mismatched."
        ) from e

    total_reward = (
        -flowCost
    )  # Note: drop edges add a positive cost, reducing total reward.

    # 6. Convert the flow into a schedule.
    schedule = {}
    unscheduled = []
    for p, t in meeting_requests:
        m_node = f"M_{p}_{t}"
        scheduled = False
        # Check all candidate routes for timeslots.
        for r in time_slots:
            a_node = f"A_{p}_{t}_{r}"
            if a_node in flowDict[m_node] and flowDict[m_node][a_node] > 0:
                schedule[(p, t)] = r
                scheduled = True
                break
        # If not scheduled via any timeslot, it must have gone the drop route.
        if not scheduled:
            unscheduled.append((p, t))

    return schedule, total_reward, unscheduled


# Example usage:
if __name__ == "__main__":
    schedule, total_reward, unscheduled = schedule_meetings_optimal(
        time_slots, teachers, parent_preferences, preferred_reward=10, drop_penalty=1000
    )

    print("--- Scheduled Meetings ---")
    for (p, t), slot in schedule.items():
        is_preferred = "✓" if slot in parent_preferences[p]["preferred_slots"] else ""
        print(f"Parent: {p:<8} Teacher: {t:<15} Timeslot: {slot:<6} {is_preferred}")

    print("\nTotal reward (after penalties):", total_reward)
    print(
        "Total meeting requests:",
        len(parent_preferences["Parent1"])
        + len(parent_preferences["Parent2"])
        + len(parent_preferences["Parent3"]),
    )  # just an example count
    print("Meetings scheduled:", len(schedule))
    print("Meetings dropped:", len(unscheduled))
    if unscheduled:
        print("Dropped meeting requests:")
        for req in unscheduled:
            print("  ", req)
