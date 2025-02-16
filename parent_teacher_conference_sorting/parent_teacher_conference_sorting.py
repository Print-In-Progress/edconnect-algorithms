import networkx as nx


def schedule_meetings_aggregator(
    meeting_requests,
    teacher_slots,
    global_timeslots,
    drop_penalty=1000,
    reschedule_penalty=50,
    parent_bonus=20,
):
    """
    Schedules parent-teacher meetings using an aggregator (time-indexed bipartite) formulation.

    - Each parent p has an aggregator node A_p with supply equal to the number of meeting requests from p.
    - For each global timeslot t, a node P_{p,t} is created for parent p (capacity 1).
    - For each teacher T and timeslot t (in teacher_slots[T] ∩ global_timeslots), a node T_{T,t} is created (capacity 1).
    - For each meeting request from parent p to teacher T with preferred time, an edge is added from P_{p,t} to T_{T,t}
      (for each candidate timeslot t) with cost = 0 if t equals the preferred time, else reschedule_penalty.
    - Bonus edges between consecutive parent's timeslot nodes (P_{p,t} → P_{p,t_next}) with cost -parent_bonus encourage consecutive meetings.
    - Teacher aggregator nodes B_T (demand = number of meeting requests for T) collect flows from teacher slot nodes.
    - A drop edge from each parent aggregator A_p to sink (with cost = drop_penalty) allows dropping requests if needed.
    """
    G = nx.DiGraph()
    source = "source"
    sink = "sink"

    # Calculate parent's supply (number of meeting requests from that parent)
    parent_requests = {}
    # Also record the meeting request info for each (parent, teacher) pair (assumed unique)
    parent_teacher_pref = {}  # key: (parent, teacher) -> preferred timeslot
    teacher_request_count = {}
    for req in meeting_requests:
        p = req["parent"]
        T = req["teacher"]
        parent_requests[p] = parent_requests.get(p, 0) + 1
        parent_teacher_pref[(p, T)] = req["preferred"]
        teacher_request_count[T] = teacher_request_count.get(T, 0) + 1

    # Create source and sink nodes.
    G.add_node(source, demand=0)
    G.add_node(sink, demand=0)

    # Create parent aggregator nodes A_p and connect source → A_p.
    for p, supply in parent_requests.items():
        node = f"A_{p}"
        G.add_node(node, demand=0)
        # Edge from source to parent's aggregator: supply = number of meeting requests from p.
        G.add_edge(source, node, capacity=supply, weight=0)

    # Create parent timeslot nodes P_{p,t} for each parent and each global timeslot.
    parent_nodes = {}  # key: (p,t) -> node name
    for p in parent_requests:
        for t in global_timeslots:
            node = f"P_{p}_{t}"
            parent_nodes[(p, t)] = node
            # Each parent's slot can be used at most once.
            # We'll connect A_p to each P_{p,t} with capacity 1.
            G.add_edge(f"A_{p}", node, capacity=1, weight=0)

    # Add bonus edges between consecutive parent's timeslot nodes.
    for p in parent_requests:
        for i in range(len(global_timeslots) - 1):
            t = global_timeslots[i]
            t_next = global_timeslots[i + 1]
            node_from = parent_nodes[(p, t)]
            node_to = parent_nodes[(p, t_next)]
            # Bonus edge with capacity 1 and negative cost rewards consecutive use.
            G.add_edge(node_from, node_to, capacity=1, weight=-parent_bonus)

    # Create teacher aggregator nodes B_T and teacher timeslot nodes T_{T,t}.
    teacher_nodes = {}  # key: (T, t) -> node name
    for T, slots in teacher_slots.items():
        for t in slots:
            if t in global_timeslots:
                node = f"T_{T}_{t}"
                teacher_nodes[(T, t)] = node
                # Each teacher timeslot has capacity 1; connect to teacher aggregator later.
                G.add_node(node, demand=0)
    # Create teacher aggregator nodes B_T and connect teacher timeslot nodes → B_T.
    for T, count in teacher_request_count.items():
        node = f"B_{T}"
        G.add_node(node, demand=0)
        for t in teacher_slots[T]:
            if t in global_timeslots and (T, t) in teacher_nodes:
                # Edge from teacher timeslot node to aggregator B_T.
                G.add_edge(teacher_nodes[(T, t)], node, capacity=1, weight=0)
        # Connect teacher aggregator to sink.
        G.add_edge(node, sink, capacity=count, weight=0)

    # For each meeting request from parent p to teacher T with preferred time pref,
    # add edges from parent's timeslot nodes to teacher's timeslot nodes.
    # (We assume each parent–teacher pair appears at most once.)
    for (p, T), pref in parent_teacher_pref.items():
        # For candidate timeslot t, add edge from P_{p,t} to T_{T,t} (if available).
        for t in teacher_slots[T]:
            if (
                t in global_timeslots
                and (p, t) in parent_nodes
                and (T, t) in teacher_nodes
            ):
                cost = 0 if t == pref else reschedule_penalty
                G.add_edge(
                    parent_nodes[(p, t)], teacher_nodes[(T, t)], capacity=1, weight=cost
                )

    # Add drop edges: from parent's aggregator A_p directly to sink (to drop a meeting request) at high cost.
    # This ensures that if a parent's meeting cannot be scheduled, it can be dropped with penalty.
    for p, supply in parent_requests.items():
        G.add_edge(f"A_{p}", sink, capacity=supply, weight=drop_penalty)

    # Set overall demand: total supply = sum of parent's requests, total demand = same.
    total_requests = sum(parent_requests.values())
    G.nodes[source]["demand"] = -total_requests
    G.nodes[sink]["demand"] = total_requests

    # Compute min-cost flow.
    flowDict = nx.min_cost_flow(G)

    # Extract schedule: for each edge from a parent's timeslot node P_{p,t} to a teacher's timeslot node T_{T,t}
    # that carries flow, record that meeting as assigned.
    schedule = []
    nonpreferred = []
    # We know the meeting requests by (p,T) pairs from parent_teacher_pref.
    for (p, T), pref in parent_teacher_pref.items():
        # Check candidate timeslots.
        assigned_t = None
        for t in teacher_slots[T]:
            if t in global_timeslots:
                pnode = parent_nodes[(p, t)]
                tnode = teacher_nodes[(T, t)]
                # Check if flow goes from pnode to tnode.
                if (
                    pnode in flowDict
                    and tnode in flowDict[pnode]
                    and flowDict[pnode][tnode] > 0
                ):
                    assigned_t = t
                    break
        if assigned_t is not None:
            schedule.append(
                {
                    "parent": p,
                    "teacher": T,
                    "timeslot": assigned_t,
                    "preferred": pref,
                    "cost": 0 if assigned_t == pref else reschedule_penalty,
                }
            )
            if assigned_t != pref:
                nonpreferred.append((p, T))
        else:
            # Meeting request dropped.
            schedule.append(
                {
                    "parent": p,
                    "teacher": T,
                    "timeslot": None,
                    "preferred": pref,
                    "cost": drop_penalty,
                }
            )

    return schedule, nonpreferred, flowDict, G
