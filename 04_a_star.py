import heapq

def a_star(graph, heuristic, start, goal):
    # Queue entries: (estimated_total_cost, cost_so_far, node, path)
    frontier = [(heuristic[start], 0, start, [start])]
    best_cost = {start: 0}

    while frontier:
        _, cost, node, path = heapq.heappop(frontier)
        if node == goal:
            return path, cost
        if cost > best_cost.get(node, float("inf")):
            continue

        for neighbor, edge_cost in graph.get(node, {}).items():
            new_cost = cost + edge_cost
            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                priority = new_cost + heuristic[neighbor]
                heapq.heappush(
                    frontier, (priority, new_cost, neighbor, path + [neighbor])
                )
    return None, float("inf")

if __name__ == "__main__":
    graph = {
        "A": {"B": 1, "C": 4},
        "B": {"A": 1, "C": 2, "D": 5},
        "C": {"A": 4, "B": 2, "D": 1},
        "D": {"B": 5, "C": 1, "E": 3},
        "E": {"D": 3}
    }
    heuristic = {"A": 6, "B": 5, "C": 3, "D": 2, "E": 0}
    path, cost = a_star(graph, heuristic, "A", "E")
    print("Path:", " -> ".join(path) if path else "No path found")
    print("Total path cost:", cost)
