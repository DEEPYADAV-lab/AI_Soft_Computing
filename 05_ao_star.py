# AO* search on a small AND-OR graph.
# Each node maps to alternatives (OR choices); each alternative is a list
# of children that must ALL be solved (an AND choice).
# This compact example assumes an acyclic graph and non-negative costs.

def ao_star(graph, edge_cost, heuristic, start):
    memo = {}
    solution = {}

    def solve(node):
        if node not in graph or not graph[node]:
            memo[node] = 0
            solution[node] = []
            return 0

        if node in memo:
            return memo[node]

        best_value = float("inf")
        best_children = None

        for option in graph[node]:
            option_cost = sum(edge_cost.get((node, child), 1) + solve(child)
                              for child in option)
            if option_cost < best_value:
                best_value = option_cost
                best_children = option

        memo[node] = best_value
        solution[node] = best_children
        return best_value

    total_cost = solve(start)
    return total_cost, solution

if __name__ == "__main__":
    # A has two OR alternatives: (B AND C) OR (D).
    # B, C and D are terminal nodes.
    graph = {
        "A": [["B", "C"], ["D"]],
        "B": [], "C": [], "D": []
    }
    edge_cost = {
        ("A", "B"): 1, ("A", "C"): 1, ("A", "D"): 4
    }
    cost, solution = ao_star(graph, edge_cost, {}, "A")
    print("Minimum solution cost:", cost)
    print("Selected children:")
    for node, children in solution.items():
        print(f"  {node}: {children}")
