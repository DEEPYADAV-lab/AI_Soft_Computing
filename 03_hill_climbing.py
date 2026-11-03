# Demonstration: maximize f(x) = -(x - 3)^2 + 10.
# Simple hill climbing accepts the first improving neighbor.
# Steepest-ascent checks all neighbors and selects the best one.

def f(x):
    return -(x - 3) ** 2 + 10

def simple_hill_climbing(start, step=0.1, max_iterations=1000):
    x = start
    for _ in range(max_iterations):
        current = f(x)
        candidates = [x - step, x + step]
        improved = False
        for candidate in candidates:
            if f(candidate) > current:
                x = candidate
                improved = True
                break
        if not improved:
            break
    return x, f(x)

def steepest_ascent_hill_climbing(start, step=0.1, max_iterations=1000):
    x = start
    for _ in range(max_iterations):
        candidates = [x - step, x + step]
        best = max(candidates, key=f)
        if f(best) <= f(x):
            break
        x = best
    return x, f(x)

if __name__ == "__main__":
    start = 0.0
    x1, y1 = simple_hill_climbing(start)
    x2, y2 = steepest_ascent_hill_climbing(start)
    print(f"Simple hill climbing: x = {x1:.2f}, f(x) = {y1:.4f}")
    print(f"Steepest-ascent hill climbing: x = {x2:.2f}, f(x) = {y2:.4f}")
