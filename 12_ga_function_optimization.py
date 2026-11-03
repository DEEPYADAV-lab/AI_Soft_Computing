import random

# Maximize f(x) = x * sin(10*pi*x) + 1 on [0, 1].
# Real-valued chromosomes avoid binary decoding.

def objective(x):
    import math
    return x * math.sin(10 * math.pi * x) + 1

def genetic_algorithm(population_size=40, generations=100,
                      mutation_rate=0.15, lower=0.0, upper=1.0):
    population = [random.uniform(lower, upper) for _ in range(population_size)]

    for _ in range(generations):
        population.sort(key=objective, reverse=True)
        new_population = population[:2]  # elitism

        while len(new_population) < population_size:
            p1, p2 = random.choices(population[:population_size // 2], k=2)
            alpha = random.random()
            child = alpha * p1 + (1-alpha) * p2

            if random.random() < mutation_rate:
                child += random.gauss(0, 0.1)
            child = max(lower, min(upper, child))
            new_population.append(child)
        population = new_population

    best = max(population, key=objective)
    return best, objective(best)

if __name__ == "__main__":
    random.seed(42)
    x, value = genetic_algorithm()
    print(f"Best x: {x:.6f}")
    print(f"Maximum function value: {value:.6f}")
