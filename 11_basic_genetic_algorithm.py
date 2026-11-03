import random

def fitness(chromosome):
    # Maximize the number of 1s (the OneMax problem).
    return sum(chromosome)

def genetic_algorithm(population_size=20, chromosome_length=12,
                      generations=50, mutation_rate=0.02):
    population = [[random.randint(0, 1) for _ in range(chromosome_length)]
                  for _ in range(population_size)]

    for generation in range(generations):
        population.sort(key=fitness, reverse=True)
        if fitness(population[0]) == chromosome_length:
            break

        # Elitism: keep the best two chromosomes.
        new_population = population[:2]
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population[:10], k=2)
            point = random.randint(1, chromosome_length - 1)
            child = parent1[:point] + parent2[point:]

            for i in range(chromosome_length):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]
            new_population.append(child)
        population = new_population

    best = max(population, key=fitness)
    return best, fitness(best)

if __name__ == "__main__":
    random.seed(42)
    best, score = genetic_algorithm()
    print("Best chromosome:", best)
    print("Fitness (number of 1s):", score)
