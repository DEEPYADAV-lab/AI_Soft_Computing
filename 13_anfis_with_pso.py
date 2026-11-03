"""
ANFIS with PSO: PSO tunes the four Gaussian premise centers (two per input).
For each particle, Sugeno consequent coefficients are fitted by least squares.
The particle fitness is mean squared error on the training data.
"""
import numpy as np

def gaussian(x, center, sigma=0.22):
    return np.exp(-0.5 * ((x-center)/sigma)**2)

def design_matrix(X, centers):
    c1, c2, c3, c4 = centers
    rows = []
    for x1, x2 in X:
        firing = [
            gaussian(x1, c1)*gaussian(x2, c3),
            gaussian(x1, c1)*gaussian(x2, c4),
            gaussian(x1, c2)*gaussian(x2, c3),
            gaussian(x1, c2)*gaussian(x2, c4)
        ]
        total = sum(firing) + 1e-12
        row = []
        for w in firing:
            w /= total
            row.extend([w*x1, w*x2, w])
        rows.append(row)
    return np.asarray(rows)

def evaluate_particle(X, y, centers):
    matrix = design_matrix(X, centers)
    coefficients, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    predictions = matrix @ coefficients
    mse = np.mean((y-predictions)**2)
    return mse, coefficients

def pso_anfis(X, y, particles=20, iterations=60, seed=42):
    rng = np.random.default_rng(seed)
    # Particle = [low center for x1, high center for x1,
    #             low center for x2, high center for x2]
    positions = rng.uniform(0.05, 0.95, size=(particles, 4))
    velocities = rng.uniform(-0.05, 0.05, size=(particles, 4))
    personal_best = positions.copy()
    personal_scores = np.array([evaluate_particle(X, y, p)[0] for p in positions])
    best_index = np.argmin(personal_scores)
    global_best = positions[best_index].copy()
    global_score = personal_scores[best_index]

    for _ in range(iterations):
        for i in range(particles):
            r1, r2 = rng.random(4), rng.random(4)
            velocities[i] = (0.7*velocities[i]
                             + 1.5*r1*(personal_best[i]-positions[i])
                             + 1.5*r2*(global_best-positions[i]))
            positions[i] = np.clip(positions[i] + velocities[i], 0.0, 1.0)
            score = evaluate_particle(X, y, positions[i])[0]

            if score < personal_scores[i]:
                personal_scores[i] = score
                personal_best[i] = positions[i].copy()
            if score < global_score:
                global_score = score
                global_best = positions[i].copy()

    mse, coefficients = evaluate_particle(X, y, global_best)
    return global_best, coefficients, mse

if __name__ == "__main__":
    rng = np.random.default_rng(10)
    X = rng.random((250, 2))
    y = 2*X[:, 0] + 3*X[:, 1] + 1

    centers, coefficients, mse = pso_anfis(X, y)
    print("Optimized Gaussian centers:", np.round(centers, 4))
    print("Sugeno consequent coefficients:", np.round(coefficients, 4))
    print(f"Training MSE: {mse:.8f}")
