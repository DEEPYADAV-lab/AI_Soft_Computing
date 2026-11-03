import numpy as np

def defuzzify(x, membership):
    x = np.asarray(x, dtype=float)
    membership = np.asarray(membership, dtype=float)
    if len(x) != len(membership) or len(x) == 0:
        raise ValueError("x and membership must have the same non-zero length")
    if np.max(membership) <= 0:
        raise ValueError("Membership values must contain at least one positive value")

    # Centroid / Center of Area
    centroid = np.sum(x * membership) / np.sum(membership)

    # Mean, smallest, and largest of maxima
    maximum = np.max(membership)
    maxima = x[np.isclose(membership, maximum)]
    mom = np.mean(maxima)
    som = np.min(maxima)
    lom = np.max(maxima)

    # Bisector: find the point splitting the discrete area approximately in half.
    area = np.trapezoid(membership, x) if hasattr(np, "trapezoid") else np.trapz(membership, x)
    cumulative = np.zeros_like(x)
    for i in range(1, len(x)):
        cumulative[i] = cumulative[i-1] + (membership[i-1] + membership[i]) * (x[i]-x[i-1]) / 2
    bisector = np.interp(area / 2, cumulative, x) if area > 0 else centroid

    return {"Centroid": centroid, "Bisector": bisector,
            "Mean of maxima": mom, "Smallest of maxima": som,
            "Largest of maxima": lom}

if __name__ == "__main__":
    x = np.linspace(0, 10, 1001)
    # Example aggregated output membership function
    mu = np.maximum(np.minimum((x - 1) / 3, 1), 0)
    mu = np.maximum(mu, np.maximum(np.minimum((8 - x) / 3, 1), 0))
    for method, value in defuzzify(x, mu).items():
        print(f"{method}: {value:.3f}")
