"""
A compact first-order Sugeno ANFIS demonstration with two inputs and four rules.
Each rule has Gaussian membership functions. Consequent parameters are fitted
by least squares (hybrid-learning consequent step).
"""
import numpy as np

def gaussian(x, center, sigma):
    return np.exp(-0.5 * ((x-center)/sigma)**2)

def train_anfis(X, y, centers=(0.25, 0.75), sigma=0.25):
    # Four rules: (low,low), (low,high), (high,low), (high,high)
    design_rows = []
    for x1, x2 in X:
        memberships = [
            gaussian(x1, centers[0], sigma) * gaussian(x2, centers[0], sigma),
            gaussian(x1, centers[0], sigma) * gaussian(x2, centers[1], sigma),
            gaussian(x1, centers[1], sigma) * gaussian(x2, centers[0], sigma),
            gaussian(x1, centers[1], sigma) * gaussian(x2, centers[1], sigma)
        ]
        total = sum(memberships) + 1e-12
        normalized = [w / total for w in memberships]
        row = []
        for w in normalized:
            row.extend([w*x1, w*x2, w])
        design_rows.append(row)
    coefficients, *_ = np.linalg.lstsq(np.asarray(design_rows), y, rcond=None)
    return coefficients

def predict_anfis(X, coefficients, centers=(0.25, 0.75), sigma=0.25):
    predictions = []
    for x1, x2 in X:
        memberships = [
            gaussian(x1, centers[0], sigma) * gaussian(x2, centers[0], sigma),
            gaussian(x1, centers[0], sigma) * gaussian(x2, centers[1], sigma),
            gaussian(x1, centers[1], sigma) * gaussian(x2, centers[0], sigma),
            gaussian(x1, centers[1], sigma) * gaussian(x2, centers[1], sigma)
        ]
        total = sum(memberships) + 1e-12
        output = 0.0
        for i, w in enumerate(memberships):
            p, q, r = coefficients[3*i:3*i+3]
            output += (w/total) * (p*x1 + q*x2 + r)
        predictions.append(output)
    return np.asarray(predictions)

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    X = rng.random((200, 2))
    y = 2*X[:, 0] + 3*X[:, 1] + 1
    coefficients = train_anfis(X, y)
    predictions = predict_anfis(X, coefficients)
    mse = np.mean((y - predictions)**2)
    print("First 5 actual values:   ", np.round(y[:5], 3))
    print("First 5 predicted values:", np.round(predictions[:5], 3))
    print(f"Training MSE: {mse:.8f}")
