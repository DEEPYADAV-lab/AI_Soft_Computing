import numpy as np
import matplotlib.pyplot as plt

def triangular(x, a, b, c):
    x = np.asarray(x, dtype=float)
    left = (x - a) / (b - a) if b != a else np.zeros_like(x)
    right = (c - x) / (c - b) if c != b else np.zeros_like(x)
    return np.maximum(0, np.minimum(left, right))

def trapezoidal(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b > a:
        mask = (x >= a) & (x < b)
        y[mask] = (x[mask] - a) / (b - a)
    y[(x >= b) & (x <= c)] = 1
    if d > c:
        mask = (x > c) & (x <= d)
        y[mask] = (d - x[mask]) / (d - c)
    return y

def gaussian(x, mean, sigma):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

if __name__ == "__main__":
    x = np.linspace(0, 10, 500)
    plt.plot(x, triangular(x, 1, 5, 9), label="Triangular")
    plt.plot(x, trapezoidal(x, 1, 3, 6, 9), label="Trapezoidal")
    plt.plot(x, gaussian(x, 5, 1.5), label="Gaussian")
    plt.title("Fuzzy Membership Functions")
    plt.xlabel("Input")
    plt.ylabel("Membership degree")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.show()
