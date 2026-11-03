import numpy as np

def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    # Left shoulder when a == b; ordinary rising slope otherwise.
    if a == b:
        y[x <= b] = 1.0
    else:
        mask = (x >= a) & (x <= b)
        y[mask] = (x[mask] - a) / (b - a)
    # Right shoulder when b == c; ordinary falling slope otherwise.
    if b == c:
        y[x >= b] = 1.0
    else:
        mask = (x >= b) & (x <= c)
        y[mask] = np.maximum(y[mask], (c - x[mask]) / (c - b))
    return np.clip(y, 0.0, 1.0)

def mamdani_tip(service, food):
    # Inputs: service and food ratings from 0 to 10.
    # Output: tip percentage from 0 to 25.
    service_low = trimf(service, 0, 0, 5)
    service_high = trimf(service, 5, 10, 10)
    food_low = trimf(food, 0, 0, 5)
    food_high = trimf(food, 5, 10, 10)

    # AND = min; OR = max
    poor_rule = min(service_low, food_low)
    good_rule = max(service_high, food_high)

    output = np.linspace(0, 25, 501)
    low_tip = trimf(output, 0, 0, 13)
    high_tip = trimf(output, 12, 25, 25)

    # Mamdani implication: clipping; aggregation: maximum.
    aggregated = np.maximum(np.minimum(poor_rule, low_tip),
                            np.minimum(good_rule, high_tip))
    if aggregated.sum() == 0:
        return 0.0
    return float(np.sum(output * aggregated) / np.sum(aggregated))

if __name__ == "__main__":
    service = float(input("Enter service quality (0-10): "))
    food = float(input("Enter food quality (0-10): "))
    if not (0 <= service <= 10 and 0 <= food <= 10):
        raise ValueError("Both inputs must be between 0 and 10.")
    print(f"Recommended tip: {mamdani_tip(service, food):.2f}%")
