"""
Heart disease classification using an Artificial Neural Network (MLP).
For a real experiment, put a heart.csv file in this script's folder.
Expected target column: 'target' or 'HeartDisease' (0 = no disease, 1 = disease).
If no CSV is found, a clearly labelled synthetic DEMO dataset is used so
the program can still be executed and understood in a lab.
"""
from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data():
    path = Path(__file__).with_name("heart.csv")
    if path.exists():
        data = pd.read_csv(path)
        target_col = "target" if "target" in data.columns else (
            "HeartDisease" if "HeartDisease" in data.columns else None)
        if target_col is None:
            raise ValueError("heart.csv must contain a 'target' or 'HeartDisease' column.")
        X = pd.get_dummies(data.drop(columns=[target_col]), drop_first=True)
        y = data[target_col].astype(int)
        print(f"Loaded real dataset file: {path}")
        return X, y

    print("WARNING: heart.csv not found. Running a SYNTHETIC demo, not medical data.")
    X, y = make_classification(n_samples=800, n_features=13, n_informative=8,
                               n_redundant=2, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X, pd.Series(y)

if __name__ == "__main__":
    X, y = load_data()
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ann = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                        solver="adam", max_iter=1000, random_state=42,
                        early_stopping=True)
    ann.fit(X_train, y_train)
    predictions = ann.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification report:\n", classification_report(y_test, predictions))
    print("Educational demonstration only; not for clinical diagnosis.")
