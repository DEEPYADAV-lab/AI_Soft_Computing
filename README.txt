# AI & Soft Computing Lab Programs (Python)

## Programs
01 BFS
02 DFS
03 Hill Climbing (simple and steepest-ascent)
04 A* search
05 AO* search (small AND-OR graph)
06 Fuzzy membership functions (triangular, trapezoidal, Gaussian)
07 Defuzzification (centroid, bisector, MOM, SOM, LOM)
08 Mamdani fuzzy inference system
09 Heart disease prediction using ANN (MLP)
10 ANFIS with first-order Sugeno inference
11 Basic Genetic Algorithm (OneMax)
12 Function optimization using Genetic Algorithm
13 ANFIS with PSO

## Install dependencies
Most programs use only Python's standard library. Install the libraries used by the fuzzy plots, ANN and ANFIS examples with:

    python -m pip install numpy matplotlib pandas scikit-learn

Run a program from this folder, for example:

    python 01_bfs.py

## Important notes
- Each numbered file is independent. Run one file at a time.
- Program 06 displays a graph using Matplotlib.
- Program 08 asks for service and food ratings from 0 to 10.
- Program 09 uses `heart.csv` in the same folder if available. The CSV must contain a `target` or `HeartDisease` column. If it is absent, the script runs a clearly labelled synthetic demonstration so it remains executable. Synthetic results are not heart-disease research results and must not be presented as real medical accuracy.
- Programs 10 and 13 use generated sample data to demonstrate ANFIS/Sugeno learning and PSO tuning. Replace the sample data with your lab dataset if one is prescribed.
- AO* is demonstrated on a small acyclic AND-OR graph to make the search logic easy to inspect.
