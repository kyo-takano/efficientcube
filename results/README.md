# Solutions

This directory contains the reported solutions obtained with the trained models (located at `../efficientcube/models`) by different beam widths.
Each problem has its own directory (`cube3`, `puzzle15`, `lightsout7`), and each file holds a set of results (`solutions`, `times`, and `num_nodes_generated`).

```python
import pickle

with open("./results/cube3/beam_width_262144.pkl", "rb") as f:
    data = pickle.load(f)
    solutions, times, num_nodes_generated = [data[k] for k in ['solutions', 'times', 'num_nodes_generated']]

mean_solution_length = sum([len(e) for e in solutions]) / len(solutions)
print(mean_solution_length)
```

Please note that we only include the *actual* (wall-clock) times taken per solution.
