# Datasets

For each problem, `deepcubea-dataset--{cube3|puzzle15|lightsout7}.json` stores space-delimited scrambles for the 1000|500 test cases generated for [DeepCubeA (Agostinelli, et al., 2019)](https://github.com/forestagostinelli/DeepCubeA). All these scrambles were generated simply by reversing the optimal solutions they released under https://github.com/forestagostinelli/DeepCubeA/blob/master/data/

We provide these files to make it easy to access the data, as the original datasets have dependencies on the codebase to which it belongs, requiring a redundant `git clone`.

**Example usage**:

```python
import json
with open('../data/deepcubea-dataset--cube3.json') as f:
    test_scrambles = json.load(f)


print(test_scrambles[0].split())
# ['B', "U'", 'R', 'U', 'U', 'R', 'U', 'D', 'R', 'F', 'R', 'B', "R'", "F'", "U'", "R'", "F'", "F'", 'U', "R'", "F'", 'D']
```
