# MCTS_Tree

A concise, self-contained reference implementation of Monte Carlo Tree Search (MCTS) in Python.

## Contents

- `mcts.py`  
  Minimal MCTS implementation featuring:
  - `State` interface (required methods)
  - `Node` and `MCTS` classes (core algorithm: UCT selection, expansion, rollout, backpropagation)
  - `export_tree_json(root, best_child_state_label, best_mean_value, max_depth=3)` for exporting the search tree as JSON
- `main.py`  
  Demonstration script running MCTS and printing a JSON export of the search tree.

## Requirements

- Python 3.8+ (no external dependencies)

## Quick Start

From the repository root (Windows PowerShell):

```powershell
python MCT_Tree\\mcts_main.py
```

## Output

- The demo executes a small number of MCTS iterations and prints a JSON object describing the search tree and the selected best next state.
- The JSON includes:
  - `best_next_state`: label of the immediate child of the root selected by visit count
  - `best_mean_value`: mean value estimate for the best child
  - `tree`: nested node objects with `state`, `visits`, `value`, and `children` (up to the exported `max_depth`)

Example (truncated):

```json
{
  "best_next_state": "S1",
  "best_mean_value": 5.0,
  "tree": {
    "state": "S0",
    "visits": 5,
    "value": 25.0,
    "children": [
      { "state": "S1", "visits": 3, "value": 15.0, "children": [...] },
      { "state": "S2", "visits": 2, "value": 6.0, "children": [] }
    ]
  }
}
```

## Integrating MCTS

To use the `MCTS` class, provide a custom `State` implementation that supports the following interface:

- `successors() -> List[State]`
- `is_terminal() -> bool`
- `get_reward() -> float` (reward at the current state)
- `clone() -> State` (deep copy for rollouts)
- `label() -> str` (human-readable state identifier)

Example usage:

```python
from MCT_Tree.mcts import MCTS

mcts = MCTS(iterations=500, exploration=1.414, rollout_depth=10, seed=42)
best_state_label, best_mean, root = mcts.search(initial_state)
print(best_state_label, best_mean)
```

## Customization

- `MCTS.__init__` parameters:
  - `iterations`: number of MCTS iterations
  - `exploration`: UCT exploration constant
  - `rollout_depth`: maximum depth for random rollouts
  - `seed`: random seed for deterministic rollouts

Replace the example state with your domain-specific `State` implementation for practical applications.
