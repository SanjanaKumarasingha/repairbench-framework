# mcts.py
import math
import random
import json
from collections import deque
from typing import Optional, List, Dict, Any, Tuple


# ---------- Minimal State interface ----------
class State:
    """
    Abstract state interface used by the MCTS implementation.

    Concrete states must implement:
      - successors() : list of next states
      - is_terminal(): True if no further successors
      - get_reward() : scalar reward for this state
      - clone()      : deep/shallow copy suitable for simulation
      - label()      : string identifier (used for logging / JSON export)
    """
    def successors(self) -> List["State"]:
        raise NotImplementedError

    def is_terminal(self) -> bool:
        raise NotImplementedError

    def get_reward(self) -> float:
        raise NotImplementedError

    def clone(self) -> "State":
        raise NotImplementedError

    def label(self) -> str:
        raise NotImplementedError


# ---------- Simple graph-based State for APR patches ----------
class MCTSState(State):
    """
    A minimal State implementation backed by an adjacency `graph` and
    a `reward` dictionary.

    - `sid`    : string state / patch id (e.g., "P0", "P12", ...)
    - `graph`  : mapping state_id -> list of child state_ids
    - `reward` : mapping state_id -> scalar reward (test + feedback)
    """

    def __init__(
        self,
        sid: str,
        graph: Optional[Dict[str, List[str]]] = None,
        reward: Optional[Dict[str, float]] = None,
    ) -> None:
        self.sid = sid
        # Prefer instance-provided graph/reward; fall back to class attributes if present; else empty.
        self._graph = graph if graph is not None else getattr(MCTSState, "GRAPH", {})
        self._reward = reward if reward is not None else getattr(MCTSState, "REWARD", {})

    def successors(self) -> List["MCTSState"]:
        return [MCTSState(child, self._graph, self._reward) for child in self._graph.get(self.sid, [])]

    def is_terminal(self) -> bool:
        return len(self._graph.get(self.sid, [])) == 0

    def get_reward(self) -> float:
        return float(self._reward.get(self.sid, 0.0))

    def clone(self) -> "MCTSState":
        return MCTSState(self.sid, self._graph, self._reward)

    def label(self) -> str:
        return self.sid


# ---------- MCTS (simple UCT) ----------
class Node:
    """
    Tree node used internally by MCTS.

    - `state`   : domain-specific State instance
    - `parent`  : parent Node or None for root
    - `children`: list of child Nodes
    - `visits`  : number of times this node was visited
    - `value`   : accumulated reward from all simulations through this node
    """

    def __init__(self, state: State, parent: Optional["Node"] = None) -> None:
        self.state = state
        self.parent = parent
        self.children: List["Node"] = []
        self.visits: int = 0
        self.value: float = 0.0
        print(f"Created node for state: {state.label()}")

    def is_fully_expanded(self) -> bool:
        """Return True if all successors of this state are already children."""
        return len(self.children) == len(self.state.successors())

    def best_child(self, c: float) -> "Node":
        """
        Select the best child according to UCT.

        Special rule at the root (self.parent is None):
          - Use the state's static reward (state.get_reward()) as the exploitation term
            so we favour the empirically best patch under the root.
          - Still use an exploration bonus to ensure each child is tried.

        At non-root nodes:
          - Use standard UCT with exploit = value / visits.
        """
        best: Optional["Node"] = None
        best_score: float = -float("inf")

        for child in self.children:
            if self.parent is None:
                # ---- ROOT NODE SELECTION: bias by static reward ----
                exploit = child.state.get_reward()

                if child.visits == 0:
                    # Force at least one visit to each child.
                    score = float("inf")
                else:
                    explore = c * math.sqrt(math.log(self.visits + 1) / child.visits)
                    score = exploit + explore

                print(
                    f"  [ROOT] Child state: {child.state.label()}, visits: {child.visits}, "
                    f"static_reward: {exploit}, score: {score}"
                )
            else:
                # ---- NON-ROOT: standard UCT ----
                if child.visits == 0:
                    score = float("inf")
                else:
                    exploit = child.value / child.visits
                    explore = c * math.sqrt(math.log(self.visits + 1) / child.visits)
                    score = exploit + explore
                    print(
                        f"  Child state: {child.state.label()}, visits: {child.visits}, "
                        f"value: {child.value}, score: {score}"
                    )

            if score > best_score:
                best, best_score = child, score
                print(f"    New best child: {child.state.label()} with score: {score}")

        if best is None:
            raise RuntimeError("best_child called on node with no children")

        return best


class MCTS:
    """
    Basic Monte Carlo Tree Search with UCT and random rollouts.

    Parameters:
      - iterations    : number of search iterations to run
      - exploration   : UCT exploration constant
      - rollout_depth : max depth of random rollout
      - seed          : RNG seed for reproducibility
    """

    def __init__(
        self,
        iterations: int = 500,
        exploration: float = 1.414,
        rollout_depth: int = 10,
        seed: int = 42,
    ) -> None:
        self.iterations = iterations
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.rng = random.Random(seed)

    def search(self, initial_state: State) -> Tuple[Optional[str], float, Node]:
        """
        Run MCTS starting from `initial_state`.

        Returns:
            best_label : label of the best leaf node in the entire tree
                         (based on mean reward; earlier best kept on ties)
            best_mean  : mean reward of that best leaf
            root       : root Node of the built search tree
        """
        root = Node(initial_state)

        for i in range(self.iterations):
            print("-----------------------------------------------------")
            print(f"\033[92mIteration {i + 1}/{self.iterations}\033[0m")
            print("\033[94mSelecting...\033[0m")
            leaf = self._select(root)
            print(f"\033[94mSimulating from leaf state: {leaf.state.label()}\033[0m")
            reward = self._simulate(leaf.state)
            print(f"Simulation reward: {reward}")
            print("\033[94mBackpropagating...\033[0m")
            self._backpropagate(leaf, reward)

        # Choose best leaf in the whole tree.
        best_leaf = _best_leaf(root)

        if best_leaf is not None and best_leaf.visits > 0:
            best_label = best_leaf.state.label()
            best_mean = best_leaf.value / best_leaf.visits
        elif best_leaf is not None:
            # Fallback: static reward if never visited (edge case)
            best_label = best_leaf.state.label()
            best_mean = best_leaf.state.get_reward()
        else:
            best_label = None
            best_mean = 0.0

        return best_label, best_mean, root

    # ---------------- Selection ----------------
    def _select(self, node: Node) -> Node:
        """
        Selection + expansion:

        Descend the tree until we reach:
          - a non-terminal node that is not fully expanded -> expand one child
          - or a terminal node -> return it for simulation.
        """
        while not node.state.is_terminal():
            print(
                f"Selecting node at state: {node.state.label()} "
                f"with {len(node.children)} children and {node.visits} visits"
            )
            if not node.is_fully_expanded():
                print(f"Expanding node at state: {node.state.label()}")
                return self._expand(node)
            print(f"Node at state: {node.state.label()} is fully expanded; selecting best child...")
            node = node.best_child(self.exploration)
        return node

    # ---------------- Expansion ----------------
    def _expand(self, node: Node) -> Node:
        """
        Expand `node` by adding the first unexpanded successor as a new child.
        """
        expanded_states = {ch.state.label() for ch in node.children}
        print(f"  Expanded states so far: {expanded_states}")
        for succ in node.state.successors():
            print(f"  Considering successor state: {succ.label()}")
            if succ.label() not in expanded_states:
                child = Node(succ, parent=node)
                node.children.append(child)
                print(f"  Added child node for state: {succ.label()} to parent state: {node.state.label()}")
                return child
        # Fallback: if somehow all successors are already children
        return node

    # ---------------- Simulation (Rollout) ----------------
    def _simulate(self, state: State) -> float:
        """
        Perform a random rollout from `state` up to `rollout_depth` or until
        reaching a terminal state. Return the reward of the terminal state reached.
        """
        s = state.clone()
        depth = 0
        print(f"  Starting rollout from state: {s.label()}")
        while (not s.is_terminal()) and depth < self.rollout_depth:
            print(f"  Rollout at state: {s.label()}, depth: {depth}")
            succs = s.successors()
            if not succs:
                break
            s = self.rng.choice(succs)
            depth += 1
        return s.get_reward()

    # ---------------- Backpropagation ----------------
    def _backpropagate(self, node: Node, reward: float) -> None:
        """
        Backpropagate `reward` from `node` up to the root, updating visits and value.
        """
        cur: Optional[Node] = node
        while cur is not None:
            cur.visits += 1
            cur.value += reward
            cur = cur.parent


# ---------- Leaf-based "best patch" helpers ----------
def _is_leaf(n: "Node") -> bool:
    return len(n.children) == 0


def _mean(n: "Node") -> float:
    return (n.value / n.visits) if n.visits > 0 else float("-inf")


def _collect_leaves(root: "Node") -> List["Node"]:
    """
    Collect all leaf nodes in the tree rooted at `root` using a DFS order.
    """
    stack = [root]
    leaves: List[Node] = []
    while stack:
        cur = stack.pop()
        if _is_leaf(cur):
            leaves.append(cur)
        else:
            # DFS: children are extended in order; last child will be popped first
            stack.extend(cur.children)
    return leaves


def _best_leaf(root: "Node") -> Optional["Node"]:
    """
    Return the best *leaf* node in the entire tree.

    Strategy:
      - Prefer leaves that have been visited at least once.
      - Among visited leaves, choose the one with the highest mean reward (value / visits).
      - On ties in mean, keep the *first* such leaf encountered in DFS order
        (so earlier winners like P03 are not overwritten by later equals like P13).
      - If no leaves have been visited at all, fall back to the leaf with
        highest static state reward (again keeping the first max on ties).
    """
    leaves = _collect_leaves(root)
    if not leaves:
        return None

    # 1) Prefer leaves that have been visited
    visited = [n for n in leaves if n.visits > 0]
    if visited:
        best = visited[0]
        best_mean = _mean(best)

        for n in visited[1:]:
            m = _mean(n)
            # STRICT > comparison -> first max is kept on ties
            if m > best_mean:
                best = n
                best_mean = m

        return best

    # 2) If no visited leaves, use static reward as fallback
    try:
        best = leaves[0]
        best_reward = float(best.state.get_reward())

        for n in leaves[1:]:
            r = float(n.state.get_reward())
            if r > best_reward:
                best = n
                best_reward = r

        return best
    except Exception:
        # Extremely defensive fallback
        return leaves[0]


# ---------- Export to JSON (best *leaf* patch only) ----------
def export_tree_json(
    root: Node,
    _ignored_best_under_root_label: Optional[str],
    _ignored_best_under_root_mean: float,
    max_depth: int = 3,
) -> Dict[str, Any]:
    """
    Export MCTS tree (limited to `max_depth`) and best leaf summary as JSON-ish dict.

    Output format:
      {
        "best_patch": "<best leaf patch_id>",
        "best_mean_value": <mean of best leaf or null if unvisited>,
        "tree": { ... }
      }
    """

    def node_to_dict(n: Node, depth: int) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "patch_id": n.state.label(),
            "visits": n.visits,
            "reward_value": round(n.value, 4),
            "children": [],
        }
        if depth < max_depth:
            d["children"] = [node_to_dict(ch, depth + 1) for ch in n.children]
        return d

    best_leaf = _best_leaf(root)
    best_patch = best_leaf.state.label() if best_leaf is not None else None
    best_mean_value = _mean(best_leaf) if (best_leaf is not None and best_leaf.visits > 0) else None

    return {
        "best_patch": best_patch,
        "best_mean_value": (round(best_mean_value, 4) if best_mean_value is not None else None),
        "tree": node_to_dict(root, 0),
    }


# ---------- Read JSON graph from APR-style file ----------
def read_json_graph(
    filepath: str,
    *,
    label_states: bool = False,
) -> Tuple[Dict[str, List[str]], Dict[str, float], str]:
    """
    Read your APR JSON (with 'Patches') and build:
      - graph: adjacency (no actions)  e.g. {"P0":["P1","P2"], ...}
      - reward: sum of test_reward + feedback_reward(None->0)
      - root: patch_id (or state label S* if label_states=True)

    If label_states=True, also returns mapping via BFS labeling (S0,S1,...).
    """
    print(f"Reading JSON graph from: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    patches: List[Dict[str, Any]] = data.get("Patches", [])
    if not isinstance(patches, list) or not patches:
        raise ValueError("Input JSON must contain non-empty list 'Patches'.")

    # Index by patch_id
    by_id: Dict[str, Dict[str, Any]] = {p["patch_id"]: p for p in patches}

    # Detect root = node never listed as a child
    all_ids = set(by_id.keys())
    all_children = {c for p in patches for c in p.get("children", [])}
    roots = list(all_ids - all_children)
    root_id = roots[0] if roots else patches[0]["patch_id"]

    # Build graph by patch id
    graph_by_patch: Dict[str, List[str]] = {
        pid: list(by_id[pid].get("children", [])) for pid in all_ids
    }

    # Build reward by patch id (test_reward + feedback_reward(None->0))
    reward_by_patch: Dict[str, float] = {}
    for pid, p in by_id.items():
        tr = float(p.get("test_reward", 0) or 0)
        fr = float(p.get("feedback_reward", 0) or 0)
        reward_by_patch[pid] = tr + fr

    if not label_states:
        # Return by patch id
        print(f"Root (patch): {root_id}")
        return graph_by_patch, reward_by_patch, root_id

    # Otherwise, relabel with S0,S1,... using BFS from root
    patch_to_state: Dict[str, str] = {}
    state_to_patch: Dict[str, str] = {}
    q: deque[str] = deque([root_id])

    while q:
        cur = q.popleft()
        if cur in patch_to_state:
            continue
        label = f"S{len(patch_to_state)}"
        patch_to_state[cur] = label
        state_to_patch[label] = cur
        for ch in by_id[cur].get("children", []):
            q.append(ch)

    # Include any disconnected nodes (if any)
    for pid in all_ids:
        if pid not in patch_to_state:
            label = f"S{len(patch_to_state)}"
            patch_to_state[pid] = label
            state_to_patch[label] = pid

    graph = {
        patch_to_state[pid]: [patch_to_state[ch] for ch in graph_by_patch[pid]]
        for pid in all_ids
    }
    rewards = {
        patch_to_state[pid]: reward_by_patch[pid]
        for pid in all_ids
    }
    root = patch_to_state[root_id]
    print(f"Root (state): {root}")
    print(f"Graph by state: {graph}")
    print(f"Reward by state: {rewards}")
    return graph, rewards, root
