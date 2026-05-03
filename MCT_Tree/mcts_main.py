# mcts_main.py
from typing import Any, Dict, List, Optional, Union
import json
from pathlib import Path

from MCT_Tree.mcts import MCTS, MCTSState, export_tree_json, read_json_graph

# --- Paths (derived at runtime now) ---
REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "mcts_tree_output"
GRAPH_OUT = OUT_DIR / "mct-format.json"


def format_apr_json_for_iterations(iteration_json_path: Path, apr_out_path: Path) -> Dict[str, Any]:
    """
    Build/Update APR-style JSON:

    - If mct-format.json DOES NOT exist: create it with a single parent node
      (patch_id = bug.patch_id, patch = buggy_function) and its children from iteration_0 candidates.
      Also materialize the child nodes (visits=0).

    - If mct-format.json EXISTS: update the existing tree by adding/merging the new child
      patch_ids under the parent (bug.patch_id) and upserting those child nodes with their code/rewards.
      DO NOT overwrite the parent's 'patch' field.
    """
    with iteration_json_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    bug = data.get("bug", {})
    iterations: List[Dict[str, Any]] = data.get("iterations", [])
    if not iterations:
        raise ValueError("No iterations found in input JSON.")

    iter0 = iterations[0]
    candidates: List[Dict[str, Any]] = iter0.get("patches", [])
    children_ids = [c.get("patch_id") for c in candidates if c.get("patch_id")]

    first_reward = candidates[0].get("test_reward", 0.0) if candidates else 0.0
    first_feedback = candidates[0].get("feedback_reward", None) if candidates else None

    apr_out_path.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # CASE 1: Update existing tree
    # ----------------------------
    if apr_out_path.exists():
        with apr_out_path.open("r", encoding="utf-8") as f:
            apr: Dict[str, Any] = json.load(f)

        patches_list: List[Dict[str, Any]] = apr.setdefault("Patches", [])
        id2node: Dict[str, Dict[str, Any]] = {p.get("patch_id"): p for p in patches_list}

        parent_id = bug.get("patch_id")
        # Ensure parent exists (don't overwrite its 'patch')
        if parent_id not in id2node:
            parent_node = {
                "patch_id": parent_id,
                "visits": 0,
                "test_reward": 0.0,
                "feedback_reward": None,
                "children": [],
            }
            patches_list.append(parent_node)
            id2node[parent_id] = parent_node

        parent_node = id2node[parent_id]
        # Merge unique children
        existing_children = parent_node.get("children", [])
        existing_set = set(existing_children)
        for cid in children_ids:
            if cid and cid not in existing_set:
                existing_children.append(cid)
                existing_set.add(cid)
        parent_node["children"] = existing_children

        # Upsert child nodes (visits=0, code/rewards). We DO NOT touch parent's 'patch'.
        for c in candidates:
            cid = c.get("patch_id")
            if not cid:
                continue
            node = id2node.get(cid)
            if node is None:
                node = {
                    "patch_id": cid,
                    "patch": c.get("patch", ""),
                    "visits": 0,
                    "test_reward": c.get("test_reward", 0.0),
                    "feedback_reward": c.get("feedback_reward", None),
                    "children": [],
                }
                patches_list.append(node)
                id2node[cid] = node
            else:
                # Keep visits at 0 by default; refresh rewards (safe).
                # node["patch"] = c.get("patch", node.get("patch", ""))  # <- keep commented to avoid overwriting
                node["test_reward"] = c.get("test_reward", node.get("test_reward", 0.0))
                node["feedback_reward"] = c.get("feedback_reward", node.get("feedback_reward", None))
                node.setdefault("children", [])

        with apr_out_path.open("w", encoding="utf-8") as f:
            json.dump(apr, f, indent=2, ensure_ascii=False)
        return apr

    # ----------------------------
    # CASE 2: Create new tree
    # ----------------------------
    patches_list = [
        {
            "patch_id": bug.get("patch_id", "P0"),
            "patch": bug.get("buggy_function", ""),  # initial parent patch = buggy function
            "visits": 0,  # default
            "test_reward": first_reward,
            "feedback_reward": first_feedback,
            "children": children_ids,
        }
    ]

    # Add each child patch (Pxx) as its own node
    for c in candidates:
        pid = c.get("patch_id")
        if not pid:
            continue
        patches_list.append({
            "patch_id": pid,
            "patch": c.get("patch", ""),
            "visits": 0,
            "test_reward": c.get("test_reward", 0.0),
            "feedback_reward": c.get("feedback_reward", None),
            "children": [],
        })

    apr = {
        "BuggyProgramId": bug.get("program_id"),
        "Mode": bug.get("mode", "single_function"),
        "Meta": bug.get("meta", {}),
        "Patches": patches_list,
        "BuggyFunction": bug.get("buggy_function", ""),
        "InitialtestcaseFailures": bug.get("initial_failures", ""),
    }

    with apr_out_path.open("w", encoding="utf-8") as f:
        json.dump(apr, f, indent=2, ensure_ascii=False)
    return apr



def _load_patch_objects(source: Union[Path, str, Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Load one or more patch objects from either:
      - a JSON file path (Path or str), or
      - a single dict, or
      - a list of dicts, or
      - a wrapper dict with "patches": [...]
    """
    # If it's a path-like, load JSON from disk
    if isinstance(source, (str, Path)):
        json_path = Path(source)
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # Already an in-memory object
        data = source

    # Case 1: single patch object
    if isinstance(data, dict) and "patch_id" in data:
        return [data]

    # Case 3: wrapper with "patches"
    if isinstance(data, dict) and "patches" in data:
        patches = data["patches"]
        if not isinstance(patches, list):
            raise ValueError('"patches" must be a list in wrapper object')
        return patches

    # Case 2: list of objects
    if isinstance(data, list):
        # Optionally, you can validate each element is dict-like
        return data

    raise ValueError("Unsupported patch source format for _load_patch_objects")



def MCTS_initialize(
    source: Union[Path, str, Dict[str, Any], List[Dict[str, Any]]],
    apr_out_path: Path = GRAPH_OUT
) -> Dict[str, Any]:
    """
    Initialize the APR-style tree from either:
      - an in-memory dict for the root patch, or
      - a list of patch dicts (first one is taken as root).

    Example in-memory source:

        {
          "patch_id": "P0",
          "patch": "...code...",
          "test_reward": 5.0,
          "feedback_reward": null
        }

    Resulting APR JSON (written to apr_out_path):

        {
          "Patches": [
            {
              "patch_id": "P0",
              "patch": "...code...",
              "visits": 0,
              "test_reward": 5.0,
              "feedback_reward": null,
              "children": []
            }
          ]
        }
    """
    apr_out_path = Path(apr_out_path)
    # If a directory was passed, write a default file inside it
    if apr_out_path.exists() and apr_out_path.is_dir():
        apr_out_path = apr_out_path / "apr_tree.json"
    # Ensure parent dir exists
    apr_out_path.parent.mkdir(parents=True, exist_ok=True)
    # Always start fresh: remove existing file so a new one is created
    if apr_out_path.exists() and apr_out_path.is_file():
        apr_out_path.unlink()
    patches = _load_patch_objects(source)

    if not patches:
        raise ValueError("MCTS_initialize: no patch objects found in source.")

    root_patch = patches[0]
    root_id = root_patch.get("patch_id")
    if not root_id:
        raise ValueError("MCTS_initialize: root patch must have 'patch_id'.")

    apr = {
        "Patches": [
            {
                "patch_id": root_id,
                "patch": root_patch.get("patch", ""),
                # "visits": 0,
                "test_reward": float(root_patch.get("test_reward", 0.0) or 0.0),
                "feedback_reward": root_patch.get("feedback_reward", None),
                "children": [],  # no children yet
            }
        ]
    }

    apr_out_path.parent.mkdir(parents=True, exist_ok=True)
    with apr_out_path.open("w", encoding="utf-8") as f:
        json.dump(apr, f, indent=2, ensure_ascii=False)

    return apr


def MCTS_tree_update(
    source: Union[Path, str, Dict[str, Any], List[Dict[str, Any]]],
    apr_out_path: Path = GRAPH_OUT,
    parent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update the existing APR tree by adding new patches as children of a parent.

    If parent_id is:
      - None         : detect the root node (node never used as a child) and attach there
      - non-None str : attach new patches under that specific parent_id

    `source` can be:
      - path/str to a JSON file, or
      - a single patch dict, or
      - a list of patch dicts, or
      - a wrapper {"patches": [ ... ]}.
    """
    apr_out_path = Path(apr_out_path)
    if not apr_out_path.exists():
        raise FileNotFoundError(f"MCTS_tree_update: APR file not found: {apr_out_path}")

    # Load existing APR tree
    with apr_out_path.open("r", encoding="utf-8") as f:
        apr: Dict[str, Any] = json.load(f)

    patches_list: List[Dict[str, Any]] = apr.setdefault("Patches", [])
    id2node: Dict[str, Dict[str, Any]] = {p.get("patch_id"): p for p in patches_list}

    # Determine parent node
    if parent_id is None:
        # Determine root (node never used as child)
        all_ids = set(id2node.keys())
        all_children = {cid for p in patches_list for cid in p.get("children", [])}
        root_candidates = list(all_ids - all_children)
        if not root_candidates:
            # Fall back to first patch
            root_id = patches_list[0].get("patch_id")
        else:
            root_id = root_candidates[0]

        if not root_id:
            raise ValueError("MCTS_tree_update: could not determine root patch_id.")

        parent_node = id2node[root_id]
    else:
        parent_node = id2node.get(parent_id)
        if parent_node is None:
            raise ValueError(f"MCTS_tree_update: parent_id '{parent_id}' not found in APR tree.")

    existing_children = parent_node.get("children", [])
    existing_set = set(existing_children)

    # Load new patches from source
    new_patches = _load_patch_objects(source)

    for p in new_patches:
        cid = p.get("patch_id")
        if not cid:
            continue

        # Attach as child of parent (if not already attached)
        if cid not in existing_set:
            existing_children.append(cid)
            existing_set.add(cid)

        # Upsert patch node
        node = id2node.get(cid)
        if node is None:
            node = {
                "patch_id": cid,
                "patch": p.get("patch", ""),
                # "visits": 0,
                "prompt": p.get("prompt"),
                "test_reward": float(p.get("test_reward", 0.0) or 0.0),
                "feedback_reward": p.get("feedback_reward", None),
                "children": [],
            }
            patches_list.append(node)
            id2node[cid] = node
        else:
            # Refresh rewards / patch safely
            node["patch"] = p.get("patch", node.get("patch", ""))
            node["test_reward"] = float(p.get("test_reward", node.get("test_reward", 0.0)) or 0.0)
            node["feedback_reward"] = p.get("feedback_reward", node.get("feedback_reward", None))
            node["prompt"]=p.get("prompt")
            node.setdefault("children", [])

    parent_node["children"] = existing_children

    # Persist updated APR tree
    with apr_out_path.open("w", encoding="utf-8") as f:
        json.dump(apr, f, indent=2, ensure_ascii=False)

    return apr


def find_best_patch_in_apr(
    apr_data: Dict[str, Any],
    *,
    apr_out_path: Path = GRAPH_OUT,
    iterations: int = 20,
    exploration: float = 1.414,
    rollout_depth: int = 5,
    seed: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Given an APR-style tree dict (with key "Patches"), run MCTS and return
    the best patch dict. Also prints the final MCTS tree JSON.

    apr_data is expected to look like:

        {
          "Patches": [
            {
              "patch_id": "P0",
              "patch": "...",
              "visits": 0,
              "test_reward": 5.0,
              "feedback_reward": null,
              "children": ["P01", "P02", ...]
            },
            ...
          ],
          ...
        }
    """
    # 1) Persist the APR tree so read_json_graph can load it
    apr_out_path = Path(apr_out_path)
    apr_out_path.parent.mkdir(parents=True, exist_ok=True)
    with apr_out_path.open("w", encoding="utf-8") as f:
        json.dump(apr_data, f, indent=2, ensure_ascii=False)

    # 2) Build graph / reward / root from the APR JSON
    graph, reward, root = read_json_graph(str(apr_out_path))
    print("\033[95m[APR] Graph / reward / root\033[0m")
    print("graph:", graph)
    print("reward:", reward)
    print("root:", root)

    # 3) Run MCTS
    mcts = MCTS(iterations=iterations,
                exploration=exploration,
                rollout_depth=rollout_depth,
                seed=seed)
    best_state, best_mean, root_node = mcts.search(
        MCTSState(root, graph, reward)
    )

    # 4) Export and print the final MCTS tree
    result = export_tree_json(root_node, best_state, best_mean, max_depth=3)
    print("\n\033[95m[MCTS] Final tree\033[0m")
    print(json.dumps(result, indent=2))

    best_patch_id = result.get("best_patch")
    if best_patch_id is None:
        print("No best_patch found in MCTS result.")
        return None

    # 5) Map patch_id -> patch dict from apr_data
    patches: List[Dict[str, Any]] = apr_data.get("Patches", [])
    id_to_patch: Dict[str, Dict[str, Any]] = {
        p.get("patch_id"): p for p in patches if p.get("patch_id") is not None
    }

    best_patch = id_to_patch.get(best_patch_id)
    if best_patch is None:
        print(f"Best patch id {best_patch_id} not found in APR data.")
        return None

    print(f"\nBest patch ID: {best_patch_id}")
    print(f"Best patch test_reward: {best_patch.get('test_reward')}")
    # If you want, you can also print the code:
    # print("Best patch code:\n", best_patch.get("patch", ""))

    return best_patch


# def _sibling_iteration_zero_path(iter1_path: Path) -> Path:
#     """Turn .../iteration_01.json into .../iteration_00.json in the same folder."""
#     if iter1_path.name.startswith("iteration_") and iter1_path.suffix == ".json":
#         return iter1_path.with_name("iteration_00.json")
#     # Fallback: just assume iteration_00.json in same dir
#     return iter1_path.parent / "iteration_00.json"


# def mcts_main(IN_PATH: Path) -> str:
#     IN_PATH = Path(IN_PATH)
#     """
#     If IN_PATH ends with iteration_01.json:
#       - build new tree from iteration_00.json (CASE 2)
#       - then update tree from iteration_01.json (CASE 1)
#     Otherwise:
#       - just build/update from IN_PATH
#     """
#     # If we were given iteration_01.json, force a fresh build from iteration_00.json
#     if IN_PATH.name == "iteration_01.json":
#         iter0 = _sibling_iteration_zero_path(IN_PATH)
#         if not iter0.exists():
#             raise FileNotFoundError(f"Expected sibling file not found: {iter0}")
#         # Ensure fresh create from 00: delete any existing graph first
#         if GRAPH_OUT.exists():
#             GRAPH_OUT.unlink()
#         # Create from 00 (CASE 2)
#         _ = format_apr_json_first_iteration(iter0, GRAPH_OUT)
#         # Update from 01 (CASE 1)
#         Tree_data = format_apr_json_first_iteration(IN_PATH, GRAPH_OUT)
#     else:
#         # Single-shot create-or-update from the provided file
#         Tree_data = format_apr_json_first_iteration(IN_PATH, GRAPH_OUT)

#     print(f"[OK] Wrote/Updated APR graph -> {GRAPH_OUT}")

#     # Run MCTS
#     GRAPH, REWARD, ROOT = read_json_graph(str(GRAPH_OUT))
#     print("\033[95mstarting tree search...\033[0m")
#     print(GRAPH, REWARD)

#     mcts = MCTS(iterations=20, exploration=1.414, rollout_depth=5, seed=1)
#     best_under_root, best_mean, root_node = mcts.search(MCTSState(ROOT, GRAPH, REWARD))

#     # Export result
#     print("\n\033[95mtree search complete.\033[0m")
#     result = export_tree_json(root_node, best_under_root, best_mean, max_depth=3)
#     print(json.dumps(result, indent=2))

#     # best patch id + code
#     best_patch = result.get("best_patch")
#     print(f"Best patch ID: {best_patch}")
#     Tree_data["Best_Patch_ID"] = best_patch

#     patch_lookup = {p.get("patch_id"): p.get("patch", "") for p in Tree_data.get("Patches", [])}
#     best_patch_code = patch_lookup.get(best_patch, Tree_data.get("BuggyFunction", ""))
#     print(f"Best patch code:\n{best_patch_code}")

#     return best_patch_code
