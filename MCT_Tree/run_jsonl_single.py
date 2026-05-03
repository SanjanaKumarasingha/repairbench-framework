#!/usr/bin/env python3
"""
Build a small MCTS tree from a single JSONL row and print the best patch.

Expected JSONL shape (only the first non-empty row is used):
{
  "buggy_code": "...",
  "generation": ["patch text 1", "patch text 2", ...],
  "evaluation": [{"reward": 5}, {"reward": 0}, ...]  # optional, aligns with generation
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from prompt_builder import PromptBuildConfig,PrpmtBuild

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MCT_Tree.mcts_main import (  # type: ignore
    MCTS_initialize,
    MCTS_tree_update,
    find_best_patch_in_apr,
)


def _coerce_reward(val: Any) -> float:
    try:
        if val is None:
            return 0.0
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return 0.0
        return float(val)
    except Exception:
        return 0.0


def _load_first_row(jsonl_path: Path) -> Dict[str, Any]:
    text = jsonl_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception:
        pass

    for line in text.splitlines():
        if line.strip():
            return json.loads(line)

    raise ValueError(f"No rows found in {jsonl_path}")


def run_single_jsonl(
    jsonl_path: Path,
    apr_out: Path,
    *,
    iterations: int,
    exploration: float,
    rollout_depth: int,
    seed: int,
) -> Dict[str, Any]:
    apr_out = Path(apr_out)
    if apr_out.exists() and apr_out.is_dir():
        apr_out = apr_out / "apr_tree.json"
    apr_out.parent.mkdir(parents=True, exist_ok=True)

    data = _load_first_row(jsonl_path)
    bug_id = data.get("identifier") or data.get("bug_id") or data.get("id") or "bug_0"
    buggy_code = data.get("buggy_code") or data.get("input") or ""
    generations: List[str] = data.get("generation") or data.get("generations") or []
    evaluations: List[Dict[str, Any]] = data.get("evaluation") or []

    # Root node is the buggy code.
    if iterations==1:
        apr_data = MCTS_initialize(
            {"patch_id": bug_id, "patch": buggy_code, "test_reward": 0.0},
            apr_out_path=apr_out,
        )
        parentId=bug_id
    else:
        apr_parentData = apr_out.parent / "parent.jsonl"
        apr_parentData.parent.mkdir(parents=True, exist_ok=True)
        with apr_parentData.open("r", encoding="utf-8") as f:
         parentId = json.loads(f.readline())["parent_id"]
        


        

      

    # Add each candidate generation as a child of the root.
    nodes: List[Dict[str, Any]] = []
    for idx, eva in enumerate(evaluations, start=1):
        reward_val = eva.get("test_reward")
        nodes.append(
            {
                "patch_id": f"{bug_id}-{iterations}-{idx}",
                "patch": eva.get("generation"),
                "test_reward": reward_val,
                "feedback_reward": None,
            }
        )

    if nodes:
        apr_data = MCTS_tree_update(nodes, apr_out_path=apr_out, parent_id=parentId)

    best_patch = find_best_patch_in_apr(
        apr_data,
        apr_out_path=apr_out,
        iterations=iterations,
        exploration=exploration,
        rollout_depth=rollout_depth,
        seed=seed,
    )
    apr_parentData = apr_out.parent / "parent.jsonl"
    apr_parentData.parent.mkdir(parents=True, exist_ok=True)
    with apr_parentData.open("w", encoding="utf-8") as f:
          f.write(json.dumps({"parent_id": best_patch.get("patch_id")}, ensure_ascii=False) + "\n")

    return {
        "identifier": bug_id,
        "buggy_code": data.get("buggy_code"),
        "fixed_code":data.get("fixed_code"),
        "prompt_strategy":data.get("prompt_strategy"),
        "best_patch":best_patch.get("patch") if best_patch else None
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MCTS on a single JSONL bug row and pick the best generated patch."
    )
    parser.add_argument("jsonl", type=Path, help="Path to JSONL file (uses the first row).")
    parser.add_argument(
        "--apr-out",
        type=Path,
        default=Path(__file__).resolve().parent / "mcts_tree_output" / "jsonl_run_tree.json",
        help="Where to write the APR tree JSON used by MCTS.",
    )
    parser.add_argument("--iterations", type=int, default=20, help="MCTS iterations.")
    parser.add_argument("--exploration", type=float, default=1.414, help="UCT exploration constant.")
    parser.add_argument("--rollout-depth", type=int, default=5, help="Maximum random rollout depth.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        help="Optional: write a JSONL line with identifier and best patch to this file.",
    )
    args = parser.parse_args()

    result = run_single_jsonl(
        args.jsonl,
        args.apr_out,
        iterations=args.iterations,
        exploration=args.exploration,
        rollout_depth=args.rollout_depth,
        seed=args.seed,
    )
    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        # best = result.get("best_patch") or {}
        payload = {
            "identifier": result.get("identifier"),
            "buggy_code": result.get("buggy_code"),
            "fixed_code": result.get("fixed_code"),
            "prompt_strategy": result.get("prompt_strategy"),
            "best_patch": result.get("best_patch"),
        }
        cfg = PromptBuildConfig(
            mask_token="<FILL_ME>",
            single_chunk=True,
            keep_buggy_code=False,
            keep_comments=False,
        )

        buggy_used, fixed_used, prompt = PrpmtBuild(payload.get("best_patch") or "", payload.get("fixed_code") or "", cfg)
        payload["prompt"] = prompt
        with args.out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        print(f"Wrote best patch summary to {args.out_jsonl}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()



# python3 MCT_Tree/run_jsonl_single.py evaluation_defects4j_infilling_repairllama-infilling.jsonl \
#   --apr-out mcts_tree_output/chart-1/jsonl_run_tree.json \
#   --iterations 1  --exploration 1.414 --rollout-depth 5 --seed 1 \
#   --out-jsonl samples_defects4j_infilling_model_name_codellama_iteration_1.