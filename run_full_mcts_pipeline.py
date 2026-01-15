#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# your prompt builder (use your updated full-context version)
from MCT_Tree.prompt_builder import PromptBuildConfig, PrpmtBuild

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MCT_Tree.mcts_main import (  # type: ignore
    MCTS_initialize,
    MCTS_tree_update,
    find_best_patch_in_apr,
)


# -------------------------
# Helpers
# -------------------------

def run_cmd(cmd, cwd=None) -> None:
    print("CMD DEBUG:", [(i, x, type(x)) for i, x in enumerate(cmd)])
    cmd = [str(x) for x in cmd]  # make it safe
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

def newest_matching(path_dir: Path, glob_pat: str) -> Optional[Path]:
    items = sorted(path_dir.glob(glob_pat), key=lambda p: p.stat().st_mtime)
    return items[-1] if items else None


def load_jsonl_rows(p: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl_rows(p: Path, rows: Iterable[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def coerce_reward(val: Any) -> float:
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


def read_parent_id(parent_path: Path, default: str) -> str:
    if not parent_path.exists():
        return default
    try:
        with parent_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_parent_id(parent_path: Path, parent_id: str) -> None:
    parent_path.parent.mkdir(parents=True, exist_ok=True)
    with parent_path.open("w", encoding="utf-8") as f:
        json.dump(parent_id, f, ensure_ascii=False)


# -------------------------
# MCTS over multi-bug JSONL
# -------------------------

def mcts_over_evaluation_jsonl(
    evaluation_jsonl: Path,
    out_samples_jsonl: Path,
    mcts_out_dir: Path,
    *,
    mcts_iterations: int,
    exploration: float,
    rollout_depth: int,
    seed: int,
    prompt_cfg: PromptBuildConfig,
) -> None:
    """
    Reads evaluation_jsonl containing MANY bugs (JSONL).
    For each bug-row:
      - initialize persistent APR tree at mcts_out_dir/<bug_id> if missing
      - add current generation nodes under stored parent_id
      - run MCTS to pick best patch
      - build next prompt = mask(diff(best_patch, fixed_code))
      - write a sample row to out_samples_jsonl
      - update parent_id to the chosen best patch id (so next outer-iter expands from it)
    """
    rows = load_jsonl_rows(evaluation_jsonl)
    out_rows: List[Dict[str, Any]] = []

    for data in rows:
        bug_id = data.get("identifier") or data.get("bug_id") or data.get("id")
        buggy_code = data.get("buggy_code")

        # skip malformed rows:
        if not bug_id:
            continue

        if buggy_code is None:
            continue

        if not isinstance(buggy_code, str) or buggy_code.strip() == "":
            continue

        buggy_code = data.get("buggy_code") or ""
        fixed_code = data.get("fixed_code") or ""
        generations: List[str] = data.get("generation") or data.get("generations") or []
        evaluations: List[Any] = data.get("evaluation") or []

        bug_dir = mcts_out_dir / str(bug_id)
        bug_dir.mkdir(parents=True, exist_ok=True)

        parent_path = bug_dir / "parent.json"
        parent_id = read_parent_id(parent_path, default=str(bug_id))

        # init tree if first time
        tree_path = bug_dir / "jsonl_run_tree.json"
        if not tree_path.exists():
            _ = MCTS_initialize(
                {"patch_id": str(bug_id), "patch": buggy_code, "test_reward": 0.0},
                apr_out_path=tree_path,
            )
            write_parent_id(parent_path, str(bug_id))
            parent_id = str(bug_id)

        # add candidates as children of parent_id
        nodes: List[Dict[str, Any]] = []
        for idx, gen in enumerate(generations, start=1):
            ev = evaluations[idx - 1] if idx - 1 < len(evaluations) else None
            if isinstance(ev, dict):
                if "test_reward" in ev:
                    reward = coerce_reward(ev.get("test_reward"))
                else:
                    reward = 0.0
            else:
                reward = coerce_reward(ev)
            nodes.append(
                {
                    "patch_id": f"{bug_id}-{parent_id}-{idx}",
                    "patch": gen,
                    "test_reward": reward,
                    "feedback_reward": None,
                }
            )

        if nodes:
            apr_data = MCTS_tree_update(nodes, apr_out_path=tree_path, parent_id=parent_id)
        else:
            # no generations => keep tree as-is
            apr_data = json.loads(tree_path.read_text(encoding="utf-8"))

        best = find_best_patch_in_apr(
            apr_data,
            apr_out_path=tree_path,
            iterations=mcts_iterations,
            exploration=exploration,
            rollout_depth=rollout_depth,
            seed=seed,
        )

        best_patch = best.get("patch") if best else None
        best_patch_id = best.get("patch_id") if best else None
        best_reward = best.get("test_reward") if best else None

        prompt = None
        if best_patch and fixed_code:
            _, _, prompt = PrpmtBuild(best_patch, fixed_code, prompt_cfg)

        # update parent id to best patch id (so next outer iteration expands from best)
        if best_patch_id:
            write_parent_id(parent_path, str(best_patch_id))

        out_rows.append(
            {
                "identifier": bug_id,
                "buggy_code": buggy_code,
                "fixed_code": fixed_code,
                "prompt_strategy": data.get("prompt_strategy") or "infilling",
                "ground_truth": data.get("ground_truth"),
                "best_patch": best_patch,
                "best_patch_id": best_patch_id,
                "best_reward": best_reward,
                "prompt": prompt,
            }
        )

    write_jsonl_rows(out_samples_jsonl, out_rows)


# -------------------------
# Main pipeline
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="defects4j")
    ap.add_argument("--prompt-strategy", default="infilling")

    ap.add_argument("--sample-model-name", default="codellama")

    ap.add_argument("--patch-strategy", default="repairllama-infilling")
    ap.add_argument("--patch-model-name", required=True, help="Path/name for repairllama model (e.g. /.../fine-tune/v7)")
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--num-return-seq", type=int, default=10)
    ap.add_argument("--num-beams", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=64)

    ap.add_argument("--outer-iters", type=int, default=5, help="How many (patch->eval->mcts) rounds to run")
    ap.add_argument("--mcts-iters", type=int, default=20, help="MCTS simulations per bug per round")
    ap.add_argument("--exploration", type=float, default=1.414)
    ap.add_argument("--rollout-depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--workdir", type=Path, default=Path("."), help="Where outputs are written / where scripts run")
    ap.add_argument("--mcts-out-dir", type=Path, default=Path("mcts_tree_output"))
    ap.add_argument("--export", action="store_true",default=True)

    args = ap.parse_args()
    wd: Path = args.workdir

    # 1) initial samples
    # run_cmd(
    #     ["python3", "generate_samples.py", args.benchmark, args.prompt_strategy, "--model-name", args.sample_model_name],
    #     cwd=wd,
    # )
    samples0 = newest_matching(wd, "samples_defects4j_infilling_model_name_codellama_iter_2.jsonl")
    if not samples0:
        raise FileNotFoundError("Could not find initial samples_*.jsonl after generate_samples.py")
    curr_samples = samples0

    prompt_cfg = PromptBuildConfig(
        mask_token="<FILL_ME>",
        single_chunk=True,
        keep_buggy_code=False,
        keep_comments=False,
    )

    # loop outer iterations
    for it in range(3, args.outer_iters + 1):
        # 2) generate patches
        before_candidates = set(wd.glob("candidates_*.jsonl"))
        if it != 1:
            run_cmd(
                [
                    "python3",
                    "generate_patches.py",
                    str(curr_samples),
                    args.patch_strategy,
                    "--model_name",
                    "ASSERT-KTH/RepairLLaMA-IR3-OR2",
                    "--n_workers",
                    str(args.n_workers),
                    "--num_return_sequences",
                    str(args.num_return_seq),
                    "--num_beams",
                    str(args.num_beams),
                    "--max_new_tokens",
                    str(args.max_new_tokens),
                ],
                cwd=wd,
            )
        after_candidates = set(wd.glob("candidates_*.jsonl"))
        new_candidates = list(after_candidates - before_candidates)
        candidates = sorted(new_candidates, key=lambda p: p.stat().st_mtime)[-1] if new_candidates else newest_matching(wd, "candidates_*.jsonl")
        if not candidates:
            raise FileNotFoundError("Could not find candidates_*.jsonl after generate_patches.py")

        # 3) evaluate patches
        before_eval = set(wd.glob("evaluation_*.jsonl"))
        run_cmd(
            ["python3", "evaluate_patches.py", args.benchmark, str(candidates), "replace", "--n_workers", str(args.n_workers)],
            cwd=wd,
        )
        after_eval = set(wd.glob("evaluation_*.jsonl"))
        new_eval = list(after_eval - before_eval)
        evaluation = sorted(new_eval, key=lambda p: p.stat().st_mtime)[-1] if new_eval else newest_matching(wd, "evaluation_*.jsonl")
        if not evaluation:
            raise FileNotFoundError("Could not find evaluation_*.jsonl after evaluate_patches.py")

        # optional export
        # model_name="f{args.patch_model_name}"+{it}
        if args.export:
            run_cmd(
                ["python3", "export_results.py", args.benchmark, str(evaluation), f"--iterations={it}"],
                cwd=wd,
            )

        # 4) MCTS over ALL bugs in evaluation jsonl -> next samples jsonl
        next_samples = wd / f"samples_{args.benchmark}_{args.prompt_strategy}_model_name_{args.sample_model_name}_iter_{it}.jsonl"
        mcts_over_evaluation_jsonl(
            evaluation_jsonl=evaluation,
            out_samples_jsonl=next_samples,
            mcts_out_dir=wd / args.mcts_out_dir,
            mcts_iterations=args.mcts_iters,
            exploration=args.exploration,
            rollout_depth=args.rollout_depth,
            seed=args.seed,
            prompt_cfg=prompt_cfg,
        )

        curr_samples = next_samples
        print(f"\n[OK] Iteration {it} -> wrote next samples: {curr_samples}")

    print("\nDone.")
    print(f"Final samples JSONL: {curr_samples}")


if __name__ == "__main__":
    main()