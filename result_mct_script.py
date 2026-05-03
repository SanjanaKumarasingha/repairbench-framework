#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Any


CATEGORY_KEYS = {
    "exact": "bugs_with_exact_match_candidates",
    "plausible": "bugs_with_plausible_candidates",
    "compilable": "bugs_with_compilable_candidates",
    "ast_match": "bugs_with_ast_match_candidates",
}


def load_json_or_jsonl(file_path: Path) -> Dict[str, Any]:
    """
    Load a file that may be:
    1. a single JSON object
    2. a JSONL file where the first non-empty line is the JSON object we need
    """
    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Empty file: {file_path}")

    # Try as full JSON first
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: parse first non-empty line as JSONL
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        if isinstance(data, dict):
            return data

    raise ValueError(f"Could not parse JSON/JSONL object from: {file_path}")


def ensure_list_of_strings(data: Dict[str, Any], key: str) -> List[str]:
    value = data.get(key, [])
    print(value)
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"Expected list for key '{key}', got {type(value).__name__}")
    return [str(x) for x in value]


def build_input_file_path(
    input_dir: Path,
    file_prefix: str,
    iteration: int,
    extension: str,
) -> Path:
    return input_dir / f"{file_prefix}{iteration}{extension}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Defects4J iteration result files and compute category growth."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing iteration files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where output files will be written.",
    )
    parser.add_argument(
        "--file-prefix",
        type=str,
        required=True,
        help=(
            "File prefix before iteration number. Example: "
            "'samples_defects4j_infilling_model_name_codellama_iter_'"
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of iterations to read.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=".jsonl",
        help="File extension. Default: .jsonl",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing files instead of failing.",
    )

    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    file_prefix: str = args.file_prefix
    k: int = args.k
    extension: str = args.extension

    output_dir.mkdir(parents=True, exist_ok=True)

    cumulative: Dict[str, Set[str]] = {category: set() for category in CATEGORY_KEYS}
    per_iteration_summary: List[Dict[str, Any]] = []

    for iteration in range(1, k + 1):
        file_path = build_input_file_path(input_dir, file_prefix, iteration, extension)

        if not file_path.exists():
            if args.skip_missing:
                print(f"[WARN] Missing file, skipping: {file_path}")
                continue
            raise FileNotFoundError(f"Missing file: {file_path}")

        data = load_json_or_jsonl(file_path)

        iteration_result: Dict[str, Any] = {
            "iteration": iteration,
            "file": str(file_path),
            "categories": {},
        }

        for category, json_key in CATEGORY_KEYS.items():
            print(category)
            bugs = set(ensure_list_of_strings(data, json_key))

            before_count = len(cumulative[category])
            before_bugs = cumulative[category].copy()

            cumulative[category].update(bugs)

            after_count = len(cumulative[category])
            new_bugs = sorted(cumulative[category] - before_bugs)
            increase = after_count - before_count

            iteration_result["categories"][category] = {
                "count_in_this_file": len(bugs),
                "count_before_cumulative": before_count,
                "count_after_cumulative": after_count,
                "increase": increase,
                "new_bugs_added": new_bugs,
            }

        per_iteration_summary.append(iteration_result)

    final_output = {
        "k": k,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "file_prefix": file_prefix,
        "extension": extension,
        "final_unique_counts": {
            category: len(bugs) for category, bugs in cumulative.items()
        },
        "final_unique_bugs": {
            category: sorted(list(bugs)) for category, bugs in cumulative.items()
        },
        "per_iteration_summary": per_iteration_summary,
    }

    # Save full aggregated JSON
    aggregated_json_path = output_dir / "aggregated_results.json"
    with aggregated_json_path.open("w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)

    # Save per-category bug lists
    for category, bugs in cumulative.items():
        out_path = output_dir / f"{category}_unique_bugs.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for bug in sorted(bugs):
                f.write(f"{bug}\n")

    # Save a compact CSV-like summary
    summary_txt_path = output_dir / "iteration_growth_summary.csv"
    with summary_txt_path.open("w", encoding="utf-8") as f:
        f.write(
            "iteration,"
            "exact_before,exact_after,exact_increase,"
            "plausible_before,plausible_after,plausible_increase,"
            "compilable_before,compilable_after,compilable_increase,"
            "ast_match_before,ast_match_after,ast_match_increase\n"
        )

        for item in per_iteration_summary:
            cats = item["categories"]
            f.write(
                f"{item['iteration']},"
                f"{cats['exact']['count_before_cumulative']},{cats['exact']['count_after_cumulative']},{cats['exact']['increase']},"
                f"{cats['plausible']['count_before_cumulative']},{cats['plausible']['count_after_cumulative']},{cats['plausible']['increase']},"
                f"{cats['compilable']['count_before_cumulative']},{cats['compilable']['count_after_cumulative']},{cats['compilable']['increase']},"
                f"{cats['ast_match']['count_before_cumulative']},{cats['ast_match']['count_after_cumulative']},{cats['ast_match']['increase']}\n"
            )

    print(f"Done. Outputs written to: {output_dir}")
    print(f"Main result file: {aggregated_json_path}")
    print(f"Summary file: {summary_txt_path}")


if __name__ == "__main__":
    main()