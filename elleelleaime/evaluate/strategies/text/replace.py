from typing import Optional, List
from unidiff import PatchSet
from pathlib import Path
from uuid import uuid4

import os, tempfile, shutil, logging, getpass

from elleelleaime.evaluate.strategies.strategy import PatchEvaluationStrategy
from elleelleaime.core.benchmarks.bug import Bug
from elleelleaime.core.utils.java.java import remove_empty_lines, remove_java_comments
from elleelleaime.core.caching.cache import Cache


class ReplaceEvaluationStrategy(PatchEvaluationStrategy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_cache = kwargs.get("use_cache", True)
        # Default cache path resolution
        default_cache = Path(__file__).parent.parent.parent.parent.parent / "cache"
        self.cache_path = kwargs.get("cache_path", default_cache)
        
        if self.use_cache:
            self.cache = Cache(self.cache_path)

    def evaluate_generation(
        self, bug: Bug, sample: dict, generation: Optional[str]
    ) -> Optional[dict]:
        # Initialize the result object with all required keys to prevent schema mismatch
        result = {
            "generation": generation,
            "exact_match": False,
            "ast_match": False,
            "compile": False,
            "test": False,
            "test_reward": 0.0,
        }

        if generation is None:
            return result

        # 1. Check Cache first
        if self.use_cache:
            evaluation = self.cache.load_from_cache_from_bug(bug, generation)
            if evaluation is not None:
                # Critical: If your old cache doesn't have 'test_reward', 
                # you may need to force a re-run or provide a default here.
                return evaluation

        # 2. Setup temporary workspace
        buggy_path = os.path.join(
            tempfile.gettempdir(),
            f"elleelleaime-{getpass.getuser()}",
            bug.get_identifier(),
            str(uuid4()),
        )

        try:
            # 3. Exact Match Logic (String-based)
            generation_no_comments = remove_java_comments(generation)
            if generation_no_comments is None:
                if self.use_cache:
                    self.cache.save_to_cache_from_bug(bug, generation, result)
                return result

            gen_clean = remove_empty_lines(generation_no_comments).splitlines()
            fix_clean = remove_empty_lines(remove_java_comments(sample["fixed_code"])).splitlines()

            # Check if lines match exactly (ignoring leading/trailing whitespace)
            result["exact_match"] = len(gen_clean) == len(fix_clean) and all(
                x.strip() == y.strip() for x, y in zip(gen_clean, fix_clean)
            )

            # If Exact Match, we shortcut the expensive compilation/test phase
            if result["exact_match"]:
                result["ast_match"] = True
                result["compile"] = True
                result["test"] = True
                result["test_reward"] = 15.0
                
                if self.use_cache:
                    self.cache.save_to_cache_from_bug(bug, generation, result)
                return result

            # 4. Deep Evaluation (Compile & Test)
            diff = PatchSet(bug.get_ground_truth())
            bug.checkout(buggy_path, fixed=False)

            # Resolve buggy file path
            if bug.is_ground_truth_inverted():
                rel_path = diff[0].target_file[2:] if diff[0].target_file.startswith("b/") else diff[0].target_file
            else:
                rel_path = diff[0].source_file[2:] if diff[0].source_file.startswith("a/") else diff[0].source_file
            
            buggy_file_path = os.path.join(buggy_path, rel_path)

            with open(buggy_file_path, "r", encoding="ISO-8859-1") as f:
                buggy_code = f.read()
                # Normalize line endings and clean for matching
                buggy_code_clean = remove_empty_lines(remove_java_comments(buggy_code))
                buggy_code_clean = buggy_code_clean.replace("\r\n", "\n").replace("\r", "\n")

            # Prepare the snippet comparison
            target_snippet = sample["buggy_code"].replace("\r\n", "\n").replace("\r", "\n")
            
            if target_snippet not in buggy_code_clean:
                logging.error(f"Snippet not found in {bug.get_identifier()}")
                return None

            # Generate candidate file content
            candidate_code = buggy_code_clean.replace(target_snippet, generation.replace("\r\n", "\n").replace("\r", "\n"))
            fixed_code = buggy_code_clean.replace(target_snippet, sample["fixed_code"].replace("\r\n", "\n").replace("\r", "\n"))

            # Write to disk for compilation
            with open(buggy_file_path, "w", encoding="ISO-8859-1", errors="replace") as f:
                f.write(candidate_code)

            # Compilation check
            compilation_result = bug.compile(buggy_path)
            result["compile"] = compilation_result.is_passing()

            if result["compile"]:
                test_result = bug.test(buggy_path)
                result["test"] = test_result.is_passing()
                
                # Calculate Reward based on pass rate
                if test_result.tests_run > 0:
                    pr = test_result.pass_rate if test_result.pass_rate is not None else 0.0
                    result["test_reward"] = pr * 10.0
                    if pr == 1.0:
                        result["test"] = True
                    
                
                # AST Matching (only if it tests successfully or for plausible candidates)
                if result["test"]:
                    result["ast_match"] = self.ast_match(fixed_code, candidate_code)
            else:
                result["test_reward"] = 0.0

            # 5. Save to cache before returning
            if self.use_cache:
                self.cache.save_to_cache_from_bug(bug, generation, result)
            return result

        except Exception as e:
            logging.error(f"Error evaluating {bug.get_identifier()}: {e}")
            return result
        finally:
            if os.path.exists(buggy_path):
                shutil.rmtree(buggy_path)

    def _evaluate_impl(self, bug: Bug, sample: dict) -> Optional[List[dict]]:
        evaluation = []
        for generation in sample.get("generation", []):
            evaluation.append(self.evaluate_generation(bug, sample, generation))
        return evaluation