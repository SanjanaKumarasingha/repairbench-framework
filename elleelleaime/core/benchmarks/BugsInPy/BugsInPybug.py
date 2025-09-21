import subprocess
import shutil
import re
import os

from elleelleaime.core.benchmarks.benchmark import Benchmark
from elleelleaime.core.benchmarks.bug import RichBug
from elleelleaime.core.benchmarks.test_result import TestResult
from elleelleaime.core.benchmarks.compile_result import CompileResult


class BugsInPyBug(RichBug):
    """
    The class for representing BugsInPy bugs
    """

    def __init__(
        self,
        benchmark: Benchmark,
        project_name: str,
        bug_id: str,
        version_id: str,  # 1 fixed, 0 buggy
        ground_truth: str,
        failing_tests: dict[str, str],
    ) -> None:
        self.project_name = project_name
        self.bug_id = bug_id
        self.version_id = version_id
        super().__init__(
            benchmark,
            f"{project_name}-{bug_id}",
            ground_truth,
            failing_tests,
            ground_truth_inverted=False,
        )

    def checkout(self, path: str, fixed: bool = False) -> bool:
        project_name, bug_id = path.rsplit("-", 1)

        # Remove the directory if it exists (inside the container)
        subprocess.run(
            f"docker exec bugsinpy-container rm -rf /bugsinpy/framework/bin/temp/{project_name}",
            shell=True,
            capture_output=True,
            check=False,  # Don't fail if directory doesn't exist
        )

        # Checkout the bug
        checkout_run = subprocess.run(
            f"docker exec bugsinpy-container /bugsinpy/framework/bin/bugsinpy-checkout -p {project_name} -v {fixed} -i {bug_id}",  # 1 fixed, 0 buggy
            shell=True,
            capture_output=True,
            check=True,
        )

        # Convert line endings to unix
        dos2unix_run = subprocess.run(
            f"docker exec bugsinpy-container find /bugsinpy/framework/bin/temp/{project_name} -type f -name '*.py' -print0 | xargs -0 -n 1 -P 4 dos2unix",
            shell=True,
            capture_output=True,
            check=False,  # Don't fail if dos2unix has issues
        )

        return checkout_run.returncode == 0

    def compile(self, path: str) -> CompileResult:
        project_name, bug_id = path.rsplit("-", 1)
        run = subprocess.run(
            f"docker exec bugsinpy-container /bugsinpy/framework/bin/bugsinpy-compile -w /bugsinpy/framework/bin/temp/{project_name}",
            shell=True,
            capture_output=True,
            check=True,
        )

        return CompileResult(run.returncode == 0)

    def test(self, path: str) -> TestResult:
        project_name, bug_id = path.rsplit("-", 1)

        run = subprocess.run(
            f"docker exec bugsinpy-container /bugsinpy/framework/bin/bugsinpy-test -w /bugsinpy/framework/bin/temp/{project_name}",
            shell=True,
            capture_output=True,
            check=False,
        )

        # Decode the output and extract the last line
        stdout_lines = run.stdout.decode("utf-8").strip().splitlines()
        last_line = stdout_lines[-1] if stdout_lines else ""

        success = False
        # Check for various success indicators in pytest output
        if "OK" in last_line or "passed" in last_line or "PASSED" in last_line:
            success = True

        print(f"{project_name=}")
        print(f"{bug_id=}")
        print(f"{stdout_lines=}")

        return TestResult(success)

    def get_src_test_dir(self, path: str) -> str:
        project_name, bug_id = path.rsplit("-", 1)
        path = f"/bugsinpy/framework/bin/temp/{project_name}/test"

        return path

    def get_failing_tests(self) -> dict[str, str]:
        """
        Gets the failing test cases and their error messages for this bug.
        For BugsInPy, this requires running the tests to get the actual failure information.
        """
        if not hasattr(self, "_failing_tests") or self._failing_tests is None:
            self._failing_tests = self._extract_failing_tests()
        return self._failing_tests

    def _extract_failing_tests(self) -> dict[str, str]:
        """
        Extracts failing test cases by running the tests for the buggy version.
        """
        try:
            # Checkout buggy version
            self.checkout(self.get_identifier(), fixed=False)

            # Run tests to get failure information
            run = subprocess.run(
                f"docker exec bugsinpy-container /bugsinpy/framework/bin/bugsinpy-test -w /bugsinpy/framework/bin/temp/{self.project_name}",
                shell=True,
                capture_output=True,
                check=False,
            )

            # Parse the test output to extract failing tests
            stdout = run.stdout.decode("utf-8")
            stderr = run.stderr.decode("utf-8")

            failing_tests = {}

            # Look for pytest-style failures
            import re

            # Pattern to match pytest failure format
            failure_pattern = r"FAILED\s+([^\s]+)::([^\s]+)\s+-\s+(.*?)(?=\n\s*FAILED|\n\s*ERROR|\n\s*===|\Z)"
            matches = re.findall(failure_pattern, stdout + stderr, re.DOTALL)

            for test_file, test_method, error_msg in matches:
                test_name = f"{test_file}::{test_method}"
                failing_tests[test_name] = error_msg.strip()

            # If no pytest failures found, try to extract from stderr
            if not failing_tests and stderr:
                # Look for assertion errors or other test failures
                assertion_pattern = r"AssertionError:\s*(.*?)(?=\n|\Z)"
                assertion_matches = re.findall(assertion_pattern, stderr)
                if assertion_matches:
                    failing_tests["test_assertion"] = assertion_matches[0]

            return failing_tests

        except Exception as e:
            print(f"Failed to extract failing tests for {self.get_identifier()}: {e}")
            return {}

    def checkout_fixed(self, path: str, fixed: bool = False) -> bool:
        """
        Fixed version of checkout that properly handles the version parameter.
        """
        project_name, bug_id = path.rsplit("-", 1)

        # Remove the directory if it exists (inside the container)
        subprocess.run(
            f"docker exec bugsinpy-container rm -rf /bugsinpy/framework/bin/temp/{project_name}",
            shell=True,
            capture_output=True,
            check=False,  # Don't fail if directory doesn't exist
        )

        # Checkout the bug with correct version parameter
        version = "1" if fixed else "0"  # 1 fixed, 0 buggy
        checkout_run = subprocess.run(
            f"docker exec bugsinpy-container /bugsinpy/framework/bin/bugsinpy-checkout -p {project_name} -v {version} -i {bug_id}",
            shell=True,
            capture_output=True,
            check=True,
        )

        # Convert line endings to unix
        dos2unix_run = subprocess.run(
            f"docker exec bugsinpy-container find /bugsinpy/framework/bin/temp/{project_name} -type f -name '*.py' -print0 | xargs -0 -n 1 -P 4 dos2unix",
            shell=True,
            capture_output=True,
            check=False,  # Don't fail if dos2unix has issues
        )

        return checkout_run.returncode == 0
