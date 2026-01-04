import subprocess
import shutil
import re
import os

from elleelleaime.core.benchmarks.benchmark import Benchmark
from elleelleaime.core.benchmarks.bug import RichBug
from elleelleaime.core.benchmarks.test_result import TestResult
from elleelleaime.core.benchmarks.compile_result import CompileResult


class Defects4JBug(RichBug):
    """
    The class for representing Defects4J bugs
    """

    def __init__(
        self,
        benchmark: Benchmark,
        pid: str,
        bid: str,
        ground_truth: str,
        failing_tests: dict[str, str],
    ) -> None:
        self.pid = pid
        self.bid = bid
        super().__init__(
            benchmark,
            f"{pid}-{bid}",
            ground_truth,
            failing_tests,
            ground_truth_inverted=True,
        )

    def checkout(self, path: str, fixed: bool = False) -> bool:
        # Remove the directory if it exists
        shutil.rmtree(path, ignore_errors=True)

        # Checkout the bug
        checkout_run = subprocess.run(
            f"{self.benchmark.get_bin()} checkout -p {self.pid} -v {self.bid}{'f' if fixed else 'b'} -w {path}",
            shell=True,
            capture_output=True,
            check=True,
        )

        # Convert line endings to unix
        # Try to check this agin when
        dos2unix_run = subprocess.run(
            f"find {path} -type f -print0 | xargs -0 -n 1 -P 4 dos2unix",
            shell=True,
            capture_output=True,
            check=False,
        )


        return checkout_run.returncode == 0 and dos2unix_run.returncode == 0

    def compile(self, path: str) -> CompileResult:
        run = subprocess.run(
            f"cd {path} && timeout {5*60} {self.benchmark.get_bin()} compile",
            shell=True,
            capture_output=True,
            check=False,
        )
        return CompileResult(run.returncode == 0)

    def test(self, path: str) -> TestResult:
    # 1) Run relevant tests first
        run_rel = subprocess.run(
            f"cd {path} && timeout {3*60} {self.benchmark.get_bin()} test -r",
            shell=True,
            capture_output=True,
            check=False,
        )
        out_rel = run_rel.stdout.decode("utf-8", errors="ignore")

        m_rel = re.search(r"Failing tests: ([0-9]+)", out_rel)
        relevant_ok = (
            run_rel.returncode == 0
            and m_rel is not None
            and int(m_rel.group(1)) == 0
        )

        if not relevant_ok:
            return TestResult(success=False)

        # 2) Run full test suite
        run = subprocess.run(
            f"cd {path} && timeout {3*60} {self.benchmark.get_bin()} test",
            shell=True,
            capture_output=True,
            check=False,
        )
        out = run.stdout.decode("utf-8", errors="ignore")

        # Parse summary
        m_summary = re.search(
            r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)",
            out,
        )

        if m_summary:
            tests_run = int(m_summary.group(1))
            failures = int(m_summary.group(2))
            errors = int(m_summary.group(3))
        else:
            tests_run = failures = errors = None

        # Check pass/fail (legacy)
        m_fail = re.search(r"Failing tests: ([0-9]+)", out)
        success = (
            run.returncode == 0
            and m_fail is not None
            and int(m_fail.group(1)) == 0
        )

        return TestResult(
            success=success,
            tests_run=tests_run,
            failures=failures,
            errors=errors,
        )
    def get_src_test_dir(self, path: str) -> str:
        run = subprocess.run(
            f"cd {path} && {self.benchmark.get_bin()} export -p dir.src.tests",
            shell=True,
            capture_output=True,
            check=True,
        )

        return run.stdout.decode("utf-8").strip()
