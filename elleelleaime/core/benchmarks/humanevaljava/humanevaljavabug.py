import subprocess
import shutil
import os
import getpass

from elleelleaime.core.benchmarks.benchmark import Benchmark
from elleelleaime.core.benchmarks.bug import Bug
from elleelleaime.core.benchmarks.test_result import TestResult
from elleelleaime.core.benchmarks.compile_result import CompileResult
import re


class HumanEvalJavaBug(Bug):
    """
    The class for representing HumanEvalJava bugs
    """

    def __init__(self, benchmark: Benchmark, bid: str, ground_truth: str) -> None:
        super().__init__(benchmark, bid, ground_truth, True)

    def checkout(self, path: str, fixed: bool = False) -> bool:
        # Remove the directory if it exists
        shutil.rmtree(path, ignore_errors=True)

        # Copy the whole benchmark into the working directory
        shutil.copytree(self.benchmark.get_path(), path)

        # If fixed=True, replace buggy file with correct file
        if fixed:
            buggy_file = f"{path}/src/main/java/humaneval/buggy/{self.get_identifier()}.java"
            correct_file = f"{path}/src/main/java/humaneval/correct/{self.get_identifier()}.java"

            shutil.copyfile(correct_file, buggy_file)

            # Change package name from humaneval.correct -> humaneval.buggy
            subprocess.run(
                f"sed -i 's/package humaneval\\.correct/package humaneval\\.buggy/g' \"{buggy_file}\"",
                shell=True,
                capture_output=True,
                check=True,
            )

        return True

    def _get_maven_env(self) -> tuple[str, str]:
        """
        Returns:
            home_dir: base writable directory
            repo_dir: maven local repository directory
        """
        username = getpass.getuser()
        home_dir = f"/tmp/elleelleaime-{username}"
        m2_dir = os.path.join(home_dir, ".m2")
        repo_dir = os.path.join(m2_dir, "repository")

        os.makedirs(repo_dir, exist_ok=True)

        return home_dir, repo_dir

    def compile(self, path: str) -> CompileResult:
        home_dir, repo_dir = self._get_maven_env()

        run = subprocess.run(
            (
                f'cd "{path}" && '
                f'HOME="{home_dir}" '
                f'timeout {5*60} mvn -Dmaven.repo.local="{repo_dir}" test-compile'
            ),
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )

        return CompileResult(run.returncode == 0)

    def test(self, path: str) -> TestResult:
        home_dir, repo_dir = self._get_maven_env()

        run = subprocess.run(
            (
                f'cd "{path}" && '
                f'HOME="{home_dir}" '
                f'timeout {30*60} mvn -Dmaven.repo.local="{repo_dir}" '
                f'test -Dtest=TEST_{self.get_identifier()}'
            ),
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )

        out_rel=run.stdout
        m_summary = re.search(
            r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)",
            out_rel,
        )
        print( m_summary)
        relevant_ok = (
            run.returncode == 0
            and m_summary is not None
            and int(m_summary.group(1)) == 0
        )
        # if not relevant_ok:
        #     return TestResult(success=False)

        # print(sucess, )
        if m_summary:
            tests_run = int(m_summary.group(1))
            failures = int(m_summary.group(2))
            errors = int(m_summary.group(3))
        else:
            tests_run = failures = errors = None
            
        m_fail = re.search(r"Failing tests: ([0-9]+)", out_rel)
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
      