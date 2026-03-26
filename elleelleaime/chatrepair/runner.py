from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4

import fire
import logging
import re
import shutil
import subprocess
import tempfile

import javalang
from unidiff import PatchSet

from elleelleaime.chatrepair.models import FrameworkChatModel
from elleelleaime.core.benchmarks.bug import RichBug
from elleelleaime.core.utils.benchmarks import get_benchmark
from elleelleaime.core.utils.java.java import (
    compute_diff,
    extract_single_function,
    find_test_class,
    get_modified_source_lines,
    get_modified_target_lines,
    get_source_filename,
    get_target_filename,
)
from elleelleaime.core.utils.jsonl import write_jsonl

INFILL_MARKER = ">>>[INFILL]<<<"
DEFAULT_OUTPUT_DIR = Path("results/chatrepair")
COMPILE_TIMEOUT_SECONDS = 5 * 60
TEST_TIMEOUT_SECONDS = 3 * 60

LINE_EXAMPLE = """<Example start>
Example buggy function and request:
The following code contains a buggy line that has been removed.
    public static boolean isSameLocalTime(Calendar cal1, Calendar cal2) {
        if (cal1 == null || cal2 == null) {
            throw new IllegalArgumentException("The date must not be null");
        }
        return (cal1.get(Calendar.MILLISECOND) == cal2.get(Calendar.MILLISECOND) &&
                cal1.get(Calendar.SECOND) == cal2.get(Calendar.SECOND) &&
                cal1.get(Calendar.MINUTE) == cal2.get(Calendar.MINUTE) &&
                >>>[INFILL]<<<
                cal1.get(Calendar.DAY_OF_YEAR) == cal2.get(Calendar.DAY_OF_YEAR) &&
                cal1.get(Calendar.YEAR) == cal2.get(Calendar.YEAR) &&
                cal1.get(Calendar.ERA) == cal2.get(Calendar.ERA) &&
                cal1.getClass() == cal2.getClass());
    }
This was the original buggy line which was removed by the infill location:
cal1.get(Calendar.HOUR) == cal2.get(Calendar.HOUR) &&
The code fails on this test:
org.apache.commons.lang3.time.DateUtilsTest::testIsSameLocalTime_Cal
on this test line:
assertFalse("LANG-677", DateUtils.isSameLocalTime(cal3, cal4));
with the following test error:
junit.framework.AssertionFailedError: LANG-677
Please provide an analysis of the problem and the expected behaviour of the correct fix, and the correct line at the infill location in the form of Java Markdown code block.

Example response:
1. Analysis of the problem:
The problem seems to arise from the comparison of hours using Calendar.HOUR. However, Calendar.HOUR represents the 12-hour clock hour whereas Calendar.HOUR_OF_DAY represents the 24-hour clock hour.

2. Expected Behavior of Correct Fix:
The correct fix should compare the 24-hour field.

3. Correct code at the Infill Location:
```java
cal1.get(Calendar.HOUR_OF_DAY) == cal2.get(Calendar.HOUR_OF_DAY) &&
```
<Example end>
"""

FUNCTION_EXAMPLE = """<Example start>
Example buggy function and request:
The following code contains a bug:
    public static boolean isSameLocalTime(Calendar cal1, Calendar cal2) {
        if (cal1 == null || cal2 == null) {
            throw new IllegalArgumentException("The date must not be null");
        }
        return (cal1.get(Calendar.MILLISECOND) == cal2.get(Calendar.MILLISECOND) &&
                cal1.get(Calendar.SECOND) == cal2.get(Calendar.SECOND) &&
                cal1.get(Calendar.MINUTE) == cal2.get(Calendar.MINUTE) &&
                cal1.get(Calendar.HOUR) == cal2.get(Calendar.HOUR) &&
                cal1.get(Calendar.DAY_OF_YEAR) == cal2.get(Calendar.DAY_OF_YEAR) &&
                cal1.get(Calendar.YEAR) == cal2.get(Calendar.YEAR) &&
                cal1.get(Calendar.ERA) == cal2.get(Calendar.ERA) &&
                cal1.getClass() == cal2.getClass());
    }
The code fails on this test:
org.apache.commons.lang3.time.DateUtilsTest::testIsSameLocalTime_Cal
on this test line:
assertFalse("LANG-677", DateUtils.isSameLocalTime(cal3, cal4));
with the following test error:
junit.framework.AssertionFailedError: LANG-677
Please provide an analysis of the problem and the expected behaviour of the correct fix, and the correct version of the function in the form of Java Markdown code block.

Example response:
1. Analysis of the problem:
The problem seems to arise from the comparison of hours using Calendar.HOUR. However, Calendar.HOUR represents the 12-hour clock hour whereas Calendar.HOUR_OF_DAY represents the 24-hour clock hour.

2. Expected Behavior of Correct Fix:
The correct fix should compare the 24-hour field.

3. Correct function:
```java
    public static boolean isSameLocalTime(Calendar cal1, Calendar cal2) {
        if (cal1 == null || cal2 == null) {
            throw new IllegalArgumentException("The date must not be null");
        }
        return (cal1.get(Calendar.MILLISECOND) == cal2.get(Calendar.MILLISECOND) &&
                cal1.get(Calendar.SECOND) == cal2.get(Calendar.SECOND) &&
                cal1.get(Calendar.MINUTE) == cal2.get(Calendar.MINUTE) &&
                cal1.get(Calendar.HOUR_OF_DAY) == cal2.get(Calendar.HOUR_OF_DAY) &&
                cal1.get(Calendar.DAY_OF_YEAR) == cal2.get(Calendar.DAY_OF_YEAR) &&
                cal1.get(Calendar.YEAR) == cal2.get(Calendar.YEAR) &&
                cal1.get(Calendar.ERA) == cal2.get(Calendar.ERA) &&
                cal1.getClass() == cal2.getClass());
    }
```
<Example end>
"""


@dataclass
class FailureDetails:
    test_name: str
    test_line: str
    error: str


@dataclass
class PromptSpec:
    identifier: str
    kind: str
    prompt: str
    buggy_function: str
    fixed_function: str
    function_with_infill: Optional[str]
    infill_placeholder_line: Optional[str]
    buggy_fragment: Optional[str]
    failure: FailureDetails


@dataclass
class ValidationResult:
    status: str
    feedback: str
    failure: Optional[FailureDetails]
    compile_stdout: str
    compile_stderr: str
    test_stdout: str
    test_stderr: str


def _leading_whitespace(text: str) -> str:
    match = re.match(r"^\s*", text)
    return match.group(0) if match else ""


def _extract_code_block(response_text: str) -> str:
    if response_text.count("```java") > 1:
        response_text = "\n".join(response_text.splitlines()[1:])

    match = re.search(r"```(?:java)?\s*([\s\S]*?)```", response_text)
    return match.group(1).strip("\n") if match else ""


def _changed_chunk(function_diff: list[str]) -> Optional[tuple[list[str], list[str], list[str], list[str]]]:
    prefix: list[str] = []
    removed: list[str] = []
    added: list[str] = []
    suffix: list[str] = []
    in_change = False
    seen_change = False

    for line in function_diff:
        if any(line.startswith(prefix_token) for prefix_token in ("---", "+++", "@@")):
            continue
        if line.startswith("-"):
            if seen_change and not in_change:
                return None
            in_change = True
            seen_change = True
            removed.append(line[1:])
        elif line.startswith("+"):
            if seen_change and not in_change:
                return None
            in_change = True
            seen_change = True
            added.append(line[1:])
        else:
            if not seen_change:
                prefix.append(line[1:])
            elif in_change:
                in_change = False
                suffix.append(line[1:])
            else:
                suffix.append(line[1:])

    if not seen_change:
        return None

    return prefix, removed, added, suffix


def _build_infill_prompt(buggy_function: str, fixed_function: str) -> Optional[tuple[str, str, str]]:
    chunk = _changed_chunk(compute_diff(buggy_function, fixed_function))
    if chunk is None:
        return None

    prefix, removed, added, suffix = chunk
    if removed and not added:
        return None

    indentation_source = removed[0] if removed else (added[0] if added else "")
    placeholder_line = f"{_leading_whitespace(indentation_source)}{INFILL_MARKER}\n"
    function_with_infill = "".join(prefix) + placeholder_line + "".join(suffix)

    if removed and added:
        kind = "single_line" if len(removed) == 1 and len(added) == 1 else "single_hunk"
        buggy_fragment = "".join(removed)
    elif added and not removed:
        kind = "single_hunk"
        buggy_fragment = ""
    else:
        return None

    return kind, function_with_infill, buggy_fragment


def _read_statement_from_line(file_path: Path, line_no: int) -> str:
    with open(file_path, "r", encoding="ISO-8859-1") as file:
        lines = file.readlines()[line_no - 1 :]

    statement = []
    for line in lines:
        statement.append(line)
        if re.sub(r'".*?"', "", line).count(";") == 1:
            break
    return "".join(statement).strip()


def _parse_failure_details(checkout_path: Path, bug: RichBug) -> Optional[FailureDetails]:
    failing_tests_path = checkout_path / "failing_tests"
    if not failing_tests_path.exists() or failing_tests_path.stat().st_size == 0:
        return None

    with open(failing_tests_path, "r", encoding="ISO-8859-1") as file:
        lines = [line.rstrip("\n") for line in file]

    if len(lines) < 2:
        return None

    test_name = lines[0].removeprefix("--- ").strip()
    error = lines[1].strip()
    class_name, method_name = test_name.split("::", 1)

    test_line_no = None
    for line in lines[2:]:
        if method_name in line:
            match = re.search(r":(\d+)\)", line)
            if match:
                test_line_no = int(match.group(1))
                break

    if test_line_no is None:
        return FailureDetails(test_name=test_name, test_line="", error=error)

    test_path = find_test_class(checkout_path, bug, class_name)
    if test_path is None:
        return FailureDetails(test_name=test_name, test_line="", error=error)

    return FailureDetails(
        test_name=test_name,
        test_line=_read_statement_from_line(test_path, test_line_no),
        error=error,
    )


def _run_shell(command: str, cwd: Path, timeout: int) -> tuple[bool, str, str]:
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "", f"timeout after {timeout} seconds"

    return (
        completed.returncode == 0,
        completed.stdout.decode("ISO-8859-1", errors="ignore"),
        completed.stderr.decode("ISO-8859-1", errors="ignore"),
    )


def _get_buggy_file_and_lines(bug: RichBug, checkout_path: Path) -> tuple[Path, list[int]]:
    diff = PatchSet(bug.get_ground_truth())
    if bug.is_ground_truth_inverted():
        return (
            checkout_path / get_target_filename(diff),
            get_modified_target_lines(diff),
        )
    return (
        checkout_path / get_source_filename(diff),
        get_modified_source_lines(diff),
    )


def _find_method_bounds(source_code: str, target_line: int) -> tuple[int, int]:
    tree = javalang.parse.parse(source_code)
    methods = sorted(
        node.position.line
        for _, node in tree.filter(javalang.tree.MethodDeclaration)
        if node.position is not None
    )

    if not methods:
        raise ValueError("No method declarations found")

    start_line = methods[-1]
    for idx, candidate_start in enumerate(methods):
        next_start = methods[idx + 1] if idx + 1 < len(methods) else None
        if next_start is None or candidate_start <= target_line < next_start:
            start_line = candidate_start
            break

    lines = source_code.splitlines(keepends=True)
    open_braces = 0
    close_braces = 0
    end_line = start_line
    for index, line in enumerate(lines[start_line - 1 :], start=start_line):
        open_braces += line.count("{")
        close_braces += line.count("}")
        end_line = index
        if open_braces > 0 and open_braces == close_braces:
            break

    return start_line, end_line


def _replace_method_in_file(file_path: Path, target_line: int, replacement: str) -> None:
    with open(file_path, "r", encoding="ISO-8859-1") as file:
        source_code = file.read()

    start_line, end_line = _find_method_bounds(source_code, target_line)
    lines = source_code.splitlines(keepends=True)
    lines[start_line - 1 : end_line] = [replacement.rstrip("\n") + "\n"]

    with open(file_path, "w", encoding="ISO-8859-1") as file:
        file.writelines(lines)


def _build_feedback(result: ValidationResult, kind: str) -> str:
    if result.status == "plausible":
        return ""

    base = "The fixed version is still not correct. "
    if result.status == "same_failure":
        feedback = base + "It still does not fix the original test failure."
    elif result.status == "new_failure" and result.failure is not None:
        feedback = (
            base
            + "The code fails on this test:\n"
            + f"{result.failure.test_name}\n"
            + "on this test line:\n"
            + f"{result.failure.test_line}\n"
            + "with the following test error:\n"
            + result.failure.error
        )
    elif result.status == "timeout":
        feedback = base + "The program timed out while executing the test cases."
    else:
        error_line = ""
        for line in result.compile_stderr.splitlines():
            if ": error:" in line:
                error_line = "error:" + line.split(": error:", 1)[1]
                break
        if error_line:
            feedback = base + "Code has the following compilation error: " + error_line
        else:
            feedback = base + "Code has compilation error."

    if kind == "single_line":
        feedback += (
            "\nPlease provide an analysis of the problem and the expected behaviour "
            "of the correct fix, and the correct line at the infill location in the "
            "form of Java Markdown code block."
        )
    elif kind == "single_hunk":
        feedback += (
            "\nPlease provide an analysis of the problem and the expected behaviour "
            "of the correct fix, and the correct hunk at the infill location in the "
            "form of Java Markdown code block."
        )
    else:
        feedback += (
            "\nPlease provide an analysis of the problem and the expected behaviour "
            "of the correct fix, and the correct version of the function in the form "
            "of Java Markdown code block."
        )

    return feedback


def _build_prompt_spec(bug: RichBug, single_function_prompt: bool) -> Optional[PromptSpec]:
    extracted = extract_single_function(bug)
    if extracted is None:
        return None

    buggy_function, fixed_function = extracted
    checkout_path = Path(
        tempfile.gettempdir(),
        "elleelleaime-chatrepair",
        bug.get_identifier(),
        str(uuid4()),
    )
    try:
        bug.checkout(str(checkout_path), fixed=False)
        _run_shell(f"{bug.benchmark.get_bin()} test", checkout_path, TEST_TIMEOUT_SECONDS)
        failure = _parse_failure_details(checkout_path, bug)
        if failure is None:
            return None
    finally:
        shutil.rmtree(checkout_path, ignore_errors=True)

    if single_function_prompt:
        kind = "single_function"
        function_with_infill = None
        infill_placeholder_line = None
        buggy_fragment = None
        example = FUNCTION_EXAMPLE
        problem_statement = "The following code contains a bug:\n"
        request = (
            "Please provide an analysis of the problem and the expected behaviour "
            "of the correct fix, and the correct version of the function in the form "
            "of Java Markdown code block."
        )
        prompt_body = buggy_function
    else:
        infill_prompt = _build_infill_prompt(buggy_function, fixed_function)
        if infill_prompt is None:
            kind = "single_function"
            function_with_infill = None
            infill_placeholder_line = None
            buggy_fragment = None
            example = FUNCTION_EXAMPLE
            problem_statement = "The following code contains a bug:\n"
            request = (
                "Please provide an analysis of the problem and the expected behaviour "
                "of the correct fix, and the correct version of the function in the form "
                "of Java Markdown code block."
            )
            prompt_body = buggy_function
        else:
            kind, function_with_infill, buggy_fragment = infill_prompt
            infill_placeholder_line = next(
                line for line in function_with_infill.splitlines(keepends=True) if INFILL_MARKER in line
            )
            example = LINE_EXAMPLE
            prompt_body = function_with_infill
            if kind == "single_line":
                problem_statement = "The following code contains a buggy line that has been removed:\n"
                request = (
                    "Please provide an analysis of the problem and the expected behaviour "
                    "of the correct fix, and the correct line at the infill location in the "
                    "form of Java Markdown code block."
                )
            else:
                problem_statement = "The following code contains a buggy hunk that has been removed:\n"
                request = (
                    "Please provide an analysis of the problem and the expected behaviour "
                    "of the correct fix, and the correct hunk at the infill location in the "
                    "form of Java Markdown code block."
                )

    prompt = (
        "You are an Automated Program Repair Tool.\n"
        "Here is an example of a repair job:\n"
        + example
        + "\n"
        + problem_statement
        + prompt_body
        + "\n"
    )
    if buggy_fragment is not None:
        prompt += (
            f"This was the original buggy {'line' if kind == 'single_line' else 'hunk'} "
            "which was removed by the infill location:\n"
            + buggy_fragment
            + "\n"
        )
    prompt += (
        "The code fails on this test:\n"
        + failure.test_name
        + "\n"
        + "on this test line:\n"
        + failure.test_line
        + "\n"
        + "with the following test error:\n"
        + failure.error
        + "\n"
        + request
    )

    return PromptSpec(
        identifier=bug.get_identifier(),
        kind=kind,
        prompt=prompt,
        buggy_function=buggy_function,
        fixed_function=fixed_function,
        function_with_infill=function_with_infill,
        infill_placeholder_line=infill_placeholder_line,
        buggy_fragment=buggy_fragment,
        failure=failure,
    )


def _candidate_function(prompt_spec: PromptSpec, patch: str) -> str:
    if prompt_spec.kind == "single_function":
        return patch
    if prompt_spec.function_with_infill is None or prompt_spec.infill_placeholder_line is None:
        raise ValueError("Missing infill prompt state")
    replacement = patch.rstrip("\n") + "\n"
    return prompt_spec.function_with_infill.replace(prompt_spec.infill_placeholder_line, replacement)


def _validate_candidate(bug: RichBug, prompt_spec: PromptSpec, patch: str) -> ValidationResult:
    checkout_path = Path(
        tempfile.gettempdir(),
        "elleelleaime-chatrepair",
        bug.get_identifier(),
        str(uuid4()),
    )

    try:
        bug.checkout(str(checkout_path), fixed=False)
        buggy_file_path, modified_lines = _get_buggy_file_and_lines(bug, checkout_path)
        _replace_method_in_file(
            buggy_file_path,
            min(modified_lines),
            _candidate_function(prompt_spec, patch),
        )

        compile_ok, compile_stdout, compile_stderr = _run_shell(
            f"{bug.benchmark.get_bin()} compile",
            checkout_path,
            COMPILE_TIMEOUT_SECONDS,
        )
        if not compile_ok:
            return ValidationResult(
                status="compile_error",
                feedback="",
                failure=None,
                compile_stdout=compile_stdout,
                compile_stderr=compile_stderr,
                test_stdout="",
                test_stderr="",
            )

        test_ok, test_stdout, test_stderr = _run_shell(
            f"{bug.benchmark.get_bin()} test",
            checkout_path,
            TEST_TIMEOUT_SECONDS,
        )
        if "timeout after" in test_stderr:
            return ValidationResult(
                status="timeout",
                feedback="",
                failure=None,
                compile_stdout=compile_stdout,
                compile_stderr=compile_stderr,
                test_stdout=test_stdout,
                test_stderr=test_stderr,
            )

        failure = _parse_failure_details(checkout_path, bug)
        if failure is None and test_ok:
            return ValidationResult(
                status="plausible",
                feedback="",
                failure=None,
                compile_stdout=compile_stdout,
                compile_stderr=compile_stderr,
                test_stdout=test_stdout,
                test_stderr=test_stderr,
            )

        if failure is not None and failure.test_name == prompt_spec.failure.test_name:
            status = "same_failure"
        else:
            status = "new_failure"

        return ValidationResult(
            status=status,
            feedback="",
            failure=failure,
            compile_stdout=compile_stdout,
            compile_stderr=compile_stderr,
            test_stdout=test_stdout,
            test_stderr=test_stderr,
        )
    finally:
        shutil.rmtree(checkout_path, ignore_errors=True)


def _conversation_record(messages: list[dict[str, str]], response_text: str, patch: str, validation: ValidationResult) -> dict:
    return {
        "messages": messages,
        "response": response_text,
        "patch": patch,
        "validation": {
            "status": validation.status,
            "feedback": validation.feedback,
            "failure": asdict(validation.failure) if validation.failure else None,
            "compile_stdout": validation.compile_stdout,
            "compile_stderr": validation.compile_stderr,
            "test_stdout": validation.test_stdout,
            "test_stderr": validation.test_stderr,
        },
    }


def _alternative_prompt(prompt_spec: PromptSpec, plausible_patches: list[str]) -> str:
    prefix = prompt_spec.prompt.split("<Example end>")[-1]
    if "Please provide" in prefix:
        prefix = prefix.split("Please provide", 1)[0].strip()

    label = "Correct version" if prompt_spec.kind == "single_function" else "plausible patch"
    intro = (
        "It can be fixed by these possible correct versions:\n"
        if prompt_spec.kind == "single_function"
        else "It can be fixed by these possible patches:\n"
    )
    plural_request = (
        "Please generate an alternative correct version of the function in the form of Java Markdown code block."
        if prompt_spec.kind == "single_function"
        else "Please generate an alternative patch in the form of Java Markdown code block."
    )
    patches_prompt = "\n".join(
        f"{label} {index + 1}:\n{patch}" for index, patch in enumerate(plausible_patches)
    )
    return prefix + "\n" + intro + patches_prompt + "\n" + plural_request


def _run_bug(
    bug: RichBug,
    model: FrameworkChatModel,
    single_function_prompt: bool,
    max_tries: int,
    max_conv_len: int,
) -> Optional[dict]:
    prompt_spec = _build_prompt_spec(bug, single_function_prompt)
    if prompt_spec is None:
        return None

    plausible_patches: list[str] = []
    attempts: list[dict] = []
    current_tries = 0

    while current_tries < max_tries and not plausible_patches:
        messages: list[dict[str, str]] = []
        feedback_prompt = prompt_spec.prompt
        current_length = 0

        while current_tries < max_tries and current_length < max_conv_len:
            messages.append({"role": "user", "content": feedback_prompt})
            response_text, raw_generation = model.complete(messages)
            messages.append({"role": "assistant", "content": response_text})
            patch = _extract_code_block(response_text)
            if not patch:
                attempts.append(
                    {
                        "phase": "search",
                        "messages": messages,
                        "response": response_text,
                        "raw_generation": raw_generation,
                        "patch": "",
                        "validation": {"status": "invalid_format"},
                    }
                )
                current_tries += 1
                break

            validation = _validate_candidate(bug, prompt_spec, patch)
            validation.feedback = _build_feedback(validation, prompt_spec.kind)
            attempts.append(
                {
                    "phase": "search",
                    **_conversation_record(messages, response_text, patch, validation),
                    "raw_generation": raw_generation,
                }
            )
            current_tries += 1
            current_length += 1

            if validation.status == "plausible":
                plausible_patches.append(patch)
                break

            feedback_prompt = validation.feedback

    while current_tries < max_tries and plausible_patches:
        alt_prompt = _alternative_prompt(prompt_spec, plausible_patches)
        messages = [{"role": "user", "content": alt_prompt}]
        response_text, raw_generation = model.complete(messages)
        messages.append({"role": "assistant", "content": response_text})
        patch = _extract_code_block(response_text)

        if not patch:
            attempts.append(
                {
                    "phase": "alternatives",
                    "messages": messages,
                    "response": response_text,
                    "raw_generation": raw_generation,
                    "patch": "",
                    "validation": {"status": "invalid_format"},
                }
            )
            current_tries += 1
            continue

        validation = _validate_candidate(bug, prompt_spec, patch)
        validation.feedback = _build_feedback(validation, prompt_spec.kind)
        duplicate = any(
            existing.replace(" ", "").replace("\n", "") == patch.replace(" ", "").replace("\n", "")
            for existing in plausible_patches
        )
        if validation.status == "plausible" and not duplicate:
            plausible_patches.append(patch)

        attempts.append(
            {
                "phase": "alternatives",
                **_conversation_record(messages, response_text, patch, validation),
                "raw_generation": raw_generation,
                "duplicate": duplicate,
            }
        )
        current_tries += 1

    return {
        "identifier": bug.get_identifier(),
        "prompt_kind": prompt_spec.kind,
        "single_function_prompt": single_function_prompt,
        "initial_prompt": prompt_spec.prompt,
        "initial_failure": asdict(prompt_spec.failure),
        "plausible_patches": plausible_patches,
        "attempts": attempts,
        "total_attempts": current_tries,
    }


def entry_point(
    strategy_name: str,
    model_name: str,
    benchmark: str = "defects4j",
    bug_id: Optional[str] = None,
    single_function_prompt: bool = True,
    max_tries: int = 24,
    max_conv_len: int = 3,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    **model_kwargs,
) -> str:
    benchmark_obj = get_benchmark(benchmark)
    if benchmark_obj is None:
        raise ValueError(f"Unknown benchmark {benchmark}")
    benchmark_obj.initialize()

    model = FrameworkChatModel(
        strategy_name=strategy_name,
        model_name=model_name,
        **model_kwargs,
    )

    results = []
    for bug in benchmark_obj.get_bugs():
        if bug_id is not None and bug.get_identifier() != bug_id:
            continue
        if not isinstance(bug, RichBug):
            continue

        logging.info("Running ChatRepair for %s", bug.get_identifier())
        result = _run_bug(
            bug,
            model,
            single_function_prompt=single_function_prompt,
            max_tries=max_tries,
            max_conv_len=max_conv_len,
        )
        if result is not None:
            results.append(result)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / (
        f"chatrepair_{benchmark}_{strategy_name}_{model_name.replace('/', '-')}"
        f"{f'_{bug_id}' if bug_id else ''}.jsonl"
    )
    write_jsonl(str(result_file), results)
    return str(result_file)


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    fire.Fire(entry_point)


if __name__ == "__main__":
    main()
