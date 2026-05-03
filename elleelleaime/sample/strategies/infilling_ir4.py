from typing import Optional, Tuple
from unidiff import PatchSet
import re

from elleelleaime.sample.strategy import PromptingStrategy
from elleelleaime.core.benchmarks.bug import Bug
from elleelleaime.core.utils.java.java import (
    extract_single_function,
    compute_diff,
    remove_java_comments,
    remove_empty_lines,
)


class InfillingPromptingIR4(PromptingStrategy):
    MODEL_DICT = {
        "codellama": {
            "mask_token": "<FILL_ME>",
            "extra_mask_token": False,
            "single_chunk": True,
        },
        "salesforce/codet5-small": {
            "mask_token": "<extra_id_0>",
            "extra_mask_token": False,
            "single_chunk": True,
        },
        "salesforce/codet5-large": {
            "mask_token": "<extra_id_0>",
            "extra_mask_token": False,
            "single_chunk": True,
        },
    }

    def __init__(self, **kwargs):
        super().__init__("infilling")

        self.model_name: str = kwargs.get("model_name", "").strip().lower()
        assert self.model_name in self.MODEL_DICT, f"Unknown model name: {kwargs.get('model_name', None)}"

        model_kwargs = self.MODEL_DICT[self.model_name]
        self.original_mask_token: str = model_kwargs["mask_token"]
        self.begin_fim: str = model_kwargs.get("begin_fim", None)
        self.end_fim: str = model_kwargs.get("end_fim", None)
        self.extra_mask_token: bool = model_kwargs.get("extra_mask_token", False)
        self.single_chunk: bool = model_kwargs.get("single_chunk", True)

        # IR4 needs buggy code kept as comments
        self.keep_buggy_code: bool = kwargs.get("keep_buggy_code", True)
        self.keep_comments: bool = kwargs.get("keep_comments", False)

    def generate_masking_prompt(self, line_to_replace: str, mask_id: int) -> str:
        mask_token = (
            self.original_mask_token.format(mask_id)
            if "{}" in self.original_mask_token
            else self.original_mask_token
        )

        leading_spaces = re.match(r"^\s*", line_to_replace)
        leading_spaces = leading_spaces.group() if leading_spaces is not None else ""
        return leading_spaces + mask_token

    def _extract_single_chunk_parts(
        self, buggy_code: str, fixed_code: str
    ) -> tuple[str, str, str, str]:
        """
        Returns:
            prefix, buggy_chunk, fixed_chunk, suffix

        This defines one suspicious region from the first changed line
        to the last changed line, which is suitable for IR4-OR2.
        """
        fdiff = compute_diff(buggy_code, fixed_code)

        prefix_parts: list[str] = []
        buggy_chunk_parts: list[str] = []
        fixed_chunk_parts: list[str] = []
        suffix_buffer: list[str] = []

        seen_change = False

        for line in fdiff:
            if any(line.startswith(x) for x in ["---", "+++", "@@"]):
                continue

            tag = line[0]
            content = line[1:]

            if tag in ["+", "-"]:
                seen_change = True

                # unchanged lines collected after the first change belong
                # inside the suspicious region until we know the region ended
                if suffix_buffer:
                    buggy_chunk_parts.extend(suffix_buffer)
                    fixed_chunk_parts.extend(suffix_buffer)
                    suffix_buffer = []

                if tag == "-":
                    buggy_chunk_parts.append(content)
                elif tag == "+":
                    fixed_chunk_parts.append(content)

            else:
                if not seen_change:
                    prefix_parts.append(content)
                else:
                    suffix_buffer.append(content)

        prefix = "".join(prefix_parts)
        buggy_chunk = "".join(buggy_chunk_parts)
        fixed_chunk = "".join(fixed_chunk_parts)
        suffix = "".join(suffix_buffer)

        return prefix, buggy_chunk, fixed_chunk, suffix

    def build_single_cloze_prompt(self, buggy_code: str, fixed_code: str) -> str:
        prefix, buggy_chunk, _, suffix = self._extract_single_chunk_parts(
            buggy_code, fixed_code
        )

        if self.keep_buggy_code:
            buggy_comment = "// buggy code\n"
            if buggy_chunk.strip():
                for line in buggy_chunk.splitlines(keepends=True):
                    buggy_comment += "//" + line

            prompt = (
                prefix
                + buggy_comment
                + f"{self.generate_masking_prompt('', 0)}\n"
                + suffix
            )
        else:
            prompt = prefix + f"{self.generate_masking_prompt('', 0)}\n" + suffix

        return prompt

    def build_single_or2_target(self, buggy_code: str, fixed_code: str) -> str:
        """
        OR2 = fixed chunk only, not the full fixed function.
        """
        _, _, fixed_chunk, _ = self._extract_single_chunk_parts(
            buggy_code, fixed_code
        )
        return fixed_chunk

    def build_multi_cloze_prompt(self, buggy_code: str, fixed_code: str) -> str:
        fdiff = compute_diff(buggy_code, fixed_code)

        prompt = ""
        mask_id = 0
        i = 0
        while i < len(fdiff):
            if any(fdiff[i].startswith(x) for x in ["---", "+++", "@@"]):
                i += 1
            elif any(fdiff[i].startswith(x) for x in ["+", "-"]):
                if self.keep_buggy_code and fdiff[i].startswith("-"):
                    prompt += "// buggy code\n//" + fdiff[i][1:]

                mask_token = self.generate_masking_prompt(fdiff[i][1:], mask_id)
                i += 1

                while i < len(fdiff) and any(fdiff[i].startswith(x) for x in ["+", "-"]):
                    if self.keep_buggy_code and fdiff[i].startswith("-"):
                        prompt += "//" + fdiff[i][1:]
                    i += 1

                prompt += f"{mask_token}\n"
                mask_id += 1
            else:
                prompt += fdiff[i][1:]
                i += 1

        if self.extra_mask_token:
            prompt += f"{self.generate_masking_prompt('', mask_id)}\n"

        if prompt == "":
            prompt = f"{self.generate_masking_prompt('', 0)}"

        return prompt

    def cloze_prompt(self, bug: Bug) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        print(f"Building cloze prompt for bug {bug.get_identifier()}")
        result = extract_single_function(bug)
        print(f"Extracted function for bug {bug.get_identifier()}: {result}\n")

        if result is None:
            return None, None, None

        buggy_code, fixed_code = result

        if not self.keep_comments:
            buggy_code_prompt = remove_java_comments(buggy_code)
            fixed_code_prompt = remove_java_comments(fixed_code)
        else:
            buggy_code_prompt = buggy_code
            fixed_code_prompt = fixed_code

        buggy_code_prompt = remove_empty_lines(buggy_code_prompt)
        fixed_code_prompt = remove_empty_lines(fixed_code_prompt)

        if self.single_chunk:
            prompt = self.build_single_cloze_prompt(
                buggy_code_prompt, fixed_code_prompt
            )
            target = self.build_single_or2_target(
                buggy_code_prompt, fixed_code_prompt
            )
        else:
            # This path is not the exact RepairLLaMA IR4-OR2 path.
            prompt = self.build_multi_cloze_prompt(
                buggy_code_prompt, fixed_code_prompt
            )
            target = fixed_code_prompt

        if self.begin_fim:
            prompt = f"{self.begin_fim}{prompt}"
        if self.end_fim:
            prompt = f"{prompt}{self.end_fim}"

        return buggy_code_prompt, target, prompt

    def prompt(self, bug: Bug) -> dict[str, Optional[str]]:
        result = {
            "identifier": bug.get_identifier(),
            "buggy_code": None,
            "fixed_code": None,   # now OR2 target when single_chunk=True
            "prompt_strategy": self.strategy_name,
            "prompt": None,
            "ground_truth": bug.get_ground_truth(),
        }

        diff = PatchSet(bug.get_ground_truth())

        if len(diff) != 1:
            return result

        (
            result["buggy_code"],
            result["fixed_code"],
            result["prompt"],
        ) = self.cloze_prompt(bug)

        return result