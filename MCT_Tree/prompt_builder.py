from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import difflib
import re


@dataclass
class PromptBuildConfig:
    # masking
    mask_token: str = "<extra_id_0>"          # "<FILL_ME>" or "<FILL_{}>" if you want ids
    extra_mask_token: bool = False         # add one extra mask at end
    single_chunk: bool = True              # True => single-cloze, False => multi-cloze

    # formatting / options
    keep_buggy_code: bool = False          # comment out removed buggy lines before mask
    keep_comments: bool = False            # if False, strip Java comments before diff/prompt
    begin_fim: Optional[str] = None        # e.g. "<PRE>"
    end_fim: Optional[str] = None          # e.g. "<SUF>"


# ------------------------
# Preprocessing helpers
# ------------------------

def _normalize_newlines(s: str) -> str:
    # Make regex/diff stable across CRLF/LF
    return (s or "").replace("\r\n", "\n").replace("\r", "\n")


def _remove_java_comments(source: str) -> str:
    """
    Removes // and /* */ comments while trying to preserve strings/chars.
    Always returns a string (falls back to original on error).
    """
    try:
        NORMAL, SINGLE_COMMENT, MULTI_COMMENT, STRING_LITERAL, CHAR_LITERAL = range(5)

        state = NORMAL
        result: List[str] = []
        i = 0

        while i < len(source):
            if state == NORMAL:
                if source[i:i + 2] == "//":
                    state = SINGLE_COMMENT
                    i += 2
                elif source[i:i + 2] == "/*":
                    state = MULTI_COMMENT
                    i += 2
                elif source[i] == '"':
                    state = STRING_LITERAL
                    result.append(source[i])
                    i += 1
                elif source[i] == "'":
                    state = CHAR_LITERAL
                    result.append(source[i])
                    i += 1
                else:
                    result.append(source[i])
                    i += 1

            elif state == SINGLE_COMMENT:
                if source[i] == "\n":
                    state = NORMAL
                    result.append(source[i])
                i += 1

            elif state == MULTI_COMMENT:
                if source[i:i + 2] == "*/":
                    state = NORMAL
                    i += 2
                else:
                    i += 1

            elif state == STRING_LITERAL:
                if source[i] == "\\" and i + 1 < len(source):
                    result.append(source[i])
                    result.append(source[i + 1])
                    i += 2
                elif source[i] == '"':
                    state = NORMAL
                    result.append(source[i])
                    i += 1
                else:
                    result.append(source[i])
                    i += 1

            elif state == CHAR_LITERAL:
                if source[i] == "\\" and i + 1 < len(source):
                    result.append(source[i])
                    result.append(source[i + 1])
                    i += 2
                elif source[i] == "'":
                    state = NORMAL
                    result.append(source[i])
                    i += 1
                else:
                    result.append(source[i])
                    i += 1

        return "".join(result)

    except Exception:
        # safest fallback: do not break pipeline
        return source


def _remove_empty_lines(source: str) -> str:
    """
    Remove whitespace-only lines. Assumes normalized newlines (\n).
    """
    return re.sub(r"(?m)^[ \t]*\n", "", source)


def _leading_ws(s: str) -> str:
    m = re.match(r"^\s*", s)
    return m.group(0) if m else ""


def _mask_token(cfg: PromptBuildConfig, mask_id: int) -> str:
    if "{}" in cfg.mask_token:
        return cfg.mask_token.format(mask_id)
    return cfg.mask_token


# ------------------------
# Full-context cloze builders (SequenceMatcher)
# ------------------------

def _build_single_cloze_prompt_full(buggy: str, fixed: str, cfg: PromptBuildConfig) -> str:
    """
    Single mask that covers from the FIRST difference to the LAST difference.
    Uses full sequence diff (not unified diff context).
    """
    a_lines = buggy.splitlines(keepends=True)
    b_lines = fixed.splitlines(keepends=True)

    sm = difflib.SequenceMatcher(a=a_lines, b=b_lines)
    opcodes = sm.get_opcodes()

    non_equal = [op for op in opcodes if op[0] != "equal"]
    if not non_equal:
        # identical: still make it infillable
        return buggy + ("" if buggy.endswith("\n") else "\n") + _mask_token(cfg, 0) + "\n"

    # first and last change on the buggy side
    _, a1, _, _, _ = non_equal[0]
    _, _, a_last_end, _, _ = non_equal[-1]

    prefix = "".join(a_lines[:a1])
    changed_chunk = "".join(a_lines[a1:a_last_end])
    suffix = "".join(a_lines[a_last_end:])

    # indentation near where mask goes
    indent = ""
    if a1 < len(a_lines) and a_lines[a1].strip():
        indent = _leading_ws(a_lines[a1])
    elif a1 > 0:
        indent = _leading_ws(a_lines[a1 - 1])

    if cfg.keep_buggy_code and changed_chunk.strip():
        buggy_comment = "// buggy code\n"
        for line in changed_chunk.splitlines(keepends=True):
            buggy_comment += "//" + line
        return prefix + buggy_comment + f"{indent}{_mask_token(cfg, 0)}\n" + suffix

    return prefix + f"{indent}{_mask_token(cfg, 0)}\n" + suffix


def _build_multi_cloze_prompt_full(buggy: str, fixed: str, cfg: PromptBuildConfig) -> str:
    """
    One mask per non-equal opcode block (replace/delete/insert).
    Preserves all unchanged lines from buggy (full context).
    """
    a_lines = buggy.splitlines(keepends=True)
    b_lines = fixed.splitlines(keepends=True)

    sm = difflib.SequenceMatcher(a=a_lines, b=b_lines)
    opcodes = sm.get_opcodes()

    out: List[str] = []
    mask_id = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            out.extend(a_lines[i1:i2])
            continue

        # indentation near this change in buggy
        indent = ""
        if i1 < len(a_lines) and a_lines[i1].strip():
            indent = _leading_ws(a_lines[i1])
        elif i1 > 0:
            indent = _leading_ws(a_lines[i1 - 1])

        # optional: show buggy chunk being changed
        if cfg.keep_buggy_code and tag in ("replace", "delete"):
            out.append("// buggy code\n")
            for line in a_lines[i1:i2]:
                out.append("//" + line)

        out.append(f"{indent}{_mask_token(cfg, mask_id)}\n")
        mask_id += 1

    if cfg.extra_mask_token:
        out.append(f"{_mask_token(cfg, mask_id)}\n")

    prompt = "".join(out)
    if prompt.strip() == "":
        prompt = _mask_token(cfg, 0)
    return prompt


# ------------------------
# Public API
# ------------------------

def PrpmtBuild(
    buggyCode: str,
    fixedCode: str,
    cfg: Optional[PromptBuildConfig] = None
) -> Tuple[str, str, str]:
    """
    Returns (buggy_code_used, fixed_code_used, prompt)

    buggy_code_used / fixed_code_used:
      - normalized newlines
      - optionally comment-stripped
      - empty lines removed
    """
    print("................................................")
    print("BuggyCode")
    print(buggyCode)
    print("..............................")
    print(fixedCode)
    

    cfg = cfg or PromptBuildConfig()

    buggy = _normalize_newlines(buggyCode)
    fixed = _normalize_newlines(fixedCode)

    if not cfg.keep_comments:
        buggy = _remove_java_comments(buggy)
        fixed = _remove_java_comments(fixed)

    buggy = _remove_empty_lines(buggy)
    fixed = _remove_empty_lines(fixed)

    if cfg.single_chunk:
        prompt = _build_single_cloze_prompt_full(buggy, fixed, cfg)
    else:
        prompt = _build_multi_cloze_prompt_full(buggy, fixed, cfg)

    if cfg.begin_fim:
        prompt = f"{cfg.begin_fim}{prompt}"
    if cfg.end_fim:
        prompt = f"{prompt}{cfg.end_fim}"
    
    print(prompt)

    return buggy, fixed, prompt