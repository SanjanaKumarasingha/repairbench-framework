from elleelleaime.chatrepair.runner import _build_infill_prompt, _extract_code_block


def test_extract_code_block_prefers_java_block():
    response = "Analysis\n```java\nreturn 1;\n```\n"
    assert _extract_code_block(response) == "return 1;"


def test_build_infill_prompt_for_single_line_replace():
    buggy = "void f() {\n    a();\n    b();\n}\n"
    fixed = "void f() {\n    a();\n    c();\n}\n"
    result = _build_infill_prompt(buggy, fixed)

    assert result is not None
    kind, function_with_infill, buggy_fragment = result
    assert kind == "single_line"
    assert ">>>[INFILL]<<<" in function_with_infill
    assert buggy_fragment.strip() == "b();"
