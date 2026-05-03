from elleelleaime.evaluate.strategies.text.replace import ReplaceEvaluationStrategy


def test_normalize_generation_reconstructs_infilling_prompt():
    strategy = ReplaceEvaluationStrategy(use_cache=False)
    sample = {
        "prompt": "public int add(int a, int b) {\n    <FILL_ME>\n}\n",
        "buggy_code": "public int add(int a, int b) {\n    return a - b;\n}\n",
        "fixed_code": "public int add(int a, int b) {\n    return a + b;\n}\n",
    }

    generation = "return a + b;"

    assert (
        strategy._normalize_generation(sample, generation)
        == "public int add(int a, int b) {\n    return a + b;\n}\n"
    )


def test_normalize_generation_keeps_full_function_output():
    strategy = ReplaceEvaluationStrategy(use_cache=False)
    sample = {
        "prompt": "public int add(int a, int b) {\n    <FILL_ME>\n}\n",
        "buggy_code": "public int add(int a, int b) {\n    return a - b;\n}\n",
        "fixed_code": "public int add(int a, int b) {\n    return a + b;\n}\n",
    }

    generation = "public int add(int a, int b) {\n    return a + b;\n}\n"

    assert strategy._normalize_generation(sample, generation) == generation
