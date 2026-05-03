from pathlib import Path

import pytest

from elleelleaime.core.benchmarks.humanevaljava.humanevaljava import HumanEvalJava


def test_initialize_fails_with_clear_error_when_benchmark_is_incomplete(tmp_path: Path):
    benchmark = HumanEvalJava(path=tmp_path)

    with pytest.raises(RuntimeError, match="git submodule update --init --recursive"):
        benchmark.initialize()
