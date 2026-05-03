from typing import Optional


class CompileResult:
    def __init__(
        self,
        result: Optional[bool],
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        returncode: Optional[int] = None,
    ) -> None:
        self.result = result
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    def is_passing(self) -> Optional[bool]:
        return self.result

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"CompileResult(result={self.result}, returncode={self.returncode}, "
            f"stdout={self.stdout!r}, stderr={self.stderr!r})"
        )
