from dataclasses import dataclass
from typing import Optional

@dataclass
class TestResult:
    success: bool
    tests_run: Optional[int] = None
    failures: Optional[int] = None
    errors: Optional[int] = None

    def is_passing(self) -> bool:
        return self.success

    @property
    def failed(self) -> Optional[int]:
        if self.failures is None or self.errors is None:
            return None
        return self.failures + self.errors

    @property
    def passed(self) -> Optional[int]:
        if self.tests_run is None or self.failures is None or self.errors is None:
            return None
        return self.tests_run - self.failures - self.errors

    @property
    def pass_rate(self) -> Optional[float]:
        if self.tests_run is None or self.tests_run == 0:
            return None
        p = self.passed
        return None if p is None else (p / self.tests_run)
