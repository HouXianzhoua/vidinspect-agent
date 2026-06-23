from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from vidinspect_agent.models import CheckResult


class BaseChecker(ABC):
    name: str

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        """Run checks against a video file."""
