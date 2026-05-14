from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class PrecheckResult:
    score: float
    confidence: ConfidenceLevel
    run_enhanced_analysis: bool
    view_type: str
    summary: str
    reasons: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

