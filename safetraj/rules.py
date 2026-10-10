"""
rules.py — deterministic, auditable checks on trajectory metrics.

Each rule compares one metric with a threshold and adds points when it fires:

  far_from_goal_end          d_final >= strict_threshold (0.30 m)                +2
  low_progress               progress < rule_progress_min (0.10 m)               +2
  omega_too_high             max |omega| > rule_omega_max (1.99 rad/s)           +2
  high_omega_jumps           max |omega_t - omega_{t-1}| > rule_domega_max       +1
                             (1.5 rad/s per sample)
  bad_heading_far_from_goal  d_final >= near_threshold (0.50 m) and final
                             heading error > rule_heading_far (1.0 rad)          +1

Result label from the points: 0-1 PASS, 2-4 REVIEW, 5+ FAIL.

The thresholds are hand-chosen heuristics (the 0.30 / 0.50 m goal thresholds
follow the thesis definitions); they are not learned or validated against
real collisions. PASS means "no rule fired", not "safe".
"""

from dataclasses import asdict, dataclass
from typing import Dict, List

from .config import SafeTrajConfig
from .metrics import TrajectoryMetrics

LABELS = ("PASS", "REVIEW", "FAIL")


@dataclass
class RuleResult:
    name: str
    triggered: bool
    value: float
    threshold: float
    points: int
    message: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CheckResult:
    label: str
    score: int
    rules: List[RuleResult]

    @property
    def triggered(self) -> List[str]:
        return [r.name for r in self.rules if r.triggered]

    def to_dict(self) -> Dict:
        return {"label": self.label, "score": self.score, "triggered": self.triggered,
                "rules": [r.to_dict() for r in self.rules]}


def label_from_score(score: int) -> str:
    if score <= 1:
        return "PASS"
    if score <= 4:
        return "REVIEW"
    return "FAIL"


def run_checks(m: TrajectoryMetrics, config: SafeTrajConfig) -> CheckResult:
    """Apply every rule to the metrics of one trajectory."""
    c = config
    rules = [
        RuleResult("far_from_goal_end", m.d_final >= c.strict_threshold, m.d_final, c.strict_threshold, 2,
                   f"final distance to goal {m.d_final:.2f} m (threshold {c.strict_threshold:.2f} m)"),
        RuleResult("low_progress", m.progress < c.rule_progress_min, m.progress, c.rule_progress_min, 2,
                   f"progress toward the goal {m.progress:+.2f} m (minimum {c.rule_progress_min:.2f} m)"),
        RuleResult("omega_too_high", m.omega_max_abs > c.rule_omega_max, m.omega_max_abs, c.rule_omega_max, 2,
                   f"max |omega| {m.omega_max_abs:.2f} rad/s (limit {c.rule_omega_max:.2f} rad/s)"),
        RuleResult("high_omega_jumps", m.domega_max > c.rule_domega_max, m.domega_max, c.rule_domega_max, 1,
                   f"largest change of omega between samples {m.domega_max:.2f} rad/s "
                   f"(limit {c.rule_domega_max:.2f})"),
        RuleResult("bad_heading_far_from_goal",
                   m.d_final >= c.near_threshold and m.heading_error_final > c.rule_heading_far,
                   m.heading_error_final, c.rule_heading_far, 1,
                   f"final heading error {m.heading_error_final:.2f} rad while {m.d_final:.2f} m from the goal"),
    ]
    score = sum(r.points for r in rules if r.triggered)
    return CheckResult(label=label_from_score(score), score=score, rules=rules)
