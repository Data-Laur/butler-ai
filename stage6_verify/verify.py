"""Stage 6: task-aware postcondition checks for simulation recovery."""

from __future__ import annotations

import math

from common.types import ActionType, SceneState, Task, VerifyResult

TABLE_DESTINATIONS = {"plate": (0.06, 0.00), "mug": (0.12, 0.20)}
POSITION_TOLERANCE_M = 0.045


def _xy_close(actual: tuple[float, float, float], expected: tuple[float, float]) -> bool:
    return math.dist(actual[:2], expected) <= POSITION_TOLERANCE_M


def verify(scene_after: SceneState, task: Task) -> VerifyResult:
    """Check observable drawer and placement postconditions.

    Fluid transfer is not represented in the current MuJoCo scene, so a pour
    is reported as a pose-only proxy rather than falsely asserting water level.
    """
    failures: list[str] = []
    notes: list[str] = []
    for step in task.steps:
        if step.action is ActionType.OPEN_DRAWER:
            if scene_after.drawers.get(step.target or "top_drawer") != "open":
                failures.append("drawer is not open")
        elif step.action is ActionType.PLACE and step.object in TABLE_DESTINATIONS:
            actual = scene_after.objects.get(step.object)
            expected = TABLE_DESTINATIONS[step.object]
            if actual is None or not _xy_close(actual, expected):
                failures.append(f"{step.object} is not at its table destination")
        elif step.action is ActionType.POUR:
            mug = scene_after.objects.get(step.into or "mug")
            if mug is None:
                failures.append("mug is not observable after pour")
            else:
                notes.append("pour checked as pose-only proxy (no fluid model)")

    if failures:
        return VerifyResult(ok=False, replan=True, details="; ".join(failures))
    return VerifyResult(ok=True, replan=False, details="; ".join(notes) or "observable postconditions satisfied")
