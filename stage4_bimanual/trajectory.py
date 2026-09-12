"""Smooth trajectory interpolation and actuation executor for MuJoCo simulation."""

from typing import Any
import numpy as np

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from stage4_bimanual.constants import DEFAULT_SUBSTEPS_PER_TRAJECTORY


class TrajectoryExecutor:
    """Executes smooth joint-space trajectories in MuJoCo with physics stepping."""

    def __init__(self, model: Any, data: Any):
        self.model = model
        self.data = data

    def interpolate(
        self,
        target_ctrl: np.ndarray | list[float],
        steps: int = DEFAULT_SUBSTEPS_PER_TRAJECTORY,
    ) -> None:
        """Smoothly interpolate actuator controls using cosine easing and step physics.

        Args:
            target_ctrl: Desired 12-element actuator target array.
            steps: Number of simulation substeps for smooth transition.
        """
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return

        start_ctrl = np.copy(self.data.ctrl)
        target_ctrl_arr = np.asarray(target_ctrl, dtype=np.float64)

        # Step simulation along smooth cosine S-curve
        for s in range(steps):
            alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / steps))
            self.data.ctrl[:] = start_ctrl + alpha * (target_ctrl_arr - start_ctrl)
            mujoco.mj_step(self.model, self.data)
