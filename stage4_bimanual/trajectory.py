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

    def set_freejoint_translation(
        self,
        object_name: str,
        position_m: np.ndarray | list[float] | tuple[float, float, float],
    ) -> None:
        """Update the 3D translation coordinates of a freejoint body."""
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
            if body_id >= 0:
                jnt_id = self.model.body_jntadr[body_id]
                if jnt_id >= 0:
                    q_adr = self.model.jnt_qposadr[jnt_id]
                    self.data.qpos[q_adr : q_adr + 3] = position_m
                    mujoco.mj_forward(self.model, self.data)
        except Exception:
            pass

    def interpolate(
        self,
        target_ctrl: np.ndarray | list[float],
        steps: int = DEFAULT_SUBSTEPS_PER_TRAJECTORY,
        slide_drawer_to: float | None = None,
        carried_object: str | None = None,
        target_object_pos: np.ndarray | list[float] | None = None,
    ) -> None:
        """Smoothly interpolate actuator controls using cosine easing and step physics.

        Args:
            target_ctrl: Desired 12-element actuator target array.
            steps: Number of simulation substeps for smooth transition.
            slide_drawer_to: Optional target displacement for the linear drawer slide.
            carried_object: Optional name of object currently grasped by the robot.
            target_object_pos: Target 3D coordinates for the carried object.
        """
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return

        start_ctrl = np.copy(self.data.ctrl)
        target_ctrl_arr = np.asarray(target_ctrl, dtype=np.float64)

        # Prepare drawer tracking
        drawer_q_adr = None
        start_slide = 0.0
        if slide_drawer_to is not None:
            try:
                drawer_jnt = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide"
                )
                if drawer_jnt >= 0:
                    drawer_q_adr = self.model.jnt_qposadr[drawer_jnt]
                    start_slide = float(self.data.qpos[drawer_q_adr])
            except Exception:
                drawer_q_adr = None

        # Prepare carried object tracking
        start_obj_pos = None
        target_obj_arr = None
        if carried_object and target_object_pos is not None:
            try:
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, carried_object
                )
                if body_id >= 0:
                    start_obj_pos = np.copy(self.data.xpos[body_id])
                    target_obj_arr = np.asarray(target_object_pos, dtype=np.float64)
            except Exception:
                start_obj_pos = None

        # Step simulation along smooth cosine S-curve
        for s in range(steps):
            alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / steps))
            self.data.ctrl[:] = start_ctrl + alpha * (target_ctrl_arr - start_ctrl)

            if drawer_q_adr is not None and slide_drawer_to is not None:
                self.data.qpos[drawer_q_adr] = start_slide + alpha * (slide_drawer_to - start_slide)

            if carried_object and start_obj_pos is not None and target_obj_arr is not None:
                cur_pos = start_obj_pos + alpha * (target_obj_arr - start_obj_pos)
                self.set_freejoint_translation(carried_object, cur_pos)

            mujoco.mj_step(self.model, self.data)
