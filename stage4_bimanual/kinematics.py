"""Kinematics utilities and Damped Least Squares (DLS) Inverse Kinematics for SO-101 arms."""

from typing import Any
import numpy as np

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from stage4_bimanual.constants import (
    ARM_A_JOINTS,
    ARM_B_JOINTS,
    ArmIdentifier,
)


class DLSInverseKinematics:
    """Damped Least Squares IK solver for 5-DOF / 6-DOF SO-101 robotic arms in MuJoCo."""

    def __init__(
        self,
        model: Any,
        data: Any,
        damping: float = 0.005,
        step_size: float = 0.4,
        max_iterations: int = 150,
        tolerance_m: float = 0.003,
    ):
        self.model = model
        self.data = data
        self.damping = damping
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tolerance_m = tolerance_m

    def solve(
        self,
        arm: ArmIdentifier,
        target_pos_m: np.ndarray | list[float] | tuple[float, float, float],
    ) -> tuple[bool, list[float], float]:
        """Compute joint angles (radians) to reach target_pos_m.

        Returns:
            (converged, joint_angles, residual_distance_m)
        """
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return False, [0.0] * 5, 999.0

        target = np.asarray(target_pos_m, dtype=np.float64)
        prefix = arm.lower()
        site_name = f"{prefix}_gripperframe"

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in MuJoCo model.")

        joint_names = ARM_A_JOINTS if arm == "A" else ARM_B_JOINTS
        jnt_qpos_indices = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in joint_names
        ]
        jnt_dof_indices = [
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in joint_names
        ]

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        converged = False

        for _ in range(self.max_iterations):
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[site_id]
            error = target - current_pos
            err_norm = float(np.linalg.norm(error))

            if err_norm < self.tolerance_m:
                converged = True
                break

            mujoco.mj_jacSite(self.model, self.data, jacp, None, site_id)
            J = jacp[:, jnt_dof_indices]

            # Damped Least Squares update: dq = J^T * (J * J^T + lambda^2 * I)^(-1) * error
            lambda_matrix = (self.damping ** 2) * np.eye(3)
            delta_q = J.T @ np.linalg.solve(J @ J.T + lambda_matrix, error)

            for i, q_idx in enumerate(jnt_qpos_indices):
                self.data.qpos[q_idx] += self.step_size * delta_q[i]

                # Clamp to joint range limits
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[i])
                limit_range = self.model.jnt_range[joint_id]
                self.data.qpos[q_idx] = np.clip(self.data.qpos[q_idx], limit_range[0], limit_range[1])

        mujoco.mj_forward(self.model, self.data)
        final_pos = self.data.site_xpos[site_id]
        final_error = float(np.linalg.norm(target - final_pos))
        joint_angles = [float(self.data.qpos[idx]) for idx in jnt_qpos_indices]

        return converged or (final_error < 0.01), joint_angles, final_error
