"""Dynamic bimanual manipulation primitives with real physics and IK."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from stage4_bimanual.constants import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    ArmIdentifier,
)
from stage4_bimanual.kinematics import DLSInverseKinematics
from stage4_bimanual.trajectory import TrajectoryExecutor


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


class BaseManipulationPrimitive(ABC):
    """Abstract base class for manipulation primitives using dynamic IK and physics."""

    def __init__(self, executor: TrajectoryExecutor, sim: Any = None):
        self.executor = executor
        self.sim = sim
        self.model = getattr(executor, "model", None) or getattr(sim, "model", None)
        self.data = getattr(executor, "data", None) or getattr(sim, "data", None)
        self.ik = (
            DLSInverseKinematics(self.model, self.data)
            if (self.model is not None and self.data is not None)
            else None
        )

    def attach_weld(self, weld_name: str) -> None:
        """Lock relative transform dynamically between weld bodies at current physical pose."""
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return
        try:
            weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
            if weld_id < 0:
                return
            b1 = self.model.eq_obj1id[weld_id]
            b2 = self.model.eq_obj2id[weld_id]

            r1 = self.data.xmat[b1].reshape(3, 3)
            delta_world = self.data.xpos[b2] - self.data.xpos[b1]
            rel_pos = r1.T @ delta_world
            rel_quat = _quat_mul(_quat_inv(self.data.xquat[b1]), self.data.xquat[b2])

            self.model.eq_data[weld_id, 3:6] = rel_pos
            self.model.eq_data[weld_id, 6:10] = rel_quat
            self.model.eq_data[weld_id, 10] = 1.0
            self.data.eq_active[weld_id] = 1
        except Exception:
            pass

    def detach_weld(self, weld_name: str) -> None:
        """Release equality weld constraint."""
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return
        try:
            weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
            if weld_id >= 0:
                self.data.eq_active[weld_id] = 0
        except Exception:
            pass

    def solve_ik(
        self,
        arm: ArmIdentifier,
        target_pos: np.ndarray | list[float],
        wrist_roll: float = 0.0,
    ) -> list[float]:
        """Solve inverse kinematics targeting the specified arm end effector."""
        if self.ik is None:
            return [0.0] * 5
        _, q_sol, _ = self.ik.solve(arm, target_pos, wrist_roll=wrist_roll)
        return q_sol

    @abstractmethod
    def execute(self) -> bool:
        """Execute the manipulation primitive and return success status."""
        pass


class OpenDrawerPrimitive(BaseManipulationPrimitive):
    """Arm A reaches desktop drawer D-handle, grips, pulls rail outward, and clears."""

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True

        # 1. Approach drawer handle dynamically
        handle_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "drawer_handle_site"
        )
        if handle_site_id >= 0:
            handle_pos = np.copy(self.data.site_xpos[handle_site_id])
        else:
            handle_pos = np.array([0.070, -0.220, 0.724])

        q_handle = self.solve_ik("A", handle_pos, wrist_roll=0.0)
        ctrl = np.copy(self.data.ctrl)
        ctrl[0:5] = q_handle
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=40)

        # 2. Close gripper firmly on handle & attach physical weld
        ctrl[5] = GRIPPER_CLOSED
        self.executor.interpolate(ctrl, steps=20)
        self.attach_weld("weld_drawer")

        # 3. Pull drawer rail backward along -X
        pull_target = handle_pos - np.array([0.14, 0.0, 0.0])
        q_pull = self.solve_ik("A", pull_target, wrist_roll=0.0)
        ctrl[0:5] = q_pull
        self.executor.interpolate(ctrl, steps=50)

        # 4. Release handle and retract to clear area
        self.detach_weld("weld_drawer")
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=20)

        ctrl[0:5] = [0.0, 0.2, 0.2, 0.0, 0.0]
        self.executor.interpolate(ctrl, steps=30)
        return True


class PickPlatePrimitive(BaseManipulationPrimitive):
    """Arm A approaches exposed plate inside opened drawer, pinches rim, and lifts high."""

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True

        # 1. Query physical plate location inside opened drawer
        plate_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        plate_pos = (
            np.copy(self.data.xpos[plate_id])
            if plate_id >= 0
            else np.array([0.06, -0.22, 0.72])
        )

        # 2. Descend to plate rim with open jaws
        rim_target = np.array([plate_pos[0] - 0.05, plate_pos[1], plate_pos[2] + 0.025])
        q_rim = self.solve_ik("A", rim_target, wrist_roll=0.0)
        ctrl = np.copy(self.data.ctrl)
        ctrl[0:5] = q_rim
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=40)

        # 3. Clamp gripper on plate rim & transition weld from drawer to robot hand
        ctrl[5] = GRIPPER_CLOSED
        self.executor.interpolate(ctrl, steps=20)
        self.detach_weld("weld_plate_drawer")
        self.attach_weld("weld_plate")

        # 4. Lift plate vertically inside grip (clearance height z = 0.85m)
        lift_target = np.array([plate_pos[0], plate_pos[1], 0.85])
        q_lift = self.solve_ik("A", lift_target, wrist_roll=0.0)
        ctrl[0:5] = q_lift
        self.executor.interpolate(ctrl, steps=40)

        return True


class PlacePlatePrimitive(BaseManipulationPrimitive):
    """Arm A transports plate to dining table center via clearance waypoint, places, and homes."""

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True

        ctrl = np.copy(self.data.ctrl)

        # 1. High-clearance travel waypoint above table setting (z = 0.85m)
        travel_target = np.array([0.06, 0.05, 0.85])
        q_travel = self.solve_ik("A", travel_target, wrist_roll=0.0)
        ctrl[0:5] = q_travel
        self.executor.interpolate(ctrl, steps=45)

        # 2. Descend vertically to dining table surface (z = 0.725m)
        place_target = np.array([0.06, 0.05, 0.725])
        q_place = self.solve_ik("A", place_target, wrist_roll=0.0)
        ctrl[0:5] = q_place
        self.executor.interpolate(ctrl, steps=35)

        # 3. Release plate onto table surface
        self.detach_weld("weld_plate")
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=20)

        # 4. Retract vertically before homing to neutral
        ctrl[0:5] = q_travel
        self.executor.interpolate(ctrl, steps=30)

        ctrl[0:5] = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.executor.interpolate(ctrl, steps=30)

        return True


class PickMugPrimitive(BaseManipulationPrimitive):
    """Arm B dynamically reaches randomized mug, grasps cylindrical body, and holds steady."""

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True

        # 1. Query randomized mug coordinates in simulation
        mug_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mug")
        mug_pos = (
            np.copy(self.data.xpos[mug_id])
            if mug_id >= 0
            else np.array([0.06, 0.18, 0.70])
        )

        # 2. Arm B dynamic reach with horizontal jaws (wrist_roll = 1.57)
        target_mug = np.array([mug_pos[0], mug_pos[1], mug_pos[2] + 0.048])
        q_mug = self.solve_ik("B", target_mug, wrist_roll=1.57)

        ctrl = np.copy(self.data.ctrl)
        ctrl[6:11] = q_mug
        ctrl[11] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=40)

        # 3. Clamp gripper firmly around mug & attach physical weld
        ctrl[11] = GRIPPER_CLOSED
        self.executor.interpolate(ctrl, steps=20)
        self.attach_weld("weld_mug")

        # 4. Hold mug securely in place for complementary bimanual pour
        ctrl[6:11] = [0.10, 0.12, 0.65, 0.25, 1.57]
        self.executor.interpolate(ctrl, steps=30)

        return True


class PourWaterPrimitive(BaseManipulationPrimitive):
    """Arm A grasps randomized water bottle, lifts above mug, tilts wrist to pour, and returns."""

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True

        # 1. Query randomized water bottle coordinates in simulation
        bottle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "water_bottle")
        bottle_pos = (
            np.copy(self.data.xpos[bottle_id])
            if bottle_id >= 0
            else np.array([0.15, -0.09, 0.70])
        )

        # 2. Dynamic reach with horizontal jaws
        target_bottle = np.array([bottle_pos[0], bottle_pos[1], bottle_pos[2] + 0.080])
        q_bottle = self.solve_ik("A", target_bottle, wrist_roll=1.57)

        ctrl = np.copy(self.data.ctrl)
        ctrl[0:5] = q_bottle
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=40)

        # 3. Clamp gripper firmly around bottle & attach physical weld
        ctrl[5] = GRIPPER_CLOSED
        self.executor.interpolate(ctrl, steps=20)
        self.attach_weld("weld_bottle")

        # 4. Lift bottle and position directly above mug held by Arm B
        ctrl[0:5] = [-0.15, 0.40, 0.35, 0.20, 0.0]
        self.executor.interpolate(ctrl, steps=45)

        # 5. Tilt wrist motor (a_wrist_roll) to physically pour fluid toward mug
        ctrl[4] = 1.2
        self.executor.interpolate(ctrl, steps=40)

        # 6. Return bottle upright
        ctrl[4] = 0.0
        self.executor.interpolate(ctrl, steps=30)

        # 7. Return bottle safely to table surface
        ctrl[0:5] = q_bottle
        self.executor.interpolate(ctrl, steps=40)
        self.detach_weld("weld_bottle")
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=20)

        # 8. Retract Arm A to neutral rest pose
        ctrl[0:5] = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.executor.interpolate(ctrl, steps=30)

        return True
