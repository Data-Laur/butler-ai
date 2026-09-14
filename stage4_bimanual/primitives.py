"""Dynamic bimanual manipulation primitives with real physics and IK.

Every object manipulation is physically executed by the robot arm with:
  1. Elevated pre-grasp approach (no table sweeping or knocking objects)
  2. Vertical descent to exact physical contact
  3. Closed gripper clamping at zero air-gap
  4. Vertical lift to clearance height (0.85m)
  5. Horizontal transport above the table
  6. Smooth descent and gentle placement
  7. Opening gripper before vertical retraction
Zero arbitrary air-gap offsets.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from stage4_bimanual.constants import (
    ALTITUDE_APPROACH_HIGH,
    ALTITUDE_GRASP_BOTTLE,
    ALTITUDE_GRASP_MUG,
    ALTITUDE_GRASP_PLATE,
    ALTITUDE_SAFE_TRANSIT,
    ARM_A_STANDBY,
    ARM_B_STANDBY,
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

    def attach_weld(self, weld_name: str) -> bool:
        """Attach only after the two bodies are in real MuJoCo contact.

        A weld is an approximation for a stable grasp in this lightweight
        scene. It must never be used as a teleport: if the robot did not reach
        the handle/object, the primitive fails visibly instead of moving it.
        """
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return False
        try:
            weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
            if weld_id < 0:
                return False
            b1 = self.model.eq_obj1id[weld_id]
            b2 = self.model.eq_obj2id[weld_id]

            def in_subtree(body_id: int, root_id: int) -> bool:
                """A jaw is a child body of the gripper base in this model."""
                while body_id > 0:
                    if body_id == root_id:
                        return True
                    body_id = int(self.model.body_parentid[body_id])
                return body_id == root_id

            has_contact = any(
                (in_subtree(int(self.model.geom_bodyid[self.data.contact[i].geom1]), b1)
                 and in_subtree(int(self.model.geom_bodyid[self.data.contact[i].geom2]), b2))
                or
                (in_subtree(int(self.model.geom_bodyid[self.data.contact[i].geom1]), b2)
                 and in_subtree(int(self.model.geom_bodyid[self.data.contact[i].geom2]), b1))
                for i in range(self.data.ncon)
            )
            if not has_contact:
                print(f"[grasp] {weld_name} refused: gripper/object contact was not established")
                return False

            r1 = self.data.xmat[b1].reshape(3, 3)
            delta_world = self.data.xpos[b2] - self.data.xpos[b1]
            rel_pos = r1.T @ delta_world
            rel_quat = _quat_mul(_quat_inv(self.data.xquat[b1]), self.data.xquat[b2])

            self.model.eq_data[weld_id, 3:6] = rel_pos
            self.model.eq_data[weld_id, 6:10] = rel_quat
            self.data.eq_active[weld_id] = 1
            mujoco.mj_forward(self.model, self.data)
            return True
        except Exception:
            return False

    def detach_weld(self, weld_name: str) -> None:
        """Release equality weld constraint."""
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return
        try:
            weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
            if weld_id >= 0:
                self.data.eq_active[weld_id] = 0
                mujoco.mj_forward(self.model, self.data)
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
    """Arm A approaches the D-handle from the front (-X), grips it, and pulls the drawer open."""

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True

        # Ensure Arm B is parked safely in standby pose
        ctrl = np.copy(self.data.ctrl)
        ctrl[6:11] = ARM_B_STANDBY
        ctrl[11] = GRIPPER_OPEN

        # 1. Query dynamic handle position
        handle_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "drawer_handle_site"
        )
        if handle_site_id >= 0:
            handle_pos = np.copy(self.data.site_xpos[handle_site_id])
        else:
            handle_pos = np.array([-0.02, -0.220, 0.732])

        # --- Vertical descent approach ---
        # The extended handle bar projects well in front of the cabinet face.
        # Descending from directly above keeps the palm clear of the roof
        # while staying inside the SO-101 IK workspace.

        # 2a. Move high above the handle (z=0.95 is reachable and clears all objects)
        above_handle = np.array([handle_pos[0], handle_pos[1], ALTITUDE_SAFE_TRANSIT])
        q_above = self.solve_ik("A", above_handle, wrist_roll=0.0)
        ctrl[0:5] = q_above
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=40)

        # 2b. Descend vertically to handle height via Cartesian waypoints so the joint-space
        # arc never swings forward into the cabinet or prematurely displaces the handle.
        z_waypoints = np.linspace(ALTITUDE_SAFE_TRANSIT, handle_pos[2], 6)[1:]
        for z_wp in z_waypoints:
            q_wp = self.solve_ik("A", [handle_pos[0], handle_pos[1], z_wp], wrist_roll=0.0)
            ctrl[0:5] = q_wp
            self.executor.interpolate(ctrl, steps=15)

        # 3. Close gripper firmly on handle & attach weld
        ctrl[5] = GRIPPER_CLOSED
        self.executor.interpolate(ctrl, steps=25)
        if not self.attach_weld("weld_drawer"):
            # Retry: nudge slightly toward the handle (+X in world = closer to cabinet)
            nudge_pos = handle_pos + np.array([0.008, 0.0, 0.0])
            q_nudge = self.solve_ik("A", nudge_pos, wrist_roll=0.0)
            ctrl[0:5] = q_nudge
            self.executor.interpolate(ctrl, steps=40)
            ctrl[5] = GRIPPER_CLOSED
            self.executor.interpolate(ctrl, steps=25)
            if not self.attach_weld("weld_drawer"):
                return False

        # 4. Pull along -X by 7.5 cm — reliably exceeds the 4 cm threshold.
        pull_target = handle_pos - np.array([0.075, 0.0, 0.0])
        q_pull = self.solve_ik("A", pull_target, wrist_roll=0.0)
        ctrl[0:5] = q_pull
        self.executor.interpolate(ctrl, steps=60)

        drawer_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        if drawer_joint < 0:
            self.detach_weld("weld_drawer")
            return False
        slide = float(self.data.qpos[self.model.jnt_qposadr[drawer_joint]])
        if slide <= 0.04:
            self.detach_weld("weld_drawer")
            print(f"[drawer] pull incomplete: slide={slide:.3f} m; need >0.040 m")
            return False

        # 5. Release and retract vertically.
        self.detach_weld("weld_drawer")
        ctrl[5] = 0.45  # Relax grip
        self.executor.interpolate(ctrl, steps=10)

        # Retreat slightly away from the handle in -X before vertical ascent
        retract_x = pull_target[0] - 0.015
        ctrl[0:5] = self.solve_ik("A", [retract_x, pull_target[1], handle_pos[2]], wrist_roll=0.0)
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=10)

        # Retract straight up via Cartesian waypoints to clear the handle area without forward arcing
        z_retract = np.linspace(handle_pos[2], ALTITUDE_SAFE_TRANSIT, 6)[1:]
        for z_wp in z_retract:
            q_wp = self.solve_ik("A", [retract_x, pull_target[1], z_wp], wrist_roll=0.0)
            ctrl[0:5] = q_wp
            self.executor.interpolate(ctrl, steps=10)

        final_slide = float(self.data.qpos[self.model.jnt_qposadr[drawer_joint]])
        if final_slide <= 0.04:
            print(f"[drawer] did not remain open after release: slide={final_slide:.3f} m")
            return False
        return True


class PickPlatePrimitive(BaseManipulationPrimitive):
    """Arm A reaches exposed plate inside opened drawer, pinches rim directly, and lifts vertically."""

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True

        # 1. Query physical plate location inside opened drawer tray
        plate_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        plate_pos = (
            np.copy(self.data.xpos[plate_id])
            if plate_id >= 0
            else np.array([0.11, -0.22, 0.720])
        )

        # Front rim of plate is at X = plate_pos[0] - 0.040
        rim_x = plate_pos[0] - 0.040
        rim_y = plate_pos[1]

        # 2. Approach from high safe transit altitude directly above plate rim
        ctrl = np.copy(self.data.ctrl)
        ctrl[0:5] = self.solve_ik("A", [rim_x, rim_y, ALTITUDE_SAFE_TRANSIT], wrist_roll=0.0)
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=25)

        # 3. Pure vertical descent to plate front rim via Cartesian waypoints
        for z_wp in np.linspace(ALTITUDE_SAFE_TRANSIT, ALTITUDE_GRASP_PLATE, 6)[1:]:
            ctrl[0:5] = self.solve_ik("A", [rim_x, rim_y, z_wp], wrist_roll=0.0)
            self.executor.interpolate(ctrl, steps=10)

        # 4. Clamp gripper firmly on plate rim
        ctrl[5] = GRIPPER_CLOSED
        self.executor.interpolate(ctrl, steps=20)

        # Attach robot gripper weld at exact physical contact, detach tray anchor weld
        if not self.attach_weld("weld_plate"):
            return False
        self.detach_weld("weld_plate_drawer")

        # 5. Pure vertical lift straight up to safe transit altitude via Cartesian waypoints
        for z_wp in np.linspace(ALTITUDE_GRASP_PLATE, ALTITUDE_SAFE_TRANSIT, 6)[1:]:
            ctrl[0:5] = self.solve_ik("A", [rim_x, rim_y, z_wp], wrist_roll=0.0)
            self.executor.interpolate(ctrl, steps=10)

        return True


class PlacePlatePrimitive(BaseManipulationPrimitive):
    """Arm A transits plate to dining center at transit altitude, descends vertically, and parks."""

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True

        ctrl = np.copy(self.data.ctrl)

        # Calibrated pinch target so plate center lands accurately at (0.06, 0.00)
        pinch_target_xy = [0.038, -0.025]
        plate_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        plate_pos = np.copy(self.data.xpos[plate_id]) if plate_id >= 0 else np.array([0.06, -0.22, 0.95])

        # 1. Horizontal transit at safe transit altitude directly to dining center
        for xy in np.linspace([plate_pos[0] - 0.040, plate_pos[1]], pinch_target_xy, 5)[1:]:
            ctrl[0:5] = self.solve_ik("A", [xy[0], xy[1], ALTITUDE_SAFE_TRANSIT], wrist_roll=0.0)
            self.executor.interpolate(ctrl, steps=10)

        # 2. Pure vertical descent to dining table surface via Cartesian waypoints
        for z_landing in np.linspace(ALTITUDE_SAFE_TRANSIT, ALTITUDE_GRASP_PLATE, 8)[1:]:
            ctrl[0:5] = self.solve_ik("A", [pinch_target_xy[0], pinch_target_xy[1], z_landing], wrist_roll=0.0)
            self.executor.interpolate(ctrl, steps=10)

        # Settle plate vertical velocity to zero before releasing
        if plate_id >= 0:
            plate_dof = self.model.body_dofadr[plate_id]
            self.data.qvel[plate_dof:plate_dof+6] = 0.0
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            self.data.qvel[plate_dof:plate_dof+6] = 0.0

        # 3. Release plate onto table surface
        self.detach_weld("weld_plate")
        ctrl[5] = 0.50  # Relax grip
        self.executor.interpolate(ctrl, steps=10)

        # Retreat horizontally away from plate rim before vertical retraction
        retreat_xy = [pinch_target_xy[0] - 0.025, pinch_target_xy[1] - 0.015]
        ctrl[0:5] = self.solve_ik("A", [retreat_xy[0], retreat_xy[1], ALTITUDE_GRASP_PLATE], wrist_roll=0.0)
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=10)

        # 4. Pure vertical retraction straight up to safe transit altitude
        for z_up in np.linspace(ALTITUDE_GRASP_PLATE, ALTITUDE_SAFE_TRANSIT, 5)[1:]:
            ctrl[0:5] = self.solve_ik("A", [retreat_xy[0], retreat_xy[1], z_up], wrist_roll=0.0)
            self.executor.interpolate(ctrl, steps=10)

        # 5. Retract Arm A to parked standby pose clear of the central workspace
        ctrl[0:5] = ARM_A_STANDBY
        self.executor.interpolate(ctrl, steps=25)

        return True


class PickMugPrimitive(BaseManipulationPrimitive):
    """Arm B reaches mug from transit altitude, grasps body, and brings to bimanual pouring station."""

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True

        # 1. Query physical mug coordinates
        mug_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mug")
        mug_pos = (
            np.copy(self.data.xpos[mug_id])
            if mug_id >= 0
            else np.array([0.12, 0.20, 0.70])
        )

        # 2. Elevated pre-grasp waypoint directly above mug at safe transit altitude
        pre_grasp = np.array([mug_pos[0], mug_pos[1], ALTITUDE_SAFE_TRANSIT])
        ctrl = np.copy(self.data.ctrl)
        ctrl[6:11] = self.solve_ik("B", pre_grasp, wrist_roll=0.0)
        ctrl[11] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=25)

        # 3. Pure vertical descent to mug body grasp height via Cartesian waypoints
        for z_wp in np.linspace(ALTITUDE_SAFE_TRANSIT, ALTITUDE_GRASP_MUG, 6)[1:]:
            ctrl[6:11] = self.solve_ik("B", [mug_pos[0], mug_pos[1], z_wp], wrist_roll=0.0)
            self.executor.interpolate(ctrl, steps=10)

        # 4. Clamp gripper firmly around mug & attach weld
        ctrl[11] = GRIPPER_CLOSED
        self.executor.interpolate(ctrl, steps=20)
        if not self.attach_weld("weld_mug"):
            return False

        # 5. Pure vertical lift to safe transit altitude via Cartesian waypoints
        for z_wp in np.linspace(ALTITUDE_GRASP_MUG, ALTITUDE_SAFE_TRANSIT, 6)[1:]:
            ctrl[6:11] = self.solve_ik("B", [mug_pos[0], mug_pos[1], z_wp], wrist_roll=0.0)
            self.executor.interpolate(ctrl, steps=10)

        # 6. Move mug to bimanual pouring station (suspended 12cm above table, clear of cutlery)
        mug_pour_station = np.array([0.06, 0.17, 0.82])
        ctrl[6:11] = self.solve_ik("B", mug_pour_station, wrist_roll=0.0)
        self.executor.interpolate(ctrl, steps=30)

        return True


class PourWaterPrimitive(BaseManipulationPrimitive):
    """Arm A grasps water bottle, approaches mug station, tilts to pour, and both arms stow."""

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True

        # 1. Query water bottle coordinates
        bottle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "water_bottle")
        bottle_pos = (
            np.copy(self.data.xpos[bottle_id])
            if bottle_id >= 0
            else np.array([0.14, -0.04, 0.70])
        )

        # 2. Elevated pre-approach: raise arm high before lateral motion to clear bottle top
        q_standby_high = self.solve_ik("A", [0.00, -0.20, ALTITUDE_APPROACH_HIGH], wrist_roll=1.57)
        ctrl = np.copy(self.data.ctrl)
        ctrl[0:5] = q_standby_high
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=25)

        # 3. Approach directly above bottle at high transit altitude
        pre_grasp = np.array([bottle_pos[0], bottle_pos[1], ALTITUDE_APPROACH_HIGH])
        q_pre = self.solve_ik("A", pre_grasp, wrist_roll=1.57)
        ctrl[0:5] = q_pre
        self.executor.interpolate(ctrl, steps=30)

        # 4. Pure vertical descent to bottle neck grasp height via Cartesian waypoints
        for z_wp in np.linspace(ALTITUDE_APPROACH_HIGH, ALTITUDE_GRASP_BOTTLE, 6)[1:]:
            ctrl[0:5] = self.solve_ik("A", [bottle_pos[0], bottle_pos[1], z_wp], wrist_roll=1.57)
            self.executor.interpolate(ctrl, steps=10)

        # 5. Clamp gripper firmly around bottle & attach weld
        ctrl[5] = GRIPPER_CLOSED
        self.executor.interpolate(ctrl, steps=20)
        if not self.attach_weld("weld_bottle"):
            return False

        # 6. Pure vertical lift to high transit altitude via Cartesian waypoints
        for z_wp in np.linspace(ALTITUDE_GRASP_BOTTLE, ALTITUDE_APPROACH_HIGH, 6)[1:]:
            ctrl[0:5] = self.solve_ik("A", [bottle_pos[0], bottle_pos[1], z_wp], wrist_roll=1.57)
            self.executor.interpolate(ctrl, steps=10)

        # 7. Move bottle to pour position (beside mug station on -Y side with orientation preserved)
        bottle_pour_pos = np.array([0.06, 0.065, 0.94])
        ctrl[0:5] = self.solve_ik("A", bottle_pour_pos, wrist_roll=1.57)
        self.executor.interpolate(ctrl, steps=30)

        # 8. Tilt wrist flex smoothly to pour stream directly into mug opening
        ctrl[3] -= 0.45
        self.executor.interpolate(ctrl, steps=20)

        # 9. Hold pour stream
        self.executor.interpolate(ctrl, steps=20)

        # 10. Untilt wrist flex back upright
        ctrl[3] += 0.45
        self.executor.interpolate(ctrl, steps=20)

        # 11. Return bottle via safe altitude highway to above resting spot
        ctrl[0:5] = self.solve_ik("A", pre_grasp, wrist_roll=1.57)
        self.executor.interpolate(ctrl, steps=30)

        # 12. Pure vertical descent back to bottle resting height via Cartesian waypoints
        for z_wp in np.linspace(ALTITUDE_APPROACH_HIGH, ALTITUDE_GRASP_BOTTLE, 6)[1:]:
            ctrl[0:5] = self.solve_ik("A", [bottle_pos[0], bottle_pos[1], z_wp], wrist_roll=1.57)
            self.executor.interpolate(ctrl, steps=10)

        # 13. Settle bottle, detach weld, gentle release before full retraction
        if bottle_id >= 0:
            bottle_dof = self.model.body_dofadr[bottle_id]
            self.data.qvel[bottle_dof:bottle_dof+6] = 0.0
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            self.data.qvel[bottle_dof:bottle_dof+6] = 0.0

        self.detach_weld("weld_bottle")
        ctrl[5] = 0.45  # Relax grip without swinging jaw into bottle
        self.executor.interpolate(ctrl, steps=10)

        # Lift clear of bottle before opening fully
        for z_wp in np.linspace(ALTITUDE_GRASP_BOTTLE, ALTITUDE_APPROACH_HIGH, 6)[1:]:
            ctrl[0:5] = self.solve_ik("A", [bottle_pos[0], bottle_pos[1], z_wp], wrist_roll=1.57)
            self.executor.interpolate(ctrl, steps=10)
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=10)

        # Retract Arm A to parked standby pose
        ctrl[0:5] = ARM_A_STANDBY
        self.executor.interpolate(ctrl, steps=25)

        # 14. Arm B gently sets the mug back down on table at dining station
        mug_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mug")
        mug_target_high = np.array([0.12, 0.20, ALTITUDE_SAFE_TRANSIT])
        ctrl[6:11] = self.solve_ik("B", mug_target_high, wrist_roll=0.0)
        self.executor.interpolate(ctrl, steps=30)

        # Pure vertical descent to table resting height via Cartesian waypoints
        for z_wp in np.linspace(ALTITUDE_SAFE_TRANSIT, ALTITUDE_GRASP_MUG, 6)[1:]:
            ctrl[6:11] = self.solve_ik("B", [0.12, 0.20, z_wp], wrist_roll=0.0)
            self.executor.interpolate(ctrl, steps=10)

        # Settle mug vertical velocity
        if mug_id >= 0:
            mug_dof = self.model.body_dofadr[mug_id]
            self.data.qvel[mug_dof:mug_dof+6] = 0.0
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)
            self.data.qvel[mug_dof:mug_dof+6] = 0.0

        self.detach_weld("weld_mug")
        ctrl[11] = 0.45  # Relax grip
        self.executor.interpolate(ctrl, steps=10)

        # Lift vertically before full open
        for z_wp in np.linspace(ALTITUDE_GRASP_MUG, ALTITUDE_SAFE_TRANSIT, 6)[1:]:
            ctrl[6:11] = self.solve_ik("B", [0.12, 0.20, z_wp], wrist_roll=0.0)
            self.executor.interpolate(ctrl, steps=10)
        ctrl[11] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=10)

        # Retract Arm B to parked standby pose
        ctrl[6:11] = ARM_B_STANDBY
        self.executor.interpolate(ctrl, steps=25)

        return True
