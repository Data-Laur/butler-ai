"""Reusable bimanual manipulation primitives for dual SO-101 robotic arms."""

from abc import ABC, abstractmethod
import numpy as np

from stage4_bimanual.constants import (
    DRAWER_SLIDE_MAX_METERS,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    WAYPOINT_BOTTLE_GRASP,
    WAYPOINT_DRAWER_CLEAR,
    WAYPOINT_HANDLE_PULL,
    WAYPOINT_HANDLE_REACH,
    WAYPOINT_MUG_APPROACH,
    WAYPOINT_MUG_GRASP,
    WAYPOINT_MUG_HOLD,
    WAYPOINT_PLATE_APPROACH,
    WAYPOINT_PLATE_GRASP,
    WAYPOINT_PLATE_TABLE_CENTER,
    WAYPOINT_POUR_POSITION,
)
from stage4_bimanual.trajectory import TrajectoryExecutor


class BaseManipulationPrimitive(ABC):
    """Abstract base class for high-level manipulation primitives."""

    def __init__(self, executor: TrajectoryExecutor):
        self.executor = executor

    @abstractmethod
    def execute(self) -> bool:
        """Execute the manipulation primitive and return success status."""
        pass


class OpenDrawerPrimitive(BaseManipulationPrimitive):
    """Arm A reaches desktop drawer D-handle, closes gripper, pulls rail open, and retracts."""

    def execute(self) -> bool:
        # 1. Approach handle with open gripper
        ctrl = np.copy(self.executor.data.ctrl)
        ctrl[0:5] = WAYPOINT_HANDLE_REACH
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=60)

        # 2. Close gripper firmly on handle
        ctrl[5] = GRIPPER_CLOSED
        self.executor.interpolate(ctrl, steps=30)

        # 3. Pull drawer outward to maximum slide extension
        ctrl[0:5] = WAYPOINT_HANDLE_PULL
        self.executor.interpolate(
            ctrl, steps=80, slide_drawer_to=DRAWER_SLIDE_MAX_METERS
        )

        # 4. Release handle and retract to clear area
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=30)
        ctrl[0:5] = WAYPOINT_DRAWER_CLEAR
        self.executor.interpolate(ctrl, steps=40)

        return True


class PickPlatePrimitive(BaseManipulationPrimitive):
    """Arm A approaches exposed plate inside opened drawer, pinches rim, and lifts."""

    def execute(self) -> bool:
        # 1. Approach above open drawer
        ctrl = np.copy(self.executor.data.ctrl)
        ctrl[0:5] = WAYPOINT_PLATE_APPROACH
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=60)

        # 2. Descend to plate rim
        ctrl[0:5] = WAYPOINT_PLATE_GRASP
        self.executor.interpolate(ctrl, steps=40)

        # 3. Grasp plate rim
        ctrl[5] = GRIPPER_CLOSED
        self.executor.interpolate(ctrl, steps=30)

        # 4. Lift plate vertically
        ctrl[0:5] = WAYPOINT_PLATE_APPROACH
        self.executor.interpolate(
            ctrl,
            steps=50,
            carried_object="plate",
            target_object_pos=np.array([0.10, -0.22, 0.82]),
        )

        return True


class PlacePlatePrimitive(BaseManipulationPrimitive):
    """Arm A carries plate from drawer to table center, sets down, releases, and homes."""

    def execute(self) -> bool:
        # 1. Transport plate to dining table center
        ctrl = np.copy(self.executor.data.ctrl)
        ctrl[0:5] = WAYPOINT_PLATE_TABLE_CENTER
        self.executor.interpolate(
            ctrl,
            steps=80,
            carried_object="plate",
            target_object_pos=np.array([0.05, 0.0, 0.715]),
        )

        # 2. Release gripper
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=30)

        # 3. Retract Arm A to neutral rest pose
        ctrl[0:5] = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.executor.interpolate(ctrl, steps=50)

        return True


class PickMugPrimitive(BaseManipulationPrimitive):
    """Arm B reaches upright mug, grasps cylindrical body, and holds steady."""

    def execute(self) -> bool:
        # 1. Arm B approach mug position
        ctrl = np.copy(self.executor.data.ctrl)
        ctrl[6:11] = WAYPOINT_MUG_APPROACH
        ctrl[11] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=60)

        # 2. Descend to mug center of mass
        ctrl[6:11] = WAYPOINT_MUG_GRASP
        self.executor.interpolate(ctrl, steps=40)

        # 3. Grasp mug body firmly
        ctrl[11] = GRIPPER_CLOSED
        self.executor.interpolate(ctrl, steps=30)

        # 4. Hold mug securely ready for complementary bimanual pour
        ctrl[6:11] = WAYPOINT_MUG_HOLD
        self.executor.interpolate(ctrl, steps=40)

        return True


class PourWaterPrimitive(BaseManipulationPrimitive):
    """Arm A grasps water bottle, positions above mug held by Arm B, tilts wrist, and returns."""

    def execute(self) -> bool:
        # 1. Arm A reach and grasp water bottle
        ctrl = np.copy(self.executor.data.ctrl)
        ctrl[0:5] = WAYPOINT_BOTTLE_GRASP
        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=60)

        ctrl[5] = GRIPPER_CLOSED
        self.executor.interpolate(ctrl, steps=30)

        # 2. Lift bottle and position above mug
        ctrl[0:5] = WAYPOINT_POUR_POSITION
        self.executor.interpolate(
            ctrl,
            steps=70,
            carried_object="water_bottle",
            target_object_pos=np.array([0.06, 0.12, 0.84]),
        )

        # 3. Tilt wrist (a_wrist_roll) to pour fluid (Complementary Action!)
        ctrl[4] = 1.2
        self.executor.interpolate(ctrl, steps=80)

        # 4. Return bottle upright
        ctrl[4] = 0.0
        self.executor.interpolate(ctrl, steps=50)

        # 5. Place bottle back onto table
        ctrl[0:5] = WAYPOINT_BOTTLE_GRASP
        self.executor.interpolate(
            ctrl,
            steps=60,
            carried_object="water_bottle",
            target_object_pos=np.array([0.12, -0.04, 0.78]),
        )

        ctrl[5] = GRIPPER_OPEN
        self.executor.interpolate(ctrl, steps=30)

        # 6. Return Arm A to neutral rest pose
        ctrl[0:5] = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.executor.interpolate(ctrl, steps=50)

        return True
