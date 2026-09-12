"""Hardware and simulation constants for dual SO-101 robotic arms."""

from dataclasses import dataclass
from typing import Final, Literal

ArmIdentifier = Literal["A", "B"]

# Actuator and joint names
ARM_A_JOINTS: Final[list[str]] = [
    "a_shoulder_pan",
    "a_shoulder_lift",
    "a_elbow_flex",
    "a_wrist_flex",
    "a_wrist_roll",
]

ARM_B_JOINTS: Final[list[str]] = [
    "b_shoulder_pan",
    "b_shoulder_lift",
    "b_elbow_flex",
    "b_wrist_flex",
    "b_wrist_roll",
]

ARM_A_ACTUATOR_OFFSET: Final[int] = 0
ARM_B_ACTUATOR_OFFSET: Final[int] = 6

ARM_A_GRIPPER_ACTUATOR: Final[int] = 5
ARM_B_GRIPPER_ACTUATOR: Final[int] = 11

# Gripper control values (STS3215 actuator angle in radians)
GRIPPER_OPEN: Final[float] = 1.0
GRIPPER_CLOSED: Final[float] = 0.1
GRIPPER_HALF: Final[float] = 0.6


@dataclass(frozen=True)
class ArmRestPose:
    """Neutral resting configuration for SO-101 arm joints."""
    shoulder_pan: float = 0.0
    shoulder_lift: float = 0.0
    elbow_flex: float = 0.0
    wrist_flex: float = 0.0
    wrist_roll: float = 0.0
    gripper: float = GRIPPER_OPEN

    def as_array(self) -> list[float]:
        return [
            self.shoulder_pan,
            self.shoulder_lift,
            self.elbow_flex,
            self.wrist_flex,
            self.wrist_roll,
            self.gripper,
        ]


# Calibrated waypoints for tabletop manipulation
WAYPOINT_HANDLE_REACH: Final[list[float]] = [-0.0, -0.268, 0.722, 0.53, 0.0]
WAYPOINT_HANDLE_PULL: Final[list[float]] = [-0.0, -0.698, 0.911, 1.306, 0.0]
WAYPOINT_DRAWER_CLEAR: Final[list[float]] = [-0.0, -0.400, 0.600, 0.60, 0.0]

WAYPOINT_PLATE_APPROACH: Final[list[float]] = [-0.0, -0.463, 0.521, 0.624, 0.0]
WAYPOINT_PLATE_GRASP: Final[list[float]] = [-0.0, -0.028, 0.552, 0.438, 0.0]
WAYPOINT_PLATE_TABLE_CENTER: Final[list[float]] = [-0.804, 0.215, 0.368, 0.216, -0.025]

WAYPOINT_MUG_APPROACH: Final[list[float]] = [0.177, -0.740, 0.714, 0.718, 0.005]
WAYPOINT_MUG_GRASP: Final[list[float]] = [0.179, -0.424, 0.802, 0.572, 0.005]
WAYPOINT_MUG_HOLD: Final[list[float]] = [0.179, -0.556, 0.776, 0.634, 0.005]

WAYPOINT_BOTTLE_GRASP: Final[list[float]] = [-0.573, 0.171, 0.254, 0.145, -0.019]
WAYPOINT_POUR_POSITION: Final[list[float]] = [-0.993, 0.544, -0.291, -0.180, 0.0]

DRAWER_SLIDE_MAX_METERS: Final[float] = 0.12
DEFAULT_SUBSTEPS_PER_TRAJECTORY: Final[int] = 60
