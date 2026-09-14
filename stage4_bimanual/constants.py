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
GRIPPER_OPEN: Final[float] = 1.60
GRIPPER_CLOSED: Final[float] = -0.10
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


# Tabletop Standard Altitude Planes (meters)
ALTITUDE_SURFACE: Final[float] = 0.70
ALTITUDE_GRASP_PLATE: Final[float] = 0.730   # Gripper pinch site height for plate rim contact
ALTITUDE_GRASP_MUG: Final[float] = 0.765     # Pinch site height for mug upper body grasp
ALTITUDE_GRASP_BOTTLE: Final[float] = 0.860  # Pinch site height for bottle neck grasp
ALTITUDE_SAFE_TRANSIT: Final[float] = 0.95   # Unobstructed 3D airspace above all tabletop objects
ALTITUDE_APPROACH_HIGH: Final[float] = 0.96  # High waypoint for approaching tall obstacles

# Standby configurations: compactly tucked back, clear of central tabletop workspace
ARM_A_STANDBY: Final[list[float]] = [-0.10, -0.80, 1.40, -0.60, 0.0]
ARM_B_STANDBY: Final[list[float]] = [ 0.10, -0.80, 1.40, -0.60, 0.0]

DRAWER_SLIDE_MAX_METERS: Final[float] = 0.15
DEFAULT_SUBSTEPS_PER_TRAJECTORY: Final[int] = 60

