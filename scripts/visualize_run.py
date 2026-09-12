"""Interactive 3D real-time visualizer for the bimanual table-setting simulation.

Run with:
    python scripts/visualize_run.py
"""

import time
from pathlib import Path
import sys
import mujoco
import mujoco.viewer
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage4_bimanual.bimanual import reset_scene
from stage4_bimanual.sim import MuJoCoSim
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


def run_interactive():
    print("Initializing MuJoCo bimanual scene...")
    sim = reset_scene(seed=0)
    if not isinstance(sim, MuJoCoSim):
        print("Error: MuJoCo is required for the visualizer.")
        return

    m = sim.model
    d = sim.data

    print("\n" + "=" * 60)
    print("Launching MuJoCo 3D Viewer...")
    print("Controls:")
    print("  - Left click + drag:  Orbit camera")
    print("  - Right click + drag: Zoom camera")
    print("  - Middle click:       Pan camera")
    print("=" * 60 + "\n")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        time.sleep(1.0)

        def sync_step(steps=1, delay=0.01):
            for _ in range(steps):
                mujoco.mj_step(m, d)
                viewer.sync()
                time.sleep(delay)

        while viewer.is_running():
            print("Step 1: Arm A reaching and opening desktop drawer...")
            ctrl = np.copy(d.ctrl)
            ctrl[0:5] = WAYPOINT_HANDLE_REACH
            ctrl[5] = GRIPPER_OPEN
            for s in range(50):
                alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / 50))
                d.ctrl[:] = (1 - alpha) * d.ctrl + alpha * ctrl
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            ctrl[5] = GRIPPER_CLOSED
            for _ in range(20):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            start_c = np.copy(d.ctrl)
            ctrl[0:5] = WAYPOINT_HANDLE_PULL
            drawer_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
            drawer_qadr = m.jnt_qposadr[drawer_jnt]
            for s in range(60):
                alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / 60))
                d.ctrl[:] = (1 - alpha) * start_c + alpha * ctrl
                d.qpos[drawer_qadr] = DRAWER_SLIDE_MAX_METERS * alpha
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            ctrl[5] = GRIPPER_OPEN
            for _ in range(20):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            ctrl[0:5] = WAYPOINT_DRAWER_CLEAR
            for _ in range(30):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            print("Step 2: Arm A picking plate from drawer...")
            ctrl[0:5] = WAYPOINT_PLATE_APPROACH
            for _ in range(40):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            ctrl[0:5] = WAYPOINT_PLATE_GRASP
            for _ in range(30):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            ctrl[5] = GRIPPER_CLOSED
            for _ in range(20):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            start_c = np.copy(d.ctrl)
            ctrl[0:5] = WAYPOINT_PLATE_APPROACH
            plate_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "plate")
            start_pos = np.copy(d.xpos[plate_bid])
            target_pos = np.array([0.10, -0.22, 0.82])
            plate_jnt = m.body_jntadr[plate_bid]
            plate_qadr = m.jnt_qposadr[plate_jnt]
            for s in range(40):
                alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / 40))
                d.ctrl[:] = (1 - alpha) * start_c + alpha * ctrl
                d.qpos[plate_qadr:plate_qadr+3] = (1 - alpha) * start_pos + alpha * target_pos
                mujoco.mj_forward(m, d)
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            print("Step 3: Arm A placing plate on table center...")
            start_c = np.copy(d.ctrl)
            ctrl[0:5] = WAYPOINT_PLATE_TABLE_CENTER
            start_pos = np.copy(d.xpos[plate_bid])
            target_pos = np.array([0.05, 0.0, 0.715])
            for s in range(50):
                alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / 50))
                d.ctrl[:] = (1 - alpha) * start_c + alpha * ctrl
                d.qpos[plate_qadr:plate_qadr+3] = (1 - alpha) * start_pos + alpha * target_pos
                mujoco.mj_forward(m, d)
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            ctrl[5] = GRIPPER_OPEN
            for _ in range(20):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            ctrl[0:5] = [0.0, 0.0, 0.0, 0.0, 0.0]
            for _ in range(30):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            print("Step 4: Arm B picking and holding mug...")
            ctrl[6:11] = WAYPOINT_MUG_APPROACH
            ctrl[11] = GRIPPER_OPEN
            for _ in range(40):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            ctrl[6:11] = WAYPOINT_MUG_GRASP
            for _ in range(30):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            ctrl[11] = GRIPPER_CLOSED
            for _ in range(20):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            ctrl[6:11] = WAYPOINT_MUG_HOLD
            for _ in range(30):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            print("Step 5: Complementary Action — Arm A pouring water into Arm B's mug...")
            ctrl[0:5] = WAYPOINT_BOTTLE_GRASP
            ctrl[5] = GRIPPER_OPEN
            for _ in range(40):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            ctrl[5] = GRIPPER_CLOSED
            for _ in range(20):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            start_c = np.copy(d.ctrl)
            ctrl[0:5] = WAYPOINT_POUR_POSITION
            bottle_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "water_bottle")
            start_pos = np.copy(d.xpos[bottle_bid])
            target_pos = np.array([0.06, 0.12, 0.84])
            bottle_jnt = m.body_jntadr[bottle_bid]
            bottle_qadr = m.jnt_qposadr[bottle_jnt]
            for s in range(45):
                alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / 45))
                d.ctrl[:] = (1 - alpha) * start_c + alpha * ctrl
                d.qpos[bottle_qadr:bottle_qadr+3] = (1 - alpha) * start_pos + alpha * target_pos
                mujoco.mj_forward(m, d)
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            # Pour: tilt wrist
            ctrl[4] = 1.2
            for _ in range(50):
                sync_step(1, 0.02)
                if not viewer.is_running(): return

            print("Returning water bottle...")
            ctrl[4] = 0.0
            for _ in range(30):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            ctrl[0:5] = WAYPOINT_BOTTLE_GRASP
            for _ in range(35):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            ctrl[5] = GRIPPER_OPEN
            for _ in range(20):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            ctrl[0:5] = [0.0, 0.0, 0.0, 0.0, 0.0]
            for _ in range(30):
                sync_step(1, 0.015)
                if not viewer.is_running(): return

            print("\nSequence complete! You can inspect the table in 3D. Press ESC or close window to exit.")
            while viewer.is_running():
                sync_step(1, 0.02)


if __name__ == "__main__":
    run_interactive()
