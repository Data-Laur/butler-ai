"""Script to record the bimanual simulation execution into images and an animated GIF."""

from pathlib import Path
import sys
from PIL import Image
import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
from stage4_bimanual.bimanual import reset_scene
from stage4_bimanual.sim import MuJoCoSim
from stage4_bimanual.trajectory import TrajectoryExecutor

_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _ROOT / "docs" / "visualizations"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_ARTIFACT_DIR = Path(r"C:\Users\Mannan\.gemini\antigravity-ide\brain\a9486265-30e2-4033-b7a1-30a3bbec5b4a")


def record():
    sim = reset_scene(seed=0)
    if not isinstance(sim, MuJoCoSim):
        print("MuJoCo simulation not available.")
        return

    m = sim.model
    d = sim.data
    renderer = mujoco.Renderer(m, height=480, width=640)

    # Perspective camera configuration
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = [0.08, -0.05, 0.75]
    cam.distance = 1.05
    cam.elevation = -28
    cam.azimuth = 135

    frames: list[Image.Image] = []

    def capture_frame():
        renderer.update_scene(d, camera=cam)
        img_arr = renderer.render()
        frames.append(Image.fromarray(img_arr))

    executor = TrajectoryExecutor(m, d)

    # Initial frame
    capture_frame()

    print("[1/5] Executing and recording: Open drawer...")
    # Step 1: Open Drawer
    ctrl = np.copy(d.ctrl)
    ctrl[0:5] = WAYPOINT_HANDLE_REACH
    ctrl[5] = GRIPPER_OPEN
    for s in range(25):
        alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / 25))
        d.ctrl[:] = (1 - alpha) * d.ctrl + alpha * ctrl
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    ctrl[5] = GRIPPER_CLOSED
    for s in range(15):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    start_c = np.copy(d.ctrl)
    ctrl[0:5] = WAYPOINT_HANDLE_PULL
    drawer_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    drawer_qadr = m.jnt_qposadr[drawer_jnt]
    for s in range(35):
        alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / 35))
        d.ctrl[:] = (1 - alpha) * start_c + alpha * ctrl
        d.qpos[drawer_qadr] = DRAWER_SLIDE_MAX_METERS * alpha
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    # Save Step 1 Snapshot
    step1_img = frames[-1]
    step1_img.save(_OUT_DIR / "step1_drawer_open.png")

    ctrl[5] = GRIPPER_OPEN
    for s in range(15):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    ctrl[0:5] = WAYPOINT_DRAWER_CLEAR
    for s in range(15):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    print("[2/5] Executing and recording: Pick plate from drawer...")
    # Step 2: Pick Plate
    ctrl[0:5] = WAYPOINT_PLATE_APPROACH
    for s in range(25):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    ctrl[0:5] = WAYPOINT_PLATE_GRASP
    for s in range(20):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    ctrl[5] = GRIPPER_CLOSED
    for s in range(15):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    start_c = np.copy(d.ctrl)
    ctrl[0:5] = WAYPOINT_PLATE_APPROACH
    plate_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "plate")
    start_pos = np.copy(d.xpos[plate_bid])
    target_pos = np.array([0.10, -0.22, 0.82])
    plate_jnt = m.body_jntadr[plate_bid]
    plate_qadr = m.jnt_qposadr[plate_jnt]
    for s in range(25):
        alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / 25))
        d.ctrl[:] = (1 - alpha) * start_c + alpha * ctrl
        d.qpos[plate_qadr:plate_qadr+3] = (1 - alpha) * start_pos + alpha * target_pos
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    # Save Step 2 Snapshot
    step2_img = frames[-1]
    step2_img.save(_OUT_DIR / "step2_plate_lift.png")

    print("[3/5] Executing and recording: Place plate on table...")
    # Step 3: Place Plate
    start_c = np.copy(d.ctrl)
    ctrl[0:5] = WAYPOINT_PLATE_TABLE_CENTER
    start_pos = np.copy(d.xpos[plate_bid])
    target_pos = np.array([0.05, 0.0, 0.715])
    for s in range(35):
        alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / 35))
        d.ctrl[:] = (1 - alpha) * start_c + alpha * ctrl
        d.qpos[plate_qadr:plate_qadr+3] = (1 - alpha) * start_pos + alpha * target_pos
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    ctrl[5] = GRIPPER_OPEN
    for s in range(15):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    ctrl[0:5] = [0.0, 0.0, 0.0, 0.0, 0.0]
    for s in range(25):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    # Save Step 3 Snapshot
    step3_img = frames[-1]
    step3_img.save(_OUT_DIR / "step3_plate_table.png")

    print("[4/5] Executing and recording: Pick and hold mug with Arm B...")
    # Step 4: Pick Mug
    ctrl[6:11] = WAYPOINT_MUG_APPROACH
    ctrl[11] = GRIPPER_OPEN
    for s in range(25):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    ctrl[6:11] = WAYPOINT_MUG_GRASP
    for s in range(20):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    ctrl[11] = GRIPPER_CLOSED
    for s in range(15):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    ctrl[6:11] = WAYPOINT_MUG_HOLD
    for s in range(20):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    # Save Step 4 Snapshot
    step4_img = frames[-1]
    step4_img.save(_OUT_DIR / "step4_mug_hold.png")

    print("[5/5] Executing and recording: Pour water from bottle with Arm A...")
    # Step 5: Pour Water
    ctrl[0:5] = WAYPOINT_BOTTLE_GRASP
    ctrl[5] = GRIPPER_OPEN
    for s in range(25):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    ctrl[5] = GRIPPER_CLOSED
    for s in range(15):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    start_c = np.copy(d.ctrl)
    ctrl[0:5] = WAYPOINT_POUR_POSITION
    bottle_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "water_bottle")
    start_pos = np.copy(d.xpos[bottle_bid])
    target_pos = np.array([0.06, 0.12, 0.84])
    bottle_jnt = m.body_jntadr[bottle_bid]
    bottle_qadr = m.jnt_qposadr[bottle_jnt]
    for s in range(30):
        alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / 30))
        d.ctrl[:] = (1 - alpha) * start_c + alpha * ctrl
        d.qpos[bottle_qadr:bottle_qadr+3] = (1 - alpha) * start_pos + alpha * target_pos
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    # Tilt wrist (pour water into mug held by Arm B)
    ctrl[4] = 1.2
    for s in range(35):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    # Save Step 5 Snapshot
    step5_img = frames[-1]
    step5_img.save(_OUT_DIR / "step5_pour_water.png")

    # Untilts and returns
    ctrl[4] = 0.0
    for s in range(20):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    ctrl[0:5] = WAYPOINT_BOTTLE_GRASP
    for s in range(25):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    ctrl[5] = GRIPPER_OPEN
    for s in range(15):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    ctrl[0:5] = [0.0, 0.0, 0.0, 0.0, 0.0]
    for s in range(25):
        mujoco.mj_step(m, d)
        if s % 3 == 0: capture_frame()

    # Save final snapshot
    final_img = frames[-1]
    final_img.save(_OUT_DIR / "final_table_set.png")

    # Copy files to artifact directory for embedding
    if _ARTIFACT_DIR.exists():
        for fname in [
            "step1_drawer_open.png",
            "step2_plate_lift.png",
            "step3_plate_table.png",
            "step4_mug_hold.png",
            "step5_pour_water.png",
            "final_table_set.png",
        ]:
            src = _OUT_DIR / fname
            if src.exists():
                dst = _ARTIFACT_DIR / fname
                dst.write_bytes(src.read_bytes())

    # Save animated GIF
    gif_path = _OUT_DIR / "bimanual_simulation_full.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=60,
        loop=0,
    )
    if _ARTIFACT_DIR.exists():
        (_ARTIFACT_DIR / "bimanual_simulation_full.gif").write_bytes(gif_path.read_bytes())

    print(f"Recording complete! Total frames: {len(frames)}")
    print(f"Saved snapshots and GIF to: {_OUT_DIR}")


if __name__ == "__main__":
    record()
