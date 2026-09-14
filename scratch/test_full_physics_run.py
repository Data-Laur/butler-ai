import sys, os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import mujoco
import numpy as np

# Modify scene in memory or load
xml_path = Path("assets/bimanual_scene.xml")
xml = xml_path.read_text(encoding="utf-8")

# 1. Update mug mesh scale and geom
xml = xml.replace(
    '<mesh name="mug_mesh" file="mug_ready.obj"/>',
    '<mesh name="mug_mesh" file="mug_ready.obj" scale="0.60 0.60 0.60"/>'
)
xml = xml.replace(
    '<geom name="mug_geom" type="cylinder" size="0.044 0.048" pos="0 0 0.048" mass="0.14" friction="1.8 0.01 0.001" rgba="0 0 0 0" group="3"/>',
    '<geom name="mug_geom" type="cylinder" size="0.024 0.028" pos="0 0 0.028" mass="0.09" friction="1.8 0.01 0.001" rgba="0 0 0 0" group="3"/>'
)

# 2. Drawer geoms group="0"
lines = xml.splitlines()
new_lines = []
for line in lines:
    if any(g in line for g in ['cabinet_', 'tray_', 'handle_']):
        line = line.replace('group="3"', 'group="0"')
    new_lines.append(line)
xml = "\n".join(new_lines)

# Write modified XML temporarily to assets/bimanual_scene_test.xml
test_xml_path = Path("assets/bimanual_scene_test.xml")
test_xml_path.write_text(xml, encoding="utf-8")

model = mujoco.MjModel.from_xml_path(str(test_xml_path))
data = mujoco.MjData(model)

# Set initial standby poses at t=0
from stage4_bimanual.constants import ARM_A_STANDBY, ARM_B_STANDBY, GRIPPER_OPEN, GRIPPER_CLOSED
data.qpos[36:41] = ARM_A_STANDBY
data.qpos[41] = GRIPPER_OPEN
data.qpos[42:47] = ARM_B_STANDBY
data.qpos[47] = GRIPPER_OPEN
data.ctrl[0:5] = ARM_A_STANDBY
data.ctrl[5] = GRIPPER_OPEN
data.ctrl[6:11] = ARM_B_STANDBY
data.ctrl[11] = GRIPPER_OPEN
mujoco.mj_forward(model, data)

from stage4_bimanual.trajectory import TrajectoryExecutor
from stage4_bimanual.kinematics import DLSInverseKinematics

ik = DLSInverseKinematics(model, data)
executor = TrajectoryExecutor(model, data)

def solve_ik(arm, target_pos, wrist_roll=0.0):
    _, q_sol, _ = ik.solve(arm, target_pos, wrist_roll=wrist_roll)
    return q_sol

def attach_weld(weld_name):
    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
    if weld_id < 0: return
    b1 = model.eq_obj1id[weld_id]
    b2 = model.eq_obj2id[weld_id]
    r1 = data.xmat[b1].reshape(3, 3)
    delta_world = data.xpos[b2] - data.xpos[b1]
    rel_pos = r1.T @ delta_world
    model.eq_data[weld_id, 3:6] = rel_pos
    data.eq_active[weld_id] = 1
    mujoco.mj_forward(model, data)

def detach_weld(weld_name):
    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
    if weld_id >= 0:
        data.eq_active[weld_id] = 0
        mujoco.mj_forward(model, data)

ctrl = np.copy(data.ctrl)

print("Starting Step 1: Open Drawer...")
# Query dynamic handle position
handle_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "drawer_handle_site")
handle_pos = np.copy(data.site_xpos[handle_site_id])
ctrl[0:5] = solve_ik("A", handle_pos + np.array([-0.05, 0.0, 0.02]))
executor.interpolate(ctrl, steps=35)
ctrl[0:5] = solve_ik("A", handle_pos)
executor.interpolate(ctrl, steps=25)
ctrl[5] = GRIPPER_CLOSED
executor.interpolate(ctrl, steps=20)
attach_weld("weld_drawer")
ctrl[0:5] = solve_ik("A", handle_pos - np.array([0.15, 0.0, 0.0]))
executor.interpolate(ctrl, steps=55)
detach_weld("weld_drawer")
ctrl[5] = GRIPPER_OPEN
executor.interpolate(ctrl, steps=20)
ctrl[0:5] = solve_ik("A", np.array([-0.08, handle_pos[1], 0.88]))
executor.interpolate(ctrl, steps=30)

print("Starting Step 2: Pick Plate...")
plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
plate_pos = np.copy(data.xpos[plate_id])
ctrl[0:5] = solve_ik("A", np.array([-0.05, plate_pos[1], 0.74]))
executor.interpolate(ctrl, steps=35)
ctrl[0:5] = solve_ik("A", np.array([plate_pos[0] - 0.05, plate_pos[1], plate_pos[2] + 0.010]))
executor.interpolate(ctrl, steps=35)
ctrl[5] = GRIPPER_CLOSED
executor.interpolate(ctrl, steps=25)
detach_weld("weld_plate_drawer")
attach_weld("weld_plate")
ctrl[0:5] = solve_ik("A", np.array([-0.06, plate_pos[1], 0.74]))
executor.interpolate(ctrl, steps=35)
ctrl[0:5] = solve_ik("A", np.array([-0.06, plate_pos[1], 0.88]))
executor.interpolate(ctrl, steps=40)

print("Starting Step 3: Place Plate...")
center_high = np.array([-0.03, -0.05, 0.88])
ctrl[0:5] = solve_ik("A", center_high)
executor.interpolate(ctrl, steps=45)
place_target = np.array([-0.03, -0.05, 0.775])
ctrl[0:5] = solve_ik("A", place_target)
executor.interpolate(ctrl, steps=40)
plate_dof = model.body_dofadr[plate_id]
data.qvel[plate_dof:plate_dof+6] = 0.0
for _ in range(25): mujoco.mj_step(model, data)
detach_weld("weld_plate")
ctrl[5] = GRIPPER_OPEN
executor.interpolate(ctrl, steps=25)
ctrl[0:5] = solve_ik("A", center_high)
executor.interpolate(ctrl, steps=30)
ctrl[0:5] = ARM_A_STANDBY
executor.interpolate(ctrl, steps=35)

print("Starting Step 4: Pick Mug with Arm B...")
mug_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
mug_pos = np.copy(data.xpos[mug_id])
print(f"  Mug initial pos: {np.round(mug_pos, 3)}")
ctrl[6:11] = solve_ik("B", [mug_pos[0], mug_pos[1], 0.88], wrist_roll=0.0)
ctrl[11] = GRIPPER_OPEN
executor.interpolate(ctrl, steps=35)
# Descend to mug body
ctrl[6:11] = solve_ik("B", [mug_pos[0], mug_pos[1], 0.728], wrist_roll=0.0)
executor.interpolate(ctrl, steps=35)
print(f"  Mug pos before grasp: {np.round(data.xpos[mug_id], 3)}")
ctrl[11] = GRIPPER_CLOSED
executor.interpolate(ctrl, steps=25)
attach_weld("weld_mug")
print(f"  Mug pos after grasp: {np.round(data.xpos[mug_id], 3)}")
# Lift mug
ctrl[6:11] = solve_ik("B", [mug_pos[0], mug_pos[1], 0.88], wrist_roll=0.0)
executor.interpolate(ctrl, steps=35)
print(f"  Mug pos after lift: {np.round(data.xpos[mug_id], 3)}")
# Bring mug to pouring station
mug_pour_pos = np.array([0.08, 0.10, 0.77])
ctrl[6:11] = solve_ik("B", mug_pour_pos, wrist_roll=0.0)
executor.interpolate(ctrl, steps=40)
print(f"  Mug pos at pour station: {np.round(data.xpos[mug_id], 3)}")

print("Starting Step 5: Pour Water with Arm A...")
bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "water_bottle")
bottle_pos = np.copy(data.xpos[bottle_id])
print(f"  Bottle initial pos: {np.round(bottle_pos, 3)}")
bottle_pre = np.array([bottle_pos[0], bottle_pos[1], 0.90])
ctrl[0:5] = solve_ik("A", bottle_pre, wrist_roll=1.57)
ctrl[5] = GRIPPER_OPEN
executor.interpolate(ctrl, steps=35)
ctrl[0:5] = solve_ik("A", [bottle_pos[0], bottle_pos[1], bottle_pos[2] + 0.075], wrist_roll=1.57)
executor.interpolate(ctrl, steps=35)
ctrl[5] = GRIPPER_CLOSED
executor.interpolate(ctrl, steps=25)
attach_weld("weld_bottle")
print(f"  Bottle pos after grasp: {np.round(data.xpos[bottle_id], 3)}")
ctrl[0:5] = solve_ik("A", bottle_pre, wrist_roll=1.57)
executor.interpolate(ctrl, steps=35)

# Pour station for bottle: Y=0.03 (mug is at Y=0.10)
bottle_pour_pos = np.array([0.08, 0.03, 0.85])
ctrl[0:5] = solve_ik("A", bottle_pour_pos, wrist_roll=0.0)
executor.interpolate(ctrl, steps=40)
print(f"  Bottle pos at pour: {np.round(data.xpos[bottle_id], 3)}, Mug pos: {np.round(data.xpos[mug_id], 3)}")
# Tilt bottle wrist towards +Y
ctrl[4] = 1.15
executor.interpolate(ctrl, steps=30)
print(f"  Bottle pos tilted: {np.round(data.xpos[bottle_id], 3)}, Mug pos: {np.round(data.xpos[mug_id], 3)}")
# Hold pour
executor.interpolate(ctrl, steps=25)
# Untilt bottle
ctrl[4] = 0.0
executor.interpolate(ctrl, steps=30)
print(f"  Bottle pos untilted: {np.round(data.xpos[bottle_id], 3)}, Mug pos: {np.round(data.xpos[mug_id], 3)}")
# Return bottle
ctrl[0:5] = solve_ik("A", bottle_pre, wrist_roll=1.57)
executor.interpolate(ctrl, steps=40)
ctrl[0:5] = solve_ik("A", [bottle_pos[0], bottle_pos[1], 0.775], wrist_roll=1.57)
executor.interpolate(ctrl, steps=35)
print(f"  Bottle pos at table descent: {np.round(data.xpos[bottle_id], 3)}")
detach_weld("weld_bottle")
ctrl[5] = GRIPPER_OPEN
executor.interpolate(ctrl, steps=20)
print(f"  Bottle pos after release: {np.round(data.xpos[bottle_id], 3)}")
ctrl[0:5] = solve_ik("A", bottle_pre, wrist_roll=1.57)
executor.interpolate(ctrl, steps=30)
ctrl[0:5] = ARM_A_STANDBY
executor.interpolate(ctrl, steps=35)
print(f"  Bottle pos after Arm A standby: {np.round(data.xpos[bottle_id], 3)}")

# Arm B sets mug back down on table
print(f"  Mug pos before return transit: {np.round(data.xpos[mug_id], 3)}")
ctrl[6:11] = solve_ik("B", [0.10, 0.18, 0.88], wrist_roll=0.0)
executor.interpolate(ctrl, steps=35)
print(f"  Mug pos high above station: {np.round(data.xpos[mug_id], 3)}")
ctrl[6:11] = solve_ik("B", [0.10, 0.18, 0.728], wrist_roll=0.0)
executor.interpolate(ctrl, steps=35)
print(f"  Mug pos after descent to table: {np.round(data.xpos[mug_id], 3)}")
mug_dof = model.body_dofadr[mug_id]
data.qvel[mug_dof:mug_dof+6] = 0.0
for _ in range(25): mujoco.mj_step(model, data)
detach_weld("weld_mug")
ctrl[11] = GRIPPER_OPEN
executor.interpolate(ctrl, steps=20)
print(f"  Mug pos after release: {np.round(data.xpos[mug_id], 3)}")
ctrl[6:11] = solve_ik("B", [0.10, 0.18, 0.88], wrist_roll=0.0)
executor.interpolate(ctrl, steps=30)
print(f"  Mug pos after lift: {np.round(data.xpos[mug_id], 3)}")
ctrl[6:11] = ARM_B_STANDBY
executor.interpolate(ctrl, steps=35)
print(f"  Mug pos after Arm B standby: {np.round(data.xpos[mug_id], 3)}")

print("\nAll 5 Steps Completed Successfully!")
print("Final Object Positions:")
for obj in ["plate", "mug", "water_bottle", "spoon", "fork"]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj)
    print(f"  {obj:15s}: {np.round(data.xpos[bid], 3)}")

# Clean up test xml
if test_xml_path.exists():
    test_xml_path.unlink()
