import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mujoco, numpy as np
from stage4_bimanual.constants import ARM_B_STANDBY, GRIPPER_OPEN, GRIPPER_CLOSED
from stage4_bimanual.kinematics import DLSInverseKinematics
from stage4_bimanual.trajectory import TrajectoryExecutor

xml_path = Path("assets/bimanual_scene_test.xml")
xml = Path("assets/bimanual_scene.xml").read_text(encoding="utf-8")
xml = xml.replace(
    '<mesh name="mug_mesh" file="mug_ready.obj"/>',
    '<mesh name="mug_mesh" file="mug_ready.obj" scale="0.60 0.60 0.60"/>'
)
xml = xml.replace(
    '<geom name="mug_geom" type="cylinder" size="0.044 0.048" pos="0 0 0.048" mass="0.14" friction="1.8 0.01 0.001" rgba="0 0 0 0" group="3"/>',
    '<geom name="mug_geom" type="cylinder" size="0.024 0.028" pos="0 0 0.028" mass="0.09" friction="1.8 0.01 0.001" rgba="0 0 0 0" group="3"/>'
)
lines = xml.splitlines()
new_lines = []
for line in lines:
    if any(g in line for g in ['cabinet_', 'tray_', 'handle_']):
        line = line.replace('group="3"', 'group="0"')
    new_lines.append(line)
xml = "\n".join(new_lines)
xml_path.write_text(xml, encoding="utf-8")

m = mujoco.MjModel.from_xml_path(str(xml_path))
d = mujoco.MjData(m)
d.qpos[42:47] = ARM_B_STANDBY
d.qpos[47] = GRIPPER_OPEN
d.ctrl[6:11] = ARM_B_STANDBY
d.ctrl[11] = GRIPPER_OPEN
mujoco.mj_forward(m, d)

ik = DLSInverseKinematics(m, d)
ex = TrajectoryExecutor(m, d)

mug_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mug")
mug_pos = np.copy(d.xpos[mug_id])
print("Mug start pos:", np.round(mug_pos, 4))

# 1. Pre grasp at 0.88
_, q_pre, _ = ik.solve("B", [mug_pos[0], mug_pos[1], 0.88], wrist_roll=0.0)
ctrl = np.copy(d.ctrl)
ctrl[6:11] = q_pre
ctrl[11] = GRIPPER_OPEN
ex.interpolate(ctrl, steps=35)
print("After pre grasp, mug pos:", np.round(d.xpos[mug_id], 4), "contacts:", d.ncon)

# 2. Descend to grasp height: Z = 0.780
_, q_grasp, _ = ik.solve("B", [mug_pos[0], mug_pos[1], 0.780], wrist_roll=0.0)
ctrl[6:11] = q_grasp
ex.interpolate(ctrl, steps=35)
print("After descent, mug pos:", np.round(d.xpos[mug_id], 4))
print("Actuator forces:", np.round(d.actuator_force[6:12], 2))

# 3. Clamp gripper
ctrl[11] = GRIPPER_CLOSED
ex.interpolate(ctrl, steps=25)
print("After clamp, mug pos:", np.round(d.xpos[mug_id], 4))

# Attach weld
weld_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "weld_mug")
b1, b2 = m.eq_obj1id[weld_id], m.eq_obj2id[weld_id]
r1 = d.xmat[b1].reshape(3, 3)
delta_world = d.xpos[b2] - d.xpos[b1]
m.eq_data[weld_id, 3:6] = r1.T @ delta_world
d.eq_active[weld_id] = 1
mujoco.mj_forward(m, d)

# 4. Lift mug to 0.88
ctrl[6:11] = q_pre
ex.interpolate(ctrl, steps=35)
print("After lift, mug pos:", np.round(d.xpos[mug_id], 4))

# 5. Move to pour station: [0.08, 0.10, 0.82]
_, q_pour, _ = ik.solve("B", [0.08, 0.10, 0.82], wrist_roll=0.0)
ctrl[6:11] = q_pour
ex.interpolate(ctrl, steps=40)
print("At pour station, mug pos:", np.round(d.xpos[mug_id], 4))

# 6. Return mug to [0.10, 0.18, 0.88]
ctrl[6:11] = q_pre
ex.interpolate(ctrl, steps=35)
print("Returned high, mug pos:", np.round(d.xpos[mug_id], 4))

# 7. Descend to table at [0.10, 0.18, 0.780]
ctrl[6:11] = q_grasp
ex.interpolate(ctrl, steps=35)
print("Returned table, mug pos:", np.round(d.xpos[mug_id], 4))

# Settle and release
d.eq_active[weld_id] = 0
ctrl[11] = GRIPPER_OPEN
ex.interpolate(ctrl, steps=20)
print("After release, mug pos:", np.round(d.xpos[mug_id], 4))

# Retract
ctrl[6:11] = q_pre
ex.interpolate(ctrl, steps=30)
ctrl[6:11] = ARM_B_STANDBY
ex.interpolate(ctrl, steps=35)
print("After standby, mug pos:", np.round(d.xpos[mug_id], 4))

if xml_path.exists():
    xml_path.unlink()
