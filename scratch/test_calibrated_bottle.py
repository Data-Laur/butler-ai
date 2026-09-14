import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mujoco, numpy as np
from stage4_bimanual.constants import ARM_A_STANDBY, GRIPPER_OPEN, GRIPPER_CLOSED
from stage4_bimanual.kinematics import DLSInverseKinematics
from stage4_bimanual.trajectory import TrajectoryExecutor

m = mujoco.MjModel.from_xml_path("assets/bimanual_scene.xml")
d = mujoco.MjData(m)
d.qpos[36:41] = ARM_A_STANDBY
d.qpos[41] = GRIPPER_OPEN
d.ctrl[0:5] = ARM_A_STANDBY
d.ctrl[5] = GRIPPER_OPEN
mujoco.mj_forward(m, d)

ik = DLSInverseKinematics(m, d)
ex = TrajectoryExecutor(m, d)

bottle_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "water_bottle")
bottle_pos = np.copy(d.xpos[bottle_id])
print("Bottle start pos:", np.round(bottle_pos, 4))

# 1. Pre grasp at Z = 0.94
bottle_pre = np.array([bottle_pos[0], bottle_pos[1], 0.94])
_, q_pre, _ = ik.solve("A", bottle_pre, wrist_roll=1.57)
ctrl = np.copy(d.ctrl)
ctrl[0:5] = q_pre
ctrl[5] = GRIPPER_OPEN
ex.interpolate(ctrl, steps=35)
print("After pre grasp, bottle pos:", np.round(d.xpos[bottle_id], 4))

# 2. Descend to bottle neck: Z = 0.870
bottle_grasp = np.array([bottle_pos[0], bottle_pos[1], 0.870])
_, q_grasp, _ = ik.solve("A", bottle_grasp, wrist_roll=1.57)
ctrl[0:5] = q_grasp
ex.interpolate(ctrl, steps=35)
print("After descent, bottle pos:", np.round(d.xpos[bottle_id], 4))
print("Actuator forces:", np.round(d.actuator_force[0:6], 2))

# 3. Clamp gripper
ctrl[5] = GRIPPER_CLOSED
ex.interpolate(ctrl, steps=25)
print("After clamp, bottle pos:", np.round(d.xpos[bottle_id], 4))

# Attach weld
weld_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "weld_bottle")
b1, b2 = m.eq_obj1id[weld_id], m.eq_obj2id[weld_id]
r1 = d.xmat[b1].reshape(3, 3)
delta_world = d.xpos[b2] - d.xpos[b1]
m.eq_data[weld_id, 3:6] = r1.T @ delta_world
d.eq_active[weld_id] = 1
mujoco.mj_forward(m, d)

# 4. Lift bottle to 0.94
ctrl[0:5] = q_pre
ex.interpolate(ctrl, steps=35)
print("After lift, bottle pos:", np.round(d.xpos[bottle_id], 4))

# 5. Move bottle to pour station: [0.08, 0.03, 0.88]
pour_station = np.array([0.08, 0.03, 0.88])
_, q_pour, _ = ik.solve("A", pour_station, wrist_roll=0.0)
ctrl[0:5] = q_pour
ex.interpolate(ctrl, steps=40)
print("At pour station, bottle pos:", np.round(d.xpos[bottle_id], 4))

# 6. Tilt wrist
ctrl[4] = 1.15
ex.interpolate(ctrl, steps=30)
print("Tilted, bottle pos:", np.round(d.xpos[bottle_id], 4))
ex.interpolate(ctrl, steps=25)
ctrl[4] = 0.0
ex.interpolate(ctrl, steps=30)
print("Untilted, bottle pos:", np.round(d.xpos[bottle_id], 4))

# 7. Return to high bottle station
ctrl[0:5] = q_pre
ex.interpolate(ctrl, steps=40)
print("Returned high, bottle pos:", np.round(d.xpos[bottle_id], 4))

# 8. Descend to table resting height: Z = 0.870
ctrl[0:5] = q_grasp
ex.interpolate(ctrl, steps=35)
print("Returned table, bottle pos:", np.round(d.xpos[bottle_id], 4))

# Settle bottle velocity
d.eq_active[weld_id] = 0
ctrl[5] = GRIPPER_OPEN
ex.interpolate(ctrl, steps=20)
print("After release, bottle pos:", np.round(d.xpos[bottle_id], 4))

# Retract Arm A
ctrl[0:5] = q_pre
ex.interpolate(ctrl, steps=30)
ctrl[0:5] = ARM_A_STANDBY
ex.interpolate(ctrl, steps=35)
print("After standby, bottle pos:", np.round(d.xpos[bottle_id], 4))
