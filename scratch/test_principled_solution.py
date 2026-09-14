import os
import mujoco
import numpy as np
import PIL.Image

os.chdir("assets")
xml = open("bimanual_scene.xml", "r").read()

# 1. Update mug mesh scale and geom
xml = xml.replace(
    '<mesh name="mug_mesh" file="mug_ready.obj"/>',
    '<mesh name="mug_mesh" file="mug_ready.obj" scale="0.60 0.60 0.60"/>'
)
xml = xml.replace(
    '<geom name="mug_geom" type="cylinder" size="0.044 0.048" pos="0 0 0.048" mass="0.14" friction="1.8 0.01 0.001" rgba="0 0 0 0" group="3"/>',
    '<geom name="mug_geom" type="cylinder" size="0.024 0.028" pos="0 0 0.028" mass="0.09" friction="1.8 0.01 0.001" rgba="0 0 0 0" group="3"/>'
)

# 2. Drawer geoms group="0" so they are visible
lines = xml.splitlines()
new_lines = []
for line in lines:
    if any(g in line for g in ['cabinet_', 'tray_', 'handle_']):
        line = line.replace('group="3"', 'group="0"')
    new_lines.append(line)
xml = "\n".join(new_lines)

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
print("Model compiled successfully! ngeom =", m.ngeom)

import sys
sys.path.insert(0, "..")
from stage4_bimanual.constants import ARM_A_STANDBY, ARM_B_STANDBY
d.qpos[36:41] = ARM_A_STANDBY
d.qpos[41] = 1.0
d.qpos[42:47] = ARM_B_STANDBY
d.qpos[47] = 1.0
d.ctrl[0:5] = ARM_A_STANDBY
d.ctrl[5] = 1.0
d.ctrl[6:11] = ARM_B_STANDBY
d.ctrl[11] = 1.0
mujoco.mj_forward(m, d)

renderer = mujoco.Renderer(m, 720, 1280)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.04, 0.0, 0.75]
cam.distance = 1.08
cam.elevation = -42.0
cam.azimuth = 165.0
renderer.update_scene(d, cam)
img = renderer.render()
PIL.Image.fromarray(img).save("C:/Users/Mannan/.gemini/antigravity-ide/brain/a9486265-30e2-4033-b7a1-30a3bbec5b4a/test_new_camera_perspective.png")
print("Rendered test_new_camera_perspective.png successfully!")
