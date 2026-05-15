"""
render_crazyflie.py -- single hover render of the Crazyflie 2 model.

Loads the MuJoCo Menagerie bitcraze_crazyflie_2 scene, sets a fixed
camera, and writes a PNG into figures/. The model is rendered at
hover; no simulation is stepped.

Run from the repository root:
    python3 render_crazyflie.py
"""
from pathlib import Path

import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parent
XML = ROOT / "mujoco_menagerie" / "bitcraze_crazyflie_2" / "scene.xml"
OUT = ROOT / "figures" / "crazyflie_platform.png"

WIDTH, HEIGHT = 1200, 800

# Menagerie's default offscreen framebuffer is 640px; bump it before
# constructing the Renderer or mujoco raises ValueError.
model = mujoco.MjModel.from_xml_path(str(XML))
model.vis.global_.offwidth = WIDTH
model.vis.global_.offheight = HEIGHT
data = mujoco.MjData(model)

cam = mujoco.MjvCamera()
cam.distance = 0.4
cam.azimuth = 135
cam.elevation = -20
cam.lookat[:] = [0, 0, 0.02]

mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, width=WIDTH, height=HEIGHT)
renderer.update_scene(data, cam)
Image.fromarray(renderer.render()).save(OUT)
print("wrote", OUT)
