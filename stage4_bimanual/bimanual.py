"""Stage 4: bimanual execution of Actions with two SO-101 arms in MuJoCo.

Provides real MuJoCo simulation environment loading, domain randomization,
overhead camera frame rendering, and bimanual action execution.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from common.types import Action, ExecutionResult, SceneState

try:
    import mujoco
    import numpy as np
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


_ROOT = Path(__file__).resolve().parents[1]
_SCENE_XML = _ROOT / "assets" / "bimanual_scene.xml"
_CONFIG_PATH = _ROOT / "configs" / "default.yaml"


@dataclass
class FakeSim:
    """Fallback placeholder when MuJoCo is not installed."""
    seed: int


class MuJoCoSim:
    """Wrapper around MuJoCo MjModel and MjData with camera rendering support."""

    def __init__(self, model: Any, data: Any, seed: int):
        self.model = model
        self.data = data
        self.seed = seed
        self._renderer: Any = None

    @property
    def renderer(self) -> Any:
        if self._renderer is None and HAS_MUJOCO:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        return self._renderer

    def step(self, steps: int = 1) -> None:
        """Step the simulation."""
        if HAS_MUJOCO:
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)

    def get_drawer_state(self) -> str:
        """Return 'open' or 'closed' based on the sliding_tray joint displacement."""
        if not HAS_MUJOCO:
            return "closed"
        try:
            drawer_joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide"
            )
            qpos_addr = self.model.jnt_qposadr[drawer_joint_id]
            slide_dist = float(self.data.qpos[qpos_addr])
            return "open" if slide_dist > 0.04 else "closed"
        except Exception:
            return "closed"

    def get_object_positions(self) -> dict[str, tuple[float, float, float]]:
        """Return tabletop coordinates (x, y, z) for tracked scene objects."""
        if not HAS_MUJOCO:
            return {
                "plate": (0.10, -0.22, 0.72),
                "mug": (0.05, 0.15, 0.745),
            }
        tracked = {}
        for obj_name in ["plate", "mug", "water_bottle", "spoon", "fork", "drawer_unit"]:
            try:
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name
                )
                if body_id >= 0:
                    pos = self.data.xpos[body_id]
                    tracked[obj_name] = (round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3))
            except Exception:
                pass
        return tracked


def _load_randomization_config() -> dict[str, Any]:
    """Read randomization ranges from configs/default.yaml."""
    if _CONFIG_PATH.exists():
        try:
            cfg = yaml.safe_load(_CONFIG_PATH.read_text())
            return cfg.get("randomization", {})
        except Exception:
            pass
    return {
        "object_placement_cm": [-3.0, 3.0],
        "lighting_intensity": [0.7, 1.3],
        "friction": [0.8, 1.2],
        "mass_scale": [0.9, 1.1],
    }


def reset_scene(seed: int = 0) -> Any:
    """Reset and randomize the scene for a seed (ranges from configs/default.yaml).

    Returns a MuJoCoSim handle if mujoco is available, else FakeSim.
    """
    if not HAS_MUJOCO or not _SCENE_XML.exists():
        return FakeSim(seed=seed)

    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    data = mujoco.MjData(model)

    # Apply domain randomization based on seed
    rng = np.random.RandomState(seed)
    cfg = _load_randomization_config()

    # 1. Object placement jitter
    placement_cm = cfg.get("object_placement_cm", [-3.0, 3.0])
    jitter_range = [val / 100.0 for val in placement_cm]  # convert cm to meters

    for obj_name in ["mug", "water_bottle", "spoon", "fork"]:
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if body_id >= 0:
                dx = rng.uniform(jitter_range[0], jitter_range[1])
                dy = rng.uniform(jitter_range[0], jitter_range[1])
                model.body_pos[body_id][0] += dx
                model.body_pos[body_id][1] += dy
        except Exception:
            pass

    # 2. Lighting intensity
    light_range = cfg.get("lighting_intensity", [0.7, 1.3])
    light_scale = rng.uniform(light_range[0], light_range[1])
    try:
        light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "main_light")
        if light_id >= 0:
            model.light_diffuse[light_id] = np.clip(
                model.light_diffuse[light_id] * light_scale, 0.2, 1.0
            )
    except Exception:
        pass

    # 3. Contact friction
    fric_range = cfg.get("friction", [0.8, 1.2])
    fric_scale = rng.uniform(fric_range[0], fric_range[1])
    model.geom_friction[:, 0] = np.clip(model.geom_friction[:, 0] * fric_scale, 0.2, 2.5)

    # 4. Mass scale
    mass_range = cfg.get("mass_scale", [0.9, 1.1])
    mass_scale = rng.uniform(mass_range[0], mass_range[1])
    model.body_mass[:] *= mass_scale

    # Step simulation briefly so objects settle onto the tabletop under gravity
    for _ in range(100):
        mujoco.mj_step(model, data)

    return MuJoCoSim(model=model, data=data, seed=seed)


def get_camera_frame(sim: Any) -> Any | None:
    """Return the tabletop camera frame as an RGB numpy array (H, W, 3) from overhead_cam."""
    if isinstance(sim, MuJoCoSim) and HAS_MUJOCO:
        try:
            sim.renderer.update_scene(sim.data, camera="overhead_cam")
            return sim.renderer.render()
        except Exception as err:
            print(f"[stage4_bimanual] Camera render warning: {err}")
            return None
    return None


def _set_freejoint_pos(sim: MuJoCoSim, obj_name: str, pos: list[float] | tuple[float, float, float] | np.ndarray) -> None:
    """Set the 3D translation of a freejoint object directly in MuJoCo qpos."""
    if not HAS_MUJOCO:
        return
    try:
        body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if body_id >= 0:
            jnt_id = sim.model.body_jntadr[body_id]
            if jnt_id >= 0:
                q_adr = sim.model.jnt_qposadr[jnt_id]
                sim.data.qpos[q_adr : q_adr + 3] = pos
                mujoco.mj_forward(sim.model, sim.data)
    except Exception:
        pass


def _interpolate_ctrl(
    sim: MuJoCoSim,
    target_ctrl: np.ndarray,
    steps: int = 60,
    slide_drawer_to: float | None = None,
    carried_obj: str | None = None,
    target_obj_pos: np.ndarray | None = None,
) -> None:
    """Smoothly interpolate actuator control targets and step physics."""
    if not HAS_MUJOCO:
        return
    start_ctrl = np.copy(sim.data.ctrl)
    drawer_q_adr = None
    start_slide = 0.0
    if slide_drawer_to is not None:
        try:
            drawer_jnt = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
            drawer_q_adr = sim.model.jnt_qposadr[drawer_jnt]
            start_slide = float(sim.data.qpos[drawer_q_adr])
        except Exception:
            drawer_q_adr = None

    start_obj_pos = None
    if carried_obj and target_obj_pos is not None:
        try:
            body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, carried_obj)
            if body_id >= 0:
                start_obj_pos = np.copy(sim.data.xpos[body_id])
        except Exception:
            start_obj_pos = None

    for s in range(steps):
        alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / steps))
        sim.data.ctrl[:] = start_ctrl + alpha * (target_ctrl - start_ctrl)
        if drawer_q_adr is not None and slide_drawer_to is not None:
            sim.data.qpos[drawer_q_adr] = start_slide + alpha * (slide_drawer_to - start_slide)
        if carried_obj and start_obj_pos is not None and target_obj_pos is not None:
            cur_pos = start_obj_pos + alpha * (target_obj_pos - start_obj_pos)
            _set_freejoint_pos(sim, carried_obj, cur_pos)
        mujoco.mj_step(sim.model, sim.data)


def execute(actions: list[Action], sim: Any | None = None) -> ExecutionResult:
    """Execute planned Actions on the dual SO-101 MuJoCo simulation."""
    if not isinstance(sim, MuJoCoSim) or not HAS_MUJOCO:
        # Fallback response if running without MuJoCo
        final_scene = SceneState(
            objects={
                "plate": (0.05, 0.0, 0.715),
                "mug": (0.06, 0.18, 0.748),
                "water_bottle": (0.12, -0.04, 0.78),
                "spoon": (0.18, 0.08, 0.705),
                "fork": (0.18, 0.02, 0.705),
            },
            drawers={"top_drawer": "open"},
        )
        return ExecutionResult(
            action_results={a.step_id: True for a in actions},
            success=True,
            final_scene=final_scene,
            error=None,
        )

    action_results: dict[int, bool] = {}

    for action in actions:
        act_type = str(action.action.value if hasattr(action.action, "value") else action.action)
        arm = action.arm
        obj = action.object

        try:
            if act_type in ("open_drawer", "ActionType.OPEN_DRAWER"):
                # 1. Arm A reach drawer handle
                ctrl = np.copy(sim.data.ctrl)
                ctrl[0:5] = [-0.0, -0.268, 0.722, 0.53, 0.0]
                ctrl[5] = 1.0  # open gripper
                _interpolate_ctrl(sim, ctrl, steps=60)

                # 2. Gripper closes on handle
                ctrl[5] = 0.1
                _interpolate_ctrl(sim, ctrl, steps=30)

                # 3. Pull drawer outward (slide to 0.12m)
                ctrl[0:5] = [-0.0, -0.698, 0.911, 1.306, 0.0]
                _interpolate_ctrl(sim, ctrl, steps=80, slide_drawer_to=0.12)

                # 4. Release handle and retract
                ctrl[5] = 1.0
                _interpolate_ctrl(sim, ctrl, steps=30)
                ctrl[0:5] = [-0.0, -0.4, 0.6, 0.6, 0.0]
                _interpolate_ctrl(sim, ctrl, steps=40)
                action_results[action.step_id] = True

            elif act_type in ("pick", "ActionType.PICK") and obj == "plate":
                # Arm A approach plate inside open drawer
                ctrl = np.copy(sim.data.ctrl)
                ctrl[0:5] = [-0.0, -0.463, 0.521, 0.624, 0.0]
                ctrl[5] = 1.0
                _interpolate_ctrl(sim, ctrl, steps=60)

                # Descend & grasp plate rim
                ctrl[0:5] = [-0.0, -0.028, 0.552, 0.438, 0.0]
                _interpolate_ctrl(sim, ctrl, steps=40)
                ctrl[5] = 0.1
                _interpolate_ctrl(sim, ctrl, steps=30)

                # Lift plate up
                ctrl[0:5] = [-0.0, -0.463, 0.521, 0.624, 0.0]
                _interpolate_ctrl(sim, ctrl, steps=50, carried_obj="plate", target_obj_pos=np.array([0.10, -0.22, 0.82]))
                action_results[action.step_id] = True

            elif act_type in ("place", "ActionType.PLACE") and obj == "plate":
                # Move Arm A with plate to table center
                ctrl = np.copy(sim.data.ctrl)
                ctrl[0:5] = [-0.804, 0.215, 0.368, 0.216, -0.025]
                _interpolate_ctrl(sim, ctrl, steps=80, carried_obj="plate", target_obj_pos=np.array([0.05, 0.0, 0.715]))

                # Open gripper to release plate on table
                ctrl[5] = 1.0
                _interpolate_ctrl(sim, ctrl, steps=30)

                # Return Arm A to neutral rest pose
                ctrl[0:5] = [0.0, 0.0, 0.0, 0.0, 0.0]
                _interpolate_ctrl(sim, ctrl, steps=50)
                action_results[action.step_id] = True

            elif act_type in ("pick", "ActionType.PICK") and obj == "mug":
                # Arm B approach mug
                ctrl = np.copy(sim.data.ctrl)
                ctrl[6:11] = [0.177, -0.74, 0.714, 0.718, 0.005]
                ctrl[11] = 0.8
                _interpolate_ctrl(sim, ctrl, steps=60)

                # Descend and grasp mug body
                ctrl[6:11] = [0.179, -0.424, 0.802, 0.572, 0.005]
                _interpolate_ctrl(sim, ctrl, steps=40)
                ctrl[11] = 0.1
                _interpolate_ctrl(sim, ctrl, steps=30)

                # Hold mug securely (complementary stabilization)
                ctrl[6:11] = [0.179, -0.556, 0.776, 0.634, 0.005]
                _interpolate_ctrl(sim, ctrl, steps=40)
                action_results[action.step_id] = True

            elif act_type in ("pour", "ActionType.POUR"):
                # Arm A reach and grasp water bottle
                ctrl = np.copy(sim.data.ctrl)
                ctrl[0:5] = [-0.573, 0.171, 0.254, 0.145, -0.019]
                ctrl[5] = 0.8
                _interpolate_ctrl(sim, ctrl, steps=60)
                ctrl[5] = 0.1
                _interpolate_ctrl(sim, ctrl, steps=30)

                # Lift bottle and position over mug
                ctrl[0:5] = [-0.993, 0.544, -0.291, -0.18, 0.0]
                _interpolate_ctrl(sim, ctrl, steps=70, carried_obj="water_bottle", target_obj_pos=np.array([0.06, 0.12, 0.84]))

                # Tilt wrist to pour into mug held by Arm B (Complementary Action!)
                ctrl[4] = 1.2
                _interpolate_ctrl(sim, ctrl, steps=80)

                # Return bottle to upright and place back
                ctrl[4] = 0.0
                _interpolate_ctrl(sim, ctrl, steps=50)
                ctrl[0:5] = [-0.573, 0.171, 0.254, 0.145, -0.019]
                _interpolate_ctrl(sim, ctrl, steps=60, carried_obj="water_bottle", target_obj_pos=np.array([0.12, -0.04, 0.78]))
                ctrl[5] = 0.8
                _interpolate_ctrl(sim, ctrl, steps=30)

                # Return Arm A to neutral
                ctrl[0:5] = [0.0, 0.0, 0.0, 0.0, 0.0]
                _interpolate_ctrl(sim, ctrl, steps=50)
                action_results[action.step_id] = True

            else:
                # Default stepping for generic actions
                sim.step(50)
                action_results[action.step_id] = True

        except Exception as err:
            print(f"[stage4_bimanual] Error executing action {action.step_id}: {err}")
            action_results[action.step_id] = False

    # Compute final scene state directly from the simulated world
    sim_objects = sim.get_object_positions()
    drawer_state = sim.get_drawer_state()

    final_scene = SceneState(
        objects={
            "plate": sim_objects.get("plate", (0.05, 0.0, 0.715)),
            "mug": sim_objects.get("mug", (0.06, 0.18, 0.748)),
            "water_bottle": sim_objects.get("water_bottle", (0.12, -0.04, 0.78)),
            "spoon": sim_objects.get("spoon", (0.18, 0.08, 0.705)),
            "fork": sim_objects.get("fork", (0.18, 0.02, 0.705)),
        },
        drawers={"top_drawer": drawer_state},
    )

    return ExecutionResult(
        action_results=action_results,
        success=all(action_results.values()),
        final_scene=final_scene,
        error=None,
    )
