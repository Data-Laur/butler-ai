"""MuJoCo simulation wrapper and domain randomization engine."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

try:
    import mujoco
    import numpy as np
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


@dataclass
class FakeSim:
    """Fallback stub when MuJoCo is not available."""
    seed: int


class DomainRandomizer:
    """Applies domain randomization per evaluation seed to prevent sim-to-real overfitting."""

    @staticmethod
    def load_config(config_path: Path) -> dict[str, Any]:
        """Read randomization parameters from YAML config."""
        if config_path.exists():
            try:
                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                return cfg.get("randomization", {})
            except Exception:
                pass
        return {
            "object_placement_cm": [-3.0, 3.0],
            "lighting_intensity": [0.7, 1.3],
            "friction": [0.8, 1.2],
            "mass_scale": [0.9, 1.1],
        }

    @classmethod
    def randomize(cls, model: Any, data: Any, seed: int, config_path: Path) -> None:
        """Apply seeded perturbations to object locations, lighting, friction, and mass."""
        if not HAS_MUJOCO:
            return

        rng = np.random.RandomState(seed)
        cfg = cls.load_config(config_path)

        # 1. Tabletop object placement jitter
        placement_cm = cfg.get("object_placement_cm", [-3.0, 3.0])
        jitter_range = [val / 100.0 for val in placement_cm]

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

        # 2. Lighting intensity variation
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

        # 3. Contact friction scaling
        fric_range = cfg.get("friction", [0.8, 1.2])
        fric_scale = rng.uniform(fric_range[0], fric_range[1])
        model.geom_friction[:, 0] = np.clip(
            model.geom_friction[:, 0] * fric_scale, 0.2, 2.5
        )

        # 4. Body mass perturbation
        mass_range = cfg.get("mass_scale", [0.9, 1.1])
        mass_scale = rng.uniform(mass_range[0], mass_range[1])
        model.body_mass[:] *= mass_scale

        # Settle simulation under gravity
        for _ in range(100):
            mujoco.mj_step(model, data)


class MuJoCoSim:
    """Encapsulates MuJoCo simulation environment, rendering, and state queries."""

    def __init__(self, model: Any, data: Any, seed: int):
        self.model = model
        self.data = data
        self.seed = seed
        self._renderer: Any = None

    @property
    def renderer(self) -> Any:
        """Lazy-initialize offscreen RGB camera renderer."""
        if self._renderer is None and HAS_MUJOCO and self.model is not None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        return self._renderer

    def step(self, steps: int = 1) -> None:
        """Advance physics simulation by the given number of timesteps."""
        if HAS_MUJOCO and self.model is not None and self.data is not None:
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)

    def get_camera_frame(self, camera_name: str = "overhead_cam") -> Any | None:
        """Render RGB image (H, W, 3) from the specified camera."""
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return None
        try:
            self.renderer.update_scene(self.data, camera=camera_name)
            return self.renderer.render()
        except Exception as err:
            print(f"[stage4_bimanual.sim] Camera render warning: {err}")
            return None

    def get_drawer_state(self) -> str:
        """Query the drawer linear slide joint displacement."""
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return "closed"
        try:
            drawer_joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide"
            )
            if drawer_joint_id >= 0:
                qpos_addr = self.model.jnt_qposadr[drawer_joint_id]
                slide_dist = float(self.data.qpos[qpos_addr])
                return "open" if slide_dist > 0.04 else "closed"
        except Exception:
            pass
        return "closed"

    def get_object_positions(self) -> dict[str, tuple[float, float, float]]:
        """Return 3D Cartesian coordinates for all tracked scene items."""
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return {
                "plate": (0.05, 0.0, 0.715),
                "mug": (0.06, 0.18, 0.748),
                "water_bottle": (0.12, -0.04, 0.78),
                "spoon": (0.18, 0.08, 0.705),
                "fork": (0.18, 0.02, 0.705),
            }

        tracked: dict[str, tuple[float, float, float]] = {}
        for obj_name in ["plate", "mug", "water_bottle", "spoon", "fork", "drawer_unit"]:
            try:
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name
                )
                if body_id >= 0:
                    pos = self.data.xpos[body_id]
                    tracked[obj_name] = (
                        round(float(pos[0]), 3),
                        round(float(pos[1]), 3),
                        round(float(pos[2]), 3),
                    )
            except Exception:
                pass
        return tracked
