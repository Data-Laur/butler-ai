"""Record the bimanual simulation using the REAL execution pipeline.

Captures frames during actual physics-based robot manipulation (no object
teleportation). Every object movement is caused by the robot arm + weld
constraints, exactly as the pipeline runs during evaluation.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import mujoco
    import numpy as np
    from PIL import Image
except ImportError as err:
    sys.exit(f"Missing dependency: {err}. Install with: pip install mujoco numpy Pillow")

from common.types import Action, ActionType
from stage4_bimanual.bimanual import reset_scene
from stage4_bimanual.primitives import (
    OpenDrawerPrimitive,
    PickMugPrimitive,
    PickPlatePrimitive,
    PlacePlatePrimitive,
    PourWaterPrimitive,
)
from stage4_bimanual.sim import MuJoCoSim
from stage4_bimanual.trajectory import TrajectoryExecutor


_OUT_DIR = _ROOT / "docs" / "visualizations"
_OUT_DIR.mkdir(parents=True, exist_ok=True)


class FrameCaptureTrajectoryExecutor(TrajectoryExecutor):
    """Wraps TrajectoryExecutor to capture frames at regular intervals during
    real physics stepping, giving us a truthful visual record of what the
    robot actually does.
    """

    def __init__(self, model, data, renderer, camera, capture_every: int = 3):
        super().__init__(model, data)
        self._renderer = renderer
        self._camera = camera
        self._capture_every = capture_every
        self.frames: list[Image.Image] = []

    def _capture(self) -> None:
        """Render one frame from the perspective camera."""
        self._renderer.update_scene(self.data, camera=self._camera)
        img_arr = self._renderer.render()
        self.frames.append(Image.fromarray(img_arr))

    def interpolate(self, target_ctrl, steps: int = 60) -> None:
        """Override: same smooth cosine interpolation + physics stepping,
        but captures frames every N substeps.
        """
        if self.model is None or self.data is None:
            return

        start_ctrl = np.copy(self.data.ctrl)
        target_ctrl_arr = np.asarray(target_ctrl, dtype=np.float64)

        for s in range(steps):
            alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / steps))
            self.data.ctrl[:] = start_ctrl + alpha * (target_ctrl_arr - start_ctrl)
            mujoco.mj_step(self.model, self.data)

            if s % self._capture_every == 0:
                self._capture()


def record(seed: int = 0) -> None:
    """Run the REAL manipulation primitives and capture every step."""
    print(f"=== Recording real simulation (seed={seed}) ===\n")

    # 1. Initialize scene
    sim = reset_scene(seed=seed)
    if not isinstance(sim, MuJoCoSim):
        sys.exit("MuJoCo simulation not available.")

    m, d = sim.model, sim.data

    # Set up perspective camera for recording
    renderer = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = [0.08, -0.05, 0.75]
    cam.distance = 1.05
    cam.elevation = -28
    cam.azimuth = 135

    # Create the frame-capturing trajectory executor
    executor = FrameCaptureTrajectoryExecutor(m, d, renderer, cam, capture_every=3)

    # Capture initial frame
    executor._capture()

    # 2. Execute each REAL primitive and capture frames throughout
    steps = [
        ("Step 1/5: Open Drawer (Arm A)", OpenDrawerPrimitive),
        ("Step 2/5: Pick Plate from Drawer (Arm A)", PickPlatePrimitive),
        ("Step 3/5: Place Plate on Table (Arm A)", PlacePlatePrimitive),
        ("Step 4/5: Pick & Hold Mug (Arm B)", PickMugPrimitive),
        ("Step 5/5: Pour Water into Mug (Arm A)", PourWaterPrimitive),
    ]

    snapshot_names = [
        "step1_drawer_open.png",
        "step2_plate_lift.png",
        "step3_plate_table.png",
        "step4_mug_hold.png",
        "step5_pour_water.png",
    ]

    for i, (desc, PrimitiveClass) in enumerate(steps):
        print(f"[{desc}]")
        frame_before = len(executor.frames)

        # Create and execute the REAL primitive with the frame-capturing executor
        primitive = PrimitiveClass(executor, sim)
        success = primitive.execute()

        frame_after = len(executor.frames)
        status = "OK" if success else "FAILED"
        print(f"  -> {status} ({frame_after - frame_before} frames captured)\n")

        # Save milestone snapshot
        if executor.frames:
            snapshot = executor.frames[-1]
            snapshot.save(_OUT_DIR / snapshot_names[i])
            print(f"  -> Saved {snapshot_names[i]}")

    # 3. Save final snapshot
    executor._capture()
    final = executor.frames[-1]
    final.save(_OUT_DIR / "final_table_set.png")
    print(f"\nSaved final_table_set.png")

    # 4. Print object positions from real physics
    positions = sim.get_object_positions()
    drawer_state = sim.get_drawer_state()
    print(f"\n=== Final Scene State (from real physics) ===")
    print(f"  Drawer: {drawer_state}")
    for name, pos in positions.items():
        print(f"  {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    # 5. Save animated GIF
    gif_path = _OUT_DIR / "bimanual_simulation_full.gif"
    if len(executor.frames) > 1:
        executor.frames[0].save(
            gif_path,
            save_all=True,
            append_images=executor.frames[1:],
            duration=60,
            loop=0,
        )
        print(f"\nGIF saved: {gif_path}")
    else:
        print("\nNot enough frames for GIF.")

    print(f"\nRecording complete! Total frames: {len(executor.frames)}")
    print(f"All outputs saved to: {_OUT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record real bimanual simulation.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    args = parser.parse_args()
    record(seed=args.seed)
