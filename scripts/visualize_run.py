"""Interactive 3D real-time visualizer for the bimanual table-setting simulation.

Uses the REAL manipulation primitives (same as the pipeline) so you can
visually verify the robot physically moves every object — no teleportation.

Run with:
    python scripts/visualize_run.py
    python scripts/visualize_run.py --seed 3
"""

import time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mujoco
import mujoco.viewer
import numpy as np

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


class ViewerSyncTrajectoryExecutor(TrajectoryExecutor):
    """Extends TrajectoryExecutor to sync with the MuJoCo interactive viewer
    after each physics step, so the user sees every movement in real time.
    """

    def __init__(self, model, data, viewer, step_delay: float = 0.012):
        super().__init__(model, data)
        self._viewer = viewer
        self._step_delay = step_delay

    def interpolate(self, target_ctrl, steps: int = 60) -> None:
        """Smooth cosine interpolation with viewer sync after each substep."""
        if self.model is None or self.data is None:
            return

        start_ctrl = np.copy(self.data.ctrl)
        target_ctrl_arr = np.asarray(target_ctrl, dtype=np.float64)

        for s in range(steps):
            if not self._viewer.is_running():
                return

            alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / steps))
            self.data.ctrl[:] = start_ctrl + alpha * (target_ctrl_arr - start_ctrl)
            mujoco.mj_step(self.model, self.data)
            self._viewer.sync()
            time.sleep(self._step_delay)


def run_interactive(seed: int = 0) -> None:
    print(f"Initializing MuJoCo bimanual scene (seed={seed})...")
    sim = reset_scene(seed=seed)
    if not isinstance(sim, MuJoCoSim):
        print("Error: MuJoCo is required for the visualizer.")
        return

    m, d = sim.model, sim.data

    print("\n" + "=" * 60)
    print("Launching MuJoCo 3D Viewer — REAL PHYSICS MODE")
    print("The robot will physically move every object.")
    print("Controls:")
    print("  - Left click + drag:  Orbit camera")
    print("  - Right click + drag: Zoom camera")
    print("  - Middle click:       Pan camera")
    print("  - ESC:                Close viewer")
    print("=" * 60 + "\n")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        time.sleep(1.0)

        if not viewer.is_running():
            return

        # Create the viewer-syncing trajectory executor
        executor = ViewerSyncTrajectoryExecutor(m, d, viewer, step_delay=0.012)

        # Execute each REAL primitive
        steps = [
            ("Step 1/5: Arm A opening drawer...", OpenDrawerPrimitive),
            ("Step 2/5: Arm A picking plate from drawer...", PickPlatePrimitive),
            ("Step 3/5: Arm A placing plate on table...", PlacePlatePrimitive),
            ("Step 4/5: Arm B picking and holding mug...", PickMugPrimitive),
            ("Step 5/5: Arm A pouring water into Arm B's mug...", PourWaterPrimitive),
        ]

        for desc, PrimitiveClass in steps:
            if not viewer.is_running():
                return

            print(desc)
            primitive = PrimitiveClass(executor, sim)
            success = primitive.execute()
            status = "OK" if success else "FAILED"
            print(f"  -> {status}")

            if not viewer.is_running():
                return

        # Print final scene state
        positions = sim.get_object_positions()
        drawer_state = sim.get_drawer_state()
        print(f"\n=== Final Scene State ===")
        print(f"  Drawer: {drawer_state}")
        for name, pos in positions.items():
            print(f"  {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        print("\nSequence complete! Inspect the table in 3D. Press ESC to exit.")

        # Keep the viewer open for inspection
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interactive 3D bimanual visualizer.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    args = parser.parse_args()
    run_interactive(seed=args.seed)
