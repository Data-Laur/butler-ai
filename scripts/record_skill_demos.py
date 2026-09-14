#!/usr/bin/env python3
"""Record visually inspectable MuJoCo demonstrations for one atomic skill.

This intentionally writes an open raw bundle (PNG frames + compressed NumPy
arrays + JSON manifest) rather than pretending to be a LeRobot dataset when
the LeRobot package is not installed.  Only episodes that pass the contact
audit are marked ``accepted_for_training``.  The bundle is designed for a
small conversion step using the installed LeRobotDataset API.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from PIL import Image

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

SKILLS = {
    "open_drawer": (OpenDrawerPrimitive, "Open the top drawer with arm A."),
    "pick_plate": (PickPlatePrimitive, "Pick up the plate from the open drawer with arm A."),
    "place_plate": (PlacePlatePrimitive, "Place the plate at the center of the table with arm A."),
    "pick_mug": (PickMugPrimitive, "Pick up and hold the mug with arm B."),
    "pour_water": (PourWaterPrimitive, "Hold the mug with arm B and pour water from the bottle with arm A."),
}


class DemonstrationExecutor(TrajectoryExecutor):
    """Record actions and two synchronized rendered views during physics steps."""

    def __init__(self, model, data, *, fps: int, width: int, height: int, contact_audit):
        super().__init__(model, data, contact_audit=contact_audit)
        self.sample_interval = max(1, round(1 / (fps * model.opt.timestep)))
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.front_camera = mujoco.MjvCamera()
        self.front_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.front_camera.lookat[:] = [-0.04, 0.00, 0.74]
        self.front_camera.distance = 1.52
        self.front_camera.elevation = -58.0
        self.front_camera.azimuth = 145.0
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.overhead_frames: list[np.ndarray] = []
        self.front_frames: list[np.ndarray] = []
        self._substeps = 0

    def capture(self) -> None:
        self.states.append(np.asarray(self.data.qpos[36:48], dtype=np.float32).copy())
        self.actions.append(np.asarray(self.data.ctrl[:12], dtype=np.float32).copy())
        self.timestamps.append(float(self.data.time))
        self.renderer.update_scene(self.data, camera="overhead_cam")
        self.overhead_frames.append(self.renderer.render().copy())
        self.renderer.update_scene(self.data, camera=self.front_camera)
        self.front_frames.append(self.renderer.render().copy())

    def interpolate(self, target_ctrl, steps: int = 60) -> None:
        start = np.copy(self.data.ctrl)
        target = np.asarray(target_ctrl, dtype=np.float64)
        for step in range(steps):
            alpha = 0.5 * (1.0 - np.cos(np.pi * (step + 1) / steps))
            self.data.ctrl[:] = start + alpha * (target - start)
            mujoco.mj_step(self.model, self.data)
            self.contact_audit.sample(self.model, self.data)
            self._substeps += 1
            if self._substeps % self.sample_interval == 0:
                self.capture()
        self.data.ctrl[:] = target
        for _ in range(min(25, max(10, steps // 3))):
            mujoco.mj_step(self.model, self.data)
            self.contact_audit.sample(self.model, self.data)
            self._substeps += 1
            if self._substeps % self.sample_interval == 0:
                self.capture()


def prepare_for_skill(sim: MuJoCoSim, skill: str, executor: DemonstrationExecutor) -> None:
    """Execute prerequisite skills in the same real scene; do not record them."""
    prerequisites = {
        "open_drawer": [],
        "pick_plate": [OpenDrawerPrimitive],
        "place_plate": [OpenDrawerPrimitive, PickPlatePrimitive],
        "pick_mug": [],
        "pour_water": [PickMugPrimitive],
    }[skill]
    for primitive_class in prerequisites:
        if not primitive_class(executor, sim).execute():
            raise RuntimeError(f"prerequisite {primitive_class.__name__} failed")
    # Start the recorded portion cleanly: prerequisite contacts are useful for
    # debugging but must not invalidate the requested atomic demonstration.
    executor.contact_audit.violations.clear()
    executor.states.clear(); executor.actions.clear(); executor.timestamps.clear()
    executor.overhead_frames.clear(); executor.front_frames.clear()
    executor.capture()


def save_episode(output_root: Path, skill: str, index: int, seed: int, executor: DemonstrationExecutor, success: bool) -> Path:
    episode_dir = output_root / skill / f"episode_{index:05d}_seed_{seed:04d}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    overhead_dir = episode_dir / "images" / "overhead"
    front_dir = episode_dir / "images" / "front"
    overhead_dir.mkdir(parents=True)
    front_dir.mkdir(parents=True)
    for frame_index, (overhead, front) in enumerate(zip(executor.overhead_frames, executor.front_frames)):
        Image.fromarray(overhead).save(overhead_dir / f"{frame_index:06d}.png")
        Image.fromarray(front).save(front_dir / f"{frame_index:06d}.png")
    np.savez_compressed(
        episode_dir / "trajectory.npz",
        observation_state=np.asarray(executor.states, dtype=np.float32),
        action=np.asarray(executor.actions, dtype=np.float32),
        timestamp=np.asarray(executor.timestamps, dtype=np.float64),
    )
    manifest = {
        "format": "bimanual_raw_demo_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": SKILLS[skill][1],
        "skill": skill,
        "seed": seed,
        "fps_requested": round(1 / (executor.sample_interval * executor.model.opt.timestep), 3),
        "state": {"name": "observation.state", "shape": [12], "meaning": "A then B: 5 joints + gripper"},
        "action": {"name": "action", "shape": [12], "meaning": "position actuator targets, A then B"},
        "cameras": ["overhead", "front"],
        "frames": len(executor.states),
        "primitive_returned_success": success,
        "contact_violations": [v.__dict__ for v in executor.contact_audit.violations],
        "accepted_for_training": bool(success and executor.contact_audit.ok),
        "rejection_reason": "" if success and executor.contact_audit.ok else executor.contact_audit.summary(),
    }
    (episode_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # A quick visual proof without requiring a GUI or video codec.
    if executor.front_frames:
        frames = [Image.fromarray(frame) for frame in executor.front_frames]
        frames[0].save(episode_dir / "front_replay.gif", save_all=True, append_images=frames[1:], duration=50, loop=0)
    return episode_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Record one visually inspectable atomic MuJoCo skill demonstration.")
    parser.add_argument("--task", required=True, choices=SKILLS)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw_skill_demos")
    parser.add_argument("--save-rejected", action="store_true", help="Save unsafe episodes for debugging; they remain rejected for training.")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    primitive_class, _ = SKILLS[args.task]
    accepted = 0
    skill_root = args.output / args.task
    start_index = len(list(skill_root.glob("episode_*"))) if skill_root.exists() else 0
    for offset in range(args.episodes):
        index = start_index + offset
        seed = args.seed + offset
        sim = reset_scene(seed)
        if not isinstance(sim, MuJoCoSim):
            raise RuntimeError("MuJoCo is required for demonstration collection")
        executor = DemonstrationExecutor(sim.model, sim.data, fps=args.fps, width=640, height=480, contact_audit=sim.contact_audit)
        try:
            prepare_for_skill(sim, args.task, executor)
            success = primitive_class(executor, sim).execute()
        except Exception as exc:
            success = False
            print(f"seed {seed}: primitive error: {exc}")
        safe = success and sim.contact_audit.ok
        if safe or args.save_rejected:
            path = save_episode(args.output, args.task, index, seed, executor, success)
            print(f"seed {seed}: {'ACCEPTED' if safe else 'REJECTED'} -> {path}")
        else:
            print(f"seed {seed}: REJECTED (not saved): {sim.contact_audit.summary()}")
        accepted += int(safe)
    print(f"Accepted training demonstrations: {accepted}/{args.episodes}")
    return 0 if accepted == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
