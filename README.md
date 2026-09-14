# ai-infra-summit-hack

Two simulated SO-101 arms set a dinner table from a natural-language voice
command — Intel Physical AI Online Challenge (lablab.ai AI Infra Summit
Hackathon, Sept 10–16, 2026).

## Overview

Voice command in, table set. Example command:

> "Open the top drawer, pick up the plate with arm A, place it on the table,
> pick up the mug with arm B, pour water into the mug with arm A."

Read [the implementation approach and honest current-status checklist](docs/IMPLEMENTATION_APPROACH.md)
before running evaluation. It documents the physics/contact gates, the
simulation-state perception baseline, and what must be completed before
claiming VLA or OpenVINO results.

For the visual, per-skill ACT data-collection workflow, see
[DATA_COLLECTION.md](docs/DATA_COLLECTION.md).

**Voice input is powered by Speechmatics** (ASR) with Claude for instruction
parsing — see `stage1_voice/`.

TODO: 2–3 sentence pitch once the demo works.

## Architecture

```
voice (Speechmatics + Claude)          stage1_voice      -> Task
  -> perception (OpenCV)               stage2_perception -> SceneState
  -> policy (LeRobot VLA + fallback)   stage3_policy     -> list[Action]
  -> bimanual execution (MuJoCo)       stage4_bimanual   -> ExecutionResult
  -> verify / recover                  stage6_verify     -> VerifyResult
  -> evaluation over seeds             stage7_eval       -> EvalReport

standalone: OpenVINO benchmark         stage5_openvino
sim helpers (MuJoCo):                  stage4_bimanual.reset_scene / get_camera_frame
```

All inter-stage types are pydantic models in [`common/types.py`](common/types.py).
Signatures are pinned in [`CONTRACTS.md`](CONTRACTS.md).

## Simulation Environment (MuJoCo)

The bimanual simulation environment is defined in [`assets/bimanual_scene.xml`](assets/bimanual_scene.xml), featuring two physical **SO-101 6-DoF robotic arms** (TheRobotStudio/SO-ARM100) mounted on a dining table facing a drawer unit, ceramic plate, mug, and water bottle.
- **Arm A (`so101_arm_a.xml`)**: Left arm mounted at `pos="0.0 -0.22 0.70"`, responsible for opening the drawer, picking/placing the plate, and grasping/pouring the water bottle.
- **Arm B (`so101_arm_b.xml`)**: Right arm mounted at `pos="0.0 0.22 0.70"`, responsible for picking and holding the mug during the pour.
- **Visual & Physical Finger Pads**: Silicone finger pads (`a_moving_finger_pad`, `a_fixed_finger_pad`, `b_moving_finger_pad`, `b_fixed_finger_pad`) rendered in yellow with calibrated friction (`1.8 0.01 0.001`).
- **Domain Randomization**: Driven by [`stage4_bimanual.bimanual.reset_scene(seed)`](stage4_bimanual/bimanual.py) using [`configs/default.yaml`](configs/default.yaml):
  - Object positions: $\pm 2.0\text{ cm}$ XY table jitter and $\pm 0.8\text{ cm}$ drawer jitter per seed.
  - Contact friction: randomized across $[0.8\times, 1.2\times]$.
  - Object mass: randomized across $[0.9\times, 1.1\times]$.
  - Lighting intensity: randomized across $[0.7\times, 1.3\times]$.

## Bimanual Coordination Strategy (`stage4-bimanual`)

The table-setting execution follows a strict 5-stage dependency-ordered pipeline:
1. **`OpenDrawerPrimitive` (Arm A)**: High-altitude transit to handle bar, vertical pinch grasp on `drawer_handle`, smooth cosine pull $+X$, and clean $+Z$ vertical lift off the handle to prevent drawer recoil.
2. **`PickPlatePrimitive` (Arm A)**: Dynamic position reading of `plate_site`, rim pinch grasp, physical weld constraint, and vertical lift out of drawer tray airspace.
3. **`PlacePlatePrimitive` (Arm A)**: High-altitude transit over the drawer partition, calibrated placement at table center `(0.06, 0.00, 0.70)`, weld release, and vertical retreat.
4. **`PickMugPrimitive` (Arm B)**: Approaches table-right workspace, grasps the `mug_handle` directly via calibrated pinch site, lifts to holding station `[0.12, 0.16, 0.84]`, maintaining stable mechanical posture with $<1\text{ mm}$ shoulder sag.
5. **`PourWaterPrimitive` (Arm A + Arm B)**:
   - Complementary action: Arm B holds the mug securely at the pouring station.
   - Arm A lifts vertically to $Z = 0.93\text{ m}$ in home airspace to clear the asymmetric open gripper, approaches and welds the water bottle at $Z = 0.855\text{ m}$.
   - Arm A aligns the bottle lip with the mug opening, rotates wrist roll $80^\circ$ to pour water.
   - Symmetrical return: un-tilts bottle, returns bottle to table base, lowers mug to tabletop with zero drop height ($Z = 0.755\text{ m}$), settles physics for 30 steps, and releases welds cleanly.

## Visual Inspection Media (GIFs & Videos)

All primitive motions and full simulation recordings are saved in [`docs/visualizations/`](docs/visualizations/):

- **Step 1: Open Drawer**: [open_drawer.gif](docs/visualizations/open_drawer.gif) | [open_drawer.mp4](docs/visualizations/open_drawer.mp4)
- **Step 2: Pick Plate**: [pick_plate.gif](docs/visualizations/pick_plate.gif) | [pick_plate.mp4](docs/visualizations/pick_plate.mp4)
- **Step 3: Place Plate**: [place_plate.gif](docs/visualizations/place_plate.gif) | [place_plate.mp4](docs/visualizations/place_plate.mp4)
- **Step 4: Pick Mug**: [pick_mug.gif](docs/visualizations/pick_mug.gif) | [pick_mug.mp4](docs/visualizations/pick_mug.mp4)
- **Step 5: Pour Water**: [pour_water.gif](docs/visualizations/pour_water.gif) | [pour_water.mp4](docs/visualizations/pour_water.mp4)
- **Complete Sequence**: [bimanual_simulation_full.gif](docs/visualizations/bimanual_simulation_full.gif) | [bimanual_simulation_full.mp4](docs/visualizations/bimanual_simulation_full.mp4)

## Dataset Collection Pipeline (`collect_lerobot_data.py`)

Generates Hugging Face LeRobot v2.0 format datasets:
```bash
python scripts/collect_lerobot_data.py --num-episodes 50
```
- **Parquet Episodes**: 12-DoF state (`observation.state`) and joint commands (`action`) stored in `data/lerobot_bimanual_v2/data/chunk-000/episode_{id:06d}.parquet`.
- **Synchronized Video**: High-speed MP4 video (`episode_{id:06d}.mp4`) saved automatically alongside each training example in `videos/chunk-000/observation.images.overhead/`.
- **Animated GIF**: Optional `--save-gif` flag creates visual GIFs for immediate inspection.
- **Dataset Replay Utility**: Re-render any existing collected episodes to MP4/GIF via:
  ```bash
  python scripts/render_dataset_videos.py --episodes 0 1 2 3 4
  ```

## Evaluation Results (Success Rate Over 10 Seeds)

Verified via official test runner:
```bash
python scripts/evaluate.py --seeds 10
```
```text
Evaluating over 10 seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  seed 0: PASS  1 attempt(s)
  seed 1: PASS  1 attempt(s)
  seed 2: PASS  1 attempt(s)
  seed 3: PASS  1 attempt(s)
  seed 4: PASS  1 attempt(s)
  seed 5: PASS  1 attempt(s)
  seed 6: PASS  1 attempt(s)
  seed 7: PASS  1 attempt(s)
  seed 8: PASS  1 attempt(s)
  seed 9: PASS  1 attempt(s)

Success rate: 10/10 = 100%
Contact audit: 0 unexpected penetrations
```

## Stage Status & What Remains

| Stage | Owner | Status | Details & Remaining Actions |
|:---|:---|:---|:---|
| **stage1_voice** | Alex | Ready | Speechmatics ASR + Claude command parser -> `Task`. |
| **stage2_perception** | Lauren | Baseline Ready | Ground-truth simulation state + OpenCV RGB baseline. |
| **stage4_bimanual** | Azeem | **COMPLETED** | 100% benchmark pass, contact audit clean, LeRobot v2.0 data collected (50 episodes + video). |
| **stage6_verify** | Abdullah | **COMPLETED** | Verified across all tolerance bounds (plate center, mug base, drawer open). |
| **stage3_policy** | Bidipta | **NEXT UP** | Train ACT / Diffusion Policy via LeRobot on the 50-episode dataset in `data/lerobot_bimanual_v2/`. |
| **stage5_openvino** | Lauren | Pending | Benchmark policy model conversion to OpenVINO IR on Intel Core Ultra. |
| **stage7_eval & Demo** | Everyone | Pending | Final end-to-end demo video recording. |

## Team

| Who | Owns |
|-----|------|
| Alex | stage1_voice, integration & submission (scripts/, contracts) |
| Lauren | stage2_perception, stage5_openvino |
| Bidipta | stage3_policy (training) |
| Azeem | stage4_bimanual (simulation, primitives, dataset collection) |
| Abdullah | stage6_verify, stage7_eval |
| Everyone | docs/ (demo video & slides) |

## Working in this repo

- One branch per stage (`stage1-voice`, `stage3-policy`, `stage4-bimanual`, ...). PRs into `main` are merged by the integration owner (Alex).
- Keep your code inside your own `stageN_*` folder.
- Don't change another stage's interface or a shared model in `common/types.py` without agreeing it in [`CONTRACTS.md`](CONTRACTS.md) first.
- `main` must always pass `python scripts/run_pipeline.py`.
