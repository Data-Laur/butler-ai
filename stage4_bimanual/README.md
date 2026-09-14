# stage4_bimanual — Owner: Azeem

Executes planned `Action` sequences with dual SO-101 robotic arms in MuJoCo physics simulation (drawer opening, plate transfer, mug grasping, and coordinated bimanual pouring).

- **Input**: `list[Action]` (+ optional `MuJoCoSim` handle).
- **Output**: `ExecutionResult` with per-action results, `ContactAudit` verification, and final `SceneState`.

## Standalone Test
```powershell
python -c "from common.types import Action, ActionType; from stage4_bimanual import execute, reset_scene; sim = reset_scene(0); print(execute([Action(step_id=1, action=ActionType.OPEN_DRAWER, arm='A')], sim))"
```

## Validated Bimanual Skills (10/10 Seeds Passing)
All 5 manipulation primitives are physically validated with convex finger-pad contact gating, DLS inverse kinematics, and continuous penetration auditing (`<8mm`):

1. `OpenDrawerPrimitive` (Arm A): Approaches handle vertically, pinches, pulls >7.5cm, lifts clear before jaw release.
2. `PickPlatePrimitive` (Arm A): Pinches plate front rim inside open drawer tray and lifts vertically.
3. `PlacePlatePrimitive` (Arm A): Transits at safe altitude, places plate at table center `(0.06, 0.00)`, and retreats cleanly.
4. `PickMugPrimitive` (Arm B): Grasps mug body, lifts to transit altitude, and holds at pouring station `(0.06, 0.17, 0.82)`.
5. `PourWaterPrimitive` (Arm A): Approaches bottle neck, grasps, tilts over mug to pour, returns bottle, and parks both arms.

## Verification & Data Collection
```powershell
# Run 10-seed evaluation harness
python scripts/evaluate.py --seeds 10

# View interactive execution
python scripts/visualize_run.py --seed 0

# Record skill demonstration
python scripts/record_skill_demos.py --task open_drawer --episodes 1 --seed 0
```

