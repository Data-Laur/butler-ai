# stage4_bimanual — owner: Azeem
Executes planned Actions with two SO-101 arms in MuJoCo (drawer opening, hand-offs, coordinated moves). Currently a stub — no MuJoCo import.
Input: `list[Action]` (+ optional sim handle). Output: `ExecutionResult` with per-step results and final `SceneState`.
Test standalone:
`python3 -c "from common.types import Action, ActionType; from stage4_bimanual import execute; print(execute([Action(step_id=1, action=ActionType.PICK, arm='A', object='plate')]))"`

## Validated atomic skill: open drawer

`OpenDrawerPrimitive` passes on seeds 0–4: the measured slide remains open
beyond 4 cm after release, with no unexpected deep contact. Detailed SO-101
meshes remain visual-only near the gripper; convex finger-pad proxies provide
the actual MuJoCo contact. The temporary grasp constraint activates only after
finger-pad/handle contact.

```powershell
python scripts/visualize_run.py --seed 0
python scripts/record_skill_demos.py --task open_drawer --episodes 1 --seed 0
```

Do not record plate, mug, or pour demonstrations yet. Their grasp approaches
are still under validation.
