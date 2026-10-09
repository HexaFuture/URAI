# Provenance of this release

This repository is a cleaned-up release of the authors' internal real-robot code. It starts a new history; the
internal history is not published.

## Source snapshot

- Taken from the internal development checkout's **working tree** on 2026-10-08 at 11:30:04 UTC.
- Base commit of that working tree: 2026-09-23 19:49 (UTC+8), "Integrate current real-robot skills and operator
  controls". That commit was recorded right after the paper's real-robot trials (2026-09-20 to 2026-09-23); the trials
  themselves ran on working-tree states before it, which were not committed individually.
- The working tree contained uncommitted changes made after the paper (bottle-cap work in progress on 2026-10-08).
  They are included where they belong to the released tools (see below) and may still change:

| File in the internal tree | State |
| --- | --- |
| `uari/service.py` | modified |
| `uari/skills/__init__.py` | modified |
| `uari/skills/bottle_cap.py` | modified |
| `uari/skills/bottle_upright.py` | new |
| `tests/test_bottle_active_holder.py` | new |

  - Included: the bottle-cap changes (an optional actively held bottle during the twist stage, with wider holder
    tolerances only in that mode; backward compatible) and their test.
  - Included: the new upright-bottle tools (`bottle_upright_*`, `bottle_body_recenter`), developed after
    the paper's trials.
- The package was renamed from the internal `uari` to `urai`, matching the paper's name, Universal Robot–Agent
  Interface.

## Arm geometry

The PiPER-X kinematics and collision model (`urai/robot/kinematics.py`, `urai/robot/arm_model.py`) are derived from
the authors' internal robot SDK as it ran on the robot on 2026-10-08, including one change made after the paper:
the tool-housing collision capsule now ends at the finger roots instead of the fingertip centre (the fingers have
their own capsules). During the paper's trials the housing capsule extended to the fingertip centre.

## What the release leaves out

- Relational keypoint constraints (ReKep-style solving, tracking and execution-time constraint monitoring).
  The released service has no code path that executes caller-supplied code.
- 32 further skills that the paper's real-robot tasks did not use (pushing, rotary, tool use, articulated objects,
  other bimanual skills, the separate `carry` tool, plug insertion with joint-current feedback), and the whip
  toss style, which was never run on the robot. The skills kept are the ones the trial records show in use:
  `pick_place`, `grasp_line`, `carry_line`, `pour`, the four `bottle_cap_*` stages, and the toss queue.
- Internal launch scripts, benchmark and search scripts, development notes and run records.
- The authors' calibration data; `urai/robot/assets/example_calibration.json` is a nominal example only.

## What the release changes

- Safety defaults (motion profile, tracking guard, table checks), network binding and token authentication; the
  paper's settings are kept in `configs/paper-settings.json`. See [safety.md](safety.md).
- The hardware runtime (PiPER driver, RealSense cameras, launcher) is a new implementation on the public
  `piper_sdk` and `pyrealsense2`; the paper's trials used the internal SDK's runtime.
- Differences between the paper's text and the code are listed in [paper-code-differences.md](paper-code-differences.md).
