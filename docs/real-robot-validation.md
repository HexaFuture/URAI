# Real-robot validation checklist

This release replaces the hardware runtime the paper's trials used (PiPER driver, cameras, launcher) and changes the
safety defaults. **None of it has been run on the robot yet.** Before the release is published or used on a robot,
a person who knows the PiPER arms runs the checks below on the real platform, in order, and records the result of
each line. Stop at the first unexpected behaviour.

What was verified without the robot: the arm geometry (kinematics, IK, collision model) is bit-identical to the
previous implementation on tens of thousands of cases; the new driver's CAN frames for connecting, joint mode,
joint references, gripper commands, leaving drag-teach mode, control takeover and motor-fault clearing are
byte-identical to the previous driver's on a silent virtual bus (gripper initialisation differs on purpose, item
2.5); planning and the 50 Hz control loop run against the kinematic simulator in the test suite. Everything that
depends on the arms and cameras answering is only verifiable here.

Conventions: **[R]** read-only, nothing moves; **[M]** the arms move. For every [M] step: emergency stop in hand,
workspace clear, nobody within reach of the arms.

## 0. Preparation (no motion)

| # | Step | Expected |
| --- | --- | --- |
| 0.1 | Write the calibration file in the release format ([calibration.md](calibration.md)). Converting an existing calibration from another format: check the result by hand (right base at about (0, −0.59, 0) m, head camera pose, wrist camera mounts, camera serial numbers). | `python -c "from urai.robot.calibration import load_calibration as l; l('calib.json')"` succeeds. |
| 0.2 | Bind the CAN interface names with udev and write the rig file if the defaults do not match ([hardware.md](hardware.md)). | `ip -details link show canleft` / `canright`: `UP`, bitrate 1000000. |
| 0.3 | Install on the robot host in a fresh environment: `pip install -e ".[robot,dev]"`. Run `pytest` and `node --test tests/*.cjs`. | All pass; hardware tests are reported as skipped. |
| 0.4 | Stop every other program that uses the arms' CAN interfaces or the cameras. | — |

## 1. Read-only bring-up [R]

Start `urai-robot --calibration calib.json` **without** `--allow-motion`.

| # | Step | Expected |
| --- | --- | --- |
| 1.1 | Start on a freshly powered rig and read `GET /api/state`. | Nothing moves, no motor clicks. `ctrl_mode` stays what it was (typically `STANDBY(0x0)`; `TEACHING_MODE(0x2)` for an arm in drag-teach; `CAN_CTRL(0x1)` only if a previous session left CAN control), `enabled` unchanged. Joint readings equal the vendor tool's within 0.1 deg; `feedback_age_s` < 0.05. Every move request is refused because motion is not allowed. |
| 1.2 | CAN identity: put the **left** arm in drag-teach mode, move joint 1 by hand, watch `GET /api/state`; repeat for the right arm. | Only the `left` (then only the `right`) joints change. Otherwise the interface names are swapped: stop and fix the udev binding. |
| 1.3 | Geometry: compare `xyz` / `rpy_deg` in `/api/state` with the TCP pose the previous system reported for the same joint readings and calibration. | Equal within 1 mm / 0.1 deg. |
| 1.4 | Silent bus: switch one arm off (or press its emergency stop) with the service running; switch it on again. | That arm's `feedback_age_s` grows past 0.5 s within a second, joints read 0 and `enabled` false, previews are refused with the CAN-feedback-expired message; after power-on the feedback is fresh again. |
| 1.5 | Cameras: `POST /api/observe`, `GET /api/observation.jpg`, `GET /api/wrist/left/image.jpg`, `GET /api/wrist/right/image.jpg`. | Overhead view; `hand_left` shows the left gripper's fingers and moves with the left arm, `hand_right` likewise. The front camera is not opened. |
| 1.6 | Fresh frames: wave a hand in front of the head camera between two observations; move an arm in drag-teach mode and capture its wrist camera twice. | Each picture shows the scene after the call; no picture repeats the previous one. |
| 1.7 | Depth: compare the depth at a block's edge with the colour edge; `POST /api/pick` on the table. | Depth edges coincide with colour edges within 2 px; picked table points lie at the calibrated table height within 1 cm. |
| 1.8 | Live view: open `GET /api/live.mjpg`, call `POST /api/observe` repeatedly; unplug and re-plug the head camera. | About 30 fps without delaying observations; after the re-plug the live view resumes within a few seconds (observations fail with a timeout meanwhile instead of returning an old picture). |
| 1.9 | Wrist extrinsics: with an arm resting at several poses, compare `runtime.camera_pose('hand_left')` with `calibration.t_world_wrist_camera('left', <firmware end pose>)` (firmware end pose from `runtime.arms['left'].piper.GetArmEndPoseMsgs()`). | Equal within 1 mm and 0.1 deg. |
| 1.10 | Draw a `pick_place` on a block and `POST /api/preview`. | The preview plans and returns. If an arm is in drag-teach mode or disabled, the preview is refused because motion is not allowed (nothing is sent). |
| 1.11 | Network: start with `--host 0.0.0.0` and no token; then with `--token`. | Without a token it refuses to start. With a token, requests without it get 401; `http://<robot>:7860/?token=...` opens the console. |
| 1.12 | `URAI_HARDWARE_TESTS=1 URAI_CALIBRATION=calib.json pytest tests/hardware` | All read-only hardware tests pass (motion tests skipped). |

## 2. Motion basics at the conservative defaults [M]

Restart with `--allow-motion` (defaults: `normal` profile, tracking guard 3 deg, table checks on).

| # | Step | Expected |
| --- | --- | --- |
| 2.1 | Start-up takeover in three situations, restarting each time: (a) both arms freshly powered, resting in the folded rest pose; (b) left arm in drag-teach mode, released and resting; (c) after a power cut that left joint 2 slightly below 0 deg. | No jump, no sag when the motors enable; `ctrl_mode` `CAN_CTRL(0x1)`, `enabled` true, `arm_status` 0, `err_code` 0 on both arms; joints within 1 deg of before the start (in (c) only what is needed to bring joint 2 back inside its limit). About 5 s per arm. The start-up never homes. |
| 2.2 | Start-up with one arm switched off. | Start-up stops with an error naming that arm and its stale feedback; nothing is commanded to it. |
| 2.3 | `POST /api/home` with both arms at home. | Grippers open, no arm motion, `completed`. |
| 2.4 | Move one arm by hand in drag-teach mode 20–30 cm from home, release it, `POST /api/home`. | After about 1 s of standstill the arm leaves drag-teach and holds where it is (no jump), then homes slowly. |
| 2.5 | Gripper initialisation: power-cycle, open one gripper to about 30 mm with the vendor tool, start with `--allow-motion`; then command 40 mm and back from the console. | The gripper does **not** close during start-up (opening unchanged within 1 mm); the commands move it. With the gripper module unplugged, start-up stops with a "no gripper feedback" error. |
| 2.6 | Controller speed: step joint 6 by 20 deg at controller speed 10 % and 40 % (hardware motion tests). | At 10 % about 1.2–1.6 s; at 40 % about 2.5–3 times faster; stops within 0.3 deg. |
| 2.7 | Draw a small free path (console brush, 10 cm in the air), preview and execute; press stop half-way through a second run. | The arm follows the preview; the run log's measured trace stays within a few degrees of the reference. On stop the arm halts within about 0.1 s and holds without creep (< 0.2 deg over 10 s); `POST /api/recover` clears the state without motion. |
| 2.8 | Tracking guard: with the `fine` profile, execute a slow path and gently hold the forearm. | Execution stops with a joint tracking error and the arm holds. |
| 2.9 | Table checks: draft a path whose fingertips would go 1 cm below the table. | Preview refuses it (table floor) or lifts the samples by at most a few millimetres and says so in its notes. |
| 2.10 | Inter-arm check: draft the left arm into the parked right arm. | Preview refuses with an inter-arm clearance message. |
| 2.11 | Protection-stop recovery: disable one arm with the vendor's documented method while it rests, then `POST /api/home`. | The motors are re-enabled at the measured pose without a jump, then the arm homes. If a motor stays disabled, the request fails with "motors [...] remain disabled". |
| 2.12 | Shutdown: stop the service (Ctrl-C) while the arms hold a pose away from home; start again with `--allow-motion`. | The arms keep holding on stop; the restart takes control without a jump. |
| 2.13 | `URAI_HARDWARE_TESTS=1 URAI_HARDWARE_MOTION=1 URAI_CALIBRATION=calib.json pytest tests/hardware`, then the drag-teach test with `URAI_TEACHING_ARM=left` (left arm in drag-teach mode). | All pass. |

## 3. The seven paper tasks [M]

Run each at least once with the default settings where the task is feasible, and once with
`configs/paper-settings.json` (required for tossing). Use the tools and arguments in [tasks.md](tasks.md). Record
outcome, duration, the preview's `retiming_factor` and any refusals.

| # | Task | Tool calls | What to watch |
| --- | --- | --- | --- |
| 3.1 | Block into bowl | `pick_place` | The holding check passes on the block; release above the bowl. |
| 3.2 | Tic-tac-toe (one game) | `pick_place` per move | Pieces land in the cells; return home between moves. |
| 3.3 | Hot-dog serving | `grasp_line` (grasp, release), `pick_place`, `carry_line`, effort 300 | Bread is not crushed; the carry height clears the plates. |
| 3.4 | Pour blocks | `pour` (tilt 110–130) | The cup is not crushed at 300 mN·m; blocks land in the container; the cup is put back. |
| 3.5 | Toss blocks into bowl (paper settings) | `POST /api/tasks`, `overhand`, then one `sidearm` | Release at the planned point (lead 0.08 s); nobody in the throwing direction. |
| 3.6 | Unscrew bottle cap | `bottle_cap_prepare`, `twist`, `place` (and `retract`) | The holder stays put during the twist (holder checked at 10 Hz); in `place` the cap arm retracts and homes before the holder moves. |
| 3.7 | Clothes folding (one fold) | two `pick_place` with `pinch: true`, one per arm, in one `PUT /api/draft` | Both task segments start together; fingertips press the cloth without pushing it away. |

## 4. Comparison with the previous system [R]

| # | Step | Expected |
| --- | --- | --- |
| 4.1 | For three drafts of section 3 (one pick-place, one pour, one bottle-cap stage), run `POST /api/preview` on the same draft and robot state with the previous system and with this release (same settings). | Same feasibility; joint trajectories within 0.1 deg, except where the documented changes apply (tool-housing capsule, defaults). |
| 4.2 | Read the run logs of section 3. | Control iterations stay at 20 ms (no scheduler-stall or deadline errors); feedback age < 0.05 s throughout. |

## 5. Backend behaviours that only the robot can show

The automated part is `tests/hardware/test_backend_on_robot.py`; run it with `-s` so the operator prompts in each
test's docstring are visible (`URAI_HARDWARE_TESTS=1 URAI_HARDWARE_MOTION=1 pytest -s -m hardware
tests/hardware/test_backend_on_robot.py`). It covers: refusal before any IK when an arm is in drag-teach mode,
stale CAN feedback (cable unplugged or arm powered off), `STANDBY` instead of CAN control, a controller error code;
waiting for 1 s of standstill before leaving drag-teach and holding the measured pose (< 0.5 deg change); a fresh
observation after control was restored; no takeover while the arm keeps being dragged; no mode change when the
wait is cancelled or `restore_can` is off; a disabled arm refused by preview but re-enabled by home; never
re-enabling an arm that is disabled **and** reports a controller error (emergency stop); and a control-deadline
overrun under CPU load ending either normally or with a deadline/stall error and a hold.

Check by hand, recording what happens:

| # | Situation | Expected |
| --- | --- | --- |
| 5.1 | During a home move the measured j3 briefly reads slightly past its nominal limit (e.g. +0.08 deg). | Execution is not refused for it. |
| 5.2 | Restore control from a drag-taught pose outside the nominal joint limits. | No clamping jump; the approach moves back inside the limits. |
| 5.3 | Cancel while control is being restored (between leaving drag-teach and the hold). | No further mode change or hold command is sent; a failed hold is reported. |
| 5.4 | Switch an arm to drag-teach mode while a preview is planning. | The preview retries once after restoring control, then refuses. |
| 5.5 | A drafted motion whose last gripper event has no hold (`hold_s`), so the jaws are still moving when the task segment ends. | Watch whether the automatic return home starts or fails with "gripper moved since preview" (the simulator shows the failure; record what the real gripper does). |
| 5.6 | Live view in stream mode while executing at 50 Hz. | Colour order correct (no blue faces), observations not delayed, control iterations stay below 250 ms apart. |
| 5.7 | `throw` profile with the default 3 deg tracking guard on a gentle toss. | Record whether healthy swings trip the guard (expected on fast swings; the paper ran with the guard off). |
| 5.8 | Real grasps: an empty close and an off-centre grasp of a block. | The empty close is reported ("疑似空抓") and the arm holds; record whether the off-centre grasp pushes the block into the jaws. |
| 5.9 | Pinch of a single layer of cloth with `pinch_depth_mm` 5 and table checks on. | The fingertips reach about 4.5 mm below the surface and lift the cloth. Re-measure the 10 mm pinch compensation of `urai/grasp.py` on your rig. |

## 6. Upright-bottle tools

The upright-bottle tools were added after the paper's trials. Test them with an empty bottle before a filled
one. Use the measured cap-to-body grip distance and inspect each preview before executing.

- Prepare with each holding arm: confirm the bottle stays vertical, the grasp is secure, the worker remains
  clear, and the holder stays in place after completion.
- Refresh the observation and perform one 10-degree twist. Check that rotation is about the bottle's vertical
  axis and the holding arm remains stationary. Increase angle and cycles only after that succeeds.
- Retract the cap arm, refresh the observation, and plan another twist. The preserved holder session must
  relocalize against the new image.
- Test `bottle_body_recenter` on an opaque label with reliable depth, using a bottle whose cap is 180 mm above
  the selected body point. Refresh the cap observation before twisting.
- Support the bottle by hand before `bottle_upright_release`. Only the holder gripper should open; neither arm
  should move or return home, and the holder session should clear.
- If using the optional final cap lift, verify cap clearance and retention. Remove or support the cap before
  retracting, since retract opens the cap gripper.
