# Differences between the paper and this code

The documentation of this repository describes what the code does. Where the paper's text (arXiv:2609.39018,
first version) describes the real-robot system differently, the difference is listed here. Several statements of
the paper's Implementation Details appendix describe the simulated (RoboDojo) implementation of the interface,
which is not part of this release.

| # | Paper | This code (real robot) |
| --- | --- | --- |
| 1 | Appendix, Tool calls: an observation identifier stays valid for 120 s and only until the robot next moves; a call citing an expired identifier is refused as stale. | Drafting refuses an identifier that is not the latest observation; there is no 120 s lifetime for observations and robot motion does not by itself invalidate one. If the robot moved since the observation, drafting is refused only when `auto_prepare.refresh_observation` is off; with it on (default and in the trials) `preview` re-observes and re-binds the draft. The 120 s lifetime applies to **previews**, and `execute` regenerates an expired preview instead of refusing. |
| 2 | Appendix, Tool calls: a dry run returns the plan without motion. | `preview` is the dry run (draft without `commit` plus preview). It may send CAN commands that restore control and hold the current pose (see [safety.md](safety.md)). |
| 3 | Appendix, Tool calls: a refusal reports which stage failed; after a tool finishes, its reply reports the stages it completed. | Refusals name the arm, the progress value and the violated check. Stage-by-stage completion is reported only by the bottle-cap placement sequence; other executions report completion, endpoint error and timing. |
| 4 | Appendix, Gripper events: dynamic release uses a calibrated release delay. | The toss release lead (0.08 s) is an empirical constant; the physical release latency is not calibrated (the planner's notes say so). |
| 5 | Appendix, Path inputs: the requested speed specifies the generated reference. | The requested speed is an upper bound: retiming to the motion profile's joint speed, acceleration and jerk limits only slows the reference down. |
| 6 | Appendix, Local execution backend: model-based checks cover the robot and the calibrated table. | Robot (joint limits, self-collision, inter-arm) checks always run. Table checks are a setting; they were **off** during the paper's trials. They are on by default in this release. |
| 7 | Appendix, AgileX platform: the table plane is fitted from depth at run time and rejected when its residual or tilt exceeds thresholds. | The table-plane fit used by the tools rejects a frame only when too few points support the plane; there are no residual or tilt thresholds. |
| 8 | Appendix, AgileX platform: wrist cameras calibrated eye-in-hand and the overhead camera eye-to-hand against a planar AprilTag board; both arm bases registered through the same board. | The overhead camera was calibrated eye-to-hand with a ChArUco board. Only one wrist camera had an eye-in-hand calibration; the other used the same mount transform. The right arm base was located by a single fingertip measurement with the overhead depth camera (translation only, 0.59 m), not through the board. |
| 9 | Appendix, AgileX platform: four RealSense cameras. | The tools observe through the overhead camera only. Wrist cameras are available through read-only endpoints; the front camera is not used by the software (video only). |
| 10 | Appendix, AgileX platform: perception models run on off-board GPU servers; the robot host runs only capture, geometry and control. | The released tools call no learned perception model; segmentation and geometry are depth-based and run on the robot host. |
| 11 | Figure 5 marks pour blocks as using a task-specific tool. | `pour` is a general pouring tool; pouring blocks uses it with a larger tilt (110–130 deg). |
| 12 | Section 4.3 (fold case): eight revised PnP versions (layer-height quantile, local thickness, wrist tilt). | Those revisions belong to the simulated tool collection. Real-robot clothes folding used `pick_place` with `pinch` and dual-arm drafts. |
| 13 | Not mentioned. | After a completed drafted motion the arms return home automatically (except bottle-cap holding stages); this time is part of the measured episode time. |
| 14 | Not mentioned. | The tracking guard (stop when a joint leaves its reference) was off during the trials (0 was the default at the time; the trial records do not store it). It defaults to 3 deg in this release. |
| 15 | Section 4: tool development by a programming agent, validated and frozen before evaluation. | This repository contains the real-robot tool collection as it existed after the trials; it does not contain the programming agent's development history or the execution-agent briefs. |

## Changes made after the paper's trials that this release includes

- Upright bottle handling: prepare, world-axis cap turning, retract, assisted release and body recentering.
- Bottle cap: an optional actively held bottle during the twist stage (wider holder tolerances only in that mode).
- Collision model: the tool-housing capsule ends at the finger roots instead of the fingertip centre.
- Conservative safety defaults, token authentication, loopback binding (see [safety.md](safety.md)).
- A new hardware runtime on `piper_sdk` and `pyrealsense2` (the trials used the authors' internal runtime). Its
  start-up enables the gripper at the measured opening instead of at 0 mm, so the gripper no longer closes when
  the service starts.
- `pour` ends at the pre-grasp point after putting the cup back; the service's automatic joint-space return home
  brings the arm back. During the trials the tool itself appended a Cartesian move back to the start pose, which
  passed at the trials' 50 mm / 40 deg tolerance but not at the default 10 mm / 3 deg near the home pose, where
  joints 2 and 3 sit on their limits.
- The fingertip-floor lift accepts a fingertip exactly on the floor (a 1e-6 mm roundoff allowance); before, an
  exact IK solution 1e-13 mm below the floor was refused. Table checks were off during the trials, so this never
  showed there.
- Readiness checks and the position hold require both the low-speed and the joint feedback frames to be fresh
  (before: only the low-speed frames).
- Every request that lacks a required field is refused with a reason instead of failing with a server error.
