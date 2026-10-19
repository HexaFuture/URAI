# Agent-in-the-loop pouring

The controller uses observed bottle-mouth geometry rather than estimating the mouth from the bottle body's segmented height. This handles an off-center grasp: the full 3D vector from TCP to mouth is transformed with the wrist, and translation compensates its motion. The **lowest pouring rim point** is placed over the receiving cup's inner opening.

## Operation

1. Grasp and lift using `pour_agent_prepare`, or begin already holding an **upright, open bottle**. The prepare skill holds position instead of automatically homing.
2. `POST /api/pour-agent/start` begins a session. Supply `arm`, actual/conservative `bottle_length_m` and `bottle_radius_m`. These describe the whole bottle envelope; mouth radius is measured separately. Defaults are 0.35 m and 0.06 m.
3. `POST /api/pour-agent/observe` with `session_id` captures fresh RGB-D and returns `observation_token`, original image dimensions and TCP state.
4. A vision agent labels 4–16 visible points on each rim: the bottle mouth and the cup's **inner** opening. Points must cover at least 120° of the rim. Never label transparent opening centers or infer hidden rims.
5. `POST /api/pour-agent/plan` with session/token, `source_rim_pixels`, `target_rim_pixels`, `confidence` produces a draft. `POST /api/preview` and `/api/execute` use the existing motion planner and controls.
6. Repeat fresh observation → agent landmarks → correction. A correction is at most 15 mm of TCP travel; inspection steps are at most 5°. The default inspection tilt is 20°. Each new observation re-estimates the TCP-to-mouth vector, including grasp slip visible between steps.
7. Once aligned at the inspection angle, a single checked motion tilts to the configured pouring angle (130° by default), holds briefly (0.5 s), and retraces to the inspection pose. **There is no model wait at the flowing pose.** The controller verifies the compiled interpolated path, not only waypoint endpoints.
8. Repeat `POST /api/pour-agent/return` → preview → execute until `return_steps_remaining` is zero. Each step reverses an original executed correction curve. The bottle ends at the initial upright hold; placement is a separate action.

The translation speed parameter `speed_m_s` follows the existing 0.8 m/s default (0.001–1.0 configurable). The actual path also respects the 8°/s angular rate and the existing acceleration envelope.

The model loop is staged visual correction, not a high-frequency visual servo at full pouring tilt. During the brief pour, compensation assumes the observed bottle-to-gripper transform remains fixed. Gripper feedback detects large opening changes, but cannot detect every axial slip; real pouring validation remains necessary.

## Agent runner

`scripts/pour_agent.py` implements the loop using a Responses-compatible vision model. It reads `OPENAI_BASE_URL` (including `/v1` if applicable), `OPENAI_API_KEY`, and `OPENAI_MODEL` from the environment. For a token-protected URAI service, set `URAI_TOKEN` (separate from the model API key). It does not record credentials. Install the optional `agent` dependency group for `requests`.

```sh
python scripts/pour_agent.py --arm left --perception rgb --source-radius-m <measured_mouth_radius> --target-radius-m <measured_inner_radius>
```

Without `--execute`, it stops after producing the first draft and causes no motion. Add `--execute` to perform the loop. The runner returns along known completed corrections if perception fails after an inspection step. A failed/cancelled physical motion requires recovery from its actual state; it is not blindly reversed.

`GET /api/pour-agent` exposes the tool catalog. The ordinary pouring skill remains available separately; selecting it does not automatically invoke this model loop.

## Records and limits

Session files are saved under the service log directory in `pour-agent-<session_id>/`: RGB JPEG, raw depth/calibration NPZ and `events.jsonl`. The runner saves model decisions, response IDs and reported usage in `pour-agent-decisions.jsonl` (change with `--log`).

Real captures require an actual fresh camera exposure timestamp/sequence. Geometry uses raw calibrated world coordinates consistently. Plans cannot reuse an old image after motion or silently rebind its landmarks to a new frame. Poor rim depth/fit, changed grip or invalid geometry requests another observation instead of fabricating a mouth position. Execution retains the robot's existing IK, collision and feedback checks.

Circular rims must be visible in RGB. The legacy depth mode additionally requires usable depth; RGB mode does not read depth for rim positioning. Bottle dimensions are supplied, not inferred from truncated body segmentation. The body clearance envelope is conservative and is not a complete scene collision model. A completed trajectory alone does **not** confirm that water reached the cup; the runner reports `water_transfer_confirmed:false` until actual outcome evidence is supplied.

## Real trial follow-up (2026-10-09)

The first grasp used a body-segmentation centroid and truncated height, and missed. Preparation now anchors its grasp to the selected raw depth surface with calibrated fingertip compensation and an explicit `grasp_inset_mm` (default 15 mm). The operator/agent selects the grasp height by clicking the visible bottle surface.

In this trial two body grasps did not hold; a narrower shoulder grasp succeeded (about 43 mm measured gripper opening). The bottle was still capped. Cap-removal planning encountered inter-arm clearance or IK failures, so no pouring/alignment motion was executed. The bottle was placed back and both arms returned home. This trial does not validate the new pouring loop's water-placement accuracy.

## RGB alignment (2026-10-09)

The agent locates each rim from RGB. `perception:rgb` fits its calibrated perspective projection using measured `source_radius_m` and `target_radius_m`; it does not gate on transparent or reflective depth. Each contour needs 6–16 distinct visible points spanning at least half the rim. The receiver is upright. Radius must refer to the same material edge as the annotation; image residual is not a guarantee of millimetric accuracy.

If source radius is unknown, `perception:rgb_probe` takes visible source contour landmarks and a bounded 10–15 mm `translation_m`. After normal preview/execute, observe again and use `perception:rgb_parallax`, the probe observation's `reference_token`, and the same ordered material landmarks in the new image. A joint two-view circle fit uses measured robot translation for metric scale. It rejects rotation, insufficient parallax and inconsistent matches. Use the resulting calibrated source radius for subsequent RGB observations.

RGB fitting preserves freshness, held-grip, immutable-plan, IK and collision requirements. Every correction requires a new image. Depth images may still be saved alongside calibration for diagnostics, but neither RGB method depends on their values. There is no video recording requirement.

### RGB live trial result

A measured 14.6 mm translation calibrated the bottle-mouth radius using two RGB views (about 13.66 mm). Four bounded alignment motions reduced the model's alignment residual from 51.7 mm to 0.5 mm. This is model residual, not independent positioning accuracy. Two 115-degree pour-and-return bursts completed (22.2 s and 22.7 s); the 125-degree endpoint failed preview and was not executed. The bottle was subsequently placed upright at its original table location, the gripper opened, and both arms were home in the final image. Video was not recorded. Water entering the receiver was not independently confirmed because the mouth/stream was occluded during the burst. 97 targeted tests passed.
