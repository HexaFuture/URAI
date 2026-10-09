# Tools, drafts and the motion contract

A URAI *tool* turns one screen-space intent (a stroke, a pair of points or a single point on the current
observation) plus a few arguments into a **draft**: a timed metric path with orientation, speed profile and gripper
events for one or both arms. Tools never touch the robot. Every draft, whether a person drew it in the console or an
agent requested it through the API, goes through the same `preview -> execute` chain and the same backend.

This matches the paper's request representation `a = (p(s), R(s), v(s), E, C)`: `p` is the `path`, `R` the
`orientation`, `v` the `speed`, `E` the `gripper_events`, and `C` the execution settings (pose tolerance and motion
profile are service-wide settings, `PUT /api/settings`, rather than per-request fields).

## Tools in this release

| Tool | Call | Stroke | Used in the paper's real-robot tasks |
| --- | --- | --- | --- |
| `pick_place` | `POST /api/skills/pick_place` or `POST /api/pick-place` | two points: object, free table spot | tic-tac-toe, block into bowl, hot-dog serving, clothes folding (`pinch`, `top_down`) |
| `grasp_line` | `POST /api/skills/grasp_line` or `POST /api/grasp` | two points across the object (the closing line) | hot-dog serving (grasp, and release at a point) |
| `carry_line` | `POST /api/skills/carry_line` | one stroke: a short grasp line across the object, then the carry route, ending where to put it down | hot-dog serving |
| `pour` | `POST /api/skills/pour` or `POST /api/pour` | two points: cup, target container | pour blocks |
| `bottle_cap_prepare`, `bottle_cap_twist`, `bottle_cap_place`, `bottle_cap_retract` | `POST /api/skills/<name>` | one point each | unscrew bottle cap |
| `toss` (task queue) | `POST /api/tasks` | object pixel or grasp line per item, plus a drop point | toss blocks into bowl |
| free drawing | console brush / `PUT /api/draft` | any path | — |
| observe, home | `POST /api/observe`, `POST /api/home` | — | all tasks |

`GET /api/skills` is the authoritative catalogue: each entry lists its stroke kind, every argument with type,
range, default and unit, and the tool's own limits. The catalogue is what the execution agent reads; the console
renders the same entries.

Common arguments of every registered skill: `speed_m_s` (target speed of free and carrying segments, 0.001–1 m/s,
default 0.8) and `approach_speed_m_s` (automatic approach from the current pose, default 0.8). Both are upper bounds:
the motion profile's joint limits retime the trajectory.

### pick_place

Grasp the object under the first point, lift by `clearance_m`, carry, and release just above the surface under the
second point. The end point must be on the table surface (within 35 mm of it). Options:
- `top_down`: keep the wrist vertical for the whole motion instead of leaning back toward the base for far targets
  (vertical grasps reach about 0.33 m from the base).
- `pinch` with `pinch_depth_mm`: skip object segmentation and pinch exactly at the first pixel, pressing the
  fingertips the given depth below the surface; used for cloth, where a single layer is about 1 mm thick.

### grasp_line

Draw a short line across the object: its midpoint (on the depth surface) locates the grasp, its direction is the
closing axis, and `inset_mm` is how far the fingertips go below the object's top. `endpoint_action` is `grasp`
(descend and close, then lift), `release` (go to the point and open) or `none`. `wrist` chooses `auto` (try a
vertical grasp, lean back toward the base when that is out of reach), `lean` or `top_down`. Soft objects such as
bread need a lower `grip_effort` (about 300 mN·m).

### carry_line

One stroke from grasp to release: the first `grasp_span_mm` of the stroke is a grasp line across the object (planned
by `grasp_line`), the rest is the carry route, and the pen's end is where the object is put down. The carry height
is computed from the highest point of the depth image along the route plus `carry_margin_m`; only obstacles the
camera sees are cleared. The whole carry keeps the placing wrist orientation, so a route that swings farther from
the base than the drop point can exceed the default pose tolerance (10 mm / 3 deg) and be refused; the paper's
trials ran with 50 mm / 40 deg.

### pour

Side-grasp the cup at `grasp_fraction` of its height (grip effort `grip_effort`, 300 mN·m by default for a paper cup),
lift, bring the cup mouth over the target container's rim circle, tilt to `tilt_deg`, hold for `hold_s`, return
upright and put the cup back where it was; the tool ends at the pre-grasp point and the service's automatic return
home takes the arm back. Cups must be at least 50 mm tall and narrower than the 70 mm jaw stroke; the end point must
lie inside the container's top-view circle.

### Upright-bottle tools

These tools keep the bottle vertical while the second arm turns the cap about the world Z axis.
They use `POST /api/skills/<name>` and the same observe, draft, preview and execute sequence.

1. Select the holding arm and call `bottle_upright_prepare` on the cap centre. Set `body_offset_mm`
   to the measured cap-to-grip distance; the bottle is carried upright into the shared workspace.
2. Observe again and call `bottle_upright_twist` on the cap centre. Each cycle turns 10–60 degrees;
   `cycles` selects 1–6 cycles. Inspect the result before another turn.
3. Use `bottle_upright_retract` to open and withdraw the cap arm, then observe again before another twist.
4. Once a person supports the bottle, select the holding arm and call `bottle_upright_release`.
   This opens the holder at its current position and clears the session; it does not place the bottle.

`bottle_body_recenter` uses a visible opaque patch on the bottle body instead of the cap. It assumes
180 mm from that point to the cap, then requires a fresh cap observation before twisting.

### bottle_cap_*

A staged two-arm procedure: `prepare` grasps the bottle with the holding arm and brings it horizontal into the
shared workspace; `twist` takes the cap with the other arm and turns it in cycles (the holder stays put and is
re-checked at 10 Hz while the cap turns); `retract` releases the cap gripper and backs off; `place` puts the
bottle back on the table, with the cap arm first retracting and returning home. Each stage is a separate tool call
with its own preview, and the session state (which arm holds, where the bottle axis is) is kept by the service and
reported in `GET /api/state` as `bottle_cap`.

### toss (task queue)

`POST /api/tasks {observation_id, items: [...], toss_style, ...}` queues up to 32 items. Each item names an object
(`object_pixel`, `object_xy` or a `grasp_pixels` line) and its placement: `place` (put it down at the landing
point), `drop` (release above the landing point) or `toss` (throw toward it); drawn grasp lines are only accepted for
`toss`. For each item the queue observes, re-locates the object, plans, checks and executes, then moves on;
any failure stops the queue and marks the remaining items skipped. `toss_style` is `overhand` (top-down grasp, fold
the arm back, swing forward and release on a precomputed joint line) or `sidearm` (swing about the base joint).
The full list of fields with ranges is the `toss` entry of `GET /api/skills`. The queue refuses toss items unless
the motion profile is `throw`: any slower profile retimes the swing so much that the object is released at the right
pose but barely moves. With the `throw` profile, healthy swings can exceed the default 3 deg tracking guard; the
paper's trials ran tossing with `configs/paper-settings.json`.

## Draft format

`PUT /api/draft {"observation_id": ..., "arms": {"left": <arm draft>, "right": <arm draft>}}`; a skill with
`commit: true` stores the same structure. One arm draft:

```json
{
  "path": {"mode": "waypoints", "points": [[0.40, 0.10, 0.20], [0.45, 0.05, 0.12]]},
  "orientation": {"mode": "keyframes", "points": [[0, 180, 0, 0], [1, 180, 0, 30]]},
  "speed": 0.05,
  "gripper_events": [{"s": 1.0, "opening_mm": 0, "hold_s": 0.6, "wait_for_arrival": true}],
  "approach": {"speed_m_s": 0.8, "clearance_m": 0.12}
}
```

- `path`: world coordinates in metres (world = left arm base frame). `mode: "waypoints"` takes 2–128 points,
  interpolated by a monotone cubic (PCHIP); `mode: "function"` takes expressions `x`, `y`, `z` of the progress
  `s ∈ [0, 1]`. Coordinates must stay within ±2 m.
- Expressions are evaluated by a bounded interpreter (never `eval`): numbers, `s`, `pi`, `x0`/`y0`/`z0` (the
  current TCP position), user `parameters`, `+ - * /`, `**` with a constant exponent in [-8, 8], and `sin`, `cos`,
  `sqrt`, `abs`, `minimum`, `maximum`; at most 512 characters and 96 syntax nodes.
- `orientation`: `hold` (keep the current orientation), `free` (only XYZ is constrained; IK chooses the rotation and
  prefers joint continuity), `keyframes` (`[s, roll, pitch, yaw]` rows in degrees, XYZ Euler, from `s = 0` to
  `s = 1`, interpolated on SO(3) by slerp), or `function` (`roll`, `pitch`, `yaw` expressions).
- `speed`: a number or expression in m/s (0.001–5), or `[[s, v], ...]` keyframes. The nominal timing is
  `dt/ds = |p'(s)| / v(s)` with an acceleration envelope (`accel_m_s2`, default 0.1 m/s²) that ramps from and to
  rest; pure rotations are timed by `angular_speed_deg_s` (default 30 deg/s); holds add their own time. The planner
  then retimes to the motion profile's joint limits, which can only slow the motion down.
- `gripper_events` (at most 16, one per progress value): `s` (progress, not time), `opening_mm` (0–70) or a pure
  pause, `hold_s` (0–3 s dwell at that pose), `effort` (1–5000 mN·m torque limit), `ramp_s` (close gradually over
  up to 2 s), `wait_for_arrival` (dispatch only after the measured joints reached the event pose; needs a hold),
  `verify: "holding"` (after a closing hold, require a measured opening between 1.5 and 67 mm, i.e. something is
  between the jaws, otherwise stop), `lead_s` (0–0.5 s early dispatch) and `on_measured` (dispatch against the
  measured progress along the reference instead of the clock; used for releases during fast swings).
- `approach`: if the path does not start at the current TCP pose, the planner prepends an approach and tries the
  routes `direct`, `lift` (rise by `clearance_m` first) and `joint` (joint-space interpolation) in turn; both arms'
  task segments start together after both reach their start poses.
- Optional: `start_hold_s` (0–60 s wait before the task segment), `air_track` (do not try the `lift` route),
  `parameters` for expressions.

Drafts are capped at 64 kB; `merge: true` keeps the other arm of the stored draft when the observation matches.

## What preview checks

On the PiPER-X backend, `POST /api/preview` solves IK for every sample (respecting `pose_tolerance`, default 10 mm
and 3 deg; the orientation tolerance is relaxed to 6 deg for drawn grasp lines, whose vertical pinch matters more
than the exact wrist angle), retimes to the motion profile, and checks the joint interpolant on a
40 ms grid: joint limits, table clearance and fingertip floor (when `table_checks` is on), self-collision and
inter-arm clearance. A refusal names the arm, the progress value and the violated check. Arbitrary scene objects are
not collision-checked. The synthetic simulation backend does not solve IK and checks nothing.

## Adding a tool

A tool is a function registered with `urai.skills.registry.register(...)` in a module imported by
`urai/skills/__init__.py`. It receives a `SkillContext` (the observation, the arm, its calibration and an IK probe)
and the validated stroke pixels and arguments, and returns `{"summary": ..., "arm": <arm draft>}` (or `"arms"`
for two-arm tools). The registry validates argument types and ranges before the planner runs. Because tools only
produce drafts, every check of the preview applies to them unchanged.

## Limitations of the upright-bottle tools

These tools were added after the paper's trials. The release's hardware runtime and these tools together
still need real-robot validation. Depth at transparent surfaces may be unreliable; body recentering
requires a visible opaque surface. A completed turn does not establish that the cap is unscrewed.
With `lift_mm > 0`, the cap gripper stays closed after the last turn; support or remove the cap before
using the retract operation, which opens that gripper. Release requires a person to support the bottle.
