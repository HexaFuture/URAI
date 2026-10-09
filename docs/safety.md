# Safety

URAI streams joint references to two position-controlled arms. It has no force control, no model of
arbitrary scene objects, and no safety rating. Read this page before the first real-robot run.

## 1. The hardware emergency stop comes first

- Keep a hardware emergency stop for both arms within reach of the person supervising every run.
- `POST /api/cancel` (the console's stop button) is a **software** stop: the 50 Hz control loop notices it within
  one cycle (20 ms), stops streaming the trajectory and holds the measured joint positions. It does not cut power,
  and it cannot stop an arm whose controller ignores the hold.
- Closing the browser does **not** stop a running motion; there is no dead-man switch.
- Nobody stands inside the reach of the arms while they move. Tossing (`/api/tasks` with `placement: "toss"`)
  releases objects at about 1 m/s: keep people and fragile things out of the throwing direction.

## 2. Network exposure

- The server binds to `127.0.0.1` by default. Binding to any other address requires a token
  (`--token` or `URAI_TOKEN`), which every request must carry as `Authorization: Bearer <token>` or
  `X-URAI-Token: <token>`; a browser opens `/?token=<token>` once and keeps an HttpOnly cookie.
- Without a token, only loopback `Host` headers are answered (this blocks DNS-rebinding pages), and
  state-changing requests from another browser origin are refused.
- The service is not designed to be exposed to the internet. Use it on a trusted network, behind a VPN or an
  SSH tunnel.
- `/docs` and `/openapi.json` are disabled; unexpected server errors return a generic message and are logged on the
  server instead of being echoed to the client.

## 3. What moves the robot

| Request | Effect on the robot |
| --- | --- |
| `POST /api/execute` | Runs a previewed plan: automatic approach, the drafted motion, then (after a completed drafted motion) an automatic return home of the arms that moved. Bottle-cap stages that must keep holding the bottle do not return home. A preview older than 120 s is regenerated from the same draft (and a fresh observation, when `auto_prepare.refresh_observation` is on) and then executed in the same call. |
| `POST /api/home` | Plans and immediately executes a joint-space move to the all-zero pose (grippers open first unless `open_grippers` is false). It may re-enable an arm that a protection stop disabled. |
| `POST /api/tasks` | Queues up to 32 pick-and-place or pick-and-toss items; each is observed, planned, checked and executed in turn without further confirmation. |
| `POST /api/preview` | **Sends CAN commands** when `auto_prepare.restore_can` is on (default): an arm in drag-teach mode is taken out of it, the arm is put into CAN joint-position mode and the measured pose is held. This happens only after both arms have stood still for 1 s; it commands no motion away from the current pose. |
| `PUT /api/settings` | Changes limits that affect later motions (motion profile, table checks, tracking guard, tolerances). |

Drafting requests (`/api/skills/<name>`, `/api/pick-place`, `/api/pour`, `/api/stroke`, `PUT /api/draft`) never
send commands to the robot. The server only moves when started with `urai-robot --allow-motion`; without it the
driver is read-only and every command that would move the arms is refused. With `--allow-motion`, start-up takes
CAN control of both arms (it requires fresh feedback from each arm first), enables the motors and holds the
measured pose; it never homes or moves the arms on its own. The gripper is enabled at its measured opening, so it
does not close at start-up. Stopping the service leaves the arms holding their pose (motors stay enabled).

## 4. Safety-relevant defaults

| Setting | Default | Paper trials (`configs/paper-settings.json`) |
| --- | --- | --- |
| `motion_profile` | `normal`: 30 deg/s, 120 deg/s², controller speed 40 % | `throw`: 150 deg/s, 2500 deg/s², controller speed 100 % |
| `tracking_limit_deg` (stop when a joint leaves its reference by more than this) | 3.0 | 0 (off; not recorded in the trial logs, 0 was the default then) |
| `table_checks` (table clearance and table collision in planning and in the 50 Hz guard) | on | off |
| `pose_tolerance` | 10 mm / 3 deg | 50 mm / 40 deg (the bottle-cap stages tighten it to 3 mm / 3 deg themselves) |
| `auto_prepare.refresh_observation`, `auto_prepare.restore_can` | on, on | on, on |

With the tracking guard off, nothing stops a blocked arm: position mode keeps pulling until the motion is cancelled
or the emergency stop is pressed. With table checks off, fingertips may be commanded onto the table (this was used to
pinch cloth in the paper trials). The `throw` profile is needed for tossing (the task queue refuses toss items under any other profile); use it
only with a clear workspace.
The tracking guard tolerates the firmware's lag behind fast references (it compares against the reference up to
0.5 s behind), but with the `throw` profile healthy motions can still exceed 3 deg, which is why the paper trials ran
with it off.

Requested Cartesian speeds (skills default to 0.8 m/s, homing to 0.8 m/s) are upper bounds: the planner retimes
every trajectory to the joint speed, acceleration and jerk limits of the active motion profile, so the profile is
what actually limits speed.

## 5. What the checks cover, and what they do not

Covered on every planned sample (40 ms grid) and on the measured joints every control cycle:
joint limits, IK residual against the pose tolerance, self-collision of the gripper with the arm's own base and
upper arm (capsule model), clearance between the two arms (30 mm margin), and, when `table_checks` is on, link
clearance above the table and the fingertip floor.

Not covered:
- objects in the scene (the point cloud is visual context only);
- the head-camera mast, cables and anything else not in the capsule model;
- contact forces: grasps close until the gripper's effort limit, set per skill (e.g. 300 mN·m for a paper cup);
- the release timing of tossed objects (the release lead time is an empirical constant, not a calibrated latency).

The self-collision check refuses only actual interpenetration (0 mm margin) so that the overhand toss can wind up
close to the base column.

## 6. Recovering after a stop

- After a protection stop or a cancelled motion, `POST /api/recover` clears the error without moving anything.
- `POST /api/home` may re-enable a disabled arm; it first waits for both arms to stand still, then re-enables
  holding the measured joints so that no old controller target causes a jump.
- Sending the PiPER `MotionCtrl_1` command with the wrong arguments outside URAI can disable all motors of an arm;
  let URAI or the vendor tools handle mode changes.
