# HTTP API

All endpoints live on one FastAPI server (`urai` for the synthetic simulation, `urai-robot` for the PiPER-X
backend or its kinematic simulator). Bodies are JSON. Refusals of the planner and the service, including missing or invalid fields, come back as
`409 {"detail": "<reason>"}`; a body that is not a JSON object as `422`; unexpected server errors as a generic `500`
whose details are only written to the server log.
`GET /api` returns a machine-readable quick start that summarizes this page.

Authentication: see [safety.md](safety.md#2-network-exposure). With a token, send
`Authorization: Bearer <token>` (or `X-URAI-Token`) on every request.

Language: refusal messages and the skill catalogue's labels are currently in Chinese, as the execution agent saw
them in the paper's real-robot trials; field names, endpoint names and this documentation are English.

## Typical call sequence

```text
POST /api/observe                    -> observation_id; GET /api/observation.jpg for the picture
GET  /api/skills                     -> skill catalogue (once)
POST /api/skills/<name>  {observation_id, arm, pixels, <arguments>, commit: true}   -> draft revision
POST /api/preview        {expected_revision}                                        -> preview_id, plan
POST /api/execute        {preview_id}                                               -> execution starts
GET  /api/state          (poll until execution.state is completed / error / cancelled)
POST /api/home           {arms: ["left", "right"]}
```

## Observation and state

| Method and path | Body | Returns | Moves the robot |
| --- | --- | --- | --- |
| `GET /api/state` | — | backend info, execution state and progress, `observation_id`, `revision`, `preview_id`, settings, both arms' state (`xyz`, `rpy_deg`, `joints_deg`, `gripper_mm`, `enabled`, `feedback_age_s`, `ctrl_mode`), `observation_stale`, task queue summary, bottle-cap session, `recovery_actions` | no |
| `GET /api/settings` / `PUT /api/settings` | any of `pose_tolerance {position_mm, orientation_deg}`, `motion_profile`, `auto_prepare {refresh_observation, restore_can}`, `table_checks`, `tracking_limit_deg` (0–90, 0 = off), `home_speed_m_s` (0.05–1.0) | current settings and `motion_profiles` | no (affects later motions) |
| `POST /api/observe` | `{preserve_draft: false}` | the new observation: `id`, `image` (data URL), intrinsics and camera pose. Clears the draft unless `preserve_draft` | no |
| `GET /api/observation` | — | the current observation | no |
| `GET /api/observation.jpg` | — | the current observation as JPEG bytes | no |
| `POST /api/pick` | `{observation_id, u, v, mode: "plane"\|"surface", z}` | world point under a pixel | no |
| `GET /api/wrist/{left\|right}/observation`, `GET /api/wrist/{arm}/image.jpg` | — | a wrist-camera frame (PiPER-X backend only); never replaces the head observation | no |
| `GET /api/live.jpg`, `GET /api/live.mjpg` | — | live head-camera JPEG / MJPEG stream (boundary `urai-frame`) | no |

## Drafting (never moves the robot)

Every drafting request names the current `observation_id`; an older one is refused (`Stale observation`). If the
robot moved since the observation and `auto_prepare.refresh_observation` is off, drafting is refused too.

| Method and path | Body | Returns |
| --- | --- | --- |
| `GET /api/skills` | — | `{skills: [...]}`: every registered skill (`name`, `label`, `group`, `summary`, `stroke`, `arms`, `inputs` with ranges and defaults, `hint`, `limits`) plus the `toss` entry, which is called through `POST /api/tasks` |
| `POST /api/skills/{name}` | `{observation_id, arm, pixels: [[u, v], ...], <skill inputs>, speed_m_s, approach_speed_m_s, clearance_m, commit, merge, expected_revision}` | the skill's report with `arm` (or `arms`) drafts; with `commit` the draft is saved and `revision` returned |
| `POST /api/grasp` | `{observation_id, arm, pixels: [end1, end2], endpoint_action: "grasp"\|"release"\|"none", inset_mm, grip_effort, top_down, clearance_m, speed_m_s}` | grasp-line draft: the line's midpoint locates the grasp, its direction the closing axis (same planner as the `grasp_line` skill) |
| `POST /api/pick-place` | `{observation_id, arm, pixels: [start, end], pinch, top_down, pinch_depth_mm, clearance_m, speed_m_s}` | pick, lift, place draft for one arm (same planner as the `pick_place` skill) |
| `POST /api/pour` | `{observation_id, arm, pixels: [cup, container], tilt_deg, hold_s, grasp_fraction, mouth_clearance_m, grip_effort, clearance_m, speed_m_s}` | pour draft (same planner as the `pour` skill) |
| `POST /api/stroke` | `{observation_id, pixels, mode: "plane"\|"surface", z, surface_clearance_m}` | a drawn stroke lifted to world coordinates (console brush) |
| `POST /api/objects` | `{observation_id, polygon}` or `{observation_id, all: true}` | raised objects the depth image sees, with nearest arm and graspability |
| `GET /api/draft`, `PUT /api/draft`, `DELETE /api/draft` | `PUT`: `{observation_id, arms: {left?, right?}, expected_revision?, merge?}` | the draft; `PUT` validates and saves it (see [tools.md](tools.md#draft-format)) |
| `POST /api/compile` | — | the compiled per-arm samples of the current draft (no IK, no checks) |
| `GET /api/drop-point`, `PUT /api/drop-point` | `{xyz}` | the console's toss/drop target |

## Planning and motion

| Method and path | Body | Effect |
| --- | --- | --- |
| `POST /api/preview` | `{expected_revision?, expected_cancel_epoch?}` | Plans the draft: automatic approach routes, IK, retiming to the motion profile, then joint limits, table, self-collision and inter-arm checks on a 40 ms grid. **May send CAN commands** to restore control (see [safety.md](safety.md#3-what-moves-the-robot)). Returns the plan with `id`, per-arm samples, `duration_s`, `notes`. |
| `POST /api/execute` | `{preview_id, cancel_epoch?}` | Executes the preview (regenerating it first if it is older than 120 s). Returns at once; poll `GET /api/state`. After a completed drafted motion the arms that moved return home, except during bottle-cap holding stages. |
| `POST /api/home` | `{arms: ["left","right"], open_grippers: true, cancel_epoch?}` | Plans and executes a return to the all-zero joint pose. |
| `POST /api/tasks` | see the `toss` entry of `GET /api/skills` | Queues pick-and-place or pick-and-toss items and runs them one after another. `GET /api/tasks` reports progress, `DELETE /api/tasks` stops the queue. |
| `POST /api/cancel` | — | Software stop: stop streaming and hold the measured pose. |
| `POST /api/recover` | — | Acknowledge an error or cancellation without moving. |

## Execution states

`execution.state` is one of `idle`, `planning`, `ready` (a preview exists), `running`, `cancelling`, `completed`,
`cancelled`, `error`. While `planning`, `stage` and `progress` describe the planner (`prepare`, `refresh`,
`compile`, `approach`, `ik`, `retiming`, `collision`); while `running`, `progress` is the fraction of the
trajectory's duration and `stage` may be `return_home`. Every executed plan is written to `--log-dir` (when given)
as one JSON record with the preview, the measured joint trace, the result and the caller (user agent, origin,
address; `kind` is `console` for same-origin browser requests and `script` otherwise).
