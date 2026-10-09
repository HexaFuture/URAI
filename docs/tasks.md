# The paper's seven real-robot tasks

The paper evaluates URAI on seven tabletop tasks on the AgileX dual-arm platform, with GPT-6 Astra (Codex CLI)
as the execution agent calling URAI tools over the HTTP API. This page maps each task to the tools and arguments the
trial records show in use. The records contain the executed drafts, not tool names, so the tool attribution below
is inferred from each draft's shape (event pattern, number of waypoints, grip efforts).

All trials ran with the settings in [`configs/paper-settings.json`](../configs/paper-settings.json): the `throw`
motion profile, table checks off, pose tolerance 50 mm / 40 deg (the bottle-cap stages use their own 3 mm / 3 deg),
automatic observation refresh and CAN restoration on. The release's defaults are more conservative; see
[safety.md](safety.md). Every executed motion ended with the automatic return home, which counts toward the episode
time reported in the paper.

| Task (paper) | Tools | Arguments and notes from the trial records |
| --- | --- | --- |
| Unscrew bottle cap | `bottle_cap_prepare` → `bottle_cap_twist` → `bottle_cap_place` (`bottle_cap_retract` when a twist has to be undone) | prepare: bottle grip effort 1000 mN·m, 0.10 m/s, 20 deg/s turn into the horizontal hold; twist: 6 cycles, cap grip effort 1500 mN·m; place: the cap arm retracts and homes before the holder puts the bottle down. Holding arm left in one trial, right in two. |
| Tic-tac-toe | `pick_place` (one call per move) | The records attributed to this task are left-arm `pick_place` calls placing pieces on a 3×3 grid about 8 cm apart (carry 0.12 m/s, approach 0.08 m/s, clearance 0.12 m); the attribution is less certain than for the other tasks. The board logic is entirely the execution agent's; URAI only moves pieces. |
| Block into bowl | `pick_place` | Left arm, carry 0.12 m/s, approach 0.08 m/s, leaning wrist (not `top_down`). The first trial includes the agent writing a small execution script that the next two reuse (paper, Appendix). |
| Hot-dog serving | `grasp_line` (grasp, and `endpoint_action: "release"` to put an item down), `pick_place`, `carry_line` | Buns and sausages are grasped with `grip_effort` 300 mN·m. |
| Toss blocks into bowl | `POST /api/tasks` with `placement: "toss"` | `toss_style: "overhand"` (the queue default) for all but one attempt (one `sidearm` attempt); release lead 0.08 s; peak fingertip speed of the throw about 0.95 m/s; both arms. Requires the `throw` motion profile. |
| Pour blocks | `pour` | `tilt_deg` 130 (default) in the first trials, 110 later; `hold_s` 2.5–3.0 s; cup grip effort 300 mN·m. |
| Clothes folding | `pick_place` with `pinch: true` for both arms in one dual-arm draft | Each fold is two pinch drafts (one per arm) submitted together with `PUT /api/draft` and executed simultaneously; fingertips pressed 2–7 mm below the cloth surface; carry 0.12 m/s. |

## Reproducing a task

1. Start the robot service with the paper settings and motion enabled (after reading [safety.md](safety.md)):

   ```sh
   urai-robot --calibration my_calibration.json --allow-motion --settings configs/paper-settings.json
   ```

2. Give the execution agent the task instruction from the paper's Appendix (task settings and success criteria)
   and tell it to read `GET /api` first. The agent briefs used in the paper's trials are not part of this release.
3. Each call follows observe → tool (with `commit`) → preview → execute → poll state; the agent decides the next call
   after every execution.

Differences between this release and the code the trials ran on are listed in
[provenance.md](provenance.md) and [paper-code-differences.md](paper-code-differences.md).
