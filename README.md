# URAI — Universal Robot–Agent Interface

URAI is the robot-side half of the paper
[*Make Code as Policy Great Again: Frontier Agents Write, Call, and Evolve Robot Tools*](https://arxiv.org/abs/2609.39018)
(arXiv:2609.39018). It exposes a collection of manipulation **tools** that a person can use from a web console and
an AI agent can call over an HTTP API, with exactly the same semantics: a tool turns a few pixels on the current
camera observation plus a handful of arguments into a complete, timed, collision-checked motion that runs locally on
the robot at 50 Hz before control returns to the caller.

This repository contains the **real-robot** implementation for a dual-arm AgileX PiPER-X platform and the tools used
in the paper's seven real-robot tasks (unscrew bottle cap, tic-tac-toe, block into bowl, hot-dog serving, toss blocks
into bowl, pour blocks, clothes folding), plus the latest upright-bottle tools. The simulated (RoboDojo) tool collection of the paper is not part of this
release.

> **Safety first.** URAI moves two position-controlled arms with no force control and no model of scene objects.
> Read [docs/safety.md](docs/safety.md) before connecting a robot. Keep the hardware emergency stop within reach.

## Contents

| Path | What it is |
| --- | --- |
| `urai/service.py`, `urai/app.py`, `urai/routes/` | The service: observations, drafts, previews, executions; the HTTP API and the console |
| `urai/trajectory.py`, `urai/approach.py` | The motion contract: metric path, SO(3) orientation, speed profile and gripper events, compiled into a timed trajectory |
| `urai/backend/` | Planning (IK, automatic approaches, time scaling, geometry checks) and the local 50 Hz execution loop |
| `urai/skills/`, `urai/transfer.py`, `urai/grasp.py`, `urai/pour.py`, `urai/toss.py`, `urai/tasks.py`, `urai/bottle_place.py` | The tools |
| `urai/robot/` | PiPER-X kinematics and collision model, calibration format, the hardware runtime (`piper_sdk`, `pyrealsense2`) and a kinematic simulator |
| `urai/static/` | The web console (no third-party code) |
| `configs/paper-settings.json` | The settings the paper's real-robot trials ran with |
| `docs/` | API, tools, safety, hardware, calibration, the paper's tasks, validation checklist, provenance |

## Hardware

- Two AgileX PiPER-X arms (6 DoF, parallel grippers with a 70 mm stroke), mounted side by side about 0.59 m apart.
- Four Intel RealSense D435 cameras: one overhead (fixed, looking down on the table; used for observations), one on
  each wrist (read-only wrist views), and one front camera (video only, not used by the software).
- One CAN adapter per arm (1 Mbit/s), a Linux host with Python ≥ 3.10.

Details, CAN setup and camera serial numbers: [docs/hardware.md](docs/hardware.md).

## Installation

```sh
git clone https://github.com/HexaFuture/URAI.git urai && cd urai
python -m venv .venv && . .venv/bin/activate      # or: uv venv && . .venv/bin/activate
pip install -e ".[robot,dev]"                       # drop "robot" on a machine without the hardware
```

The core needs `numpy`, `scipy`, `fastapi`, `uvicorn`, `pillow` and `opencv-python-headless`; the `robot` extra adds
`piper_sdk`, `python-can` and `pyrealsense2`. Node.js ≥ 20 is only needed to run the console's tests.

## Quick start without hardware

Two simulations are included:

```sh
urai                  # synthetic backend: one block on a grey table, no IK, no collision checks; port 7861
urai-robot --sim      # the real PiPER-X backend on a kinematic simulator: real IK, checks, retiming and the
                      # 50 Hz control loop, simulated arms and a rendered RGB-D tabletop scene; port 7860
```

Open `http://127.0.0.1:<port>/` for the console, or `curl http://127.0.0.1:<port>/api` for the agent quick start.

## Calibration

URAI needs the pose of the right arm base in the left arm's base frame (the world frame), the overhead camera's pose
in the world frame and each wrist camera's mount on its flange. The file format, the frames and a calibration
procedure with standard tools are in [docs/calibration.md](docs/calibration.md);
`urai/robot/assets/example_calibration.json` is a nominal example, not a real calibration.

## Running on the robot

```sh
urai-robot --calibration my_calibration.json              # read-only: state, cameras, planning; no motion
urai-robot --calibration my_calibration.json --allow-motion
```

The server binds to `127.0.0.1:7860`. To reach it from another machine, either tunnel the port (recommended) or bind
another address with a token: `URAI_TOKEN=<secret> urai-robot ... --host 0.0.0.0`, then open
`http://<robot>:7860/?token=<secret>` once in the browser and send `Authorization: Bearer <secret>` from scripts.

Defaults are conservative (`normal` motion profile, tracking guard 3 deg, table checks on). The paper's trials ran
with relaxed settings, kept in `configs/paper-settings.json` (`--settings configs/paper-settings.json`); tossing
needs them. See [docs/safety.md](docs/safety.md#4-safety-relevant-defaults).

Before relying on this release on a robot, work through [docs/real-robot-validation.md](docs/real-robot-validation.md):
the hardware runtime of this release is a new implementation and has not yet been run on the robot.

## Using URAI

**In the console**, choose an arm and a tool, draw on the camera picture (for `pick_place`: from the object to a free
spot on the table), check the preview and execute. The console and an agent use the same endpoints.

**From an agent or a script**, the loop is observe → tool → preview → execute → poll:

```sh
H=http://127.0.0.1:7860
OBS=$(curl -s -X POST $H/api/observe -H 'Content-Type: application/json' -d '{}' | jq -r .id)
curl -s $H/api/observation.jpg -o shot.jpg                       # look at it, choose pixels
REV=$(curl -s -X POST $H/api/skills/pick_place -H 'Content-Type: application/json' \
      -d "{\"observation_id\":\"$OBS\",\"arm\":\"left\",\"pixels\":[[396,226],[560,400]],\"commit\":true}" | jq .revision)
PID=$(curl -s -X POST $H/api/preview -H 'Content-Type: application/json' -d "{\"expected_revision\":$REV}" | jq -r .id)
curl -s -X POST $H/api/execute -H 'Content-Type: application/json' -d "{\"preview_id\":\"$PID\"}"
curl -s $H/api/state | jq .execution.state                         # until completed / error / cancelled
```

`GET /api` returns this sequence in machine-readable form, and `GET /api/skills` lists every tool with its stroke,
arguments (ranges and defaults) and limits. An agent needs nothing else to start.

## Tools

| Tool | Stroke | Purpose |
| --- | --- | --- |
| `pick_place` | object → free table spot | grasp, lift, carry, place; `pinch` for cloth, `top_down` for a vertical wrist |
| `grasp_line` | a short line across the object | grasp along a drawn closing axis, or release at a point |
| `carry_line` | one stroke: grasp line, then the route | grasp, carry along a drawn route at a safe height, put down |
| `pour` | cup → container | side grasp, tilt over the container, return the cup |
| `bottle_cap_prepare` / `_twist` / `_retract` / `_place` | a point on the cap | two-arm cap unscrewing in stages |
| `bottle_upright_prepare` / `_twist` / `_retract` / `_release`, `bottle_body_recenter` | a point on the cap or body | upright bottle handling and cap turning |
| toss queue (`POST /api/tasks`) | objects + drop point | pick and throw (overhand or sidearm) or place, item after item |
| free drawing (console / `PUT /api/draft`) | any path | metric path, orientation keyframes, speed profile, gripper events |

The motion contract, the draft format and how to add a tool: [docs/tools.md](docs/tools.md).
The HTTP API: [docs/api.md](docs/api.md).

## Relation to the paper

- Method, Section 4: the request `a = (p(s), R(s), v(s), E, C)` is the draft format of [docs/tools.md](docs/tools.md);
  bounded expressions, SO(3) interpolation, the timing rule and local 50 Hz execution are implemented in
  `urai/trajectory.py` and `urai/backend/`.
- Real-robot experiments: which tools and arguments each of the seven tasks used is in [docs/tasks.md](docs/tasks.md).
- Where the paper's text and the code differ: [docs/paper-code-differences.md](docs/paper-code-differences.md).
- How this release relates to the code the trials ran on: [docs/provenance.md](docs/provenance.md).

## Development

```sh
pytest                                   # unit, planning, simulator and API tests; hardware tests are skipped
node --test tests/*.cjs                  # console modules against a real local service
URAI_HARDWARE_TESTS=1 pytest -m hardware tests/hardware                       # on the robot, read-only
URAI_HARDWARE_TESTS=1 URAI_HARDWARE_MOTION=1 pytest -m hardware tests/hardware # on the robot, moves the arms
```

The test suite uses no mocks: planning tests run on the real arm models, execution tests drive the real backend
against the kinematic simulator, and API tests call the real application.

## License

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The bundled PiPER-X URDF comes unmodified from
AgileX Robotics' `agx_arm_urdf` repository under the MIT License (see NOTICE).

## Citation

```bibtex
@article{ge2026urai,
  title   = {Make Code as Policy Great Again: Frontier Agents Write, Call, and Evolve Robot Tools},
  author  = {Ge, Shijia and Zhou, Alex and Zeng, Jianshu and Wan, Yexing and Wu, Di and Zheng, Zelin and
             Wang, Yazhe and Jia, Zhiqi and Shangguan, Xuan and Zhu, Jay and Liu, Yijun and He, Lingyu and
             Wu, Sihang and He, Xiao and Gao, Hongcheng},
  journal = {arXiv preprint arXiv:2609.39018},
  year    = {2026}
}
```
