# Hardware

This page describes the robot URAI's real-robot backend drives, how to connect it, and how to start the
service on it (`urai-robot`). The same entry point also runs a kinematic simulator of the rig (`--sim`) for
development without hardware. Calibration is described in [calibration.md](calibration.md), the safety model in
[safety.md](safety.md), and the bring-up checklist in [real-robot-validation.md](real-robot-validation.md).

## Bill of materials

| Item | Count | Role |
| --- | --- | --- |
| AgileX PiPER-X arm with the standard parallel gripper (70 mm stroke) | 2 | Left and right arm on one table. The left arm's base frame is the world frame. |
| USB-CAN adapter (the one shipped with the arm; SocketCAN `gs_usb`) | 2 | One CAN bus per arm, 1 Mbit/s. |
| Intel RealSense D435 | 4 | One fixed **head** camera looking at the workspace, one **wrist** camera on each arm, and one **front** camera that only records video for documentation. The software uses the head and the two wrist cameras; it never opens the front camera. |
| Linux host | 1 | Runs `urai-robot`. Python 3.10+, USB 3 ports for the cameras, two free USB ports for the CAN adapters. |

Software dependencies of the robot side are an optional extra:

```bash
pip install -e ".[robot]"     # piper_sdk (MIT), python-can, pyrealsense2 (Apache-2.0)
```

The simulator needs none of them.

## CAN buses

Each arm is on its own CAN bus at 1 Mbit/s:

```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

(`piper_sdk` ships helper scripts for the same, e.g. `can_activate.sh`.)

**Bind the interface names to the adapters.** The kernel numbers CAN adapters (`can0`, `can1`) in USB
enumeration order, which can change after a reboot or a re-plug; a swapped pair sends the left arm's commands to
the right arm. Give each adapter a fixed name from its serial number with a udev rule and use those names in the
rig file (the defaults are `canleft` and `canright`):

```bash
udevadm info -a -p /sys/class/net/can0 | grep -m1 'ATTRS{serial}'   # serial of the adapter currently called can0
```

```text
# /etc/udev/rules.d/60-piper-can.rules
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="<serial of the left arm's adapter>",  NAME="canleft"
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="<serial of the right arm's adapter>", NAME="canright"
```

Reload the rules (`sudo udevadm control --reload && sudo udevadm trigger`, or re-plug the adapters), bring the
renamed interfaces up at 1 Mbit/s (for example from a small systemd unit) and **check the binding once**: start
`urai-robot` without `--allow-motion`, put one arm in drag-teach mode, move it by hand and confirm in
`GET /api/state` that the joints of that arm (and only that arm) change.

Only one process may use an arm's CAN interface and the cameras at a time; stop any other robot software first.

## Cameras

List the serial numbers of the attached cameras:

```bash
python -c "from urai.robot.cameras import connected_serials; print(connected_serials())"
```

Write the head camera's serial into `head_camera.serial` of the calibration file and the wrist cameras' serials
into `wrist_cameras.left.serial` and `wrist_cameras.right.serial` (see [calibration.md](calibration.md)). Serial
numbers are strings; quote them.

Each camera streams colour (BGR8) and depth (Z16) at 30 fps; depth is aligned to colour, so both share the colour
intrinsics, which are read from the device. Defaults: colour 1280x720 for all three cameras, depth 848x480. Depth
is returned in metres with invalid pixels set to NaN. Every capture returns a frame that arrived at the host after
the call (and that differs from the previous picture), so a picture never shows the arm's previous pose. The head
camera can run a frame pump (`head_stream`, on by default) that feeds the live view without delaying
observations.

Several high-resolution RealSense streams on one USB 3 controller can drop frames; spread the cameras over
controllers, or lower the resolutions in the rig file if captures time out.

## Rig file

Constants that are not part of the calibration live in `urai.robot.rig.RigConfig`. Override any of them with a
JSON object passed as `--rig`; unknown fields and wrong types are refused.

| Field | Default | Meaning |
| --- | --- | --- |
| `left_interface`, `right_interface` | `"canleft"`, `"canright"` | CAN interface of each arm. |
| `speed_percent` | `20` | Controller speed percentage selected at start-up. Executions select their own per motion profile. |
| `grip_effort`, `open_effort` | `1000`, `1000` | Gripper torque limits for closing and opening, in 0.001 N·m (firmware range 0–5000). |
| `max_opening_mm` | `70.0` | Gripper stroke written to the gripper at start-up (the firmware accepts 0, 70 or 100). |
| `head_width`, `head_height` | `1280`, `720` | Head camera colour resolution. |
| `wrist_width`, `wrist_height` | `1280`, `720` | Wrist camera colour resolution. |
| `depth_width`, `depth_height` | `848`, `480` | Depth resolution of all cameras before alignment. |
| `head_stream` | `true` | Run the head camera's frame pump. |
| `table_z_mm` | `3.5` | Table surface height in the arm base frames (mm). |
| `link_clearance_mm` | `60.0` | Minimum height of the joint origins above the table (mm). |
| `fingertip_bias_mm_left`, `fingertip_bias_mm_right` | `0.0` | Real minus modelled fingertip height per arm (mm). |
| `fingertip_below_table_mm` | `5.0` | How far a fingertip may reach below the table (mm). |
| `ik_position_tol_mm` | `2.0` | IK position residual above which a pose counts as unreachable (mm). |

Example:

```json
{"left_interface": "can_left", "right_interface": "can_right", "grip_effort": 1500}
```

## Starting the service

```bash
urai-robot --calibration calib.json                          # observe and plan; nothing moves
urai-robot --calibration calib.json --allow-motion           # the console and the API may move the arms
urai-robot --calibration calib.json --rig rig.json --allow-motion --settings configs/paper-settings.json \
           --log-dir runs/
urai-robot --calibration calib.json --host 0.0.0.0 --token "$URAI_TOKEN"   # reachable from the network
```

The server binds to 127.0.0.1 on port 7860 by default. Binding to any other address requires a token
(`--token` or `URAI_TOKEN`); without one `urai-robot` refuses to start before it touches the hardware.

### Motion gate

`--allow-motion` is the only way to let URAI command the robot, and it is fixed for the life of the process.

- **Without it** the arm drivers are read-only. URAI reads joint, gripper and controller feedback and the cameras;
  every request that would take control, move an arm or move a gripper is refused. The only frames written to the
  CAN buses are the parameter queries `piper_sdk` sends when it opens a port.
- **With it**, start-up brings both arms under control **without moving them**. For each arm it first requires
  fresh joint and driver feedback (a silent bus reads as all-zero joints, and a hold commanded from that reading
  would drive the arm to the zero pose), then: switches the arm to the motion-output role, ends drag-teach mode,
  enables the motors, selects CAN joint-position mode and commands a hold at the measured joints (clamped into the
  joint limits); re-enables motors that a protection stop had disabled, again bracketed by holds at the measured
  joints; writes the gripper parameters and enables the gripper at its measured opening; and selects joint-position
  mode at `speed_percent`. Start-up never homes the arms. Arms in drag-teach mode are taken out of it, so make sure
  nobody is holding an arm when the service starts.

Stopping the service closes the CAN ports and the cameras; it does not disable the motors, so the arms keep
holding their last pose.

### Simulator

```bash
urai-robot --sim                               # bundled nominal example calibration
urai-robot --sim --calibration calib.json      # your rig's geometry
```

`--sim` runs the same backend (IK, collision checks, retiming, the 50 Hz control loop) on simulated arms and
synthetic cameras; motion is always allowed. The simulated arms are ideal joint position servos whose joint speed
is limited to the controller speed percentage of the firmware's 3.0 rad/s limit; they are always under CAN
control, enabled and reporting fresh feedback. The grippers close until they meet an object between the fingers
(the measured opening is then the object's width), carry that object and release it when they open again; a
released object settles upright on the surface below it. The cameras ray-cast a table with two blocks, a bowl, a
cup and a capped bottle, plus both arms drawn as capsules, with nominal D435 intrinsics. The same runtime is
available from Python:

```python
from urai.robot.simulated import build_simulated_runtime
runtime = build_simulated_runtime()          # or (calibration, rig, scene, allow_motion=...)
frame = runtime.capture("head")              # rgb, depth_m (NaN = no return), k, t_world_cam
```

## Hardware tests

The tests under `tests/hardware/` run on the real rig only:

```bash
URAI_HARDWARE_TESTS=1 URAI_CALIBRATION=calib.json pytest tests/hardware           # read-only checks
URAI_HARDWARE_TESTS=1 URAI_HARDWARE_MOTION=1 URAI_CALIBRATION=calib.json pytest tests/hardware
```

`URAI_RIG=rig.json` passes a rig file. The motion tests take control of both arms, step joint 6 by 3 degrees and
move each gripper by 10 mm; clear the workspace and keep the emergency stop in reach. The drag-teach test is
operator-assisted: put one arm in drag-teach mode and set `URAI_TEACHING_ARM=left` or `right`.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `feedback ... s old` at start-up or `CAN feedback expired` when planning | The arm is unpowered, its emergency stop is pressed, the CAN interface is down or bound to the wrong adapter. A silent bus reads as all-zero joints and disabled motors; only the feedback age tells the two apart. |
| `ctrl_mode` shows `TEACHING_MODE(0x2)` | The arm is in drag-teach mode. It ignores joint references until it leaves it; previews restore CAN control once the arm has stood still for 1 s (see safety.md). |
| `no gripper feedback` at start-up | The gripper module does not answer after its parameters were written: check the wrist connector. |
| Camera capture times out | USB bandwidth or a loose cable; the head camera's frame pump rebuilds its pipeline by itself after a stall. |
