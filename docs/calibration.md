# Calibration

URAI drives a dual PiPER-X rig: two AgileX PiPER-X arms on one table, a fixed head camera looking
at the workspace and one camera on each wrist. Everything the system needs to place the arms and
cameras in one frame is stored in a single JSON calibration file. This page specifies the file and
describes one way to produce it with public tools (OpenCV).

The package ships a **nominal example**, `urai/robot/assets/example_calibration.json`
(`urai.robot.calibration.example_calibration_path()`). It is meant for the simulator and as a
template. It is not a measured calibration and must not be used on hardware.

## Frames and units

| Frame | Definition |
|---|---|
| `world` | The base frame of the left arm: the URDF `base_link` of the left PiPER-X (z up along the joint1 axis; at zero joint angles the gripper points along +x). |
| `base` (per arm) | The URDF `base_link` of that arm. For the left arm it is the world frame. |
| `link6` (per arm) | The URDF `link6` frame. It is the frame of the end pose reported by the arm firmware (`GetArmEndPoseMsgs` in `piper_sdk`). The gripper closes along link6 +x and approaches along link6 +z; the fingertips are 142.5 mm along +z. |
| `camera` | The optical frame of the camera's colour stream, OpenCV convention: x right, y down, z forward along the optical axis. Depth is registered to the colour stream, so the colour intrinsics and this frame also apply to depth pixels. |

A transform named `T_a_b` is a 4x4 homogeneous matrix that maps a point expressed in frame `b`
into frame `a` (equivalently, the pose of `b` in `a`). Every translation in the file is in
**metres**. Rotations must be proper (orthonormal, determinant +1).

Internally the kinematics work in millimetres and degrees (`urai.robot.kinematics`); conversions
happen at the `ArmModel` boundary and in `Calibration.t_world_wrist_camera`.

## File format

```json
{
  "format": "urai-calibration",
  "version": 1,
  "notes": "free text, optional",
  "right_arm_base": {
    "T_world_base_m": [[1, 0, 0, 0], [0, 1, 0, -0.59], [0, 0, 1, 0], [0, 0, 0, 1]]
  },
  "head_camera": {
    "serial": "123456789012",
    "T_world_camera_m": [[...], [...], [...], [0, 0, 0, 1]]
  },
  "wrist_cameras": {
    "left":  {"serial": "123456789013", "T_link6_camera_m": [[...], [...], [...], [0, 0, 0, 1]]},
    "right": {"serial": "123456789014", "T_link6_camera_m": [[...], [...], [...], [0, 0, 0, 1]]}
  }
}
```

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `format` | string | | Must be `"urai-calibration"`. |
| `version` | integer | | Must be `1`. |
| `notes` | string, optional | | Free text: date, setup notes, residuals of the calibration runs. |
| `right_arm_base.T_world_base_m` | 4x4 numbers | m | Pose of the right arm's `base_link` in the world (left base) frame. |
| `head_camera.serial` | string | | Serial number of the head camera (quote it; RealSense serials are digit strings). |
| `head_camera.T_world_camera_m` | 4x4 numbers | m | Pose of the head camera's colour optical frame in the world frame. |
| `wrist_cameras.left.serial`, `wrist_cameras.right.serial` | string | | Serial numbers of the wrist cameras. |
| `wrist_cameras.<arm>.T_link6_camera_m` | 4x4 numbers | m | Pose of that wrist camera's colour optical frame in the same arm's `link6` frame. |

Both wrist cameras are required and are calibrated independently; there is no fallback from one
arm's wrist calibration to the other.

At run time a wrist camera's world pose is

```
T_world_camera = T_world_base(arm) · T_base_link6(q) · T_link6_camera
```

where `T_base_link6(q)` comes from the measured joints (`PiperXKinematics.fk`) or the firmware end
pose; `Calibration.t_world_wrist_camera(arm, t_base_link6_mm)` implements it.

### Validation

`urai.robot.calibration.load_calibration(path)` rejects a file, with a message naming the field,
when:

- a required field is missing or an unknown field is present (typos are not ignored);
- `format` or `version` does not match;
- a serial is not a non-empty string;
- a matrix is not 4 rows of 4 numbers, contains NaN/inf, or its bottom row is not `[0, 0, 0, 1]`;
- a rotation block is not orthonormal (largest entry of `|RᵀR − I|` above `1e-5`) or is a
  reflection (determinant −1).

Write matrices with full double precision (Python `repr` of floats, as `Calibration.save` does);
rounding to 6 decimals stays within the tolerance, coarser rounding does not.

### Rig constants that are not in this file

`urai.robot.rig.RigConfig` holds constants that are measured or chosen separately: the table height
in the base frame (`table_z_mm`), the minimum joint clearance above the table, per-arm fingertip
height biases, CAN interface names, gripper efforts and camera resolutions.
`build_models(calibration, rig)` combines both into the two arm models.

## Calibration procedure

The steps below use a ChArUco board and OpenCV. Use the camera's factory intrinsics (RealSense
devices report them through `librealsense`) and the colour stream at the resolution URAI runs.

The hand-eye solver is `cv2.calibrateHandEye`. The package installs OpenCV 4 (>=4.8, <5),
which provides both the hand-eye solver and the ChArUco API used below.

### 0. Board detection

Print a ChArUco board, measure the printed square size, and glue it to a flat rigid plate.

```python
import cv2
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard((7, 5), square_m, marker_m, dictionary)   # measured sizes, metres
detector = cv2.aruco.CharucoDetector(board)

corners, ids, _, _ = detector.detectBoard(image)
object_points, image_points = board.matchImagePoints(corners, ids)
ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist)      # board pose in camera
```

`(rvec, tvec)` is `T_camera_board`. Reject views with few corners (for example fewer than 12) or a
large reprojection error, and prefer views where the board is not seen edge-on.

### 1. Wrist cameras (eye-in-hand)

For each arm separately:

1. Fix the board on the table, within view of the wrist camera over a range of arm poses.
2. Move the arm through 15 to 30 poses that differ in position and in orientation (rotate about
   all three axes, by 20 degrees or more between poses where possible). At each pose wait until the
   arm is still, then record the joints `q_i` and an image.
3. For each pose compute `T_base_link6_i = fk(q_i)` (convert mm to m) and `T_camera_board_i` from
   the image.
4. Solve

   ```python
   R, t = cv2.calibrateHandEye(
       [T[:3, :3] for T in T_base_link6], [T[:3, 3] for T in T_base_link6],     # gripper -> base
       [T[:3, :3] for T in T_camera_board], [T[:3, 3] for T in T_camera_board], # target -> camera
       method=cv2.CALIB_HAND_EYE_PARK)
   ```

   The result is `T_link6_camera` (camera -> gripper). Store it as
   `wrist_cameras.<arm>.T_link6_camera_m`.
5. Check consistency: `T_base_link6_i · T_link6_camera · T_camera_board_i` is the board pose in the
   base frame and must be the same for every `i`; its spread measures the calibration error.
   Comparing two solver methods
   (`CALIB_HAND_EYE_PARK`, `CALIB_HAND_EYE_TSAI`, `CALIB_HAND_EYE_DANIILIDIS`) is a cheap sanity check.

### 2. Head camera (eye-to-hand)

1. Attach the board rigidly to the left gripper (or clamp it between the fingers), facing the head
   camera.
2. Move the left arm through 15 to 30 varied poses inside the head camera's view, recording `q_i`
   and a head camera image at each.
3. Pass the **inverted** arm poses to the same solver:

   ```python
   T_link6_base = [np.linalg.inv(T) for T in T_base_link6]
   R, t = cv2.calibrateHandEye(
       [T[:3, :3] for T in T_link6_base], [T[:3, 3] for T in T_link6_base],
       [T[:3, :3] for T in T_camera_board], [T[:3, 3] for T in T_camera_board],
       method=cv2.CALIB_HAND_EYE_PARK)
   ```

   With base -> gripper poses in place of gripper -> base, the output is the camera pose in the
   base frame, `T_base_camera`. Because the left base is the world frame, store it directly as
   `head_camera.T_world_camera_m`.
4. Check consistency: `T_link6_base_i · T_base_camera · T_camera_board_i` (the board pose on the
   gripper) must be constant over `i`.

### 3. Right arm base (base-to-base registration)

Either method gives `T_world_right_base`:

- **Through the head camera.** Repeat step 2 with the board on the right gripper. This yields the
  head camera pose in the right base frame, `T_right_base_camera`. Then

  ```
  T_world_right_base = T_world_camera · inverse(T_right_base_camera)
  ```

- **By touching common points.** Mark at least four well-spread, non-collinear points on the table.
  Touch each point with the fingertip of each arm and record the fingertip position in that arm's
  base frame (`PiperXKinematics.link_origins(q)[-1]`, mm). Fit the rigid transform that maps the
  right-arm points onto the left-arm points (Kabsch/Umeyama without scale). The residuals show the
  touching accuracy.

Doing both and comparing the results is a good check.

### 4. Table height

Touch the table with a fingertip at several places within reach and read the fingertip height
`link_origins(q)[-1][2]` in the base frame. Their mean is `RigConfig.table_z_mm`. If the touches of
one arm consistently sit above or below the model, record the difference as that arm's fingertip
bias (`fingertip_bias_mm_left` / `fingertip_bias_mm_right`, real minus model height).

### 5. Verification

Before running tasks, project the model into the images: compute the fingertip positions of both
arms in the world frame (`ArmModel.fk_tcp_world`), project them into the head camera and into each
wrist camera with the calibrated transforms and the colour intrinsics, and overlay them on live
images while moving the arms. The projected points should stay on the real fingertips within a
few pixels in all cameras.
