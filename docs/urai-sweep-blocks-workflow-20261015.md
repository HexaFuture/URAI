# Sweep Blocks：实机工作流与 URAI 接口

日期：2026-10-15。平台：FoldRobot / AgileX 双臂。

本轮结果：中心区域四块积木（两个红块、蓝块、橙块）全部扫入簸箕，提起确认后，将装有积木的簸箕和扫帚放回桌面，双臂松爪退离。左侧大堆积木不在本轮目标范围内。

最终照片：[final-on-table.jpg](../artifacts/sweep-20261015/final-on-table.jpg)。阶段执行器：[run_stage.py](../artifacts/sweep-20261015/run_stage.py)。

## 1. 工作流与完成条件

1. **观测并分配角色**：定位扫帚手柄、簸箕手柄、刷毛、簸箕前沿和目标积木。根据物体所在位置及可达性分配持扫帚臂和持簸箕臂，不固定左右手。本轮最终左手持扫帚、右手持簸箕。
2. **扫帚交接**：给出臂抓手柄的一段，接收臂抓另一段手柄。接收臂合爪持稳后，给出臂才松爪并退开。两臂均抓手柄，不能把刷头当成接收抓点。
3. **调整扫帚姿态**：刷毛朝下，刷毛下沿接近桌面；夹爪滑轨、指尖和腕部在桌面上方。主摄遮挡刷头时，用手部相机侧视确定刷毛方向。不要把“逆时针 90°”当成不依赖相机视角的固定指令。
4. **抓簸箕手柄**：另一臂夹住手柄并短距离提起，确定簸箕随夹爪运动。不要抓盘沿，否则会占用入口并增加碰撞风险。
5. **让簸箕入口贴桌**：入口朝向积木，检查实际前沿高度。本轮一度悬空约 30 mm，积木钻到下面；通过退出积木上方、调整手柄高度及倾角，才建立有效入口。降低 TCP 不保证降低前沿：簸箕后部支撑在桌上时可能翘起前沿。
6. **分区短扫**：刷毛从积木后方向簸箕入口推进，进盘时随底面适当抬升；随后抬刷并退回。根据剩余积木位置横向换位。本轮逐次收进 1、2、3、4 块。长行程被双臂碰撞检查拒绝后，改用较短行程，未解除检查。
7. **装载确认**：图像确认目标积木在簸箕内部，而非遮在盘底或停在入口外。轨迹返回 completed 只表示运动完成，不等于收集成功。
8. **收尾必须落桌**：如需展示，可提起装载簸箕；展示后降回受支撑的桌面位置，松爪、退离。扫帚也放到空闲桌面，松爪、退离。不能以双臂悬空持工具作为最终完成状态。

完成条件：`目标积木在簸箕内 + 簸箕在桌面 + 扫帚在桌面 + 双臂已释放工具并退离`。

## 2. 本轮接口的真实状态

目前完成的是**已有 URAI 接口驱动的分阶段组合流程**，没有新增注册 `sweep_blocks` 一键技能。

| 能力 | 现有接口 | 适用范围 |
|---|---|---|
| 技能目录 | `GET /api/skills` | 读取当前注册技能、参数和限制 |
| 画线抓取 | `POST /api/skills/grasp_line` | 从本轮观测的手柄抓取线生成草稿 |
| 通用交接 | `POST /api/skills/handover` | 已有双臂交接规划；有物体长度、夹爪间距等限制，不保证适用于扫帚 |
| 通用扫拢 | `POST /api/skills/sweep` | 以闭合夹爪作推扫工具；不是本轮持扫帚和簸箕的完整技能 |
| 定制持工具轨迹 | `urai.skills.motion.draft(...)` | 本轮手柄交接、刷毛对齐、簸箕摆放、短扫、落桌的基础接口 |
| 保存双臂草稿 | `PUT /api/draft` | 一臂执行，另一臂以固定轨迹保持当前位置 |
| 预览并校验 | `POST /api/preview` | 全路径 IK、关节及自碰撞/双臂碰撞检查 |
| 开始执行 | `POST /api/execute` | 执行预览生成的计划 |
| 查询结果 | `GET /api/state` | 等待 completed；error/cancelled 时不进入依赖该结果的下一阶段 |

当前部署源码：`<URAI_ROOT>/urai/app.py`、`skills/__init__.py`、`skills/motion.py`、`skills/bimanual.py`、`skills/planar.py`。

机器人本机 API：`http://127.0.0.1:7860`。本轮局域网地址：`http://127.0.0.1:7860`，下次连接应重新确认可达性。开发执行通道是本次进程的 `http://127.0.0.1:45195/exec`，不是稳定的业务技能地址。

## 3. 标准技能调用顺序

```text
POST /api/observe {}
GET  /api/observation             → 获取 observation_id 与图像尺寸
GET  /api/observation.jpg         → 在这张观测图上选点
POST /api/skills/<name>           → 生成草稿，commit=true 时保存
POST /api/preview                 → 返回计划 id
POST /api/execute                 → 用 preview_id 启动
GET  /api/state                   → 轮询至完成，再进入下一阶段
```

画线抓取请求模板，变量必须来自当前观测：

```python
payload = {
    "observation_id": observation_id,
    "arm": selected_arm,
    "pixels": [[u1, v1], [u2, v2]],
    "inset_mm": 7,
    "grip_effort": 1000,
    "wrist": "auto",
    "speed_m_s": 0.15,
    "approach_speed_m_s": 0.15,
    "commit": True,
}
# POST /api/skills/grasp_line, JSON = payload
# POST /api/preview, JSON = {"expected_revision": skill_result["revision"]}
# POST /api/execute, JSON = {
#     "preview_id": preview_result["id"],
#     "cancel_epoch": current_state["cancel_epoch"],
# }
```

像素必须对应 observation.jpg 的原始尺寸；live.jpg 可能缩放，不可直接混用坐标。运动之后重新观测再选新抓点。`commit` 只保存草稿，不会让机器人运动。

## 4. 持工具动作的草稿接口

```python
from urai.skills.motion import draft, open_event

moving = draft(
    points=world_xyz_waypoints,      # 米，机器人世界坐标
    keyframes=[[0, *start_rpy], [1, *goal_rpy]],  # xyz Euler，度
    events=gripper_events,
    speed_m_s=0.075,
    clearance_m=0.08,
    approach_speed_m_s=0.15,
    angular_speed_deg_s=35,
)
holding = draft(
    [other_xyz, other_xyz],
    [[0, *other_rpy], [1, *other_rpy]],
    [], 0.08, 0.08,
)
body = {
    "observation_id": observation_id,
    "arms": {moving_arm: moving, holding_arm: holding},
}
# PUT /api/draft, JSON = body；然后 preview → execute。
```

释放动作：工具已有桌面支撑后，建立一条离开桌面的退让轨迹，在起点设置 `open_event(0, opening_mm=50, hold_s=0.7)`。不要在下降前提前松爪。

`grip_effort` 的源码单位是 0.001 N·m，是执行器参数，并非接触力估计。本轮手柄抓取主要使用 900–1000；刷扫使用约 0.075 m/s；落桌使用约 0.06 m/s。这是本场景已执行参数，不代表换工具或换布局后可直接沿用。

## 5. 本轮阶段执行器

机器人上的文件：`<URAI_ROOT>/artifacts/sweep-20261015/run_stage.py`。

支持 `kind="grasp" | "transfer" | "move"`。其中 move 接口为：

```json
{
  "label": "unique-stage-name",
  "kind": "move",
  "arm": "left",
  "points": [[0.28, -0.32, 0.15]],
  "rpy": [-170, -4.6, -99.1],
  "speed": 0.075,
  "hold_other": true,
  "events": [],
  "execute": false
}
```

上例仅展示结构和本轮坐标尺度，默认示例为**只预览**。场景或标定改变后不得直接回放这些绝对坐标。`rpy` 省略时保持当前实际姿态；`hold_other=true` 把另一臂加入保持轨迹。执行器先观测、保存草稿、预览，再执行；每段保存 `*-plan.json` 和 `*-result.json`。它使用开发进程内的 backend 执行，因此新集成优先使用前述公开 HTTP execute 路径，以沿用标准执行记录。

## 6. 成功片段与复用边界

| 阶段 | 作用 |
|---|---|
| 166–169 | 手柄交接完成；手部相机侧视确认刷毛向下 |
| 170–171 | 抓簸箕手柄、试提 |
| 183–185 | 抬高手柄、压低前沿，簸箕向目标靠近 |
| 187、188、190、191 | 短行程分区补扫，最终四块入簸箕 |
| 192 | 提起已装载簸箕 |
| 193–194 | 装载簸箕落桌、松爪、退离 |
| 195–196 | 扫帚落桌、松爪、退离 |

应复用动作顺序、工具坐标关系、双臂保持和执行接口；每次重新获取抓点、刷毛接触边、簸箕入口和目标块的位置。不要把本轮 Euler 数值或 TCP 高度当成工具的通用工作姿态。

后续若封装注册技能，建议命名 `sweep_blocks`，包含角色选择、可选手柄交接、工具对齐、分区扫入、落桌释放等阶段，并默认 `return_tools_to_table=true`。这只是后续接口建议，本轮未注册该名称，也未验证一次调用自动完成任意布局。
