# 一次 HTTP 调用执行技能

`POST /api/skills/{name}/run` 接收原技能参数，服务器依次生成草稿、执行完整 preview 检查并调用 execute。调用方无需再提交 preview 或 execute。仍需先通过 `/api/observe` 取得新图像，并在该图像上选择像素。

```bash
curl --fail-with-body --max-time 130 \
  -X POST http://ROBOT_IP:7860/api/skills/grasp_line/run \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "grasp-block-001",
    "observation_id": "CURRENT_OBSERVATION_ID",
    "arm": "right",
    "pixels": [[480, 300], [510, 300]],
    "endpoint_action": "grasp",
    "wrist": "auto",
    "inset_mm": 10,
    "grip_effort": 600,
    "speed_m_s": 0.8,
    "approach_speed_m_s": 0.8,
    "wait_s": 120
  }'
```

上面的像素只是格式示例，必须替换成新图上横跨积木的两个夹持端点。`grasp_line` 的两点决定两指闭合方向；`pick_place` 的两点则是抓取中心与放置点。每个新动作使用新 request_id。

- `wait_s` 默认 20，范围 0–120 秒。任务在等待上限内结束时，HTTP 200 返回终态；否则 HTTP 202 返回当前状态，动作继续在服务端运行。
- 不要发送 `execute`、`merge` 或 `commit:false`；这个入口本来就表示执行一个完整技能。
- 相同 request_id、相同动作参数返回原任务，不会重复规划或执行；可以改变 wait_s。相同 ID 不同动作参数会被拒绝。
- 返回 state 为 `completed`、`error`、`cancelled` 或 `unknown`。HTTP 200 表示已返回终态，不代表抓取成功，仍要检查 state 和 result。空抓自动回位返回 `error`、`outcome=empty_grasp`、`ready_for_next=true`。
- 通信中断或等待超时后，查询 `GET /api/skill-runs/{request_id}?wait_s=20`。不需要再次 execute。`unknown` 或 `not_found` 不代表没有执行，不要换 ID 盲重试。
- 仍通过原 `/api/cancel` 取消，原分步接口继续可用。新调用运行期间拒绝其它草稿/技能编辑，防止插入第二个动作。
- 仍沿用现有技能的回位和持物策略：普通抓取结束后保持夹爪回位；有专用持物阶段的技能保留原行为。这里不改变夹持力、速度、碰撞检查或 IK 容差。
- `timings_s` 包含 plan、preview、dispatch、execution_wait 和 total。记录保存在现有服务日志目录下的 `skill_calls/`。

验证：在隔离副本上，新增接口与原技能、执行、空抓测试合计 58 项通过；集成测试使用 SimulationBackend，未将测试动作发送到真机。
