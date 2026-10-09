"""The skills the console already had, declared in the registry so one catalogue lists everything.

Each planner stays where it is; this module only says what the skill is called, what it draws and what numbers
it takes, and hands the context's calibration to the existing entry point. The original endpoints are unchanged.
"""
from ..grasp import grasp_from_line, grasp_terms
from ..pour import CUP_GRIP_EFFORT, POUR_MOUTH_CLEARANCE_M, pour_from_stroke
from ..transfer import MAX_PINCH_DEPTH_M, PINCH_DEPTH_M, pick_place_from_stroke
from .motion import BLOCK_GRIP_EFFORT
from .registry import boolean, choice, integer, number, register


def transfer_arguments(ctx):
    return {'table_z': ctx.table_z, 'fingertip_bias_mm': ctx.fingertip_bias_mm,
            'preferred_yaw_deg': ctx.preferred_yaw_deg, 'base_xy': ctx.base_xy,
            'clearance_m': ctx.clearance_m, 'speed_m_s': ctx.speed_m_s}


@register(name='pick_place', label='两点抓放', group='抓取搬运', stroke='two_points',
          summary='起点抓住物体，终点放到空闲桌面：张开 → 抓住 → 抬起 → 平移 → 放下 → 松手',
          hint='起点画在物体上，终点画在空闲桌面。',
          limits='终点必须是桌面（离桌面 35 mm 内）。物体可见宽度只作记录不再据此拒绝——分割会把把手、挨着的东西和深度光晕算进去，夹口张开就是 70 mm，夹不夹得住看画面判断。',
          inputs={'pinch': boolean('按落笔处捏取', False,
                                   hint='不做物体分割，就在起点像素把桌面上的薄物（衣服、纸张）捏起来，指尖压到桌面下“捏取下探”那么深。'),
                  'top_down': boolean('全程竖直抓取', False, hint='手腕全程朝下，不向基座后仰（远处可能够不到）。'),
                  'pinch_depth_mm': number('捏取下探', 0, MAX_PINCH_DEPTH_M*1000, PINCH_DEPTH_M*1000, step=1, unit=' mm',
                                           hint='只在捏取时使用：一层布只有 1 mm，指尖停在桌面上方就是空夹。')})
def plan_pick_place(ctx, ink, pinch, top_down, pinch_depth_mm):
    # The same three options as POST /api/pick-place, passed through unchanged.
    result = pick_place_from_stroke(ctx.frame, ink, **transfer_arguments(ctx), pinch=pinch, top_down=top_down,
                                    pinch_depth_mm=pinch_depth_mm)
    return {**result, 'summary': f'两点抓放 · 物体宽约 {result["object_width_mm"]:.0f} mm'}


@register(name='grasp_line', label='画线抓取', group='抓取搬运', stroke='two_points',
          summary='跨过物体画一条短线：线中点定位、线方向决定两指闭合方向，下探后闭爪',
          inputs={'inset_mm': number('下探物体表面', 0, 50, 10, step=1, unit=' mm'),
                  'endpoint_action': choice('终点动作', [('grasp', '闭爪抓取'), ('release', '张开松手'),
                                                         ('none', '保持夹爪')], 'grasp'),
                  'grip_effort': integer('夹持力矩', 50, 5000, BLOCK_GRIP_EFFORT, unit=' ‰N·m'),
                  'wrist': choice('手腕姿态', [('auto', '自动：竖直不可达时倾斜侧抓'), ('lean', '向基座后仰'), ('top_down', '全程竖直向下')], 'auto')},
          hint='短线是抓取标注，不是移动路径；软的东西（面包、纸杯）把力矩调到 300 左右。'
               '默认自动尝试竖直与倾斜侧抓，两指保持沿画线对齐；需要固定竖直时可手动选择。',
          limits='超长画线按最大开口 70 mm。自动姿态保持两指沿画线对齐，闭合后提起 4 cm；不判断是否抓住。闭合走 0.5 秒渐进。'
                 '选了全程竖直，超过基座约 0.33 m 的抓取线可能解不出来，预览会报不可达。')
def plan_grasp_line(ctx, ink, inset_mm, endpoint_action, grip_effort, wrist='lean'):
    # The UI defaults to auto. Direct carry reuse keeps its two-point lean grasp because
    # that planner constructs its own lift and carry route after the contact.
    result = grasp_from_line(ctx.frame, ink, inset_mm, ctx.clearance_m, ctx.speed_m_s, endpoint_action,
                             preferred_yaw_deg=ctx.preferred_yaw_deg, base_xy=ctx.base_xy,
                             grip_effort=grip_effort, top_down=wrist == 'top_down',
                             reach_probe=ctx.probe if wrist=='auto' and ctx.has_kinematics and callable(getattr(ctx.backend,'pose_error',None)) else None,
                             seed_joints=ctx.current[ctx.arm]['joints_deg'],pose_tolerance=ctx.pose_tolerance,
                             **grasp_terms(ctx.arm, ctx.model, getattr(ctx.backend, 'table_checks', False)))
    if wrist=='auto' and endpoint_action=='grasp':
        from ..grasp_solver import append_grasp_lift
        append_grasp_lift(result)
    return {**result, 'summary': f'画线抓取 · 线宽 {result["width_mm"]:.0f} mm · 下探 {result["actual_inset_mm"]:.0f} mm'}


@register(name='pour', label='倒水', group='容器', stroke='two_points',
          summary='侧抓起点的杯子，倒进终点的容器，回正后放回原处',
          inputs={'tilt_deg': number('倾角', 60, 170, 130, step=5, unit='°'),
                  'hold_s': number('倾倒停留', .5, 3, 2.5, step=.5, unit=' s'),
                  'grasp_fraction': number('抓取高度比例', .2, .7, .30, step=.05),
                  'mouth_clearance_m': number('杯口离沿', .02, .20, POUR_MOUTH_CLEARANCE_M, step=.01, unit=' m'),
                  'grip_effort': integer('夹持力矩', 50, 5000, CUP_GRIP_EFFORT, unit=' ‰N·m')},
          hint='从杯子画到目标容器；纸杯力矩 300 左右。',
          limits='杯高至少 50 mm；终点必须落在容器的鸟瞰圆内。')
def plan_pour(ctx, ink, tilt_deg, hold_s, grasp_fraction, mouth_clearance_m, grip_effort):
    reach = {}
    if ctx.has_kinematics:
        reach = {'reach_probe': ctx.probe, 'pose_tolerance': ctx.pose_tolerance}
    result = pour_from_stroke(ctx.frame, ink, table_z=ctx.table_z, fingertip_bias_mm=ctx.fingertip_bias_mm,
                              preferred_yaw_deg=ctx.preferred_yaw_deg, base_xy=ctx.base_xy,
                              clearance_m=ctx.clearance_m, speed_m_s=ctx.speed_m_s, tilt_deg=tilt_deg,
                              hold_s=hold_s, grasp_fraction=grasp_fraction, mouth_clearance_m=mouth_clearance_m,
                              grip_effort=grip_effort, **reach)
    return {**result, 'summary': f'倒水 · 杯径约 {result["cup_diameter_mm"]:.0f} mm · 倾角 {result["tilt_deg"]:.0f}°'}
