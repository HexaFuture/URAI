"""Atomic manipulation skills: one registry the GUI renders and an agent calls.

Every skill module registers itself on import, so adding a skill is a module plus one line here.
"""
from .registry import SKILLS, Skill, boolean, choice, describe_all, get, integer, number, register, stroke_pixels
from .context import SkillContext, skill_context
from . import existing    # noqa: F401  pick-place, grasp line and pour
from . import carry       # noqa: F401  one-stroke carry: grasp line, drawn route and drop point in one draft
from . import bottle_cap  # noqa: F401  staged bimanual bottle opening: hold, twist, retract and place
from . import bottle_upright  # noqa: F401  upright holding, cap turning, retreat and assisted release

__all__ = ['SKILLS', 'Skill', 'SkillContext', 'boolean', 'choice', 'describe_all', 'get', 'integer', 'number',
           'plan', 'register', 'skill_context', 'stroke_pixels']


def plan(name, ctx, data):
    """Run one registered skill: validate its stroke and numbers, then hand them to its planner.

    Returns the skill's report with the draft under arm (single-armed) or arms (bimanual).
    """
    skill = get(name)
    if skill.arms == 'single' and ctx.arm not in ('left', 'right'):
        raise ValueError('请选择左臂或右臂')
    speed_specs = skill.describe()['inputs']
    if 'speed_m_s' in data:
        ctx.speed_m_s = skill._parse_one('speed_m_s', speed_specs['speed_m_s'], data['speed_m_s'])
    approach_speed = speed_specs['approach_speed_m_s']['default']
    if 'approach_speed_m_s' in data:
        approach_speed = skill._parse_one('approach_speed_m_s', speed_specs['approach_speed_m_s'], data['approach_speed_m_s'])
    ink = stroke_pixels(ctx.frame, data.get('pixels'), skill.stroke)
    result = skill.plan(ctx, ink, **skill.parse(data))
    if not isinstance(result, dict) or 'summary' not in result:
        raise ValueError(f'技能 {name} 没有返回 summary')
    if skill.arms == 'single' and 'arm' not in result:
        raise ValueError(f'技能 {name} 没有返回 arm 草稿')
    if skill.arms == 'dual' and set(result.get('arms', {})) - {'left', 'right'}:
        raise ValueError(f'技能 {name} 的 arms 草稿只能是 left 和 right')
    if skill.arms == 'dual' and not result.get('arms'):
        raise ValueError(f'技能 {name} 没有返回 arms 草稿')
    if approach_speed is not None:
        drafts = result.get('arms') or {ctx.arm: result['arm']}
        for draft in drafts.values():
            draft.setdefault('approach', {})['speed_m_s'] = approach_speed
    return {'observation_id': ctx.frame.id, 'skill': name, 'skill_label': skill.label, **result}
