"""The registry every atomic manipulation skill is declared in: one description the GUI and an agent both read.

A skill turns one screen-space intent (a stroke, a lasso or a point) plus a few numbers into the same draft the
brush and the pour skill already produce: waypoints, orientation keyframes, gripper events and an approach. It
never touches the robot; execution stays with the existing draft -> preview -> execute chain.
"""
import numpy as np

STROKE_KINDS = {
    'two_points': '从起点画到终点，只用两端',
    'stroke': '整条笔迹都参与',
    'lasso': '圈住一片区域',
    'point': '在一个点上落笔',
}
MAX_STROKE_PIXELS = 4096
SKILLS = {}


def number(label, minimum, maximum, default, step=None, unit='', hint=''):
    """A numeric input the GUI renders as a spin box and an agent reads as a range."""
    return {'type': 'number', 'label': label, 'min': float(minimum), 'max': float(maximum),
            'default': float(default), 'step': None if step is None else float(step), 'unit': unit, 'hint': hint}


def integer(label, minimum, maximum, default, unit='', hint=''):
    return {'type': 'integer', 'label': label, 'min': int(minimum), 'max': int(maximum),
            'default': int(default), 'step': 1, 'unit': unit, 'hint': hint}


def choice(label, options, default, hint=''):
    """``options`` is a list of ``(value, label)`` pairs."""
    return {'type': 'choice', 'label': label, 'options': [{'value': v, 'label': t} for v, t in options],
            'default': default, 'hint': hint}


def boolean(label, default, hint=''):
    return {'type': 'boolean', 'label': label, 'default': bool(default), 'hint': hint}


class Skill:
    """One atomic action: its intent stroke, its numbers and the planner that turns them into a draft."""

    def __init__(self, name, label, group, summary, plan, stroke='two_points', arms='single', inputs=None,
                 hint='', limits='', references=()):
        if stroke not in STROKE_KINDS:
            raise ValueError(f'{name}: stroke 需为 {sorted(STROKE_KINDS)} 之一')
        if arms not in ('single', 'dual'):
            raise ValueError(f'{name}: arms 需为 single 或 dual')
        self.name, self.label, self.group, self.summary, self.plan = name, label, group, summary, plan
        self.stroke, self.arms, self.inputs = stroke, arms, dict(inputs or {})
        self.hint, self.limits, self.references = hint, limits, tuple(references)
        for key, spec in self.inputs.items():
            if spec.get('type') not in ('number', 'integer', 'choice', 'boolean'):
                raise ValueError(f'{name}.{key}: 输入类型需为 number、integer、choice 或 boolean')

    def describe(self):
        """The JSON an agent or the GUI reads: what the skill does, what it draws and what it accepts."""
        return {'name': self.name, 'label': self.label, 'group': self.group, 'summary': self.summary,
                'stroke': self.stroke, 'stroke_hint': STROKE_KINDS[self.stroke], 'arms': self.arms,
                'inputs': {
                    'speed_m_s': number('移动速度', .001, 1., .8, step=.005, unit=' m/s', hint='空载与搬运段的目标速度；接触段使用下方专用速度。'),
                    'approach_speed_m_s': number('接近速度', .001, 1., .8, step=.005, unit=' m/s'),
                    **{key: dict(spec) for key, spec in self.inputs.items()}},
                'hint': self.hint, 'limits': self.limits, 'references': list(self.references)}

    def defaults(self):
        return {key: spec['default'] for key, spec in self.inputs.items()}

    def parse(self, data):
        """Validate the caller's numbers against this skill's declared inputs; unknown or out-of-range fails."""
        values = {}
        for key, spec in self.inputs.items():
            if key not in data or data[key] is None:
                values[key] = spec['default']
                continue
            values[key] = self._parse_one(key, spec, data[key])
        return values

    @staticmethod
    def _parse_one(key, spec, raw):
        label = spec['label']
        if spec['type'] == 'boolean':
            if not isinstance(raw, bool):
                raise ValueError(f'{label}（{key}）需为 true 或 false')
            return raw
        if spec['type'] == 'choice':
            allowed = [option['value'] for option in spec['options']]
            if raw not in allowed:
                raise ValueError(f'{label}（{key}）只接受 {"、".join(map(str, allowed))}')
            return raw
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f'{label}（{key}）需为数字') from None
        if not np.isfinite(value) or not spec['min'] <= value <= spec['max']:
            raise ValueError(f'{label}（{key}）需为 {spec["min"]:g}–{spec["max"]:g}{spec["unit"]}')
        if spec['type'] == 'integer':
            if value != int(value):
                raise ValueError(f'{label}（{key}）需为整数')
            return int(value)
        return value


def register(**declaration):
    """Decorator that puts one planner in the registry: ``@register(name='push', label='推', ...)``."""
    def decorate(plan):
        skill = Skill(plan=plan, **declaration)
        if skill.name in SKILLS:
            raise ValueError(f'技能 {skill.name} 重复注册')
        SKILLS[skill.name] = skill
        return plan
    return decorate


def get(name):
    if name not in SKILLS:
        raise ValueError(f'没有名为 {name} 的技能，可用：{"、".join(sorted(SKILLS))}')
    return SKILLS[name]


def describe_all():
    """Every registered skill, grouped in declaration order; this is the agent's tool list."""
    return [skill.describe() for skill in SKILLS.values()]


def stroke_pixels(frame, pixels, kind):
    """Validate the intent stroke for ``kind`` and return it as an (n, 2) float array of pixels inside the frame."""
    ink = np.asarray(pixels if pixels is not None else [], dtype=float)
    if ink.ndim != 2 or ink.shape[1] != 2 or not len(ink) or not np.isfinite(ink).all():
        raise ValueError('请在画面上落笔：需要有效的像素点')
    if len(ink) > MAX_STROKE_PIXELS:
        raise ValueError(f'笔迹最多 {MAX_STROKE_PIXELS} 个点')
    height, width = frame.depth.shape
    if np.any(ink < 0) or np.any(ink[:, 0] >= width) or np.any(ink[:, 1] >= height):
        raise ValueError('笔迹超出相机画面')
    if kind == 'point':
        return ink[:1]
    if kind == 'lasso':
        if len(ink) < 3:
            raise ValueError('请圈住一片区域：至少 3 个点')
        return ink
    if len(ink) < 2 or np.linalg.norm(ink[-1] - ink[0]) < 12:
        raise ValueError('请从起点画到明显不同的终点')
    return ink[[0, -1]] if kind == 'two_points' else ink
