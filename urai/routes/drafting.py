"""Drafting: strokes and skills become drafts. Nothing here moves the robot."""
from __future__ import annotations

import copy
import numpy as np
from fastapi import APIRouter
from .common import arm_geometry, reach_probe, require_arm, require_current_observation, require_unmoved

#: Lateral exclusion around each gripper when listing objects: a gripper resting on the table segments like one.
ARM_EXCLUSION_M = .06
#: Taller raised regions are arm parts, cables or containers, not things the toss queue picks up.
MAX_QUEUE_OBJECT_HEIGHT_MM = 80.


def build_router(service):
    router = APIRouter()

    @router.post('/api/stroke')
    def stroke(data: dict):
        from ..stroke import lift_stroke
        with service.lock:
            service._editable()
            require_current_observation(service, data)
            mode = data.get('mode', 'plane')
            # Surface-mode painting defaults to a 50 mm air gap; callers may request another bounded offset.
            clearance = data.get('surface_clearance_m', .05 if mode == 'surface' else 0.)
            return lift_stroke(service.frame, data.get('pixels'), mode, data.get('z', .15), clearance)

    @router.post('/api/grasp')
    def grasp(data: dict):
        """Grasp-line draft: the stroke's midpoint locates the grasp, its direction sets the closing axis."""
        from ..grasp import grasp_from_line, grasp_terms
        with service.lock:
            service._editable()
            require_current_observation(service, data)
            arm = require_arm(data)
            current = service.backend.state()
            model, base_xy, _, _ = arm_geometry(service, arm, current)
            probe = None
            if model is not None:
                def probe(xyz, rotation, seed):
                    return service.backend.pose_error(arm, xyz, rotation, seed, service.pose_tolerance)
            return grasp_from_line(service.frame, data.get('pixels'), data.get('inset_mm', 10),
                                   data.get('clearance_m', .12), data.get('speed_m_s', .8),
                                   data.get('endpoint_action', 'grasp'),
                                   preferred_yaw_deg=current[arm]['rpy_deg'][2], base_xy=base_xy,
                                   grip_effort=data.get('grip_effort'), top_down=bool(data.get('top_down')),
                                   reach_probe=probe, seed_joints=current[arm]['joints_deg'],
                                   pose_tolerance=(service.pose_tolerance['position_mm'],
                                                   service.pose_tolerance['orientation_deg']),
                                   **grasp_terms(arm, model, service.backend.table_checks))

    @router.post('/api/pick-place')
    def pick_place(data: dict):
        """Tabletop pick, lift and place draft from the two stroke endpoints (``pinch`` / ``top_down`` grasps)."""
        from ..transfer import pick_place_from_stroke
        with service.lock:
            service._editable()
            require_current_observation(service, data)
            arm = require_arm(data)
            current = service.backend.state()
            require_unmoved(service, current, '画抓放线')
            model, base_xy, table_z_m, bias_mm = arm_geometry(service, arm, current)
            return pick_place_from_stroke(
                service.frame, data.get('pixels'), table_z=table_z_m, fingertip_bias_mm=bias_mm,
                preferred_yaw_deg=current[arm]['rpy_deg'][2], base_xy=base_xy,
                clearance_m=data.get('clearance_m', .12), speed_m_s=data.get('speed_m_s', .8),
                pinch=bool(data.get('pinch')), top_down=bool(data.get('top_down')),
                **({} if data.get('pinch_depth_mm') is None else {'pinch_depth_mm': float(data['pinch_depth_mm'])}))

    @router.post('/api/pour')
    def pour(data: dict):
        from ..pour import CUP_GRIP_EFFORT, POUR_MOUTH_CLEARANCE_M, pour_from_stroke
        with service.lock:
            service._editable()
            require_current_observation(service, data)
            arm = require_arm(data)
            current = service.backend.state()
            require_unmoved(service, current, '画倒水线')
            model, base_xy, table_z_m, bias_mm = arm_geometry(service, arm, current)
            return pour_from_stroke(
                service.frame, data.get('pixels'), table_z=table_z_m, fingertip_bias_mm=bias_mm,
                preferred_yaw_deg=current[arm]['rpy_deg'][2], base_xy=base_xy,
                clearance_m=data.get('clearance_m', .12), speed_m_s=data.get('speed_m_s', .8),
                tilt_deg=data.get('tilt_deg', 130.), hold_s=data.get('hold_s', 2.5),
                grasp_fraction=data.get('grasp_fraction', .30),
                mouth_clearance_m=data.get('mouth_clearance_m', POUR_MOUTH_CLEARANCE_M),
                grip_effort=data.get('grip_effort', CUP_GRIP_EFFORT),
                **reach_probe(service, arm, current))

    @router.get('/api/skills')
    def skills():
        """The skill catalogue: what each skill does, what it draws and which arguments it takes."""
        from ..skills import describe_all
        from ..tasks import describe_toss
        return {'skills': [*describe_all(), describe_toss()]}

    @router.post('/api/skills/{name}')
    def run_skill(name: str, data: dict):
        """Plan one registered skill from a screen-space stroke; with commit the draft is saved for preview."""
        from ..skills import plan, skill_context
        with service.lock:
            service._editable()
            require_current_observation(service, data)
            arm = require_arm(data)
            current = service.backend.state()
            require_unmoved(service, current, '用技能')
            context = skill_context(service, arm, clearance_m=data.get('clearance_m', .12),
                                    speed_m_s=data.get('speed_m_s', .8))
            result = plan(name, context, data)
            if data.get('commit'):
                arms = result.get('arms') or {arm: result['arm']}
                saved_draft = {'observation_id': service.frame.id, 'arms': arms}
                if data.get('expected_revision') is not None:
                    saved_draft['expected_revision'] = data['expected_revision']
                result = {**result, 'revision': service.set_draft(saved_draft, merge=bool(data.get('merge')))['revision']}
            return result

    @router.post('/api/objects')
    def objects(data: dict):
        from ..tasks import arm_bases, arm_body_reason, nearest_arm
        from ..transfer import objects_in_polygon
        with service.lock:
            require_current_observation(service, data)
            polygon = data.get('polygon')
            if data.get('all'):
                # No lasso: every raised object in the picture (the arms are filtered below).
                h, w = service.frame.depth.shape
                polygon = [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]
            found = objects_in_polygon(service.frame, polygon, table_z=service.table_z())
            bases = arm_bases(service.backend)
            current = service.backend.state()
            for item in found:
                item['arm'] = nearest_arm(bases, item['center_xy'])
                body = arm_body_reason(service.backend.models, current, [*item['center_xy'], item['top_z']])
                if body:
                    item['graspable'] = False
                    item['reason'] = body
                elif item['graspable'] and item['height_mm'] > MAX_QUEUE_OBJECT_HEIGHT_MM:
                    # Camera cables and brackets hang outside the arm's capsule model and segment as tall objects.
                    item['graspable'] = False
                    item['reason'] = f"高约 {item['height_mm']:.0f} mm，超过投放物体上限 {MAX_QUEUE_OBJECT_HEIGHT_MM:.0f} mm（机械臂部件或容器）"
                for arm in ('left', 'right'):
                    if np.linalg.norm(np.asarray(current[arm]['xyz'][:2])-item['center_xy']) < ARM_EXCLUSION_M:
                        item['graspable'] = False
                        item['reason'] = f"紧贴{'左臂' if arm == 'left' else '右臂'}当前位置，先移开机械臂"
            return {'observation_id': service.frame.id, 'objects': found}

    @router.get('/api/drop-point')
    def drop_point():
        with service.lock:
            return {'xyz': service.drop_point}

    @router.put('/api/drop-point')
    def set_drop_point(data: dict):
        return service.set_drop_point(data.get('xyz'))

    @router.get('/api/draft')
    def draft():
        with service.lock:
            return copy.deepcopy(service.draft)

    @router.put('/api/draft')
    def set_draft(data: dict):
        # merge says how to save; it is not part of the draft and must not end up stored in it.
        merge = bool(data.pop('merge', False))
        return service.set_draft(data, merge=merge)

    @router.delete('/api/draft')
    def clear_draft():
        return service.clear_draft()

    @router.post('/api/compile')
    def compile_draft():
        from ..service import public
        from ..trajectory import compile_arms
        with service.lock:
            service._editable()
            if not service.draft:
                raise ValueError('Create a trajectory draft first')
            current = service.backend.state()
            return {'arms': public(compile_arms(service.draft['arms'], current))}

    return router
