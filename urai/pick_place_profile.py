"""Compact, calibrated two-point transfers for measured small rigid objects."""
import copy
import numpy as np
from .far_transfer import refine_transfer


def optimize_pick_place(result, parameters):
    # Leave cloth, explicit vertical grasps and larger/uncertain objects on their existing profile.
    options=parameters.get('grasp_options')
    if options is None or result.get('pinch') or result.get('top_down'):
        return result
    width=float(result['object_width_mm']);height=float(result['object_height_mm'])
    if not np.isfinite([width,height]).all() or not (4<=width<=60 and 3<=height<=40):
        return result
    r=copy.deepcopy(result)
    source=np.array(r.get('source_contact_xyz',r['source_xyz']),dtype=float)
    dest=np.array(r.get('target_contact_xyz',r['target_xyz']),dtype=float)
    # Depth backgrounds can overestimate low blocks; retain 3 mm above calibrated table.
    extra=min(6.,max(0.,float(r['finger_z_mm'])-3.))
    source[2]-=extra/1000
    points=np.asarray(r['arm']['path']['points']);hover=float(max(points[:,2]))
    placement=dict(grasp=source,place=dest,hover=np.r_[source[:2],hover],arrival=np.r_[dest[:2],hover],placement='place')
    yaw=float(r['arm']['orientation']['points'][0][-1])
    r=refine_transfer(r,placement,[],yaw,parameters['base_xy'],parameters.get('clearance_m',.12),options)
    opening=float(np.clip(width+14,35,70))
    for event in r['arm']['gripper_events']:
        if event.get('opening_mm',0)>0:
            event['opening_mm']=opening;event['hold_s']=.4
        else:event['hold_s']=.8
    r.update(pick_place_profile='compact_calibrated',opening_mm=opening,extra_inset_mm=extra,
             finger_z_mm=float(r['finger_z_mm'])-extra)
    return r
