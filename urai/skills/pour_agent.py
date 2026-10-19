"""Grasp/lift stage for visually re-localized pouring; never tips from a guessed mouth."""
import numpy as np
from .registry import register,number,integer
from .motion import draft,open_event,close_event
from .bottle_cap import _compensation
from ..pour import approach_candidates,side_grasp_rotation,SIDE_GRASP_PITCH_DEG,rpy_deg

@register(name='pour_agent_prepare',label='视觉倒水① · 抓起并保持',group='容器',stroke='point',
 summary='抓起瓶身；随后由 agent 重新定位瓶口和杯口，闭环对准后倾倒。',
 hint='点可抓瓶身。抓起后调用 /api/pour-agent/start，不从瓶身分割推算瓶口。',
 inputs={'grasp_inset_mm':number('从可见瓶身向内夹入',0,25,15,1,' mm'),
 'lift_mm':number('抬起高度',20,100,60,5,' mm'),
 'grip_effort':integer('夹持力矩',50,2000,1000),
 'approach_speed_m_s':number('接近速度',.001,1.,.8,.005,' m/s'),
 'speed_m_s':number('速度',.001,1.,.8,.005,' m/s')})
def prepare(ctx,ink,grasp_inset_mm,lift_mm,grip_effort,speed_m_s,approach_speed_m_s):
    surface=np.asarray(ctx.frame.pick(*ink[0],mode='surface'),float)
    if surface[2]<ctx.table_z+.05:raise ValueError('请点瓶身不透明标签上的可见夹持位置')
    last_error=None
    for _,direction in approach_candidates(surface[:2],ctx.base_xy):
        for pitch in SIDE_GRASP_PITCH_DEG:
            R=side_grasp_rotation(direction,pitch)
            grasp=surface+R@np.array([0.,_compensation(ctx),grasp_inset_mm/1000])
            points=[grasp-R[:,2]*.07,grasp,grasp+[0,0,lift_mm/1000]]
            seed=ctx._seed;ok=True
            for point in points:
                if ctx.has_kinematics:
                    pe,re,seed=ctx.backend.pose_error(ctx.arm,point,R,seed,{'position_mm':3.,'orientation_deg':3.})
                    if pe>3 or re>3:ok=False;last_error=(pe,re);break
            if ok:break
        if ok:break
    else:raise ValueError(f'瓶身接近/抓起姿态不可达：{last_error}')
    spec=draft(points,[[0,*rpy_deg(R)],[1,*rpy_deg(R)]],[open_event(0),close_event(.5,effort=grip_effort)],speed_m_s,ctx.clearance_m,approach_speed_m_s=approach_speed_m_s,angular_speed_deg_s=10.)
    spec['pour_agent']={'phase':'prepare','arm':ctx.arm}
    return {'arm':spec,'summary':'抓起后保持；等待 agent 对瓶口与杯口进行视觉对准。',
            'next_tool':'POST /api/pour-agent/start','grasp_xyz':grasp.tolist(),'observed_grasp_surface_xyz':surface.tolist(),
            'mouth_geometry_measured':False}
