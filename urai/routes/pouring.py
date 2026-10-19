"""Agent-assisted pouring endpoints using the normal motion pipeline."""
from fastapi import APIRouter

def build_router(service):
    router = APIRouter()
    @router.get('/api/pour-agent')
    def pour_agent_tools():
        return {'state':service.pour_agent.status(), 'workflow':[
            'Optional: skills/pour_agent_prepare -> preview -> execute (grasp/lift and hold)',
            'start -> observe -> agent labels source rim and target inner rim -> plan -> preview -> execute',
            'Re-observe after each correction; pour action tips briefly then returns without waiting for the model',
            'After poured: repeat return -> preview -> execute until return_steps_remaining=0; reverse each original correction, keeping grip'],
            'tools':['POST /api/pour-agent/start','POST /api/pour-agent/observe','POST /api/pour-agent/plan','POST /api/pour-agent/return'],
            'start_schema':{'arm':'left|right','tilt_deg':'60..150 (default 130)','inspection_tilt_deg':'0..30 (default 20)',
                            'bottle_length_m':'actual full length or conservative envelope (default .35)',
                            'bottle_radius_m':'body radius, separate from observed mouth (default .06)'},
            'plan_schema':{'session_id':'start response id','observation_token':'observe response token',
                           'source_rim_pixels':'4..16 [u,v] on visible bottle rim, spanning >=120 degrees',
                           'target_rim_pixels':'4..16 [u,v] on receiving cup inner rim','confidence':'0..1, >=.8 required',
                           'perception':'depth (legacy) | rgb | rgb_probe | rgb_parallax',
                           'rgb':'6..16 visible contour points spanning >=180 deg; source_radius_m and target_radius_m measured on the labeled edge; target upright',
                           'rgb_probe':'source_rim_pixels plus translation_m (10..15 mm); preview/execute then observe again',
                           'rgb_parallax':'reference_token from probe, same ordered source landmarks in new RGB, target_radius_m; no depth required'},
            'limits':'No robot action in start/observe/plan. Standard preview+execute required. RGB and depth plus decisions are saved.'}

    @router.post('/api/pour-agent/start')
    def pour_agent_start(data:dict):
        with service.lock:return service.pour_agent.start(data)

    @router.post('/api/pour-agent/observe')
    def pour_agent_observe(data:dict):
        with service.lock:return service.pour_agent.observe(data)

    @router.post('/api/pour-agent/plan')
    def pour_agent_plan(data:dict):
        with service.lock:return service.pour_agent.plan(data)

    @router.post('/api/pour-agent/return')
    def pour_agent_return(data:dict):
        with service.lock:return service.pour_agent.return_upright(data)

    return router
