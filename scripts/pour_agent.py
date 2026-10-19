"""Vision agent for URAI's observed-mouth loop. --execute explicitly enables motion.

Credentials are read from environment and never persisted. Every iteration gets a
new RGB-D observation; model output is only rim annotations, never joint commands.
"""
import argparse,json,os,time
import requests

PROMPT='''You locate visible physical rims for a robot pouring task. Return only JSON with source_rim_pixels,
target_rim_pixels, confidence, visible, and note. source is the open bottle held by the robot;
target is the receiving drinking cup. Each pixel list must contain 6 to 12 distinct [u,v]
points ON visible rim material, distributed over at least 180 degrees of its circumference.
Use the INNER edge of the receiving cup. Never mark the transparent opening center, bottle
handle, neck collar, or gripper. Coordinates refer to the supplied original image dimensions.
Do not infer occluded pixels or substitute remembered positions. If either rim cannot be
identified, return visible:false, empty lists and confidence below 0.8. You only annotate;
the calibrated RGB controller determines all motion. The measured radius must match the circular edge you annotate. Report uncertainty honestly.'''

class VisionAgent:
    def __init__(self,base_url,api_key,model):
        self.url=base_url.rstrip('/')+'/responses';self.key=api_key;self.model=model
    def locate(self,observation):
        r=requests.post(self.url,headers={'Authorization':'Bearer '+self.key},json={
          'model':self.model,'instructions':PROMPT,'reasoning':{'effort':'low'},'max_output_tokens':900,
          'input':[{'role':'user','content':[{'type':'input_text','text':f"Image {observation['width']} x {observation['height']}. Locate the two rims in this fresh image."},
           {'type':'input_image','image_url':observation['image']}]}]},timeout=80)
        r.raise_for_status();result=r.json()
        text=''.join(c.get('text','') for item in result.get('output',[]) for c in item.get('content',[]) if c.get('type')=='output_text')
        text=text.strip()
        if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
        value=json.loads(text)
        return value,{'model':result.get('model'),'usage':result.get('usage'),'response_id':result.get('id')}

class Robot:
    def __init__(self,url,token=None):
        self.url=url.rstrip('/')+'/api/'
        self.headers={'Authorization':'Bearer '+token} if token else {}
    def post(self,path,data):
        r=requests.post(self.url+path,json=data,headers=self.headers,timeout=180);r.raise_for_status();return r.json()
    def state(self):
        r=requests.get(self.url+'state',headers=self.headers,timeout=10);r.raise_for_status();return r.json()
    def execute(self,revision):
        preview=self.post('preview',{'expected_revision':revision})
        self.post('execute',{'preview_id':preview['id']})
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            e=self.state()['execution']
            if e['state'] not in ('running','planning','cancelling'):
                if e['state']!='completed':raise RuntimeError(e.get('error',e['state']))
                return e
            time.sleep(.1)
        self.post('cancel',{});raise RuntimeError('Motion timeout; cancelled')

def run(robot,agent,options,*,execute=False,record=print,max_steps=64):
    session=robot.post('pour-agent/start',options);sid=session['id'];record({'event':'start','session':session})
    moved=False
    def return_to_start():
        for _ in range(max_steps+1):
            returned=robot.post('pour-agent/return',{'session_id':sid})
            robot.execute(returned['revision'])
            record({'event':'return_step','remaining':returned.get('return_steps_remaining',0)})
            if not returned.get('return_steps_remaining',0):return
        raise RuntimeError('Return path budget exceeded')
    def finish_without_pour(state,note):
        if execute and moved:
            try:
                return_to_start()
                return {'state':state,'session_id':sid,'note':note,'returned_upright':True}
            except Exception as exc:
                record({'event':'return_error','error':str(exc)})
                return {'state':'needs_attention','session_id':sid,'note':note,'return_error':str(exc)}
        return {'state':state,'session_id':sid,'note':note}
    for i in range(max_steps):
        try:
            obs=robot.post('pour-agent/observe',{'session_id':sid})
            decision,usage=agent.locate(obs);record({'event':'decision','iteration':i,'observation_id':obs['observation_id'],'decision':decision,**usage})
            if not decision.get('visible') or float(decision.get('confidence',0))<.8:
                return finish_without_pour('needs_view',decision.get('note','Rim not visible'))
            plan=robot.post('pour-agent/plan',{'session_id':sid,'observation_token':obs['observation_token'],
                 **{k:decision[k] for k in ('source_rim_pixels','target_rim_pixels','confidence')},
                 **{k:options[k] for k in ('perception','source_radius_m','target_radius_m') if k in options}})
        except Exception as exc:
            record({'event':'perception_or_plan_error','error':str(exc)})
            return finish_without_pour('observation_failed',str(exc))
        record({'event':'planned',**plan})
        if not execute:return {'state':'preview_available','plan':plan,'motion_executed':False}
        # A motion error has a different recovery policy: do not blindly reverse a
        # trajectory whose endpoint was never reached, or release the held bottle.
        try:robot.execute(plan['revision'])
        except Exception as exc:
            return {'state':'needs_attention','session_id':sid,'motion_error':str(exc)}
        moved=True
        if plan['mode']=='pour':
            try:return_to_start()
            except Exception as exc:return {'state':'needs_attention','session_id':sid,'return_error':str(exc)}
            return {'state':'returned_upright','session_id':sid,'water_transfer_confirmed':False}
    return finish_without_pour('iteration_limit','Reached the observation/correction budget')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:7860');p.add_argument('--arm',choices=['left','right'],default='left')
    p.add_argument('--robot-token',default=os.environ.get('URAI_TOKEN'),help='URAI access token; defaults to URAI_TOKEN')
    p.add_argument('--model',default=os.environ.get('OPENAI_MODEL'));p.add_argument('--execute',action='store_true')
    p.add_argument('--tilt-deg',type=float,default=130);p.add_argument('--bottle-length-m',type=float,default=.35);p.add_argument('--bottle-radius-m',type=float,default=.06)
    p.add_argument('--perception',choices=['rgb','depth'],default='rgb');p.add_argument('--source-radius-m',type=float);p.add_argument('--target-radius-m',type=float)
    p.add_argument('--log',default='pour-agent-decisions.jsonl');args=p.parse_args()
    if args.perception=='rgb' and (args.source_radius_m is None or args.target_radius_m is None):p.error('RGB requires measured --source-radius-m and --target-radius-m; source radius can be calibrated by rgb_probe/rgb_parallax')
    key=os.environ.get('OPENAI_API_KEY');base=os.environ.get('OPENAI_BASE_URL')
    if not key or not base or not args.model:p.error('Set OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL (or --model)')
    with open(args.log,'a') as f:
        def record(x):f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();print(json.dumps(x,ensure_ascii=False),flush=True)
        result=run(Robot(args.url,args.robot_token),VisionAgent(base,key,args.model),{'arm':args.arm,'tilt_deg':args.tilt_deg,
                  'bottle_length_m':args.bottle_length_m,'bottle_radius_m':args.bottle_radius_m,'perception':args.perception,**({'source_radius_m':args.source_radius_m,'target_radius_m':args.target_radius_m} if args.perception=='rgb' else {})},execute=args.execute,record=record)
        record({'event':'finished',**result})
