"""Recover a measured empty gripper without replaying a stale grasp."""
import copy,math,uuid

class EmptyGraspError(RuntimeError):
    def __init__(self,arm,opening_mm):
        self.arm=arm;self.opening_mm=float(opening_mm)
        super().__init__(f'{arm}：夹爪空抓（反馈 {opening_mm:.1f} mm），停止搬运并自动回位')


def recover_empty_grasp(service,plan,error,progress):
    from .backend import Cancelled
    if service.cancel_event.is_set():raise Cancelled()
    arm=error.arm
    if arm not in plan.get('arms',{}):raise ValueError('Empty-grasp arm is not in the active plan')
    start=service.backend.state();opening=float(start[arm]['gripper_mm'])
    if not math.isfinite(opening) or not 0<=opening<=1.5:
        raise ValueError('夹爪反馈已改变，未确认空爪，不自动张开回位')
    with service.lock:
        service.execution.update(state='running',stage='empty_grasp_return_home',recovery_reason=str(error),progress=0.)
    def home_progress(**detail):
        if service.cancel_event.is_set():raise Cancelled()
    service.backend.prepare(service.cancel_event,home_progress,restore_can=False,restore_disabled=False)
    start=service.backend.state()
    opening=float(start[arm]['gripper_mm'])
    if not math.isfinite(opening) or not 0<=opening<=1.5:raise ValueError('回位前夹持反馈变化')
    home=service.backend.plan_home(start,[arm],pose_tolerance=copy.deepcopy(service.pose_tolerance),
        motion_profile=service.motion_profile,progress=home_progress,open_grippers=True,speed_m_s=service.home_speed_m_s)
    if service.cancel_event.is_set():raise Cancelled()
    home.update(kind='home',start=start,id=uuid.uuid4().hex)
    returned=service.backend.execute(home,start,service.cancel_event,progress)
    if service.cancel_event.is_set():raise Cancelled()
    if not returned.get('completed'):raise ValueError('空抓回位未完成')
    frame=service.backend.capture();current=service.backend.state()
    result={'completed':False,'outcome':'empty_grasp','return_home':{k:v for k,v in returned.items() if k!='trace'},'ready_for_next':True}
    with service.lock:
        service.frame=frame;service.observation_start=current;service.draft={};service.preview_data=None
        service.revision+=1
        service.execution.update(state='idle',stage='ready_after_empty_grasp',progress=0.,last_error=str(error),result=result,draft_cleared=True)
        service.execution.pop('error',None)
    return result
