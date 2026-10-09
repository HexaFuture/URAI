/** Pure helpers for toss task requests and the task queue panel. */
export const STATUS_LABELS={queued:'排队',observing:'观测',planning:'规划',running:'执行',completed:'完成',failed:'失败',skipped:'跳过',cancelled:'已取消'};
export const TERMINAL_STATES=new Set(['completed','failed','skipped','cancelled']);
const TONES={queued:'pending',observing:'active',planning:'active',running:'active',completed:'done',failed:'failed',skipped:'muted',cancelled:'muted'};

/** Request body for POST /api/tasks from the selected graspable objects. */
export function buildTaskRequest({observationId,objects,selected,landing=null,placement=null,arm=null,tossStyle=null}){
 const items=[];
 for(const index of [...selected].sort((a,b)=>a-b)){
  const object=objects[index];
  if(!object?.graspable)continue;
  const item={object_pixel:object.pixel.slice(),label:`物体 ${object.index+1}`};
  if(landing)item.landing_xyz=landing.slice();
  if(placement)item.placement=placement;
  if(arm)item.arm=arm;
  items.push(item);
 }
 if(!items.length)throw Error('没有可夹取的选中物体');
 return {observation_id:observationId,items,...(tossStyle?{toss_style:tossStyle}:{})};
}

export function describeItem(item){
 const source=item.landing_source==='drop_point'?'投放点':'落点';
 return {number:item.index+1,arm:item.arm==='left'?'左臂':'右臂',label:item.label||'',
  status:STATUS_LABELS[item.status]||item.status,detail:item.error||source,tone:TONES[item.status]||'pending'};
}

export function queueFinished(previous,current){
 return Boolean(previous?.active)&&Boolean(current)&&!current.active;
}

export function summarizeQueue(view){
 const items=view?.items||[];
 if(!items.length)return '';
 const count=status=>items.filter(item=>item.status===status).length;
 if(view.active){
  const current=items.find(item=>!TERMINAL_STATES.has(item.status));
  const step=current?` · 第 ${items.indexOf(current)+1} 项${STATUS_LABELS[current.status]||current.status}`:'';
  return `队列执行中 ${count('completed')}/${items.length}${step}`;
 }
 const parts=[['completed','完成'],['failed','失败'],['cancelled','已取消'],['skipped','跳过']].filter(([status])=>count(status)>0).map(([status,label])=>`${label} ${count(status)}`);
 return `队列结束：${parts.join(' · ')}`;
}
