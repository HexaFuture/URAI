/** The end of a drawn path closes the gripper (grasp), opens it (release) or leaves it as it is (none). A grasp
 * line that ends in a grasp also opens the gripper before it descends; any other ending drops that pre-opening. */
export function withEndpointAction(events,action,{graspLine=false}={}){
 if(!['grasp','release','none'].includes(action))throw Error('未知终点动作');
 const next=events.filter(e=>Number(e.s)!==1&&!e.pregrasp_open).map(e=>({...e}));
 if(graspLine&&action==='grasp'&&!next.some(e=>Number(e.s)===0))next.push({s:0,opening_mm:70,pregrasp_open:true});
 if(action!=='none')next.push({s:1,opening_mm:action==='grasp'?0:70,hold_s:.6,wait_for_arrival:true});
 return next.sort((a,b)=>Number(a.s)-Number(b.s));
}
export function endpointLabel(events){
 const end=events.filter(e=>Number(e.s)===1).at(-1);
 return !end?'保持夹爪状态':Number(end.opening_mm)===0?'抓取 · 闭合':Number(end.opening_mm)===70?'松手 · 张开':`开度 ${end.opening_mm} mm`;
}
export class EndpointActionDialog{
 constructor(el){
  this.el=el;this.pending=null;this.title=el.querySelector('#endpoint-choice-title');
  for(const button of el.querySelectorAll('[data-end-action]'))button.onclick=()=>this.finish(button.dataset.endAction==='cancel'?null:button.dataset.endAction);
  el.addEventListener('cancel',e=>{e.preventDefault();this.finish(null);});
  el.addEventListener('close',()=>{if(!el.open)this.finish(null);});
 }
 /** A tapped release point (pointOnly) offers release or hold: there is nothing between the fingers to grasp. */
 choose({arm,revision,pointOnly=false}){
  if(this.pending)throw Error('请先完成当前终点动作选择');
  for(const button of this.el.querySelectorAll('[data-end-action]'))button.disabled=pointOnly&&button.dataset.endAction==='grasp';
  this.title.textContent=`${arm==='left'?'左臂':'右臂'} · ${pointOnly?'放置点：到位后松手或保持夹爪':'到终点后做什么？'}`;
  return new Promise(resolve=>{
   this.pending={revision,resolve};this.el.showModal();this.title.focus();
  });
 }
 finish(action,reason=''){
  const pending=this.pending;if(!pending)return;this.pending=null;
  if(this.el.open)this.el.close();pending.resolve({action,reason});
 }
 cancelIfChanged(state){
  if(!this.pending||state.revision<this.pending.revision)return;
  if(['running','planning','cancelling'].includes(state.execution?.state))this.finish(null,'已有操作开始，已取消本次选择。');
  else if(state.revision!==this.pending.revision)this.finish(null,'其他窗口更新了草稿或设置，已取消本次选择。');
 }
}
