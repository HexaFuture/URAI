import {apiClient} from './api.mjs';
import {runStrokeTask,runDraftExecution,executePreparedPreview} from './automatic.mjs';
import {GripperPoseEditor,rpyMatrix,matrixRpy,sampleOrientation,transform,gripperSegments} from './orientation.mjs';
import {EndpointActionDialog,withEndpointAction,endpointLabel} from './endpoint.mjs';
import {buildTaskRequest,describeItem,queueFinished,summarizeQueue} from './queue.mjs';
import {LivePainter,liveStatusLabel} from './live.mjs';
const $ = id => document.getElementById(id);
const colors = {left:'#efb368',right:'#87cbdc'};
let state=null, frame=null, photo=null, active='left', selected=0, mode='gui', compiled=null, preview=null;
let arms={}, functionArms={}, editVersion=0, syncChain=Promise.resolve(), drag=null, speedDrag=null, busy=false;
let mainRect=null, sideMap=null, speedMap=null, lastRemoteRevision=-1, pickingStart=false;
let toleranceDirty=false;
let drawTool='pick-place', brush=null, strokeUndo={};
let automaticRun=null, automaticAwaitingFinish=false;
let clearingPaths=false;
let pendingStrokes=[], drainingStroke=false, queuedDraftWaiting=false;
// Two-armed orchestration: with the dual-arm box ticked, the arm drawn first is held in the draft until the
// other one is drawn too, and only then do both run - the two hands start their task segments at one instant.
let dualArms=new Set();
let poseEditor=null,poseUndo=null,endpointDialog=null;
let livePainter=null,finishNotice=null;

const api=apiClient();
function message(text,error=false){$('message').textContent=text;$('message').classList.toggle('error',error);}
function running(){return ['running','cancelling','planning'].includes(state?.execution?.state)||Boolean(state?.task_queue?.active);}
function renderPlanning(){
 const e=state?.execution;if(!['preview','home'].includes(e?.kind)||!['planning','cancelling'].includes(e.state))return;
 const name={left:'左臂',right:'右臂'},routes={direct:'直接接近',lift:'抬升接近',joint:'关节空间接近'};
 const stages={prepare:'检查控制模式与观测',settle:'等待双臂停止拖动并稳定 1 秒',restore:'恢复 CAN 控制并保持当前位置',refresh:'自动采集最新 RGB-D，保留世界坐标路径',compile:'生成三维采样轨迹',approach:'规划当前位置到起点的接近段',ik:'逆解与位姿核验',retiming:'关节速度、加速度与运动时长',collision:'桌面、自碰撞与双臂净空'};
 if(e.kind==='home')stages.approach='关节空间回零规划';
 const seconds=e.started_at?Math.max(e.elapsed_s||0,(Date.now()/1000-e.started_at)):e.elapsed_s||0;
 const candidate=e.candidate?`方案 ${e.candidate}/${e.candidates}${e.routes?'（'+Object.entries(e.routes).map(([a,r])=>name[a]+'：'+routes[r]).join('，')+'）':''} · `:'';
 const step=`${e.arm?name[e.arm]+' ':''}${stages[e.stage]||'轨迹检查'}${e.total>1?' '+e.done+'/'+e.total:''}`;
 message(e.state==='cancelling'?`正在取消检查 · 已用 ${seconds.toFixed(1)} 秒`:`${candidate}${step} · 已用 ${seconds.toFixed(1)} 秒`);
}
function renderTolerance(){
 const automatic=state?.auto_prepare||{refresh_observation:true,restore_can:true};
 $('auto-refresh').checked=automatic.refresh_observation;$('auto-can').checked=automatic.restore_can;
 $('table-checks').checked=Boolean(state?.table_checks);
 if(document.activeElement!==$('tracking-limit'))$('tracking-limit').value=state.tracking_limit_deg;
 const t=state?.pose_tolerance||{position_mm:10,orientation_deg:3};
 if(!toleranceDirty){$('tolerance-position').value=t.position_mm;$('tolerance-orientation').value=t.orientation_deg;}
 if(document.activeElement!==$('queue-home-speed'))$('queue-home-speed').value=state?.home_speed_m_s??.8;
 $('motion-profile').value=state.motion_profile;
 const limits=state?.backend?.motion_limits;
 $('motion-profile-status').textContent=limits?`${limits.label} · 关节 ≤ ${limits.joint_speed_deg_s}°/s · 执行档位 ${limits.controller_speed_percent}%`:'';
 $('tolerance-status').textContent=toleranceDirty?'尚未应用；点击应用，或重新预览时保存。':`已应用：位置 ${t.position_mm} mm · 姿态 ${t.orientation_deg}°`;
}
/** The profile list comes from the service, so the page offers exactly the profiles the planner accepts. */
async function loadMotionProfiles(){
 const settings=await api('settings');
 $('motion-profile').replaceChildren(...settings.motion_profiles.map(profile=>{const option=document.createElement('option');option.value=profile.profile;option.textContent=profile.label;return option;}));
 $('motion-profile').value=settings.motion_profile;
}
async function saveTolerance(){
 if(!$('tolerance-position').value.trim()||!$('tolerance-orientation').value.trim())throw Error('请填写位置和姿态容差。');
 const value={position_mm:Number($('tolerance-position').value),orientation_deg:Number($('tolerance-orientation').value)};
 const r=await api('settings','PUT',{pose_tolerance:value});
 state.pose_tolerance=r.pose_tolerance;lastRemoteRevision=r.revision;toleranceDirty=false;preview=null;renderTolerance();
}
function enabled(){return ['left','right'].filter(a=>$('enable-'+a).checked);}
function draftIssue(d){
 const chosen=Object.entries(d?.arms||{});
 if(!chosen.length)return '请选择至少一只执行臂。';
 const missing=chosen.filter(([a,spec])=>spec?.path?.mode==='waypoints'&&(spec.path.points?.length||0)<2).map(([a,spec])=>`${a==='left'?'左臂':'右臂'}（${spec.path.points?.length||0} 个点）`);
 return missing.length?`${missing.join('、')}还没有完整路径：请按住鼠标画一条线，或取消对应的执行勾选。`:null;
}
function freshArm(a){return {points:[state.arms[a].xyz.slice()],poses:[state.arms[a].rpy_deg.slice()],orientation:false,orientationMode:'free',speed:[[0,.8],[1,.8]],events:[],approach:{speed_m_s:.8,clearance_m:.12}};}
function defaultFunction(a){
 return {path:{mode:'function',x:'x0 + 0.02*s',y:'y0',z:'z0 + 0.01*sin(pi*s)'},speed:'0.8',orientation:{mode:'free'},gripper_events:[],approach:{speed_m_s:.8,clearance_m:.12}};
}
function setMode(next){mode=next;pickingStart=false;$('gui-mode').setAttribute('aria-pressed',next==='gui');$('api-mode').setAttribute('aria-pressed',next==='api');$('function-panel').hidden=next!=='api';$('brush-toolbar').hidden=next!=='gui';$('grasp-options').hidden=next!=='gui'||drawTool!=='grasp';$('pour-options').hidden=next!=='gui'||drawTool!=='pour';$('select-options').hidden=next!=='gui'||drawTool!=='select';invalidate();renderFields();drawAll();}
function invalidate(){preview=null;editVersion++;$('execute').disabled=true;$('duration').textContent='草稿已修改';$('plan-notes').replaceChildren();}
function draft(){
 const result={observation_id:frame.id,arms:{}};
 for(const a of enabled()){
  if(mode==='api') result.arms[a]=structuredClone(functionArms[a]);
  else{
   const m=arms[a],n=m.points.length;
   result.arms[a]={path:{mode:'waypoints',points:m.points.map(p=>p.slice())},speed:m.speedSpec??m.speed.map(k=>k.slice()),
    orientation:m.orientation?{mode:'keyframes',points:m.poses.map((r,i)=>[i/Math.max(n-1,1),...r])}:{mode:m.orientationMode||'free'},
    gripper_events:m.events.map(e=>({...e})),start_hold_s:m.startHold||0,approach:{...m.approach},...(m.graspLine?{grasp_line:true}:{}),...(m.airTrack?{air_track:true}:{}),...(m.bottleCap?{bottle_cap:m.bottleCap,angular_speed_deg_s:m.angularSpeed}:{})};
  }
 }
 return result;
}
function queueSync(){
 invalidate();const version=editVersion;
 syncChain=syncChain.catch(()=>{}).then(async()=>{
  if(version!==editVersion||!frame||running()) return;
  const d=draft();
  try{
   const r=await api('draft','PUT',d);lastRemoteRevision=r.revision;
   const issue=draftIssue(d);if(issue){compiled=null;drawAll();updateControls();message(issue);return;}
   const c=await api('compile','POST');
   if(version===editVersion){compiled=c;drawAll();message('草稿已更新。预览与检查通过后可执行。');}
  }catch(e){if(version===editVersion) message(e.message,true);}
  finally{updateControls();}
 });
 return syncChain;
}
async function refresh(){
 if(busy||running())return;
 const editing=frame?{active,selected,mode,text:$('function-editor').value}:null;
 busy=true;updateControls();message('正在采集 RGB-D 并重建侧视图…');
 try{
  await syncChain;
  clearDualOrchestration();
  const f=await api('observe','POST',{preserve_draft:true});await loadObservation(f);
  state=await api('state');lastRemoteRevision=state.revision;
  for(const a of ['left','right']){arms[a]??=freshArm(a);functionArms[a]??=defaultFunction(a);}
  compiled=null;strokeUndo={};invalidate();
  const d=await api('draft'),kept=Boolean(d.arms&&Object.keys(d.arms).length);
  if(kept)await adoptRemoteDraft(d,editVersion,true);
  if(editing){active=editing.active;selected=editing.selected;$('arm').value=active;setMode(editing.mode);$('function-editor').value=editing.text;}
  else $('function-editor').value=JSON.stringify(functionArms[active],null,2);
  $('plane-z').value=state.arms[active].xyz[2].toFixed(3);renderFields();drawAll();
  message(drawTool==='pick-place'?'图像已更新。从物体画到空闲桌面：第一点抓，抬起，第二点放。':drawTool==='pour'?'图像已更新。从杯子画到目标容器，松开后自动侧抓、倾倒、放回。':kept?'图像与深度已更新，路径草稿已保留。重新预览会从当前位置规划到路径起点。':'观测已更新。按住鼠标画线，松开选择终点动作；系统会自动接近笔画起点。');
 }catch(e){message(e.message,true);$('empty-view').textContent=e.message;}
 finally{busy=false;updateControls();}
}
async function observationImage(f){const p=new Image();await new Promise((res,rej)=>{p.onload=res;p.onerror=rej;p.src=f.image;});return p;}
function installObservation(f,p){frame=f;photo=p;$('empty-view').hidden=true;$('depth-quality').textContent=`有效深度 ${(100*f.valid_depth_fraction).toFixed(1)}%`;$('observation-time').textContent='深度观测 '+new Date(f.timestamp*1000).toLocaleTimeString();}
async function loadObservation(f){installObservation(f,await observationImage(f));}
async function adoptPreparedPlan(plan){
 if(plan.observation&&frame?.id!==plan.observation_id){
  const nextPhoto=await observationImage(plan.observation);
  if(state?.revision>plan.revision)throw Error('准备期间草稿已变化，请重新预览');
  installObservation(plan.observation,nextPhoto);
 }
 lastRemoteRevision=plan.revision;
}
function updateControls(){
 const locked=clearingPaths||busy||running(),issue=frame&&arms[active]?draftIssue(draft()):null;
 $('draft-readiness').textContent=issue&&mode==='gui'&&['pick-place','pour','grasp','select'].includes(drawTool)?'':issue||'';
 $('observe').disabled=locked;$('preview').disabled=locked||!frame||Boolean(issue);
 $('execute').disabled=locked||toleranceDirty||!preview||!state?.backend.motion_allowed||(state?.observation_stale&&!state?.auto_prepare?.refresh_observation);
 $('execute').textContent=state?.observation_stale&&state?.auto_prepare?.refresh_observation?'更新观测并执行':'执行已预览动作';
 $('cancel').disabled=!pendingDrop&&!pendingStrokes.length&&!brush&&!automaticRun&&!state?.task_queue?.active&&!['running','cancelling','planning'].includes(state?.execution?.state);
 $('cancel').textContent=state?.execution?.kind==='preview'&&['planning','cancelling'].includes(state.execution.state)?'取消检查':'停止轨迹并保持';
 for(const id of ['arm','enable-left','enable-right','gui-mode','api-mode','apply-function','reset-path','remove-point','add-event','orientation-enabled','base-speed','approach-speed','approach-clearance','pick-start'])$(id).disabled=locked;
 $('pick-start').disabled=locked||mode!=='gui';
 $('orientation-enabled').disabled=locked||mode!=='gui';
 $('choose-endpoint').disabled=locked||mode!=='gui'||!arms[active]||arms[active].points.length<2;
 $('base-speed').disabled=locked||mode!=='gui';
 for(const id of ['queue-grab-speed','queue-toss-approach-speed','queue-toss-home-speed','queue-approach-speed','queue-throw-speed','queue-carry-speed','queue-windup-speed','queue-home-speed','queue-grasp-hold'])$(id).disabled=locked;
 $('motion-profile').disabled=locked;
 document.querySelectorAll('#skill-speed-fields input,#skill-speed-fields select').forEach(field=>field.disabled=locked);
 $('auto-refresh').disabled=locked;$('auto-can').disabled=locked;$('table-checks').disabled=locked;$('tracking-limit').disabled=locked;
 $('tolerance-position').disabled=locked;$('tolerance-orientation').disabled=locked;
 $('apply-tolerance').disabled=locked||!toleranceDirty;
 for(const id of ['pick-place-tool','pour-tool','brush-tool','grasp-tool','skill-tool','select-tool','edit-tool','skill-name','grasp-inset','grasp-effort','pour-tilt','pour-hold','pour-grasp-fraction','pour-clearance','pour-effort','pick-place-topdown','pick-place-pinch','pick-place-pinch-depth'])$(id).disabled=locked||mode!=='gui';
 $('auto-execute').disabled=locked;$('dual-arm').disabled=locked;
 $('undo-stroke').disabled=locked||mode!=='gui'||!strokeUndo[active];
 for(const id of ['pick-mode','plane-z','side-axis','live','event-s','event-opening','function-editor'])$(id).disabled=locked;
 for(const k of ['x','y','z'])$('point-'+k).disabled=locked||mode!=='gui';
 $('plane-z').disabled=locked||$('pick-mode').value==='surface';
 $('live').disabled=locked;
 $('home').disabled=locked||!state?.backend?.motion_allowed;
 const recoverable=['error','cancelled'].includes(state?.execution?.state)&&!locked;
 $('recover').disabled=!recoverable;
 $('recover-home').disabled=!recoverable||!state?.backend?.motion_allowed;
 for(const k of ['roll','pitch','yaw'])$(k).disabled=locked||mode!=='gui'||!arms[active]?.orientation;
 for(const id of ['pose-frame','pose-scope','pose-current','pose-down'])$(id).disabled=locked||mode!=='gui';
 document.querySelectorAll('#point-list button,#event-list button,#bottle-stage-actions button').forEach(b=>b.disabled=locked);
 for(const a of ['left','right']){$('arm-'+a).disabled=locked;$('arm-'+a).setAttribute('aria-pressed',a===active);}
 $('editing-arm').textContent=`当前画笔：${active==='left'?'左臂':'右臂'}${enabled().includes(active)?'':' · 未勾选执行'}`;
 $('editing-arm').className=active+'-arm';
 drawPoseEditor();
 renderQueuePanel();
}
let skillCatalogue=[],selectedSkill=null;
/** The registry is the catalogue an agent calls; the page renders the same descriptions and inputs. */
async function loadSkills(){
 // An entry with its own `call` (the toss queue, POST /api/tasks) is not planned through /api/skills/<name>.
 try{skillCatalogue=((await api('skills')).skills||[]).filter(skill=>!skill.call);}catch(e){skillCatalogue=[];}
 const select=$('skill-name');select.replaceChildren();
 const groups=new Map();
 for(const skill of skillCatalogue){if(!groups.has(skill.group))groups.set(skill.group,[]);groups.get(skill.group).push(skill);}
 for(const [group,list] of groups){
  const optgroup=document.createElement('optgroup');optgroup.label=group;
  for(const skill of list){const option=document.createElement('option');option.value=skill.name;option.textContent=skill.label;optgroup.append(option);}
  select.append(optgroup);
 }
 select.disabled=!skillCatalogue.length;
 selectedSkill=skillCatalogue[0]||null;
 if(selectedSkill)select.value=selectedSkill.name;
 renderSkillInputs();
}
/** One form per skill, built from the declared inputs: the page never hard-codes a skill's numbers. */
function renderSkillInputs(){
 const host=$('skill-fields'),speedHost=$('skill-speed-fields');host.replaceChildren();speedHost.replaceChildren();
 if(!selectedSkill){$('skill-note').textContent='服务没有返回技能目录。';return;}
 for(const [key,spec] of Object.entries(selectedSkill.inputs||{})){
  if(['speed_m_s','approach_speed_m_s'].includes(key)&&!selectedSkill.name.startsWith('bottle_cap_'))continue;
  const label=document.createElement('label');label.textContent=spec.label+(spec.unit||'');
  let field;
  if(spec.type==='choice'){field=document.createElement('select');for(const option of spec.options){const el=document.createElement('option');el.value=option.value;el.textContent=option.label;field.append(el);}field.value=spec.default;}
  else if(spec.type==='boolean'){field=document.createElement('input');field.type='checkbox';field.checked=Boolean(spec.default);}
  else{field=document.createElement('input');field.type='number';field.min=spec.min;field.max=spec.max;if(spec.step)field.step=spec.step;field.value=spec.default;}
  field.id='skill-input-'+key;field.dataset.skillInput=key;
  if(spec.hint)field.title=spec.hint;
  label.append(field);(key.includes('speed')?speedHost:host).append(label);
 }
 if(selectedSkill.name.startsWith('bottle_cap_')){
  const actions=document.createElement('div');actions.id='bottle-stage-actions';actions.className='full';actions.style.gridColumn='1 / -1';
  actions.style.display='flex';actions.style.flexWrap='wrap';actions.style.gap='8px';
  for(const [name,label] of [['bottle_cap_prepare','① 夹瓶身'],['bottle_cap_twist','② 拧瓶盖'],['bottle_cap_place','③ 放回桌面'],['bottle_cap_retract','松爪退开重试']]){
   if(!skillCatalogue.some(s=>s.name===name))continue;
   const button=document.createElement('button');button.type='button';button.textContent=label;
   button.setAttribute('aria-pressed',selectedSkill.name===name);
   button.onclick=()=>{selectSkill(name);if(name==='bottle_cap_place')message('请选择桌面上的瓶底落点：转正 → 放下 → 松爪 → 回位。');};
   actions.append(button);
  }
  host.prepend(actions);
 }
 $('skill-note').textContent=[selectedSkill.stroke_hint,selectedSkill.hint,selectedSkill.limits].filter(Boolean).join(' · ');
 updateSkillUI();
}
/** One place changes the chosen skill, from the dropdown or from a bottle-cap stage button. */
function selectSkill(name){
 if(name!==undefined)$('skill-name').value=name;
 selectedSkill=skillCatalogue.find(s=>s.name===$('skill-name').value)||null;
 renderSkillInputs();renderFields();drawAll();
}
/** The numbers this stroke carries, read once when the pointer goes down. */
function skillInputs(){
 const values={};
 for(const [key,spec] of Object.entries(selectedSkill?.inputs||{})){
  const field=$('skill-input-'+key);if(!field)continue;
  values[key]=spec.type==='boolean'?field.checked:spec.type==='choice'?field.value:Number(field.value);
 }
 return values;
}
function updateSkillUI(){
 const skills={
  'pick-place':['两点抓放','从物体画一笔到空闲桌面。第一点抓，抬起来，第二点放。',['① 抓住','↑ 抬起','② 放下']],
  grasp:['画线抓取','跨过物体画一条短线，用线的方向指定两指闭合方向。',['画夹持线','选终点动作']],
  brush:['轨迹画笔','画出末端要经过的路径，松开后选择终点抓取、松手或保持。',['画路径','选终点动作']],
  pour:['倒水','从杯子画到目标容器，自动侧抓、倾倒，再放回原处。',['侧抓','倾倒','放回']],
  select:['画线投放','画线指定抓取位置和闭合方向，再点击本次投放终点；不沿用旧落点。',['画抓取线','点击终点','投放']],
  skill:['原子技能','在下拉里选一个原子动作，按它的提示画一笔。',['选技能','画一笔','执行']],
  edit:['控制点微调','拖动主视图或侧视图中的轨迹点，调整路径与夹爪朝向。',['选轨迹点','调整位置或姿态']]
 };
 let [title,description,steps]=mode==='api'?['函数 API','用路径、姿态与速度函数定义动作。应用后预览，再执行。',['写函数','应用','预览执行']]:skills[drawTool];
 if(mode==='gui'&&drawTool==='skill'&&selectedSkill){title=selectedSkill.label;description=[selectedSkill.summary,selectedSkill.hint].filter(Boolean).join(' · ');steps=['选技能','画一笔',autoExecuteEnabled()?'自动执行':'预览执行'];}
 $('speed-console-context').textContent=`${active==='left'?'左臂':'右臂'} · ${title}`;
 $('skill-speed-group').hidden=mode!=='gui'||drawTool!=='skill'||!$('skill-speed-fields').children.length;
 $('skill-speed-title').textContent=`${selectedSkill?.label||'当前技能'} · 专用速度`;
 $('skill-title').textContent=title;$('skill-description').textContent=description;
 $('skill-steps').replaceChildren(...steps.map(text=>{const el=document.createElement('span');el.textContent=text;return el;}));
 for(const id of ['auto-execute','dual-arm'])$(id).closest('label').hidden=mode!=='gui'||['select','edit'].includes(drawTool);
 $('execution-hint').textContent=mode==='api'||drawTool==='edit'?'修改后请手动预览与执行。':drawTool==='select'?'提交队列后自动执行；可随时停止。':autoExecuteEnabled()?(dualArmEnabled()?'双臂：两条臂各画一笔，画完第二笔才一起执行；Esc 或停止可取消。':'画完后自动检查并执行；Esc 或停止可取消。'):'只生成草稿，点击预览后手动执行。';
 $('main-hint').textContent=mode==='gui'&&!pickingStart?description:$('main-hint').textContent;
 $('action-context').textContent=(preview?'待执行：'+Object.keys(preview.arms).map(a=>a==='left'?'左臂':'右臂').join('、'):'当前：'+(active==='left'?'左臂':'右臂'))+' · '+title;
}
function renderFields(){
 const m=arms[active];if(!m)return;
 selected=Math.max(0,Math.min(selected,m.points.length-1));
 const p=m.points[selected],r=m.poses[selected];
 $('point-count').textContent=mode==='gui'?`${m.points.length} 个点`:'函数轨迹';
 $('point-list').replaceChildren();
 if(mode==='gui')m.points.forEach((p,i)=>{const b=document.createElement('button');b.textContent=i===0?'0 起点':String(i);b.className=i===selected?'selected':'';b.addEventListener('click',()=>{selected=i;$('scrub').value=i/Math.max(1,m.points.length-1);renderFields();drawAll();});$('point-list').append(b);});
 ['x','y','z'].forEach((k,i)=>{$('point-'+k).value=p[i].toFixed(4);$('point-'+k).disabled=mode!=='gui'||running();});
 ['roll','pitch','yaw'].forEach((k,i)=>{$(k).value=r[i].toFixed(2);$(k).disabled=mode!=='gui'||!m.orientation||running();});
 const orientationMode=mode==='api'?(functionArms[active]?.orientation?.mode||'hold'):(m.orientation?'keyframes':m.orientationMode||'free');
 $('orientation-enabled').value=orientationMode;
 $('orientation-hint').textContent=orientationMode==='free'?'只约束 XYZ，IK 自动调整 RPY 并优先保持关节连续；预览显示求解后的朝向。':orientationMode==='hold'?'保持当前夹爪朝向，按姿态容差检查。':orientationMode==='function'?'朝向由 API 中的 roll / pitch / yaw 函数指定。':'选择不同轨迹点设置朝向，预览中显示工具坐标轴。';
 $('base-speed').value=m.speed[0][1];
 const approach=(mode==='gui'?m:functionArms[active])?.approach||{};
 $('queue-approach-speed').value=$('approach-speed').value=approach.speed_m_s??.8;$('approach-clearance').value=approach.clearance_m??.12;
 $('pick-start').setAttribute('aria-pressed',pickingStart);$('pick-start').disabled=mode!=='gui'||running();
 $('main-hint').textContent=mode==='gui'?(pickingStart?'点击主画面设置起点；虚线为自动接近段':drawTool==='pour'?'从杯子画到目标容器 · 松开后自动侧抓、倾倒、放回':drawTool==='select'?'画抓取线 → 点击本次终点 · 超长按最大开口':drawTool==='grasp'?'跨过物体画短线，松开选择终点动作':drawTool==='brush'?'按住画线，松开选择终点动作 · 虚线为自动接近段':'拖动控制点微调位置，空白处点击补点'):'函数轨迹与画笔入口使用同一执行器';
 updateSkillUI();
 $('event-list').replaceChildren();
 const events=mode==='gui'?m.events:(functionArms[active]?.gripper_events||[]);
 $('endpoint-summary').textContent='终点：'+endpointLabel(events);
 events.forEach((e,i)=>{const row=document.createElement('div');row.className='event-row';const text=document.createElement('span');text.textContent=`${Math.round(e.s*100)}% · ${e.opening_mm===undefined?'纯停顿':e.opening_mm+' mm'}${e.hold_s?' · 停留 '+e.hold_s+' s':''}`;const b=document.createElement('button');b.textContent='×';b.setAttribute('aria-label',`删除夹爪事件 ${i+1}`);b.onclick=()=>{events.splice(i,1);if(mode==='api')$('function-editor').value=JSON.stringify(functionArms[active],null,2);renderFields();queueSync();};row.append(text,b);$('event-list').append(row);});
 updateControls();
}
function canvas(id){const el=$(id),r=el.getBoundingClientRect(),d=window.devicePixelRatio||1;if(el.width!==Math.round(r.width*d)||el.height!==Math.round(r.height*d)){el.width=Math.round(r.width*d);el.height=Math.round(r.height*d);}const c=el.getContext('2d');c.setTransform(d,0,0,d,0,0);c.clearRect(0,0,r.width,r.height);return {c,w:r.width,h:r.height};}
function project(xyz){
 if(!frame)return null;const t=frame.t_world_camera,k=frame.k,p=xyz.map((v,i)=>v-t[i][3]);
 const q=[0,1,2].map(j=>p.reduce((acc,v,i)=>acc+v*t[i][j],0));if(q[2]<=0)return null;
 return [k[0][0]*q[0]/q[2]+k[0][2],k[1][1]*q[1]/q[2]+k[1][2]];
}
function mainProject(xyz){const p=project(xyz);return p&&mainRect?[mainRect.x+p[0]*mainRect.scale,mainRect.y+p[1]*mainRect.scale]:null;}
function userPlan(a){const p=compiled?.arms?.[a];return p?.user_trajectory||p;}
function getPath(a){return userPlan(a)?.xyz||(mode==='gui'?arms[a]?.points:[])||[];}
function line(c,points,color,width=2,dash=[]){c.beginPath();let started=false;for(const p of points){if(!p||!Number.isFinite(p[0])||!Number.isFinite(p[1])){started=false;continue;}if(!started){c.moveTo(...p);started=true;}else c.lineTo(...p);}c.strokeStyle=color;c.lineWidth=width;c.setLineDash(dash);c.stroke();c.setLineDash([]);}
function dot(c,p,color,r=4){if(!p)return;c.beginPath();c.arc(p[0],p[1],r,0,2*Math.PI);c.fillStyle=color;c.fill();}
function rotation(rpy){return rpyMatrix(rpy);}
function armMarker(c,p,a){
 if(!p||!Number.isFinite(p[0])||!Number.isFinite(p[1]))return;
 c.save();c.strokeStyle=colors[a];c.lineWidth=2;c.fillStyle='#101419';c.beginPath();
 if(a==='left')c.arc(...p,7,0,2*Math.PI);else{c.moveTo(p[0],p[1]-8);c.lineTo(p[0]+8,p[1]);c.lineTo(p[0],p[1]+8);c.lineTo(p[0]-8,p[1]);c.closePath();}
 c.fill();c.stroke();
 const text=(a==='left'?'左臂':'右臂')+' · 当前';
 c.font='600 12px sans-serif';const width=c.measureText(text).width;
 const x=a==='left'?p[0]+12:p[0]-width-12,y=p[1]+(a==='left'?12:-18);
 c.fillStyle='#101419e8';c.fillRect(x-5,y-12,width+10,21);c.fillStyle=colors[a];c.fillText(text,x,y+3);c.restore();
}
function drawTrajectories(c,proj,handles=true){
 for(const a of ['left','right']){
  const path=getPath(a);if(!path.length)continue;
  const color=colors[a],shown=enabled().includes(a);c.globalAlpha=shown?1:.35;
  const planned=compiled?.arms?.[a]?.approach;
  const current=state?.arms[a]?.xyz;
  const connector=planned?.xyz||(current&&Math.hypot(...path[0].map((v,i)=>v-current[i]))>.0005?[current,path[0]]:[]);
  if(connector.length)line(c,connector.map(proj),color,2,[7,5]);
  line(c,path.map(proj),color,a===active?2.5:1.5);
  if(mode==='gui')arms[a].points.forEach((p,i)=>{const end=i===arms[a].points.length-1;if(!handles&&i!==0&&!end)return;const q=proj(p);if(!q)return;dot(c,q,color,a===active&&i===selected?6:3.5);if(a===active||i===0){c.fillStyle=color;c.font='11px sans-serif';const name=a==='left'?'左臂':'右臂';c.fillText(i===0?name+'起点':handles?String(i):name+'终点',q[0]+8,q[1]-9);}});
  const fraction=Number($('scrub').value),i=Math.min(path.length-1,Math.round(fraction*(path.length-1))),p=path[i];
  dot(c,proj(p),'#fff',4);
  const r=userPlan(a)?.rotation_matrices?.[i]||rotation(mode==='gui'&&arms[a]?.orientation?arms[a].poses[i]||state.arms[a].rpy_deg:state.arms[a].rpy_deg);
  if(a===active){for(const segment of gripperSegments(state.arms[a].gripper_mm/1000))line(c,segment.map(v=>proj(transform(r,v).map((x,k)=>x+p[k]))),'#dce7f1',1.5,[3,2]);}
  for(let j=0;j<3;j++)line(c,[proj(p),proj(p.map((v,k)=>v+.035*r[k][j]))],['#ed8d8d','#93d5a3','#88bfee'][j],2);
  for(const e of (mode==='gui'?arms[a].events:functionArms[a]?.gripper_events||[])){
   const q=proj(path[Math.min(path.length-1,Math.round(e.s*(path.length-1)))]);if(q){c.strokeStyle=color;c.strokeRect(q[0]-5,q[1]-5,10,10);}
  }
  c.globalAlpha=1;
 }
 for(const a of ['left','right']){const current=state?.arms[a]?.xyz;if(current)armMarker(c,proj(current),a);}
}
function drawMain(){const {c,w,h}=canvas('main-view');if(!frame||!photo)return;const scale=Math.min(w/frame.width,h/frame.height),rw=frame.width*scale,rh=frame.height*scale;mainRect={x:(w-rw)/2,y:(h-rh)/2,scale};c.drawImage(livePainter?.current()||photo,mainRect.x,mainRect.y,rw,rh);c.save();if(brush)c.globalAlpha=.25;drawTrajectories(c,mainProject,drawTool==='edit'||pickingStart);c.restore();if(brush){const ink=brush.pixels.map(p=>[mainRect.x+p[0]*scale,mainRect.y+p[1]*scale]);c.lineCap='round';c.lineJoin='round';line(c,ink,colors[brush.arm],3);dot(c,ink[0],colors[brush.arm],5);dot(c,ink.at(-1),'#fff',3);}drawQueueOverlay(c);}
function drawSide(){
 const {c,w,h}=canvas('side-view');if(!frame||w<100||h<100)return;
 const axis=$('side-axis').value,pad={left:44,right:14,top:20,bottom:32},pw=w-pad.left-pad.right,ph=h-pad.top-pad.bottom;
 const xmin=axis==='yz'?-1.10:-.1,xmax=axis==='yz'?.45:1.05,zmin=-.06,zmax=.75;
 sideMap={xmin,xmax,zmin,zmax,...pad,pw,ph};
 const proj=p=>axis==='iso'?[w*.5+(p[0]-.4)*pw*.75-(p[1]+.3)*pw*.55,h*.73-(p[2]) *ph*.9+(p[0]-.4)*ph*.22+(p[1]+.3)*ph*.13]:[pad.left+((axis==='yz'?p[1]:p[0])-xmin)/(xmax-xmin)*pw,pad.top+(zmax-p[2])/(zmax-zmin)*ph];
 sideMap.project=proj;c.font='10px sans-serif';c.strokeStyle='#2b3642';c.fillStyle='#9cabbb';
 if(axis!=='iso'){
  for(let i=0;i<=4;i++){const z=zmin+(zmax-zmin)*i/4,y=pad.top+ph*(1-i/4);line(c,[[pad.left,y],[w-pad.right,y]],'#26313d',.7);c.fillText(z.toFixed(2),4,y+3);const x=xmin+(xmax-xmin)*i/4,px=pad.left+pw*i/4;c.fillText(x.toFixed(2),px-10,h-13);}
  c.fillText('Z (m)',5,12);c.fillText((axis==='xz'?'X':'Y')+' (m)',w-42,h-1);
 }
 c.save();c.beginPath();c.rect(pad.left,pad.top,pw,ph);c.clip();
 frame.points.forEach((p,i)=>{if(p[2]<zmin||p[2]>zmax)return;const q=proj(p),rgb=frame.colors[i];c.fillStyle=`rgba(${rgb[0]},${rgb[1]},${rgb[2]},.58)`;c.fillRect(q[0],q[1],1.5,1.5);});
 drawTrajectories(c,proj);c.restore();
}
function drawSpeed(){
 const {c,w,h}=canvas('speed-view'),left=43,right=16,top=12,bottom=26,pw=w-left-right,ph=h-top-bottom,max=1;
 speedMap={left,top,pw,ph,max};const proj=p=>[left+p[0]*pw,top+(1-p[1]/max)*ph];
 c.font='10px sans-serif';c.fillStyle='#a0acbb';
 for(let i=0;i<=3;i++){const v=max*i/3,y=proj([0,v])[1];line(c,[[left,y],[w-right,y]],'#2b3541',.7);c.fillText(v.toFixed(2),5,y+3);}
 for(let i=0;i<=4;i++)c.fillText(`${i*25}%`,left+pw*i/4-9,h-8);
 if(!arms[active])return;
 const p=userPlan(active);
 const requested=p?Math.max(...p.requested_speed_m_s):Math.max(...arms[active].speed.map(k=>k[1]));
 const duration=p?p.time_s.at(-1)-p.time_s[0]:0;
 $('speed-summary').textContent=`目标最高 ${(requested*100).toFixed(1)} cm/s（${requested.toFixed(3)} m/s）${preview&&duration>0?` · 计划平均 ${(p.path_length_m/duration*100).toFixed(1)} cm/s · 路径 ${(p.path_length_m*100).toFixed(1)} cm / ${duration.toFixed(1)} s`:' · 预览后显示计划速度'}`;
 if(p){line(c,p.s.map((s,i)=>proj([s,p.requested_speed_m_s[i]])),colors[active],2);if(preview){const actual=p.s.slice(1).map((s,i)=>[s,Math.hypot(...p.xyz[i+1].map((v,j)=>v-p.xyz[i][j]))/(p.time_s[i+1]-p.time_s[i])]);line(c,actual.map(proj),'#ccd4df',1,[4,3]);}}
 else line(c,arms[active].speed.map(proj),colors[active],2);
 if(mode==='gui')arms[active].speed.forEach(k=>dot(c,proj(k),colors[active],4.5));
 const x=left+Number($('scrub').value)*pw;line(c,[[x,top],[x,h-bottom]],'#fff',1,[3,3]);
}
function drawAll(){drawMain();drawSide();drawSpeed();drawPoseEditor();}
function poseReference(a,index){
 const m=arms[a],p=userPlan(a),i=p?Math.round(index/Math.max(1,m.points.length-1)*(p.rotation_matrices.length-1)):0;
 return p?.rotation_matrices?.[i]?matrixRpy(p.rotation_matrices[i]):state.arms[a].rpy_deg.slice();
}
function drawPoseEditor(){
 if(!poseEditor||!state||!arms[active]||!$('pose-view').getBoundingClientRect().width)return;
 const m=arms[active],r=m.orientation?m.poses[selected]:poseReference(active,selected);
 poseEditor.setPose(r,{enabled:mode==='gui'&&!running()&&(!busy||poseEditor.dragging),frame:$('pose-frame').value,
  opening:state.arms[active].gripper_mm/1000,label:`${active==='left'?'左臂':'右臂'} · ${$('pose-scope').value==='path'?'整条路径':selected+' 号轨迹点'}`});
}
function beginPoseEdit(){
 const m=arms[active];poseUndo=structuredClone({arm:active,poses:m.poses,orientation:m.orientation,orientationMode:m.orientationMode});
 if(!m.orientation)m.poses=m.points.map((p,i)=>poseReference(active,i));
 busy=true;invalidate();compiled=null;$('scrub').value=selected/Math.max(1,m.points.length-1);updateControls();
}
function changePose(rpy){
 const m=arms[active];m.orientation=true;m.orientationMode='keyframes';
 if($('pose-scope').value==='path')m.poses=m.points.map(()=>rpy.slice());else m.poses[selected]=rpy.slice();
 compiled=null;renderFields();drawAll();
}
async function commitPoseEdit(){
 updateControls();
 try{await queueSync();}
 catch(e){message(e.message,true);}finally{poseUndo=null;busy=false;renderFields();drawAll();}
}
function cancelPoseEdit(){
 if(poseUndo){const {arm,...saved}=poseUndo;Object.assign(arms[arm],saved);}poseUndo=null;busy=false;compiled=null;renderFields();drawAll();message('已取消朝向拖动。');
}
function pointEvent(e,el){const r=el.getBoundingClientRect();return [e.clientX-r.left,e.clientY-r.top];}
function commitMotionInputs(){
 const speed=Number($('base-speed').value),clearance=Number($('approach-clearance').value);
 if(!$('base-speed').value.trim()||!Number.isFinite(speed)||speed<.001||speed>1)throw Error('移动速度需为 0.001–1.0 m/s');
 if(!$('approach-clearance').value.trim()||!Number.isFinite(clearance)||clearance<.02||clearance>.30)throw Error('抬起余量需为 0.02–0.30 m');
 const m=arms[active];
 const approach=Number($('queue-approach-speed').value);
 if(!Number.isFinite(approach)||approach<.001||approach>1)throw Error('接近速度需为 0.001–1.0 m/s');
 if(speed!==m.speed[0][1]){delete m.speedSpec;m.speed=m.speed.map(k=>[k[0],speed]);}
 m.approach.clearance_m=clearance;m.approach.speed_m_s=approach;
}
function nearest(list,position,proj,radius=14){let best=-1,d=radius;list.forEach((p,i)=>{const q=proj(p);if(!q)return;const v=Math.hypot(q[0]-position[0],q[1]-position[1]);if(v<d){d=v;best=i;}});return best;}
async function pickMain(pos,move=false){
 if(!mainRect||!frame)return;const u=(pos[0]-mainRect.x)/mainRect.scale,v=(pos[1]-mainRect.y)/mainRect.scale;
 if(u<0||v<0||u>=frame.width||v>=frame.height)return;
 const m=arms[active],d=await api('pick','POST',{observation_id:frame.id,u,v,mode:$('pick-mode').value,z:move?m.points[selected][2]:Number($('plane-z').value)});
 if(pickingStart){m.points[0]=d.xyz;selected=0;pickingStart=false;}else if(move)m.points[selected]=d.xyz;else{m.points.push(d.xyz);m.poses.push(m.poses.at(-1).slice());selected=m.points.length-1;}
 renderFields();drawAll();queueSync();
}
function brushPixel(e,clamp=false){
 if(!mainRect||!frame)return null;
 const p=pointEvent(e,$('main-view')),u=(p[0]-mainRect.x)/mainRect.scale,v=(p[1]-mainRect.y)/mainRect.scale;
 if(!clamp&&(u<0||v<0||u>=frame.width||v>=frame.height))return null;
 return [Math.max(0,Math.min(frame.width-1,u)),Math.max(0,Math.min(frame.height-1,v))];
}
function addInk(e,force=false){
 if(!brush||brush.finished||e.pointerId!==brush.pointerId)return;
 const p=brushPixel(e,true),last=brush.pixels.at(-1);
 if(!p||(!force&&Math.hypot(p[0]-last[0],p[1]-last[1])<1))return;
 if(brush.pixels.length>=4096)brush.pixels=brush.pixels.filter((_,i)=>i%2===0||i===brush.pixels.length-1);
 // A grasp line (grasp and toss tools) or a two-point skill keeps only the first and the latest sample; every other
 // tool keeps the whole ink.
 if(['grasp','select'].includes(brush.tool)||(brush.tool==='skill'&&brush.skill?.stroke==='two_points'))brush.pixels=[brush.pixels[0],p];else brush.pixels.push(p);
}
function cancelBrush(explicit=false){
 if(!brush||(brush.finished&&!explicit))return;
 const stroke=brush;stroke.cancelled=true;
 // Keep the UI locked until the pending freeze/generation drains; a second stroke must
 // not race adoption of the cancelled stroke's observation.
 if(!stroke.finished){brush=null;busy=false;}
 if($('main-view').hasPointerCapture(stroke.pointerId))$('main-view').releasePointerCapture(stroke.pointerId);
 // The frozen observation still lands on the server; adopt it so later strokes do not carry a stale id.
 if(stroke.freeze&&!stroke.finished){busy=true;stroke.freeze.then(adoptFrozenObservation).catch(e=>message(e.message,true)).finally(()=>{busy=false;updateControls();drawAll();});}
 updateControls();drawAll();message('已取消这一笔，原路径保留。');
}
function freezeObservation(){
 // Every stroke captures a fresh frame, except the strokes of a dual-arm pair: both grasps must hold on one
 // picture (the robot has not moved in between), and the service merges only drafts of the same observation.
 if(dualArms.size)return Promise.resolve(frame);
 return api('observe','POST',{preserve_draft:true});
}
async function adoptFrozenStroke(stroke){
 await adoptFrozenObservation(await stroke.freeze);
 stroke.observationId=frame.id;
}
async function adoptFrozenObservation(f){
 if(frame?.id!==f.id)installObservation(f,await observationImage(f));
 state=await api('state');lastRemoteRevision=state.revision;
}
// A pinch has no measured object: say so instead of reporting the nominal width as if it were measured.
function graspSummary(result){
 return (result.pinch?`捏取落笔处（未分割物体，指尖压桌 ${(-result.finger_z_mm).toFixed(0)} mm）`:`物体宽约 ${result.object_width_mm.toFixed(0)} mm`)+(result.top_down?' · 全程竖直':'');
}
async function finishPickPlace(stroke){
 return finishAutomaticStroke(stroke,{run:options=>runStrokeTask({...options,route:'pick-place'}),label:'两点抓放',
  request:{arm:stroke.arm,observation_id:stroke.observationId,pixels:stroke.pixels,clearance_m:arms[stroke.arm].approach.clearance_m,speed_m_s:arms[stroke.arm].speed[0][1],pinch:stroke.pinch,top_down:stroke.topDown,pinch_depth_mm:stroke.pinchDepth},
  locating:'正在定位起点与落点，生成两点抓放…',executing:'两点抓放执行中：抓住 → 抬起 → 平移 → 放下到位 → 松手。',
  summary:result=>`两点抓放 · ${graspSummary(result)} · 抓住 → 抬起 → 平移 → 放下 → 松手`});
}
async function finishSkill(stroke){
 const skill=stroke.skill;
 if(!skill){brush=null;busy=false;message('请先选择一个原子技能。',true);renderFields();drawAll();return;}
 await finishAutomaticStroke(stroke,{run:options=>runStrokeTask({...options,route:'skills/'+skill.name}),label:skill.label,independent:skill.name.startsWith('bottle_cap_'),
  request:{arm:stroke.arm,observation_id:stroke.observationId,pixels:stroke.pixels,
   clearance_m:arms[stroke.arm].approach.clearance_m,speed_m_s:arms[stroke.arm].speed[0][1],...stroke.inputs},
  locating:`正在按“${skill.label}”解析这一笔…`,executing:`${skill.label}执行中…`,
  summary:result=>result.summary});
 if(automaticAwaitingFinish&&skill.name==='bottle_cap_prepare')finishNotice='瓶子已水平持住。选择“拧瓶盖②”，在新画面点瓶盖中心；持瓶臂保持。';
 if(automaticAwaitingFinish&&skill.name==='bottle_cap_place')finishNotice='瓶子已放置，持瓶臂已回位。';
 if(automaticAwaitingFinish&&skill.name==='bottle_cap_twist')finishNotice='拧转完成，拧盖臂已松开退开。点击「③ 放回桌面」，再点桌面落点；持瓶臂继续保持。';
}
async function finishPour(stroke){
 return finishAutomaticStroke(stroke,{run:options=>runStrokeTask({...options,route:'pour'}),label:'自动倒水',
  request:{arm:stroke.arm,observation_id:stroke.observationId,pixels:stroke.pixels,clearance_m:arms[stroke.arm].approach.clearance_m,speed_m_s:arms[stroke.arm].speed[0][1],
   tilt_deg:Number($('pour-tilt').value),hold_s:Number($('pour-hold').value),grasp_fraction:Number($('pour-grasp-fraction').value),
   mouth_clearance_m:Number($('pour-clearance').value),grip_effort:Number($('pour-effort').value)},
  locating:'正在定位杯子与目标容器，生成倒水动作…',executing:'自动倒水执行中：侧抓 → 提起 → 倾倒 → 回正 → 放回 → 撤回。',
  summary:result=>`自动倒水 · 杯径约 ${result.cup_diameter_mm.toFixed(0)} mm · 杯高约 ${result.cup_height_mm.toFixed(0)} mm · 倾角 ${result.tilt_deg}° · 停留 ${result.hold_s} s`});
}
/** A generated draft replaces the stroke arm's path; only that arm executes. */
function adoptAutomaticDraft(arm,spec,saved){return adoptDraftSpecs({[arm]:spec},saved);}
/** A generated draft replaces each named arm's path; only those arms execute. Bimanual skills name both. */
function adoptDraftSpecs(specs,saved){
 for(const [arm,spec] of Object.entries(specs)){
  const m=arms[arm];strokeUndo[arm]=structuredClone(m);
  const previousPose=m.poses?.[0]?.slice()||[0,0,0];
  m.points=spec.path.points;m.orientationMode=spec.orientation?.mode||'hold';m.orientation=m.orientationMode==='keyframes';
  m.poses=m.points.map((p,i)=>m.orientation?sampleOrientation(spec.orientation.points,i/Math.max(1,m.points.length-1)):previousPose.slice());
  m.events=spec.gripper_events;m.startHold=spec.start_hold_s||0;m.graspLine=spec.grasp_line===true;m.bottleCap=spec.bottle_cap;m.angularSpeed=spec.angular_speed_deg_s;
  // A skill may hand back a speed curve (keyframes) instead of one number; keep the curve as it is and turn a
  // single number into a flat two-point curve.
  m.speed=Array.isArray(spec.speed)?spec.speed.map(k=>k.slice()):[[0,spec.speed],[1,spec.speed]];delete m.speedSpec;m.approach=spec.approach;m.airTrack=spec.air_track===true;
  functionArms[arm]=structuredClone(spec);
 }
 // In a two-armed orchestration the arm drawn earlier stays on the execution list until its partner arrives.
 for(const a of ['left','right'])$('enable-'+a).checked=(a in specs)||(dualArmEnabled()&&dualArms.has(a));
 lastRemoteRevision=saved.revision;
 selected=0;compiled=null;brush=null;renderFields();drawAll();
}
async function showAutomaticPlan(plan,label){
 await adoptPreparedPlan(plan);
 compiled=plan;preview=null;drawAll();$('duration').textContent=`${label} ${plan.duration_s.toFixed(1)} s`;
 $('plan-notes').replaceChildren();for(const note of plan.notes){const p=document.createElement('p');p.textContent=note;$('plan-notes').append(p);}
 message(`检查通过，正在自动执行${label}…`);
}
/** Shared state machine for strokes the server interprets: one run token, cancel via Esc or the stop button. */
async function finishAutomaticStroke(stroke,{run,request,label,locating,executing,summary,independent=false}){
 const token={cancelled:false},cancelEpoch=stroke.cancelEpoch??state.cancel_epoch;automaticRun=token;updateControls();
 try{
  await syncChain;
  if(toleranceDirty)await saveTolerance();
  message(locating);
  if(stroke.cancelled)throw Error('这一笔已取消');
  const dual=!independent&&dualArmEnabled();
  if(independent)clearDualOrchestration();
  let fired=false;
  await run({api,revision:lastRemoteRevision,cancelEpoch,request,merge:dual,approachSpeed:arms[stroke.arm].approach.speed_m_s,
   // Which arms a stroke drafted is only known once the route answers - a bimanual skill drafts both at once,
   // which satisfies the pair in one stroke. Until both arms are in, the draft is kept and nothing runs.
   execute:drafted=>{
    for(const a of Object.keys(drafted))dualArms.add(a);
    fired=autoExecuteEnabled()&&(!dual||dualArms.size>=2);
    return fired;
   },
   isCancelled:()=>token.cancelled||Boolean(stroke.cancelled),
   onDraft(result,saved){adoptDraftSpecs(result.arms||{[stroke.arm]:result.arm},saved);$('brush-status').textContent=summary(result);$('brush-status').classList.remove('error');},
   onPreview:plan=>showAutomaticPlan(plan,label)});
  if(fired)clearDualOrchestration();
  else if(dual&&dualArms.size<2)switchToWaitingArm();
  automaticAwaitingFinish=fired;
  message(fired?executing:dual&&dualArms.size<2?dualWaitingMessage()
          :`${label}草稿已生成。点击「预览与检查」，再执行。`);
 }catch(e){preview=null;message(e.message,true);}
 finally{automaticRun=null;brush=null;busy=false;await poll();renderFields();drawAll();}
}
async function finishBrush(e){
 if(!brush||brush.finished||e.pointerId!==brush.pointerId)return;
 addInk(e,true);const stroke=brush;stroke.finished=true;
 if($('main-view').hasPointerCapture(e.pointerId))$('main-view').releasePointerCapture(e.pointerId);
 if(stroke.deferred){
  pendingStrokes.push(stroke);brush=null;busy=false;
  updatePendingStrokes();updateControls();drawAll();
  message(`已排队 ${pendingStrokes.length} 笔，当前动作及回位完成后依次处理。`);
  return;
 }
 await finishStroke(stroke);
}
async function finishStroke(stroke){
 if(stroke.freeze){
  message('正在等待落笔时采集的新观测…');
  try{await adoptFrozenStroke(stroke);}
  catch(err){brush=null;busy=false;updateControls();drawAll();message('落笔时更新观测失败，这一笔已丢弃：'+err.message,true);return;}
 }
 if(stroke.cancelled){brush=null;busy=false;updateControls();drawAll();return;}
 // A tap specifies one release location, not a pickup-and-place pair.
 if(['pick-place','grasp'].includes(stroke.tool)&&stroke.pixels?.length&&
    stroke.pixels.every(p=>Math.hypot(p[0]-stroke.pixels[0][0],p[1]-stroke.pixels[0][1])<(stroke.tool==='grasp'?2:12))){
  const point=stroke.pixels[0].slice();stroke.tool='grasp';stroke.pointPlacement=true;stroke.pixels=[point,point.slice()];
 }
 if(stroke.tool==='pick-place'){await finishPickPlace(stroke);return;}
 if(stroke.tool==='pour'){await finishPour(stroke);return;}
 if(stroke.tool==='skill'){await finishSkill(stroke);return;}
 if(stroke.tool==='select'){await finishSelect(stroke);return;}
 message(stroke.tool==='grasp'?'请先选择抓取、松手或保持，再规划对应动作…':'正在把笔迹转换为三维轨迹…');
 try{
  await syncChain;
  const isGrasp=stroke.tool==='grasp',revision=lastRemoteRevision;
  const choice=await endpointDialog.choose({arm:stroke.arm,revision,pointOnly:Boolean(stroke.pointPlacement)});
  if(stroke.cancelled)return;
  if(choice.action===null){message(choice.reason||'已取消这一笔，原路径保留。');return;}
  const result=await api(isGrasp?'grasp':'stroke','POST',isGrasp?{arm:stroke.arm,observation_id:stroke.observationId,pixels:stroke.pixels,endpoint_action:choice.action,inset_mm:stroke.inset,grip_effort:stroke.effort,clearance_m:arms[stroke.arm].approach.clearance_m,speed_m_s:arms[stroke.arm].speed[0][1]}:{observation_id:stroke.observationId,pixels:stroke.pixels,mode:stroke.pickMode,z:stroke.z});
  if(stroke.cancelled)return;
  if(frame.id!==result.observation_id)throw Error('观测已更新，请重新画线。');
  const latest=await api('state');
  if(stroke.cancelled)return;
  if(stroke.cancelEpoch!==undefined&&latest.cancel_epoch!==stroke.cancelEpoch)throw Error('这一笔已取消，请重新画线。');
  if(latest.revision!==revision||['running','planning','cancelling'].includes(latest.execution.state)||state.revision>revision||(state.revision===revision&&running()))throw Error('草稿或机器人状态已变化，请重新画线。');
  const m=arms[stroke.arm];
  strokeUndo[stroke.arm]=structuredClone(m);
  const resetOrientation=m.orientation;
  if(isGrasp){
   const spec=result.arm;m.points=spec.path.points;
   m.orientationMode=spec.orientation.mode;m.orientation=m.orientationMode==='keyframes';
   m.poses=m.orientation?spec.orientation.points.map(k=>k.slice(1)):m.points.map(()=>state.arms?.[stroke.arm]?.rpy_deg?.slice()||stroke.pose?.slice()||[0,0,0]);
   m.events=spec.gripper_events;m.startHold=spec.start_hold_s;m.approach={...spec.approach,speed_m_s:m.approach.speed_m_s};m.speed=[[0,spec.speed],[1,spec.speed]];delete m.speedSpec;
  }else{
   m.points=result.points;m.poses=result.points.map(()=>stroke.pose.slice());m.orientation=false;m.startHold=0;
   if(resetOrientation)m.orientationMode='free';
  }
  m.graspLine=isGrasp&&choice.action==='grasp';
  m.events=withEndpointAction(m.events,choice.action,{graspLine:isGrasp});
  if(isGrasp&&choice.action!=='none'){
   const planned=result.arm.gripper_events.find(e=>Number(e.s)===1&&e.opening_mm===(choice.action==='grasp'?0:70));
   if(planned)m.events=m.events.map(e=>Number(e.s)===1?{...e,...planned}:e);
  }
  if(isGrasp)m.startHold=choice.action==='grasp'?.6:0;
  if(isGrasp&&choice.action==='grasp'&&result.lift_xyz){
   m.points.push(result.lift_xyz);m.poses.push(m.poses.at(-1).slice());
   const close=result.arm.gripper_events.find(e=>Number(e.s)===1&&e.opening_mm===0);
   m.events=m.events.map(e=>({...e,...(Number(e.s)===1?close:{}),s:Number(e.s)/2}));
  }
  selected=0;compiled=null;brush=null;renderFields();drawAll();
  const dual=dualArmEnabled();
  if(dual)dualArms.add(stroke.arm);
  const automatic=autoExecuteEnabled()&&(!dual||dualArms.size>=2);
  const ending=automatic?'画线即执行：自动预览并执行':dual?'等另一条臂画完，两只手一起执行':'预览后执行';
  if(isGrasp&&result.grasp_mode==='position_only'){
   $('brush-status').textContent=`按画线中点定位 · 目标表面上方 ${result.release_clearance_mm} mm → 到位后${endpointLabel(m.events)}；${ending}`;$('brush-status').classList.remove('error');
  }else if(isGrasp){$('brush-status').textContent=`${result.grasp_mode==='tilted_side'?`倾斜侧抓 ${result.grasp_tilt_deg}° · `:result.grasp_mode==='top_down'?'竖直抓取 · ':''}抓取线宽 ${result.width_mm.toFixed(1)} mm${result.width_clamped?'（画线超长，已按最大开口）':''} · 表面 Z ${(result.surface_xyz[2]*1000).toFixed(1)} mm · ${choice.action==='grasp'?'上方张开':'保持夹爪'} → 下探 ${(result.actual_inset_mm??stroke.inset).toFixed(1)} mm${result.table_limited?'（已按桌面边界调整）':''} → 到位后${choice.action==='grasp'&&result.lift_xyz?'闭爪并提起 4 cm':endpointLabel(m.events)}；${ending}`;$('brush-status').classList.remove('error');}
  else{
   const coarse=result.simplification_error_mm>1.001;
   $('brush-status').textContent=`终点：${endpointLabel(m.events)} · 已生成 ${result.control_point_count} 个控制点 · 折线简化偏差 ${result.simplification_error_mm.toFixed(1)} mm${coarse?'，复杂笔迹有明显简化，请检查预览或缩短这一笔':' · 侧视图可调整高度'}${resetOrientation?' · 重画已改为姿态自由，可重新设置朝向关键帧':''} · ${ending}`;
   $('brush-status').classList.toggle('error',coarse);
  }
  if(automatic){await executeDraftAutomatically(stroke.arm,stroke);clearDualOrchestration();}
  else{await queueSync();if(dual){switchToWaitingArm();message(dualWaitingMessage());}}
 }catch(err){message(err.message,true);}
 finally{brush=null;busy=false;renderFields();drawAll();}
}
function autoExecuteEnabled(){return $('auto-execute').checked;}
function dualArmEnabled(){return $('dual-arm').checked;}
function clearDualOrchestration(){dualArms=new Set();}
/** After one arm of a pair is drawn, the brush moves to the arm still missing: the next stroke is that one. */
function switchToWaitingArm(){
 const next=['left','right'].find(a=>!dualArms.has(a));
 if(next&&next!==active){active=next;$('arm').value=next;}
 return next;
}
const ARM_LABELS={left:'左臂',right:'右臂'};
function dualWaitingMessage(){
 const ready=[...dualArms].map(a=>ARM_LABELS[a]).join('、');
 const waiting=['left','right'].filter(a=>!dualArms.has(a)).map(a=>ARM_LABELS[a]).join('');
 return `${ready}已就位，草稿留着。再画一笔${waiting}，两只手一起执行。`;
}
/** "Draw to execute": the stroke arm's finished draft is saved, previewed and executed once without confirmation. */
async function executeDraftAutomatically(arm,stroke=null){
 const token={cancelled:false},cancelEpoch=stroke?.cancelEpoch??state.cancel_epoch;automaticRun=token;updateControls();
 try{
  if(toleranceDirty)await saveTolerance();
  for(const a of ['left','right'])$('enable-'+a).checked=a===arm||(dualArmEnabled()&&dualArms.has(a));
  invalidate();
  const d=draft(),issue=draftIssue(d);
  if(issue)throw Error(issue);
  message('画线即执行：正在保存草稿并自动预览…');
  await runDraftExecution({api,draft:d,revision:lastRemoteRevision,cancelEpoch,isCancelled:()=>token.cancelled||Boolean(stroke?.cancelled),
   onSaved(saved){lastRemoteRevision=saved.revision;},
   onPreview:plan=>showAutomaticPlan(plan,'画线即执行')});
  automaticAwaitingFinish=true;message('画线即执行：轨迹执行中，完成后自动更新图像。');
 }catch(e){preview=null;message(e.message,true);}
 finally{automaticRun=null;busy=false;await poll();renderFields();drawAll();}
}
/** Everything a stroke uses is read when the pen goes down; later edits to the panels do not change it. */
function startStroke(e,pixel){
 return {arm:active,tool:drawTool,skill:selectedSkill,inputs:skillInputs(),
  inset:Number($('grasp-inset').value),effort:Number($('grasp-effort').value),topDown:$('pick-place-topdown').checked,pinch:$('pick-place-pinch').checked,pinchDepth:Number($('pick-place-pinch-depth').value),
  pointerId:e.pointerId,pixels:[pixel],observationId:frame.id,cancelEpoch:state.cancel_epoch,
  pickMode:$('pick-mode').value,z:Number($('plane-z').value),pose:state.arms[active].rpy_deg.slice(),finished:false};
}
$('main-view').addEventListener('pointerdown',e=>{
 if(clearingPaths||mode!=='gui'||(busy&&!settingDrop)||!frame||e.button!==0||!e.isPrimary)return;
 if(settingDrop){e.preventDefault();placeDropPoint(e).catch(err=>message(err.message,true));return;}
 if(running()||pendingStrokes.length){
  if(!['pick-place','pour','brush','grasp','skill','select'].includes(drawTool))return;
  if(state.execution.state==='cancelling')return;
  e.preventDefault();const pixel=brushPixel(e);if(!pixel)return;
  if(pendingStrokes.length>=16){message('已有 16 笔待执行，请等待队列完成。',true);return;}
  brush={...startStroke(e,pixel),motion:structuredClone({speed:arms[active].speed,approach:arms[active].approach}),deferred:true};
  busy=true;e.currentTarget.setPointerCapture(e.pointerId);updateControls();drawMain();
  message('继续画下一笔：松开后排队，不影响当前动作。');return;
 }
 e.preventDefault();const p=pointEvent(e,e.currentTarget);
 if(pickingStart){pickMain(p).catch(e=>message(e.message,true));return;}
 if(drawTool==='select'){beginSelectPointer(e);return;}
 if(['pick-place','pour','brush','grasp','skill'].includes(drawTool)){
  // Canvas pointerdown prevents the default blur; commit typed values before locking inputs.
  try{commitMotionInputs();}catch(err){message(err.message,true);return;}
  const pixel=brushPixel(e);if(!pixel)return;
  brush=startStroke(e,pixel);
  // Freeze one fresh RGB-D now: the stroke is drawn over the live picture or a stale snapshot.
  if(livePainter?.running||state.observation_stale){brush.freeze=freezeObservation();brush.freeze.catch(()=>{});}
  busy=true;invalidate();e.currentTarget.setPointerCapture(e.pointerId);updateControls();drawMain();message(drawTool==='pick-place'?'正在画抓放线…起点抓，终点放，Esc 取消。':drawTool==='pour'?'正在画倒水线…从杯子到目标容器，松开即自动执行，Esc 取消。':drawTool==='grasp'?'正在画抓取线…松开选择终点动作，Esc 取消。':drawTool==='skill'?`正在画${selectedSkill?.label||'技能'}…松开即自动执行，Esc 取消。`:'正在画线…松开选择终点动作，Esc 取消这一笔。');return;
 }
 const i=nearest(arms[active].points,p,mainProject);
 if(i>=0){selected=i;drag={kind:'main',start:p,moved:false};e.currentTarget.setPointerCapture(e.pointerId);renderFields();drawAll();}
 else pickMain(p).catch(e=>message(e.message,true));
});
$('main-view').addEventListener('pointermove',e=>{if(brush){const events=e.getCoalescedEvents?.()||[];for(const sample of events.length?events:[e])addInk(sample);drawMain();return;}if(drag?.kind==='main'){const p=pointEvent(e,e.currentTarget);if(Math.hypot(p[0]-drag.start[0],p[1]-drag.start[1])>3)drag.moved=true;}});
$('main-view').addEventListener('pointerup',e=>{if(brush){finishBrush(e);return;}if(drag?.kind==='main'&&drag.moved)pickMain(pointEvent(e,e.currentTarget),true).catch(e=>message(e.message,true));drag=null;});
for(const event of ['pointercancel','lostpointercapture'])$('main-view').addEventListener(event,e=>{if(brush?.pointerId===e.pointerId)cancelBrush();drag=null;});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){if(automaticRun||running())$('cancel').click();else cancelBrush(true);}});
$('side-view').addEventListener('pointerdown',e=>{
 if(mode!=='gui'||running()||busy||!sideMap||$('side-axis').value==='iso')return;
 const i=nearest(arms[active].points,pointEvent(e,e.currentTarget),sideMap.project);
 if(i>=0){selected=i;renderFields();drawAll();drag={kind:'side'};e.currentTarget.setPointerCapture(e.pointerId);}
});
$('side-view').addEventListener('pointermove',e=>{
 if(drag?.kind!=='side')return;const [,y]=pointEvent(e,e.currentTarget),m=sideMap;
 arms[active].points[selected][2]=Math.max(m.zmin,Math.min(m.zmax,m.zmax-(y-m.top)/m.ph*(m.zmax-m.zmin)));
 compiled=null;invalidate();renderFields();drawAll();
});
$('side-view').addEventListener('pointerup',()=>{if(drag?.kind==='side')queueSync();drag=null;});
$('speed-view').addEventListener('pointerdown',e=>{
 if(mode!=='gui'||running()||busy||!arms[active])return;
 const [x,y]=pointEvent(e,e.currentTarget),m=speedMap,s=Math.max(0,Math.min(1,(x-m.left)/m.pw)),v=Math.max(.001,Math.min(1,(1-(y-m.top)/m.ph)*m.max));
 delete arms[active].speedSpec;const k=arms[active].speed;let i=k.findIndex(p=>Math.abs(p[0]-s)<.035);
 if(i<0){k.push([s,v]);k.sort((a,b)=>a[0]-b[0]);i=k.findIndex(p=>p[0]===s);}else k[i][1]=v;
 speedDrag=i;invalidate();compiled=null;drawSpeed();e.currentTarget.setPointerCapture(e.pointerId);
});
$('speed-view').addEventListener('pointermove',e=>{if(speedDrag===null)return;const [x,y]=pointEvent(e,e.currentTarget),m=speedMap,k=arms[active].speed;const i=speedDrag;k[i][1]=Math.max(.001,Math.min(1,(1-(y-m.top)/m.ph)*m.max));if(i>0&&i<k.length-1)k[i][0]=Math.max(k[i-1][0]+.001,Math.min(k[i+1][0]-.001,(x-m.left)/m.pw));drawSpeed();});
$('speed-view').addEventListener('pointerup',()=>{if(speedDrag!==null)queueSync();speedDrag=null;});
$('observe').onclick=refresh;
for(const id of ['tolerance-position','tolerance-orientation'])$(id).oninput=()=>{toleranceDirty=true;invalidate();renderTolerance();updateControls();};
$('motion-profile').onchange=async()=>{
 if(busy||running())return;
 const profile=$('motion-profile').value;busy=true;invalidate();updateControls();
 try{const r=await api('settings','PUT',{motion_profile:profile});state.motion_profile=r.motion_profile;lastRemoteRevision=r.revision;state=await api('state');renderTolerance();message('运动档位已保存，请重新预览。');}
 catch(e){renderTolerance();message(e.message,true);}finally{busy=false;updateControls();}
};
$('apply-tolerance').onclick=async()=>{
 if(busy||running())return;busy=true;updateControls();
 try{await saveTolerance();message('位姿容差已保存，请按新设置重新预览轨迹。');}
 catch(e){message(e.message,true);}finally{busy=false;updateControls();}
};
for(const id of ['auto-refresh','auto-can'])$(id).onchange=async()=>{
 if(busy||running())return;busy=true;updateControls();
 try{
  const r=await api('settings','PUT',{auto_prepare:{refresh_observation:$('auto-refresh').checked,restore_can:$('auto-can').checked}});
  state.auto_prepare=r.auto_prepare;lastRemoteRevision=r.revision;preview=null;
  message('自动准备设置已保存，在下一次画线、预览或执行时生效。');
 }catch(e){message(e.message,true);}finally{busy=false;renderTolerance();updateControls();}
};
$('table-checks').onchange=async()=>{
 if(busy||running())return;busy=true;updateControls();
 try{
  const r=await api('settings','PUT',{table_checks:$('table-checks').checked});
  state.table_checks=r.table_checks;lastRemoteRevision=r.revision;preview=null;
  message(r.table_checks?'已开启桌面净空与桌面碰撞检测，请重新预览。':'已关闭桌面净空与桌面碰撞检测：规划不再拦截或抬高贴桌路径，请重新预览。');
 }catch(e){message(e.message,true);}finally{busy=false;renderTolerance();updateControls();}
};
$('tracking-limit').onchange=async()=>{
 if(busy||running())return;const value=Number($('tracking-limit').value);
 if(!Number.isFinite(value)||value<1||value>90)return message('跟踪误差上限需在 1–90 度之间。',true);
 busy=true;updateControls();
 try{const r=await api('settings','PUT',{tracking_limit_deg:value});state.tracking_limit_deg=r.tracking_limit_deg;lastRemoteRevision=r.revision;preview=null;message(`执行中的关节跟踪误差上限已设为 ${r.tracking_limit_deg}°，请重新预览。`);}
 catch(e){message(e.message,true);}finally{busy=false;renderTolerance();updateControls();}
};
$('gui-mode').onclick=()=>{setMode('gui');queueSync();};
$('api-mode').onclick=()=>{setMode('api');$('function-editor').value=JSON.stringify(functionArms[active],null,2);queueSync();};
for(const tool of ['pick-place','pour','brush','grasp','skill','select','edit'])$(tool+'-tool').onclick=()=>{
 if(busy||running())return;
 drawTool=tool;pickingStart=false;
 $('more-tools').open=false;
 for(const t of ['pick-place','pour','brush','grasp','skill','select','edit'])$(t+'-tool').setAttribute('aria-pressed',tool===t);$('grasp-options').hidden=tool!=='grasp';$('pick-place-options').hidden=tool!=='pick-place';$('pour-options').hidden=tool!=='pour';$('skill-options').hidden=tool!=='skill';$('select-options').hidden=tool!=='select';$('main-view').dataset.tool=tool;if(tool==='edit'){$('trajectory-editor').open=true;$('spatial-panel').open=true;}renderFields();drawAll();updateControls();
};
$('skill-name').onchange=()=>selectSkill();
$('undo-stroke').onclick=()=>{
 if(busy||running()||!strokeUndo[active])return;
 arms[active]=structuredClone(strokeUndo[active]);delete strokeUndo[active];selected=0;compiled=null;renderFields();drawAll();queueSync();$('brush-status').textContent='已恢复重画前的路径和姿态。';$('brush-status').classList.remove('error');
};
$('arm').onchange=()=>{active=$('arm').value;selected=0;$('function-editor').value=JSON.stringify(functionArms[active],null,2);renderFields();drawAll();};
for(const a of ['left','right'])$('arm-'+a).onclick=()=>{$('arm').value=a;pickingStart=false;$('arm').onchange();};
for(const a of ['left','right'])$('enable-'+a).onchange=()=>{queueSync();drawAll();updateControls();};
$('apply-function').onclick=async()=>{try{functionArms[active]=JSON.parse($('function-editor').value);await queueSync();renderFields();}catch(e){message(e.message,true);}};
$('function-editor').oninput=()=>{invalidate();};
$('side-axis').onchange=drawSide;
$('pick-mode').onchange=()=>{updateControls();$('brush-status').textContent=$('pick-mode').value==='surface'?'画线将读取桌面或物体表面的实际深度':'画线将落在指定的固定高度平面';};
$('scrub').oninput=()=>{$('phase').textContent=Math.round(Number($('scrub').value)*100)+'%';drawAll();};
['x','y','z'].forEach((k,i)=>$('point-'+k).onchange=()=>{const v=Number($('point-'+k).value);if(!Number.isFinite(v))return;arms[active].points[selected][i]=v;compiled=null;queueSync();drawAll();});
['roll','pitch','yaw'].forEach((k,i)=>$(k).onchange=()=>{const v=Number($(k).value);if(!$(k).value.trim()||!Number.isFinite(v))return;arms[active].poses[selected][i]=v;renderFields();queueSync();drawAll();});
$('orientation-enabled').onchange=()=>{const m=arms[active];m.orientationMode=$('orientation-enabled').value;m.orientation=m.orientationMode==='keyframes';renderFields();queueSync();};
$('pose-frame').onchange=drawPoseEditor;$('pose-scope').onchange=drawPoseEditor;
for(const [id,preset] of [['pose-current',()=>state.arms[active].rpy_deg.slice()],['pose-down',()=>[180,0,0]]])$(id).onclick=()=>{
 if(busy||running()||mode!=='gui')return;beginPoseEdit();changePose(preset());commitPoseEdit();
};
$('queue-approach-speed').onchange=()=>{
 const v=Number($('queue-approach-speed').value);
 if(!Number.isFinite(v)||v<.001||v>1){$('queue-approach-speed').value=String(arms[active].approach.speed_m_s);message('接近速度需为 0.001–1.0 m/s，已恢复上次有效值。',true);return;}
 arms[active].approach.speed_m_s=v;$('approach-speed').value=v;queueSync();
};
$('queue-home-speed').onchange=async()=>{
 try{const r=await api('settings','PUT',{home_speed_m_s:Number($('queue-home-speed').value)});state.home_speed_m_s=r.home_speed_m_s;lastRemoteRevision=r.revision;preview=null;message('回位速度已保存，用于后续所有技能。');}
 catch(e){$('queue-home-speed').value=state.home_speed_m_s??.8;message(e.message,true);}
};
$('base-speed').onchange=()=>{const v=Number($('base-speed').value);if(!Number.isFinite(v)||v<.001||v>1){$('base-speed').value=String(arms[active].speed[0][1]);message('移动速度需为 0.001–1.0 m/s，已恢复上次有效值。',true);return;}delete arms[active].speedSpec;arms[active].speed=arms[active].speed.map(k=>[k[0],v]);queueSync();};
for(const id of ['base-speed','approach-clearance'])$(id).oninput=()=>invalidate();
$('pick-start').onclick=()=>{pickingStart=!pickingStart;renderFields();};
for(const [id,key] of [['approach-speed','speed_m_s'],['approach-clearance','clearance_m']])$(id).onchange=()=>{
 const m=mode==='gui'?arms[active]:functionArms[active];(m.approach??={})[key]=Number($(id).value);
 if(id==='approach-speed')$('queue-approach-speed').value=$(id).value;
 if(mode==='api')$('function-editor').value=JSON.stringify(m,null,2);queueSync();
};
$('clear-paths').onclick=async()=>{
 if(clearingPaths)return;
 cancelDropSelection();
 clearingPaths=true;$('clear-paths').disabled=true;
 pendingStrokes=[];queuedDraftWaiting=false;updatePendingStrokes();
 automaticAwaitingFinish=false;finishNotice=null;
 if(brush)cancelBrush(true);
 if(automaticRun)automaticRun.cancelled=true;
 clearDualOrchestration();invalidate();updateControls();
 try{
  await api('tasks','DELETE');await api('cancel','POST');
  message('正在停止并清除轨迹…');
  const deadline=Date.now()+20000;
  do{
   state=await api('state');
   if(!running()&&!automaticRun&&!drainingStroke)break;
   if(Date.now()>deadline)throw Error('停止尚未完成，请稍后再清除轨迹');
   await new Promise(resolve=>setTimeout(resolve,150));
  }while(true);
  await syncChain;
  const cleared=await api('draft','DELETE');lastRemoteRevision=cleared.revision;
  selection=null;settingDrop=false;tasksView=null;strokeUndo={};
  for(const arm of ['left','right'])functionArms[arm]=defaultFunction(arm);
  clearExecutedPath();$('function-editor').value=JSON.stringify(functionArms[active],null,2);
  $('plan-notes').replaceChildren();$('brush-status').textContent='';$('duration').textContent='尚未预览';
  renderQueuePanel();message('已清除双臂轨迹、预览和待执行笔画。');
 }catch(e){message('清除未完成：'+e.message,true);}
 finally{clearingPaths=false;$('clear-paths').disabled=false;updateControls();drawAll();}
};
$('reset-path').onclick=()=>{arms[active]=freshArm(active);delete strokeUndo[active];selected=0;compiled=null;renderFields();queueSync();drawAll();};
$('remove-point').onclick=()=>{if(selected===0)return;arms[active].points.splice(selected,1);arms[active].poses.splice(selected,1);selected=Math.max(0,selected-1);compiled=null;renderFields();queueSync();drawAll();};
$('choose-endpoint').onclick=async()=>{
 if(busy||running()||mode!=='gui')return;
 busy=true;updateControls();
 try{
  await syncChain;invalidate();const arm=active,revision=lastRemoteRevision;
  const choice=await endpointDialog.choose({arm,revision});
  if(choice.action===null){message(choice.reason||'已取消终点动作选择。');return;}
  const latest=await api('state');
  if(latest.revision!==revision||['running','planning','cancelling'].includes(latest.execution.state)||state.revision>revision||(state.revision===revision&&running()))throw Error('草稿或机器人状态已变化，请重新选择。');
  const m=arms[arm];m.events=withEndpointAction(m.events,choice.action,{graspLine:m.graspLine});
  if(m.graspLine)m.startHold=choice.action==='grasp'?.6:0;
  compiled=null;renderFields();drawAll();await queueSync();
 }catch(e){message(e.message,true);}finally{busy=false;renderFields();drawAll();}
};
$('add-event').onclick=()=>{const e={s:Number($('event-s').value),opening_mm:Number($('event-opening').value)};if(!(e.s>=0&&e.s<=1&&e.opening_mm>=0&&e.opening_mm<=70))return message('夹爪事件进度须在 0–1，开度在 0–70 mm。',true);const m=mode==='gui'?arms[active]:functionArms[active];const key=mode==='gui'?'events':'gripper_events';(m[key]??=[]).push(e);m[key].sort((a,b)=>a.s-b.s);if(mode==='api')$('function-editor').value=JSON.stringify(m,null,2);renderFields();queueSync();};
$('preview').onclick=async()=>{
 if(busy||running())return;busy=true;updateControls();message('正在准备轨迹检查…稍后显示具体阶段与进度。');
 const cancelEpoch=state.cancel_epoch;
 try{
  if(mode==='api')functionArms[active]=JSON.parse($('function-editor').value);
  if(toleranceDirty)await saveTolerance();
  await syncChain;const d=draft(),issue=draftIssue(d);if(issue)throw Error(issue);
  const saved=await api('draft','PUT',d);lastRemoteRevision=saved.revision;
  compiled=await api('compile','POST');drawAll();
  preview=await api('preview','POST',{expected_revision:saved.revision,expected_cancel_epoch:cancelEpoch});await adoptPreparedPlan(preview);compiled=preview;drawAll();
  $('duration').textContent=`总计 ${preview.duration_s.toFixed(2)} s${preview.approach_duration_s>0?' · 接近 '+preview.approach_duration_s.toFixed(2)+' s':''}`;
  $('plan-notes').replaceChildren();for(const note of preview.notes){const p=document.createElement('p');p.textContent=note;$('plan-notes').append(p);}
  message(state.backend.mode==='real'?'实机轨迹预览通过。点击执行后机器人将运动。':'模拟轨迹预览通过。');
 }catch(e){preview=null;message(e.message,true);}finally{busy=false;state=await api('state').catch(()=>state);updateControls();}
};
$('execute').onclick=async()=>{if(!preview||busy)return;busy=true;updateControls();try{
 const p=preview;preview=null;
 await executePreparedPreview({api,plan:p,cancelEpoch:state.cancel_epoch,onPreview:async plan=>{await adoptPreparedPlan(plan);compiled=plan;drawAll();}});
 message('轨迹执行中，机器人本地控制时序。');
}catch(e){message(e.message,true);}finally{busy=false;await poll();}};
async function recoverExecution(returnHome=false){
 if(busy||running())return;
 busy=true;preview=null;pendingStrokes=[];queuedDraftWaiting=false;updatePendingStrokes();
 automaticAwaitingFinish=false;finishNotice=null;updateControls();
 try{
  await syncChain.catch(()=>{});
  await api('recover','POST');
  state=await api('state');lastRemoteRevision=state.revision;
  if(returnHome){
   await api('home','POST',{arms:['left','right'],cancel_epoch:state.cancel_epoch,open_grippers:false});
   automaticAwaitingFinish=true;finishNotice='回原位完成，可以重新观测并开始。';
   message('已清错，保持夹爪开度并回原位…');
  }else message('错误已清除，请重新观测、规划后执行。');
 }catch(e){message(e.message,true);}
 finally{busy=false;await poll();updateControls();}
}
$('recover').onclick=()=>recoverExecution();
$('recover-home').onclick=()=>recoverExecution(true);
$('home').onclick=async()=>{
 if(busy||running())return;
 // The top bar has no arm selector beside it, so this is a whole-robot reset: an arm already at zero
 // simply does not move, and one left out in the workspace is the reason the button was reached for.
 const chosen=['left','right'];
 busy=true;preview=null;updateControls();message('正在准备回原位：先张开夹爪，再关节空间回到原位…');
 try{
  await syncChain.catch(()=>{});
  await api('home','POST',{arms:chosen,cancel_epoch:state.cancel_epoch});
  finishNotice='回原位已完成，图像已更新。';automaticAwaitingFinish=true;
  message(`回原位执行中：${chosen.map(a=>a==='left'?'左臂':'右臂').join('、')}先张开夹爪，再回到原位。`);
 }catch(e){message(e.message,true);}
 finally{busy=false;await poll();updateControls();}
};
$('cancel').onclick=async()=>{cancelDropSelection();pendingStrokes=[];queuedDraftWaiting=false;updatePendingStrokes();if(brush)cancelBrush(true);if(automaticRun)automaticRun.cancelled=true;clearDualOrchestration();try{const checking=state?.execution?.kind==='preview';await api('cancel','POST');message(checking?'正在取消轨迹检查…':'正在停止后续轨迹并保持当前位置…');await poll();}catch(e){message(e.message,true);}};

async function adoptRemoteDraft(d,expectedVersion=editVersion,allowBusy=false){
 const canApply=()=>(allowBusy||!busy)&&editVersion===expectedVersion;
 if(!canApply())return false;
 if(!d?.arms||!Object.keys(d.arms).length){clearExecutedPath();return true;}
 let nextFrame=frame,nextPhoto=photo,nextCompiled=null,notice=draftIssue(d);
 if(!frame||frame.id!==d.observation_id){nextFrame=await api('observation');nextPhoto=await observationImage(nextFrame);}
 if(!notice){try{nextCompiled=await api('compile','POST');}catch(e){notice=e.message;}}
 // A pending remote read must not replace a stroke started while awaiting it.
 if(!canApply()||nextFrame.id!==d.observation_id)return false;
 if(nextFrame!==frame)installObservation(nextFrame,nextPhoto);
 compiled=nextCompiled;preview=null;strokeUndo={};
 for(const a of ['left','right']){
  $('enable-'+a).checked=Boolean(d.arms[a]);if(!d.arms[a])continue;
  functionArms[a]=d.arms[a];
  if(d.arms[a].path.mode==='waypoints'){
   const spec=d.arms[a],points=spec.path.points,curve=compiled?.arms?.[a];
   arms[a]={points,poses:points.map(()=>state.arms[a].rpy_deg.slice()),orientation:spec.orientation?.mode==='keyframes',orientationMode:spec.orientation?.mode||'hold',
    speed:Array.isArray(spec.speed)?spec.speed:Number.isFinite(Number(spec.speed))&&Number(spec.speed)>0?[[0,Number(spec.speed)],[1,Number(spec.speed)]]:curve?curve.s.flatMap((s,i)=>i%25===0?[[s,curve.requested_speed_m_s[i]]]:[]):[[0,.8],[1,.8]],speedSpec:spec.speed,
    bottleCap:spec.bottle_cap,angularSpeed:spec.angular_speed_deg_s,events:spec.gripper_events||[],startHold:spec.start_hold_s||0,graspLine:spec.grasp_line===true,airTrack:spec.air_track===true,approach:spec.approach||{speed_m_s:.8,clearance_m:.12}};
   if(spec.orientation?.mode==='keyframes')arms[a].poses=points.map((_,i)=>sampleOrientation(spec.orientation.points,i/Math.max(1,points.length-1)));
  }
 }
 const next=Object.values(d.arms).some(a=>a.path.mode==='function'||a.orientation?.mode==='function')?'api':'gui';setMode(next);
 active=Object.keys(d.arms)[0];$('arm').value=active;$('function-editor').value=JSON.stringify(functionArms[active],null,2);renderFields();drawAll();updateControls();
 const failed=state?.execution?.state==='error';message(failed?state.execution.error:notice||'已同步 API 更新的共享草稿。',failed);return true;
}
/** The service drops a trajectory once it has run; the page stops drawing it at the same moment. */
function clearExecutedPath(){
 if(!state)return;
 for(const a of ['left','right'])arms[a]=freshArm(a);
 strokeUndo={};compiled=null;preview=null;selected=0;
 renderFields();drawAll();updateControls();
}
function updatePendingStrokes(){
 $('pending-strokes').textContent=pendingStrokes.length?`待执行笔画：${pendingStrokes.length} 笔 · 停止会清空队列`:'';
}
async function drainPendingStroke(){
 if(clearingPaths||!pendingStrokes.length||drainingStroke||busy||brush||automaticRun)return;
 if(['error','cancelled','cancelling'].includes(state.execution.state)||pendingStrokes[0].cancelEpoch!==state.cancel_epoch){
  pendingStrokes=[];queuedDraftWaiting=false;updatePendingStrokes();return;
 }
 if(running()){queuedDraftWaiting=false;return;}
 if(queuedDraftWaiting)return;
 drainingStroke=true;busy=true;
 const stroke=pendingStrokes.shift();brush=stroke;updatePendingStrokes();
 try{
  const fresh=await api('observe','POST',{preserve_draft:true});
  await adoptFrozenObservation(fresh);
  if(stroke.cancelEpoch!==state.cancel_epoch||stroke.cancelled)throw Error('排队笔画已取消');
  if(running())throw Error('机器人已有新动作，排队笔画已停止');
  active=stroke.arm;$('arm').value=active;
  arms[active].speed=structuredClone(stroke.motion.speed);
  arms[active].approach=structuredClone(stroke.motion.approach);
  stroke.deferred=false;stroke.observationId=frame.id;stroke.finished=true;
  brush=stroke;
  await finishStroke(stroke);
  state=await api('state');
  // A generated draft awaiting endpoint choice/manual approval must not be overwritten by the next stroke.
  queuedDraftWaiting=!running();
  if(queuedDraftWaiting){pendingStrokes=[];updatePendingStrokes();}
 }catch(e){pendingStrokes=[];message(e.message,true);updatePendingStrokes();}
 finally{brush=null;busy=false;drainingStroke=false;updateControls();drawAll();}
}
async function poll(){
 try{
  const before=state?.execution?.state;state=await api('state');
  endpointDialog?.cancelIfChanged(state);
  renderTolerance();
  $('connection').textContent=state.backend.mode==='real'?'● 实机':'● 模拟环境';
  const names={idle:'待机',ready:'预览就绪',planning:'轨迹检查中',running:'执行中',cancelling:'正在保持',cancelled:'已取消并保持',completed:'执行完成',error:'需要调整'};
  $('execution-state').textContent=names[state.execution.state]||state.execution.state;
  if(state.execution.kind==='preview'&&state.execution.state==='cancelling')$('execution-state').textContent='取消检查中';
  if(state.execution.kind==='preview'&&state.execution.state==='cancelled')$('execution-state').textContent='检查已取消';
  $('execution-progress').value=state.execution.progress||0;
  $('arm-status').replaceChildren();for(const a of ['left','right']){const row=document.createElement('div');row.className='robot-row';const name=document.createElement('span');name.textContent=a==='left'?'左臂':'右臂';name.style.color=colors[a];const txt=document.createElement('code');txt.textContent=state.arms[a].xyz.map(v=>v.toFixed(3)).join(' / ');row.append(name,txt);$('arm-status').append(row);}
  $('capabilities').textContent=`${state.backend.mode==='real'?'实机连接':'模拟模式'} · ${state.backend.motion_allowed?'运动接口已启用':'运动接口未启用'} · 位置控制`;
  if(state.execution.state!==before){
   if(state.execution.state==='completed'){
    if(state.execution.draft_cleared&&!brush)clearExecutedPath();
    message(state.execution.result?.holding_position?'动作行程已完成，末端保持；请检查实际效果后继续。':state.execution.draft_cleared?'轨迹执行完成，末端已到位。这条轨迹已清除，可以直接画下一笔。'
                                         :'轨迹执行完成，末端已到位。更新图像会保留路径，可从当前位置重新预览。');
   }
   if(state.execution.state==='cancelled')message(state.execution.kind==='preview'?'轨迹检查已取消，草稿保留。':'轨迹已取消，保持当前位置。');
   if(state.execution.state==='error'&&!busy)message(state.execution.error,true);
  }
  renderPlanning();
  if(state.observation_stale&&!running()&&!busy){if(!state.auto_prepare?.refresh_observation)preview=null;if(['idle','ready'].includes(state.execution.state))message(state.auto_prepare?.refresh_observation?'机器人位置已变化，下次画线、预览或执行时将自动刷新观测并保留路径。':'机器人位置已变化。更新图像会保留路径，再从当前位置重新预览。');}
  if(!busy&&!running()&&lastRemoteRevision>=0&&state.revision!==lastRemoteRevision){const revision=state.revision,version=editVersion;if(await adoptRemoteDraft(await api('draft'),version))lastRemoteRevision=revision;}
  if(state.execution.state==='error'&&!busy)message(state.execution.error,true);
  if(automaticAwaitingFinish&&!busy&&['completed','cancelled','error'].includes(state.execution.state)){automaticAwaitingFinish=false;const notice=finishNotice;finishNotice=null;if(state.execution.state==='completed'){await refresh();message(notice||'自动动作已执行完成，图像已更新。可以继续画下一笔。');}}
  updateControls();
  await drainPendingStroke();
 }catch(e){$('connection').textContent='连接中断';$('execute').disabled=true;message('连接失败：'+e.message,true);}
}
setInterval(()=>poll(),1200);
// Live RGB only changes what the canvas shows; geometry always comes from the observation frame.
livePainter=new LivePainter({paint:drawMain,status:(s,stats)=>{$('image-mode').textContent=liveStatusLabel(s,stats);}});
$('live').onchange=()=>{if($('live').checked)livePainter.start();else{livePainter.stop();drawMain();}};
// The stream starts with the page; the checkbox turns it off. Geometry still comes from the observation frozen
// when the pen goes down.
if($('live').checked)livePainter.start();
// ---- Toss items and the sequential task queue ----
let selection=null,settingDrop=false,tasksView=null,pendingDrop=null;
function cancelDropSelection(){
 const pending=pendingDrop;pendingDrop=null;settingDrop=false;if(pending)pending.resolve(null);
}
function requestDropPoint(observationId,index,total){
 if(pendingDrop)throw Error('请先完成或取消本次终点选择');
 if(frame?.id!==observationId)throw Error('观测已变化，请重新画抓取线');
 busy=true;settingDrop=true;
 const result=new Promise(resolve=>{pendingDrop={resolve,observationId,index,total};});
 renderQueuePanel();updateControls();
 message(`请点击第 ${index+1}/${total} 项的投放终点；这次点选只用于这一项。Esc 取消。`);
 return result;
}
function selectionVisible(){return Boolean(selection&&frame&&selection.observationId===frame.id&&drawTool==='select');}
function selectionPixelToCanvas(p){return mainRect?[mainRect.x+p[0]*mainRect.scale,mainRect.y+p[1]*mainRect.scale]:null;}
function drawQueueOverlay(c){
 if(!mainRect||!frame||!selectionVisible())return;
 const polygon=selection.polygon.map(selectionPixelToCanvas);line(c,[...polygon,polygon[0]],'#dce7f1',1.5,[6,4]);
 c.save();c.font='600 12px sans-serif';c.textAlign='center';c.textBaseline='middle';
 for(const object of selection.objects){
  const q=selectionPixelToCanvas(object.pixel);if(!q)continue;
  const chosen=selection.selected.has(object.index),color=object.graspable?colors[object.arm]||'#dce7f1':'#8a949f';
  c.beginPath();c.arc(q[0],q[1],11,0,2*Math.PI);c.fillStyle=chosen?color:'#101419d9';c.fill();c.strokeStyle=color;c.lineWidth=2;c.stroke();
  c.fillStyle=chosen?'#101419':color;c.font='600 12px sans-serif';c.textAlign='center';c.fillText(String(object.index+1),q[0],q[1]);
  if(!object.graspable){c.textAlign='left';c.font='11px sans-serif';c.fillStyle='#8a949f';c.fillText(object.reason,q[0]+15,q[1]);}
 }
 c.restore();
}
function beginSelectPointer(e){
 const pixel=brushPixel(e);if(!pixel)return;
 // Record a direct grasp line against the observation frozen on pointer down.
 brush={arm:active,tool:'select',pointerId:e.pointerId,pixels:[pixel],observationId:frame.id,finished:false};
 if(livePainter?.running||state.observation_stale){brush.freeze=freezeObservation();brush.freeze.catch(()=>{});}
 busy=true;e.currentTarget.setPointerCapture(e.pointerId);updateControls();drawMain();
 message('正在画抓取线…松开后点击本次投放终点。');
}
async function placeDropPoint(e){
 const pending=pendingDrop,pixel=brushPixel(e);if(!pending||!pixel)return;
 settingDrop=false;updateControls();
 try{
  if(frame?.id!==pending.observationId)throw Error('观测已变化，请取消后重新画抓取线');
  const picked=await api('pick','POST',{observation_id:pending.observationId,u:pixel[0],v:pixel[1],mode:'surface'});
  if(pendingDrop!==pending)return;
  pendingDrop=null;pending.resolve(picked.xyz.slice());
 }catch(err){
  if(pendingDrop===pending){settingDrop=true;message('终点未选中：'+err.message+'。请重新点击，或按 Esc 取消。',true);}
 }finally{renderQueuePanel();updateControls();drawAll();}
}
async function finishSelect(stroke){
 try{
  if(stroke.pixels.length<2)throw Error('请画一条抓取线，线中点为抓取位置');
  const pixels=[stroke.pixels[0],stroke.pixels.at(-1)];
  const pixel=pixels[0].map((v,i)=>(v+pixels[1][i])/2);
  await submitTasks({observation_id:stroke.observationId,
   speed_m_s:Number($('queue-grab-speed').value),clearance_m:Number($('approach-clearance').value),
   toss_style:$('queue-toss-style').value,
   items:[{object_pixel:pixel,grasp_pixels:pixels,arm:stroke.arm,placement:'toss',label:'画线抓取投放'}]});
 }catch(err){message(err.message,true);}
 finally{brush=null;busy=false;renderQueuePanel();updateControls();drawAll();}
}

async function detectAllObjects(){
 if(!frame)throw Error('请先更新图像');
 const result=await api('objects','POST',{observation_id:frame.id,all:true});
 if(frame.id!==result.observation_id)throw Error('观测已更新，请重新识别。');
 const graspable=result.objects.filter(o=>o.graspable);
 selection={observationId:result.observation_id,polygon:[[0,0],[frame.width-1,0],[frame.width-1,frame.height-1],[0,frame.height-1]],objects:result.objects,selected:new Set(graspable.map(o=>o.index))};
 if(!graspable.length){message(result.objects.length?'桌面上找到的凸起都不能夹（机械臂本体或太宽）。':'桌面上没有找到凸起的物体。',true);return;}
 message(`桌面 ${result.objects.length} 个凸起，可夹 ${graspable.length} 个，已全部选中。点编号切换选中；点“投放选中物体”后逐项选择终点。`);
}
async function submitTasks(body){
 body={...body,items:body.items.map(item=>({...item}))};
 for(let i=0;i<body.items.length;i++){
  if(body.items[i].landing_xyz)continue;
  const landing=await requestDropPoint(body.observation_id,i,body.items.length);
  if(!landing)throw Error('本次投放已取消，未启动执行。');
  body.items[i].landing_xyz=landing;
 }
 await syncChain;
 if(toleranceDirty)await saveTolerance();
 body={...body,speed_m_s:Number($('queue-grab-speed').value),approach_speed_m_s:Number($('queue-toss-approach-speed').value),toss_speed_m_s:Number($('queue-throw-speed').value),
  carry_speed_m_s:Number($('queue-carry-speed').value),windup_speed_m_s:Number($('queue-windup-speed').value),
  home_speed_m_s:Number($('queue-toss-home-speed').value),grasp_hold_s:Number($('queue-grasp-hold').value)};
 tasksView=await api('tasks','POST',body);
 state=await api('state');
 message(`已提交 ${body.items.length} 个投放任务，队列开始执行：观测 → 定位 → 规划 → 执行。`);
}
function renderQueuePanel(){
 if(selection&&frame&&selection.observationId!==frame.id)selection=null;
 const graspable=selection?.objects?.filter(o=>o.graspable).length??0;
 $('selection-status').textContent=selection?.objects?.length?`识别到 ${selection.objects.length} 个物体 · 已选 ${selection.selected.size}/${graspable}`:'';
 $('drop-point').setAttribute('aria-pressed',settingDrop);
 $('drop-point').textContent=pendingDrop?'取消本次终点选择':'画线后点选终点';
 $('drop-point-status').textContent=pendingDrop?`等待第 ${pendingDrop.index+1}/${pendingDrop.total} 项终点：点击主画面`:'每次投放均需重新点击终点，不沿用上次位置';
 const list=$('queue-list');list.replaceChildren();
 const items=tasksView?.items||[];
 $('queue-panel').hidden=drawTool!=='select'&&!items.length&&!state?.task_queue?.active;
 for(const item of items){
  const row=describeItem(item),el=document.createElement('div');el.className='queue-row '+row.tone;
  const head=document.createElement('span');head.textContent=`${row.number} · ${row.arm} · ${row.status}`;head.style.color=colors[item.arm];
  const detail=document.createElement('span');detail.className='queue-detail';detail.textContent=row.detail;detail.title=row.detail;
  el.append(head,detail);list.append(el);
 }
 if(!items.length){const p=document.createElement('p');p.className='hint';p.textContent='画抓取线后，点击本次投放终点；不要求物体凸起。';list.append(p);}
 $('queue-summary').textContent=summarizeQueue(tasksView);
 const locked=busy||running();
 $('queue-stop').disabled=!state?.task_queue?.active;
 $('queue-drop-selected').disabled=locked||!selection?.selected?.size||selection.observationId!==frame?.id;
 $('queue-clear').disabled=locked||!selection;
 $('drop-point').disabled=!pendingDrop;
 $('detect-all').disabled=locked||mode!=='gui'||!frame;
}
async function pollTasks(){
 if(!state)return;
 try{
  const next=await api('tasks');const finished=queueFinished(tasksView,next);tasksView=next;
  if(!finished){renderQueuePanel();return;}
  state=await api('state');selection=null;renderQueuePanel();updateControls();
  const summary=summarizeQueue(next),failed=next.items.find(item=>item.status==='failed');
  await refresh();
  message(failed?`${summary} · 第 ${failed.index+1} 项：${failed.error}`:`${summary}，图像已更新。`,Boolean(failed));
 }catch{}
}
setInterval(pollTasks,1000);
$('drop-point').onclick=()=>{cancelDropSelection();renderQueuePanel();updateControls();};
$('queue-stop').onclick=async()=>{cancelDropSelection();pendingStrokes=[];queuedDraftWaiting=false;updatePendingStrokes();
 try{tasksView=await api('tasks','DELETE');state=await api('state');message('正在停止队列并保持当前位置…');}
 catch(err){message(err.message,true);}finally{renderQueuePanel();updateControls();}
};
$('queue-drop-selected').onclick=async()=>{
 if(busy||running()||!selection)return;busy=true;updateControls();
 try{await submitTasks(buildTaskRequest({observationId:frame.id,objects:selection.objects,selected:selection.selected,placement:'toss',tossStyle:$('queue-toss-style').value}));}
 catch(err){message(err.message,true);}finally{busy=false;renderQueuePanel();updateControls();drawAll();}
};
$('queue-clear').onclick=()=>{selection=null;renderQueuePanel();drawAll();message('已清除选择。');};
$('detect-all').onclick=async()=>{
 if(busy||running()||!frame)return;if(drawTool!=='select')$('select-tool').click();busy=true;updateControls();
 try{await detectAllObjects();}catch(err){message(err.message,true);}finally{busy=false;renderQueuePanel();updateControls();drawAll();}
};
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&pendingDrop){cancelDropSelection();renderQueuePanel();message('已取消本次投放。');}});

new ResizeObserver(drawAll).observe(document.querySelector('.workspace'));
document.querySelectorAll('.disclosure').forEach(el=>el.addEventListener('toggle',()=>requestAnimationFrame(drawAll)));
endpointDialog=new EndpointActionDialog($('endpoint-choice'));
const AUTO_EXECUTE_KEY='urai.autoExecute';
function storedAutoExecute(){try{return localStorage.getItem(AUTO_EXECUTE_KEY)!=='off';}catch{return true;}}
$('auto-execute').checked=storedAutoExecute();
$('auto-execute').onchange=()=>{
 const on=$('auto-execute').checked;
 try{localStorage.setItem(AUTO_EXECUTE_KEY,on?'on':'off');}catch{}
 updateSkillUI();message(on?'画完自动执行已开启。两点抓放、倒水和原子技能会自动完成，轨迹画笔与抓取线在选择终点动作后执行。':'画完自动执行已关闭：所有单笔技能只生成草稿，手动预览后执行。');
};
const DUAL_ARM_KEY='urai.dualArm';
function storedDualArm(){try{return localStorage.getItem(DUAL_ARM_KEY)==='on';}catch{return false;}}
$('dual-arm').checked=storedDualArm();
$('dual-arm').onchange=()=>{
 const on=$('dual-arm').checked;
 try{localStorage.setItem(DUAL_ARM_KEY,on?'on':'off');}catch{}
 // A half-drawn pair does not survive the switch: either way the count starts again from no arm.
 clearDualOrchestration();
 updateSkillUI();
 message(on?'双臂已开启：先画一条臂，再画另一条，两笔都画完才一起执行，两只手同时动作。'
          :'双臂已关闭：每画完一笔就按单臂处理。');
};
poseEditor=new GripperPoseEditor($('pose-view'),{begin:beginPoseEdit,change:changePose,commit:commitPoseEdit,cancel:cancelPoseEdit});
await poll();
await loadSkills();
if(state){
 try{
  await loadMotionProfiles();
  if(state.observation_id){await loadObservation(await api('observation'));arms={left:freshArm('left'),right:freshArm('right')};functionArms={left:defaultFunction('left'),right:defaultFunction('right')};const d=await api('draft');if(d.arms)await adoptRemoteDraft(d);else{renderFields();drawAll();}}
  else await refresh();
  lastRemoteRevision=state.revision;
 }catch(e){message(e.message,true);}
}
