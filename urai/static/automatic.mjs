/** One explicit stroke owns one draft and one single-use execution request. */
const LABELS={pour:'自动倒水','pick-place':'两点抓放'};

function cancellation(isCancelled,label){
 return ()=>{if(isCancelled())throw Error(`${label}已取消`);};
}
function consistent(current,{observationId,revision,cancelEpoch}){
 return current.observation_id===observationId&&current.revision===revision&&current.cancel_epoch===cancelEpoch&&!['planning','running','cancelling'].includes(current.execution.state);
}
async function previewThenExecute({api,revision,cancelEpoch,check,onPreview}){
 const plan=await api('preview','POST',{expected_revision:revision,expected_cancel_epoch:cancelEpoch});check();
 await onPreview(plan);check();
 // Never retry this request: a lost response does not mean the robot did not start.
 await api('execute','POST',{preview_id:plan.id,cancel_epoch:cancelEpoch});
 return plan;
}

/** The server interprets the stroke (pick-place, pour or a named skill); the draft is then saved, previewed and
 * executed once. A bimanual skill answers with one draft per arm, every other route with the stroke arm's draft. */
export async function runStrokeTask({api,route,request,revision,cancelEpoch,isCancelled,onDraft,onPreview,
                                     execute=true,merge=false,approachSpeed}){
 const check=cancellation(isCancelled,LABELS[route]||'自动执行');
 check();
 const result=await api(route,'POST',request);check();
 const arms=result.arms||{[request.arm]:result.arm};
 if(approachSpeed!==undefined){
  if(!Number.isFinite(approachSpeed)||approachSpeed<.001||approachSpeed>1)throw Error('接近速度需为 0.001–1.0 m/s');
  // A bottle-cap stage keeps its own slower approach.
  for(const spec of Object.values(arms))spec.approach={...spec.approach,speed_m_s:spec.bottle_cap?Math.min(spec.approach?.speed_m_s??approachSpeed,.15):approachSpeed};
 }
 const current=await api('state');check();
 if(result.observation_id!==request.observation_id||!consistent(current,{observationId:request.observation_id,revision,cancelEpoch}))throw Error('观测、草稿或机器人状态已变化，请重新画线');
 // `merge` keeps the arms this draft does not name, so a two-armed move can be drawn one arm at a time.
 const saved=await api('draft','PUT',{observation_id:request.observation_id,arms,expected_revision:revision,
                                      ...(merge?{merge:true}:{})});check();
 onDraft(result,saved);check();
 // `execute` can be a decision instead of a flag: the second arm of a pair is what turns the run on.
 if(!(typeof execute==='function'?execute(arms):execute))return {result,plan:null};
 const plan=await previewThenExecute({api,revision:saved.revision,cancelEpoch,check,onPreview});
 return {result,plan};
}

/** A draft the page already computed (a brush stroke) runs the same guarded single execution. */
export async function runDraftExecution({api,draft,revision,cancelEpoch,isCancelled,onSaved,onPreview}){
 const check=cancellation(isCancelled,'画线即执行');
 check();
 const current=await api('state');check();
 if(!consistent(current,{observationId:draft.observation_id,revision,cancelEpoch}))throw Error('观测、草稿或机器人状态已变化，请重新画线');
 const saved=await api('draft','PUT',{...draft,expected_revision:revision});check();
 await onSaved(saved);check();
 const plan=await previewThenExecute({api,revision:saved.revision,cancelEpoch,check,onPreview});
 return {saved,plan};
}

/** Re-prepare a requested manual action before its single execute request. */
export async function executePreparedPreview({api,plan,cancelEpoch,onPreview}){
 const current=await api('state');
 if(current.revision!==plan.revision||current.observation_id!==plan.observation_id||current.cancel_epoch!==cancelEpoch)throw Error('观测、草稿或取消状态已变化，请重新预览');
 const options=current.auto_prepare||{};
 const recover=current.observation_stale&&options.refresh_observation||options.restore_can&&Object.values(current.arms||{}).some(a=>a.ctrl_mode?.includes('TEACHING'));
 if(recover){
  plan=await api('preview','POST',{expected_revision:plan.revision,expected_cancel_epoch:cancelEpoch});
  await onPreview(plan);
 }
 await api('execute','POST',{preview_id:plan.id,cancel_epoch:cancelEpoch});
 return plan;
}
