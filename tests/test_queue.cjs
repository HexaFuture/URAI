// Toss request bodies and the task queue panel: node --test tests/test_queue.cjs
const test=require('node:test');
const assert=require('node:assert/strict');

async function load(){return import('../urai/static/queue.mjs');}

test('task request only submits selected graspable objects with landing and placement',async()=>{
 const {buildTaskRequest}=await load();
 const objects=[{index:0,pixel:[10,10],graspable:true},{index:1,pixel:[20,20],graspable:false,reason:'太宽'},{index:2,pixel:[30,30],graspable:true}];
 const body=buildTaskRequest({observationId:'obs',objects,selected:new Set([2,1,0]),landing:[.4,.1,0],placement:'drop'});
 assert.deepEqual(body,{observation_id:'obs',items:[
  {object_pixel:[10,10],label:'物体 1',landing_xyz:[.4,.1,0],placement:'drop'},
  {object_pixel:[30,30],label:'物体 3',landing_xyz:[.4,.1,0],placement:'drop'}]});
 body.items[0].object_pixel[0]=99;assert.equal(objects[0].pixel[0],10);
 const drop=buildTaskRequest({observationId:'obs',objects,selected:new Set([0]),arm:'left'});
 assert.deepEqual(drop.items,[{object_pixel:[10,10],label:'物体 1',arm:'left'}]);
 assert.throws(()=>buildTaskRequest({observationId:'obs',objects,selected:new Set([1])}),/可夹/);
 for(const tossStyle of ['overhand','sidearm']){
  const toss=buildTaskRequest({observationId:'obs',objects,selected:new Set([0]),placement:'toss',tossStyle});
  assert.deepEqual(toss,{observation_id:'obs',toss_style:tossStyle,items:[{object_pixel:[10,10],label:'物体 1',placement:'toss'}]});
 }
});

test('panel rows map queue states to operator labels and keep failure reasons',async()=>{
 const {describeItem,STATUS_LABELS}=await load();
 assert.deepEqual(Object.keys(STATUS_LABELS),['queued','observing','planning','running','completed','failed','skipped','cancelled']);
 const row=describeItem({index:2,arm:'left',status:'failed',error:'放置位置被其他物体占用',landing_source:'drop_point',label:'物体 3'});
 assert.equal(row.number,3);assert.equal(row.arm,'左臂');assert.equal(row.status,'失败');
 assert.equal(row.detail,'放置位置被其他物体占用');assert.equal(row.tone,'failed');
 const queued=describeItem({index:0,arm:'right',status:'queued',error:null,landing_source:'item',label:'物体 1'});
 assert.equal(queued.status,'排队');assert.equal(queued.detail,'落点');assert.equal(queued.tone,'pending');
 assert.equal(describeItem({index:0,arm:'right',status:'completed',error:null,landing_source:'drop_point'}).detail,'投放点');
 assert.equal(describeItem({index:0,arm:'right',status:'running',error:null}).tone,'active');
 assert.equal(describeItem({index:0,arm:'right',status:'skipped',error:null}).tone,'muted');
});

test('queue completion is detected once and summarised',async()=>{
 const {queueFinished,summarizeQueue}=await load();
 const running={active:true,items:[{status:'completed'},{status:'running'}]};
 const done={active:false,items:[{status:'completed'},{status:'failed',error:'x'}]};
 assert.equal(queueFinished(null,running),false);
 assert.equal(queueFinished(running,running),false);
 assert.equal(queueFinished(running,done),true);
 assert.equal(queueFinished(done,done),false);
 assert.equal(summarizeQueue(done),'队列结束：完成 1 · 失败 1');
 assert.equal(summarizeQueue({active:false,items:[{status:'completed'},{status:'completed'}]}),'队列结束：完成 2');
 assert.equal(summarizeQueue({active:true,items:[{status:'completed'},{status:'running'},{status:'queued'}]}),'队列执行中 1/3 · 第 2 项执行');
 assert.equal(summarizeQueue({active:false,items:[]}),'');
});
