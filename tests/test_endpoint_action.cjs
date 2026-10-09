// End-of-path gripper action of a brush stroke: node --test tests/test_endpoint_action.cjs
const test=require('node:test');
const assert=require('node:assert/strict');
const source=()=>import('../urai/static/endpoint.mjs');

test('endpoint choice replaces previous endpoint commands and preserves interior events',async()=>{
 const {withEndpointAction}=await source();
 const before=[{s:0,opening_mm:70},{s:.6,opening_mm:25},{s:1,opening_mm:0},{s:1,opening_mm:45}],saved=structuredClone(before);
 assert.deepEqual(withEndpointAction(before,'release'),[...before.slice(0,2),{s:1,opening_mm:70,hold_s:.6,wait_for_arrival:true}]);
 assert.deepEqual(withEndpointAction(before,'grasp'),[...before.slice(0,2),{s:1,opening_mm:0,hold_s:.6,wait_for_arrival:true}]);
 assert.deepEqual(withEndpointAction(before,'none'),before.slice(0,2));
 assert.deepEqual(before,saved);
 assert.throws(()=>withEndpointAction(before,'unknown'));
});

test('release or hold on a grasp line removes template preopening and grasp restores it',async()=>{
 const {withEndpointAction}=await source();
 const before=[{s:0,opening_mm:70,pregrasp_open:true},{s:.5,opening_mm:25},{s:1,opening_mm:0}];
 const release=withEndpointAction(before,'release',{graspLine:true});
 assert.deepEqual(release.map(e=>e.s),[.5,1]);
 assert.equal(release.at(-1).opening_mm,70);assert.equal(release.at(-1).wait_for_arrival,true);
 assert.ok(release.at(-1).hold_s>=.6);
 const grasp=withEndpointAction(release,'grasp',{graspLine:true});
 assert.equal(grasp[0].pregrasp_open,true);assert.equal(grasp.at(-1).opening_mm,0);
 assert.deepEqual(withEndpointAction(grasp,'none',{graspLine:true}),[before[1]]);
});

test('the endpoint label names what the gripper does at the end of the path',async()=>{
 const {endpointLabel,withEndpointAction}=await source();
 assert.equal(endpointLabel(withEndpointAction([],'grasp')),'抓取 · 闭合');
 assert.equal(endpointLabel(withEndpointAction([],'release')),'松手 · 张开');
 assert.equal(endpointLabel(withEndpointAction([{s:.5,opening_mm:30}],'none')),'保持夹爪状态');
 assert.equal(endpointLabel([{s:1,opening_mm:35}]),'开度 35 mm');
});
