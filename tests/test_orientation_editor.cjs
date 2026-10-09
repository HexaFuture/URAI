const test=require('node:test');
const assert=require('node:assert/strict');
const math=import('../urai/static/orientation.mjs');

test('drag rotations compose in the chosen world or tool frame',async()=>{
  const {rpyMatrix,rotatePose,matrixRpy}=await math;
  const r=rpyMatrix([25,40,60]);
  for(const frame of ['world','tool']){
    const changed=rotatePose(r,2,Math.PI/2,frame);
    const roundtrip=rpyMatrix(matrixRpy(changed));
    changed.flat().forEach((v,i)=>assert.ok(Math.abs(v-roundtrip.flat()[i])<1e-8));
    const restored=rotatePose(changed,2,-Math.PI/2,frame);
    restored.flat().forEach((v,i)=>assert.ok(Math.abs(v-r.flat()[i])<1e-8));
  }
  assert.notDeepEqual(rotatePose(r,2,.5,'world'),rotatePose(r,2,.5,'tool'));
});

test('Euler readout roundtrips at the pitch singularities',async()=>{
  const {rpyMatrix,matrixRpy}=await math;
  for(const p of [-90,-89.9999,0,89.9999,90]){
    const r=rpyMatrix([35,p,-70]),back=rpyMatrix(matrixRpy(r));
    r.flat().forEach((v,i)=>assert.ok(Math.abs(v-back.flat()[i])<1e-7));
  }
});

test('gripper drawing uses tool Z as reach and X as closing direction',async()=>{
  const {gripperSegments}=await math;
  const points=gripperSegments(.06).flat();
  assert.ok(points.every(p=>p[2]<=0));
  assert.ok(points.some(p=>p[0]===.03&&p[2]===0));
  assert.ok(points.some(p=>p[0]===-.03&&p[2]===0));
});
