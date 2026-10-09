const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=v=>v.map(x=>x/(Math.hypot(...v)||1));
export const transform=(r,p)=>r.map(row=>dot(row,p));
const product=(a,b)=>a.map(row=>b[0].map((_,j)=>dot(row,b.map(r=>r[j]))));
const basis=i=>[0,1,2].map(j=>Number(i===j));
const column=(r,i)=>r.map(row=>row[i]);
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
export function rpyMatrix(rpy){
 const [r,p,y]=rpy.map(v=>v*Math.PI/180),cr=Math.cos(r),sr=Math.sin(r),cp=Math.cos(p),sp=Math.sin(p),cy=Math.cos(y),sy=Math.sin(y);
 return [[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]];
}
export function matrixRpy(r){
 const pitch=Math.asin(Math.max(-1,Math.min(1,-r[2][0]))),singular=Math.abs(Math.cos(pitch))<1e-8;
 return [singular?Math.atan2(-r[1][2],r[1][1]):Math.atan2(r[2][1],r[2][2]),pitch,singular?0:Math.atan2(r[1][0],r[0][0])].map(v=>v*180/Math.PI);
}
function rpyQuaternion(rpy){
 const [r,p,y]=rpy.map(v=>v*Math.PI/360),cr=Math.cos(r),sr=Math.sin(r),cp=Math.cos(p),sp=Math.sin(p),cy=Math.cos(y),sy=Math.sin(y);
 return [sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy];
}
export function sampleOrientation(points,s){
 let i=0;while(i<points.length-2&&s>points[i+1][0])i++;
 const a=points[i],b=points[Math.min(i+1,points.length-1)],t=Math.max(0,Math.min(1,(s-a[0])/(b[0]-a[0]||1)));
 const q0=rpyQuaternion(a.slice(1));let q1=rpyQuaternion(b.slice(1)),d=dot(q0,q1);
 if(d<0){q1=q1.map(v=>-v);d=-d;}
 const angle=Math.acos(Math.min(1,d)),k=angle<1e-8?[1-t,t]:[Math.sin((1-t)*angle)/Math.sin(angle),Math.sin(t*angle)/Math.sin(angle)];
 const q=q0.map((v,j)=>v*k[0]+q1[j]*k[1]),n=Math.hypot(...q),[x,y,z,w]=q.map(v=>v/n);
 return matrixRpy([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]);
}
function axisRotation(axis,angle){
 const [x,y,z]=unit(axis),c=Math.cos(angle),s=Math.sin(angle),v=1-c;
 return [[x*x*v+c,x*y*v-z*s,x*z*v+y*s],[y*x*v+z*s,y*y*v+c,y*z*v-x*s],[z*x*v-y*s,z*y*v+x*s,z*z*v+c]];
}
export function rotatePose(r,axis,angle,frame='world'){
 return product(axisRotation(frame==='tool'?column(r,axis):basis(axis),angle),r);
}
export function gripperSegments(opening=.04){
 const x=Math.max(.006,Math.min(.07,opening))/2;
 return [[[-.033,0,-.14],[.033,0,-.14]],[[-.033,0,-.14],[-.033,0,-.065]],
 [[.033,0,-.14],[.033,0,-.065]],[[-.033,0,-.065],[.033,0,-.065]],
 [[-x,0,-.065],[-x,0,0]],[[x,0,-.065],[x,0,0]],
 [[-x,0,0],[-x+.005,0,0]],[[x,0,0],[x-.005,0,0]]];
}
function box(x0,x1,y0,y1,z0,z1){
 const v=[[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]];
 return [[0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]].map(f=>f.map(i=>v[i]));
}

export class GripperPoseEditor{
 constructor(canvas,callbacks){
  this.el=canvas;this.cb=callbacks;this.r=rpyMatrix([0,0,0]);this.enabled=false;this.frame='tool';this.opening=.04;this.rings=[];this.drag=null;
  this.view=unit([1,-1,.8]);this.right=unit(cross([0,0,1],this.view));this.up=cross(this.view,this.right);
  canvas.addEventListener('pointerdown',e=>this.down(e));
  canvas.addEventListener('pointermove',e=>this.move(e));
  canvas.addEventListener('pointerup',e=>this.finish(e,false));
  canvas.addEventListener('pointercancel',e=>this.finish(e,true));
  canvas.addEventListener('lostpointercapture',e=>this.finish(e,true));
  canvas.addEventListener('keydown',e=>{if(e.key==='Escape'&&this.drag){e.preventDefault();this.finish(e,true);}});
  this.resize=new ResizeObserver(()=>this.draw());this.resize.observe(canvas);
 }
 get dragging(){return Boolean(this.drag);}
 setPose(rpy,{enabled,frame='tool',opening=.04,label=''}={}){
  if(!this.drag)this.r=rpyMatrix(rpy);
  this.enabled=Boolean(enabled);this.frame=frame;this.opening=opening;this.label=label;this.draw();
 }
 point(e){const b=this.el.getBoundingClientRect();return [e.clientX-b.left,e.clientY-b.top];}
 project(p){return [this.cx+this.scale*dot(this.right,p),this.cy-this.scale*dot(this.up,p)];}
 ball(p){const x=(p[0]-this.cx)/this.radius,y=(this.cy-p[1])/this.radius,z=Math.sqrt(Math.max(0,1-x*x-y*y));return unit([x,y,z]);}
 angle(p,ring){
  const u=[dot(this.right,ring.u),-dot(this.up,ring.u)],v=[dot(this.right,ring.v),-dot(this.up,ring.v)];
  const det=u[0]*v[1]-u[1]*v[0],x=p[0]-this.cx,y=p[1]-this.cy;
  return Math.abs(det)<.06?null:Math.atan2((u[0]*y-u[1]*x)/det,(x*v[1]-y*v[0])/det);
 }
 down(e){
  if(!this.enabled||this.drag||e.button!==0||!e.isPrimary)return;
  const p=this.point(e);let hit=null,distance=9;
  for(const ring of this.rings)for(const q of ring.points){const d=Math.hypot(p[0]-q[0],p[1]-q[1]);if(d<distance){distance=d;hit=ring;}}
  if(!hit&&Math.hypot(p[0]-this.cx,p[1]-this.cy)>this.radius*.82)return;
  e.preventDefault();this.el.focus();
  this.drag={id:e.pointerId,start:this.r.map(r=>r.slice()),ring:hit,lastAngle:hit?this.angle(p,hit):null,total:0,point:p,ball:this.ball(p)};
  this.el.setPointerCapture(e.pointerId);this.cb.begin?.();this.draw();
 }
 move(e){
  if(!this.drag||e.pointerId!==this.drag.id)return;
  e.preventDefault();const d=this.drag,p=this.point(e);
  if(d.ring){
   const angle=this.angle(p,d.ring);
   d.total+=angle===null||d.lastAngle===null?(p[0]-d.point[0]-p[1]+d.point[1])*.012:wrap(angle-d.lastAngle);
   d.lastAngle=angle;d.point=p;this.r=product(axisRotation(d.ring.axis,d.total),d.start);
  }else{
   const b=this.ball(p),axis=cross(d.ball,b),angle=Math.acos(Math.max(-1,Math.min(1,dot(d.ball,b))));
   const world=[0,1,2].map(i=>this.right[i]*axis[0]+this.up[i]*axis[1]+this.view[i]*axis[2]);
   this.r=Math.hypot(...axis)<1e-10?d.start:product(axisRotation(world,angle),d.start);
  }
  this.cb.change?.(matrixRpy(this.r));this.draw();
 }
 finish(e,cancel){
  if(!this.drag||(e.pointerId!==undefined&&e.pointerId!==this.drag.id))return;
  const d=this.drag;this.drag=null;
  if(this.el.hasPointerCapture(d.id))this.el.releasePointerCapture(d.id);
  if(cancel){this.r=d.start;this.cb.cancel?.();}else this.cb.commit?.(matrixRpy(this.r));
  this.draw();
 }
 draw(){
  const rect=this.el.getBoundingClientRect(),w=rect.width,h=rect.height;if(!w||!h)return;
  const ratio=window.devicePixelRatio||1;this.el.width=Math.round(w*ratio);this.el.height=Math.round(h*ratio);
  const c=this.el.getContext('2d');c.setTransform(ratio,0,0,ratio,0,0);c.clearRect(0,0,w,h);
  this.cx=w/2;this.cy=h/2+8;this.radius=Math.min(w*.38,h*.37);this.scale=this.radius/.13;
  c.fillStyle='#a0acbb';c.font='11px sans-serif';c.fillText(this.label||'拖动夹爪或旋转环',10,18);
  const color=['#ed8d8d','#93d5a3','#88bfee'];this.rings=[];
  for(let i=0;i<3;i++){
   const r=this.drag?.start||this.r,u=this.frame==='tool'?column(r,(i+1)%3):basis((i+1)%3),v=this.frame==='tool'?column(r,(i+2)%3):basis((i+2)%3),axis=this.frame==='tool'?column(r,i):basis(i);
   const points=Array.from({length:129},(_,j)=>this.project(u.map((x,k)=>.13*(x*Math.cos(j*Math.PI/64)+v[k]*Math.sin(j*Math.PI/64)))));
   this.rings.push({u,v,axis,points,index:i});c.beginPath();points.forEach((p,j)=>j?c.lineTo(...p):c.moveTo(...p));c.strokeStyle=color[i];c.globalAlpha=this.enabled ? .65 : .2;c.lineWidth=this.drag?.ring?.index===i?3:1.6;c.stroke();
  }
  c.globalAlpha=this.enabled?1:.45;
  const x=Math.max(.006,Math.min(.07,this.opening))/2,parts=[
   ...box(-.033,.033,-.025,.025,-.1425,-.065).map(face=>({face,color:'#53677a'})),
   ...box(-x-.006,-x+.006,-.009,.009,-.065,0).map(face=>({face,color:'#b2c2cf'})),
   ...box(x-.006,x+.006,-.009,.009,-.065,0).map(face=>({face,color:'#b2c2cf'}))];
  const world=p=>transform(this.r,[p[0],p[1],p[2]+.066]);
  parts.forEach(p=>{p.vertices=p.face.map(world);p.depth=p.vertices.reduce((s,v)=>s+dot(v,this.view),0)/4;});
  parts.sort((a,b)=>a.depth-b.depth).forEach(p=>{c.beginPath();p.vertices.map(v=>this.project(v)).forEach((v,i)=>i?c.lineTo(...v):c.moveTo(...v));c.closePath();c.fillStyle=p.color;c.fill();c.strokeStyle='#101a24';c.lineWidth=1;c.stroke();});
  const tcp=world([0,0,0]),origin=this.project(tcp);
  for(let j=0;j<3;j++){
   const end=this.project(tcp.map((v,k)=>v+.035*this.r[k][j]));c.strokeStyle=color[j];c.lineWidth=2.5;c.beginPath();c.moveTo(...origin);c.lineTo(...end);c.stroke();c.fillStyle=color[j];c.font='bold 10px sans-serif';c.fillText(['X','Y','Z'][j],end[0]+3,end[1]-3);
  }
  c.fillStyle='#fff';c.beginPath();c.arc(...origin,3,0,2*Math.PI);c.fill();c.globalAlpha=1;
  this.el.style.cursor=this.drag?'grabbing':this.enabled?'grab':'default';
 }
}
