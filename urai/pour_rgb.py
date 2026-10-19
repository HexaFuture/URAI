"""Calibrated RGB circular-rim fitting. Metric scale comes from a known radius.

No depth pixel is read. The caller supplies landmarks on the same circular edge
as its measured radius; pose ambiguity is resolved with the tracked axis prior.
"""
import numpy as np
from scipy.optimize import least_squares


def rgb_rim(frame, pixels, radius_m, normal_hint, *, horizontal=False, fixed_normal=False):
    p=np.asarray(pixels,float);radius=float(radius_m);hint=np.asarray(normal_hint,float)
    if p.ndim!=2 or p.shape[1]!=2 or not 6<=len(p)<=16 or not np.isfinite(p).all():
        raise ValueError('RGB rim needs 6..16 finite contour pixels')
    if len(np.unique(p,axis=0))!=len(p):raise ValueError('RGB rim pixels must be distinct')
    h,w=frame.rgb.shape[:2]
    if np.any(p<0) or np.any(p[:,0]>=w) or np.any(p[:,1]>=h):raise ValueError('RGB contour outside image')
    if not np.isfinite(radius) or not .008<=radius<=.10:raise ValueError('RGB radius must be a measured 8..100 mm')
    if hint.shape!=(3,) or not np.isfinite(hint).all() or np.linalg.norm(hint)<.1:raise ValueError('Invalid RGB axis prior')
    hint=hint/np.linalg.norm(hint)
    if horizontal:hint=np.array([0.,0.,1.])
    spread=np.linalg.svd(p-p.mean(axis=0),compute_uv=False)
    if spread[1]<2.:raise ValueError('RGB contour is too narrow to locate reliably')
    origin=frame.t[:3,3];rays=(np.linalg.solve(frame.k,np.c_[p,np.ones(len(p))].T).T @ frame.t[:3,:3].T)
    # The longest diameter is almost invariant to circle tilt for initialization.
    diameter=max(np.linalg.norm(p[i]-p[j]) for i in range(len(p)) for j in range(i))
    if diameter<12.:raise ValueError('RGB rim is too small in the image')
    z0=2*radius*frame.k[0,0]/diameter
    c0=origin+np.mean(rays,axis=0)*z0
    aux=np.eye(3)[np.argmin(np.abs(hint))];b1=np.cross(hint,aux);b1/=np.linalg.norm(b1);b2=np.cross(hint,b1)
    fixed = horizontal or fixed_normal
    def unpack(v):
        n=hint if fixed else hint+v[3]*b1+v[4]*b2
        return v[:3],n/np.linalg.norm(n)
    def residual(v):
        c,n=unpack(v);den=rays@n
        den=np.where(abs(den)<1e-7,np.copysign(1e-7,den),den)
        d=((c-origin)@n)/den;points=origin+d[:,None]*rays
        # Radial residual normalized to approximate image pixels.
        return (np.linalg.norm(points-c,axis=1)-radius)*frame.k[0,0]/np.maximum(d,.05)
    candidates=[]
    for a,b in ([(0,0)] if fixed else [(0,0),(.5,0),(-.5,0),(0,.5),(0,-.5)]):
        v=np.r_[c0] if fixed else np.r_[c0,a,b]
        bound=3 if fixed else 5
        lo=np.full(bound,-2.);hi=np.full(bound,2.)
        if not fixed:lo[3:]=-.8;hi[3:]=.8
        fit=least_squares(residual,v,bounds=(lo,hi),max_nfev=200,xtol=1e-10,ftol=1e-10,gtol=1e-10)
        c,n=unpack(fit.x);rms=float(np.sqrt(np.mean(residual(fit.x)**2)))
        angle=float(np.arccos(np.clip(n@hint,-1,1)))
        if fit.success and np.all((c-origin)@n/(rays@n)>0):candidates.append((rms+.08*angle,rms,c,n))
    if not candidates:raise ValueError('RGB circular pose did not converge')
    _,rms,c,n=min(candidates,key=lambda v:v[0])
    if rms>2.:raise ValueError('RGB rim reprojection exceeds 2 pixels; relabel the circular edge')
    d=((c-origin)@n)/(rays@n);pts=origin+d[:,None]*rays
    offsets=pts-c;u=np.cross(n,np.eye(3)[np.argmin(np.abs(n))]);u/=np.linalg.norm(u);v=np.cross(n,u)
    angles=np.sort(np.mod(np.arctan2(offsets@v,offsets@u),2*np.pi));span=2*np.pi-max(np.diff(np.r_[angles,angles[0]+2*np.pi]))
    if span<np.pi:raise ValueError('RGB contour must cover at least half the circular rim')
    evidence={'method':'calibrated_rgb_known_radius','radius_m':radius,'reprojection_rms_px':rms,
              'axis_prior_angle_deg':float(np.rad2deg(np.arccos(np.clip(n@hint,-1,1)))),
              'scale_source':'supplied_measured_radius','depth_used':False,'normal_fixed':bool(fixed)}
    # This field is only circle model residual, not an accuracy/uncertainty claim.
    return {'center':c,'normal':n,'radius':radius,'residual_m':0.},evidence


def triangulated_rim(frame, before_pixels, after_pixels, displacement, normal_hint):
    """Same material landmarks across a measured pure robot translation."""
    a=np.asarray(before_pixels,float);b=np.asarray(after_pixels,float);shift=np.asarray(displacement,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=2 or not 6<=len(a)<=16 or not np.isfinite(np.r_[a.ravel(),b.ravel(),shift]).all():
        raise ValueError('RGB parallax needs 6..16 paired material landmarks')
    if shift.shape!=(3,) or not .010<=np.linalg.norm(shift)<=.030:raise ValueError('RGB parallax baseline must be 10..30 mm')
    rays=[]
    for px in (a,b):
        if np.any(px<0) or np.any(px[:,0]>=frame.rgb.shape[1]) or np.any(px[:,1]>=frame.rgb.shape[0]):raise ValueError('Parallax landmark outside image')
        r=np.linalg.solve(frame.k,np.c_[px,np.ones(len(px))].T).T @ frame.t[:3,:3].T
        rays.append(r/np.linalg.norm(r,axis=1)[:,None])
    points=[];gaps=[];parallax_angles=[]
    for ra,rb in zip(*rays):
        angle=np.arccos(np.clip(ra@rb,-1,1));parallax_angles.append(angle)
        if angle<np.deg2rad(.6):raise ValueError('Insufficient RGB parallax')
        scales=np.linalg.lstsq(np.c_[-ra,rb],shift,rcond=None)[0]
        if np.any(scales<=0):raise ValueError('RGB parallax lies behind camera')
        pa=frame.t[:3,3]+scales[0]*ra+shift;pb=frame.t[:3,3]+scales[1]*rb
        gaps.append(np.linalg.norm(pa-pb));points.append((pa+pb)/2)
    if max(gaps)>.0015:raise ValueError('RGB correspondences disagree with measured translation')
    # Joint reprojection fit pools the short-baseline triangulation noise across
    # the physical circle instead of treating noisy independent depths as a plane.
    points=np.asarray(points);center=points.mean(axis=0)
    _,_,basis=np.linalg.svd(points-center,full_matrices=False)
    hint=np.asarray(normal_hint,float);hint/=np.linalg.norm(hint)
    u=np.cross(hint,np.eye(3)[np.argmin(abs(hint))]);u/=np.linalg.norm(u);v=np.cross(hint,u)
    initial_angles=np.arctan2((points-center)@v,(points-center)@u)
    initial_radius=np.median(np.linalg.norm(points-center,axis=1))
    from scipy.spatial.transform import Rotation
    def circle(params):
        c=params[:3];Q=Rotation.from_rotvec(params[3]*u+params[4]*v).as_matrix()
        aa=params[6:];r=params[5]
        return c+r*(np.cos(aa)[:,None]*(Q@u)+np.sin(aa)[:,None]*(Q@v)),Q@hint
    def errors(params):
        world,n=circle(params)
        return np.r_[(frame.project(world-shift)-a).ravel(),(frame.project(world)-b).ravel()]
    lower=np.r_[[-2.]*3,[-1.2]*2,.008,[-20.]*len(a)]
    upper=np.r_[[2.]*3,[1.2]*2,.100,[20.]*len(a)]
    fit=least_squares(errors,np.r_[center,0,0,np.clip(initial_radius,.009,.099),initial_angles],bounds=(lower,upper),max_nfev=300)
    rms=float(np.sqrt(np.mean(fit.fun**2)))
    if not fit.success or rms>1.5:raise ValueError('RGB parallax circle reprojection exceeds 1.5 pixels')
    world,n=circle(fit.x)
    angles=np.sort(np.mod(fit.x[6:],2*np.pi))
    if 2*np.pi-max(np.diff(np.r_[angles,angles[0]+2*np.pi]))<np.pi:raise ValueError('RGB parallax needs half-rim coverage')
    rim={'center':fit.x[:3],'normal':n,'radius':float(fit.x[5]),'residual_m':0.}
    return rim,{'method':'rgb_measured_translation_parallax','reprojection_rms_px':rms,'radius_m':rim['radius'],'depth_used':False,'baseline_m':float(np.linalg.norm(shift)),
                'max_ray_gap_m':float(max(gaps)),'min_parallax_deg':float(np.rad2deg(min(parallax_angles)))}
