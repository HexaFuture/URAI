"""What a skill planner is handed: one arm, the current observation and the arm's own kinematics.

Without it every endpoint would repeat the same assembly (model, table height, fingertip bias, base and reach
probe). One context keeps every skill on the same calibration and lets a skill ask the real arm whether a pose
is reachable before it commits to a route.
"""
import numpy as np
from ..transfer import leveled_cloud, locate_container, locate_object


class SkillContext:
    """One arm's view of the scene. probe is None in simulation, where there is no calibrated model."""

    def __init__(self, frame, arm, backend, service=None, model=None, current=None,
                 clearance_m=.12, speed_m_s=.8):
        self.frame, self.arm, self.backend, self.service = frame, arm, backend, service
        clearance_m, speed_m_s = float(clearance_m), float(speed_m_s)
        if not np.isfinite([clearance_m, speed_m_s]).all() or not .02 <= clearance_m <= .30:
            raise ValueError('抬起余量需为 0.02-0.30 m')
        if not .001 <= speed_m_s <= 1.:
            raise ValueError('移动速度需为 0.001-1.0 m/s')
        self.clearance_m, self.speed_m_s = clearance_m, speed_m_s
        self.model = model
        self.current = current or backend.state()
        self.table_z = float(getattr(model, 'table_z_mm', 0.))/1000
        self.fingertip_bias_mm = float(getattr(model, 'fingertip_bias_mm', 0.))
        # The real arm's base decides how a grasp leans and which side the wrist takes; the simulation has no
        # base, so the arm's current TCP plays that role, exactly as the transfer and pour endpoints do.
        self.base_xy = np.asarray(model.base_xy if model is not None else self.current[arm]['xyz'][:2], dtype=float)
        self.preferred_yaw_deg = float(self.current[arm]['rpy_deg'][2])
        self.pose_tolerance = (30., 60.)
        if service is not None:
            self.pose_tolerance = (service.pose_tolerance['position_mm'], service.pose_tolerance['orientation_deg'])
        self._cloud = None
        self._seed = list(self.current[arm]['joints_deg'])

    @property
    def cloud(self):
        """The levelled point cloud of the whole frame, computed once per context."""
        if self._cloud is None:
            self._cloud, _ = leveled_cloud(self.frame, self.table_z)
        return self._cloud

    def locate(self, pixel):
        """The raised object under pixel: centre, top height, jaw axis, mask and points."""
        return locate_object(self.frame, pixel, self.table_z, self.preferred_yaw_deg, self.cloud)

    def container(self, pixel):
        """The container whose rim circle contains pixel, or the object under it."""
        return locate_container(self.frame, pixel, self.table_z, self.preferred_yaw_deg, self.cloud)

    def probe(self, xyz, rotation, seed=None):
        """(position_mm, rotation_deg, joints) of the arm's best IK solution for this pose, seeded from seed.

        Returns None when the backend has no calibrated model for this arm (the simulation): a skill then has to
        pick its candidate on geometry alone, as pour_route does without a probe.
        """
        if self.model is None:
            return None
        tolerance = {'position_mm': self.pose_tolerance[0], 'orientation_deg': self.pose_tolerance[1]}
        return self.backend.pose_error(self.arm, xyz, rotation, self._seed if seed is None else seed, tolerance)

    @property
    def has_kinematics(self):
        return self.model is not None

    def other_arm(self):
        """The same scene as seen by the other arm, for the bimanual skills."""
        other = 'right' if self.arm == 'left' else 'left'
        context = SkillContext(self.frame, other, self.backend, self.service,
                               model=self.backend.models.get(other), current=self.current,
                               clearance_m=self.clearance_m, speed_m_s=self.speed_m_s)
        context._cloud = self._cloud   # the levelled cloud is the scene's, not the arm's
        return context


def skill_context(service, arm, frame=None, clearance_m=.12, speed_m_s=.8):
    """Build the context for arm from the running service, under the caller's service lock."""
    return SkillContext(frame or service.frame, arm, service.backend, service, service.backend.models.get(arm),
                        clearance_m=clearance_m, speed_m_s=speed_m_s)
