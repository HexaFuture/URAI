# Backpack flap opening

For a bag whose main zipper is already open, `bag_panel_lift` takes two pixels defining the fingertip closing
direction across the upper fabric panel. It grasps, lifts 100 mm by default, and holds without homing. This is
not a stroke along the bag's rim. Defaults are a 40 mm initial opening, 18 mm inset, effort 1200, 0.15 m/s lift
speed, and 0.12 m/s approach speed. The result includes `return_poses` for reversing the held-panel path and
`bag_open_verified=false`; the agent must inspect the resulting opening before packing with the other arm.

The alternative staged flap tool below uses a small test lift before peeling.

Use `bag_flap_grasp` for the fabric upper panel, not a handle, lower panel, zipper pull or the whole bag. This is an agent-assisted geometry tool, not an automatic bag detector.

From a fresh RGB frame, mark the two ends of a short free-lip segment and a third point inside the upper flap toward its hinge. The UI accepts a bent stroke: follow the lip, then turn into the flap. The tool extracts the bend, samples depth on the fabric side, offsets the pinch 5 mm inward, and closes perpendicular to the lip. It uses the arm's mirrored fingertip correction; it does not silently replace an unreachable vertical fabric pinch with an arbitrary side grasp.

Defaults: 24 mm initial opening, 4 mm shallow inset, 20 mm vertical test lift, effort 800, 0.15 m/s test-lift speed, 0.12 m/s approach speed. The standard draft/preview/execute path preserves collision and precision checks and holds the selected arm afterward. `bag_open_verified` remains false.

After the test lift, obtain a new RGB/depth observation. Compare both panels with the pre-grasp frame: if the lower panel rose with the upper panel, return the panel to the approach location, release, and reselect the upper free lip. Jaw opening alone does not prove an open bag.

For a visually separated flap, the agent may call:

```python
from urai.skills import skill_context
from urai.skills.bag import plan_peel
service.observe()
ctx = skill_context(service, arm)
report = plan_peel(ctx, probe_report, upper_pixel, lower_pixel)
# Adopt report['arm'] into a draft with service.frame.id, preview and execute normally.
```

`probe_report` is the actual result of the earlier grasp planner; pixels identify the newly observed upper/lower lip. The tool rejects reuse of the old observation, a changed holder pose, or less than 12 mm vertical separation. This depth check is only a geometric sanity check: the agent must still recognize the correct layers and check lower-panel motion.

The peel lifts first (80 mm), then moves toward the observed flap interior (60 mm) while rotating about the lip tangent (30 degrees). It holds the flap rather than automatically homing. It does not declare the bag open. Before packing, the agent must see the main compartment opening, define a drop point within that opening, and use the other arm. Retreat the packing arm before lowering/releasing the flap and homing the holder.

Validation: bag geometry, mirrored correction, dense UI stroke, shallow inset, observation freshness, layer separation, unchanged holder, IK refusal, and hold-without-homing tests pass. Physical opening success with this new planner has not yet been established.
