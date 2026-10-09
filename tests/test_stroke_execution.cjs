// The page's stroke pipeline (urai/static/automatic.mjs) against the real service with its simulation backend:
// node --test tests/test_stroke_execution.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const {startService} = require('./support/urai_server.cjs');

// The simulated head camera sees one block (pixels u 350-440, v 180-260) on an empty table.
const BLOCK = [395, 220], TABLE = [600, 220];
const BRUSH = [[480, 330], [540, 350], [600, 370]];
const ACROSS_BLOCK = [[395, 170], [395, 270]];

let service, automatic, endpoint;
test.before(async () => {
  service = await startService();
  automatic = await import('../urai/static/automatic.mjs');
  endpoint = await import('../urai/static/endpoint.mjs');
});
test.after(async () => { await service.stop(); });

/** A new observation and no draft; an error or a stop left by the previous test is acknowledged first. */
async function freshScene() {
  const before = await service.api('state');
  if (['error', 'cancelled'].includes(before.execution.state)) await service.api('recover', 'POST');
  const frame = await service.api('observe', 'POST', {preserve_draft: false});
  return {frame, state: await service.api('state')};
}

/** The real client, with the route of every request written down in order. */
function recorded(calls) {
  return (route, method, data) => { calls.push(route); return service.api(route, method, data); };
}

function pickPlaceRequest(frame, extra = {}) {
  return {arm: 'left', observation_id: frame.id, pixels: [BLOCK, TABLE], clearance_m: .12, speed_m_s: .8, ...extra};
}

/** The draft the page builds from a brush stroke: the service's control points plus a release at the end. */
async function brushDraft(frame, speed = .8) {
  const stroke = await service.api('stroke', 'POST', {observation_id: frame.id, pixels: BRUSH, mode: 'surface', z: .15});
  return {observation_id: frame.id, arms: {left: {
    path: {mode: 'waypoints', points: stroke.points}, speed: [[0, speed], [1, speed]], orientation: {mode: 'free'},
    gripper_events: endpoint.withEndpointAction([], 'release'), start_hold_s: 0,
    approach: {speed_m_s: .8, clearance_m: .12}}}};
}

test('a pick-place stroke is drafted by the service, previewed and executed exactly once', async () => {
  const {frame, state} = await freshScene();
  const calls = [], shown = [];
  let saved = null;
  const {result, plan} = await automatic.runStrokeTask({
    api: recorded(calls), route: 'pick-place', request: pickPlaceRequest(frame),
    revision: state.revision, cancelEpoch: state.cancel_epoch, isCancelled: () => false,
    onDraft: (answer, draft) => { saved = draft; }, onPreview: preview => { shown.push(preview); }});
  assert.deepEqual(calls, ['pick-place', 'state', 'draft', 'preview', 'execute']);
  assert.equal(result.observation_id, frame.id);
  assert.deepEqual(Object.keys(saved.draft.arms), ['left']);
  assert.equal(saved.revision, state.revision + 1);
  assert.deepEqual(shown.map(p => p.id), [plan.id]);
  const done = await service.waitFor(['completed', 'error']);
  assert.equal(done.execution.state, 'completed', JSON.stringify(done.execution));
  assert.equal(done.execution.draft_cleared, true, 'a trajectory that ran is dropped from the shared draft');
});

test('draft-only mode saves the drafted arm and never plans or executes', async () => {
  const {frame, state} = await freshScene();
  const calls = [];
  const out = await automatic.runStrokeTask({
    api: recorded(calls), route: 'pick-place', request: pickPlaceRequest(frame), execute: false,
    revision: state.revision, cancelEpoch: state.cancel_epoch, isCancelled: () => false,
    onDraft() {}, onPreview() { throw new Error('draft-only mode must not preview'); }});
  assert.equal(out.plan, null);
  assert.deepEqual(calls, ['pick-place', 'state', 'draft']);
  const shared = await service.api('draft');
  assert.deepEqual(Object.keys(shared.arms), ['left']);
  assert.equal((await service.api('state')).execution.state, 'idle');
});

test('the pinch and top-down switches reach the planner and come back in the answer the page reports', async () => {
  const {frame, state} = await freshScene();
  const {result} = await automatic.runStrokeTask({
    api: service.api, route: 'pick-place', execute: false,
    request: pickPlaceRequest(frame, {pinch: true, top_down: true, pinch_depth_mm: 5}),
    revision: state.revision, cancelEpoch: state.cancel_epoch, isCancelled: () => false,
    onDraft() {}, onPreview() {}});
  assert.equal(result.pinch, true);
  assert.equal(result.top_down, true);
  assert.ok(result.finger_z_mm < 0, `a pinch closes below the table surface, got ${result.finger_z_mm} mm`);
  const plain = await freshScene();
  const object = await automatic.runStrokeTask({
    api: service.api, route: 'pick-place', execute: false, request: pickPlaceRequest(plain.frame),
    revision: plain.state.revision, cancelEpoch: plain.state.cancel_epoch, isCancelled: () => false,
    onDraft() {}, onPreview() {}});
  assert.equal(object.result.pinch, false);
  assert.ok(object.result.object_width_mm > 0, 'a grasp reports the measured object width');
});

test('another client changing the draft stops the stroke before it overwrites anything', async () => {
  const {frame, state} = await freshScene();
  const theirs = await service.api('draft', 'PUT', await brushDraft(frame));
  const calls = [];
  await assert.rejects(automatic.runStrokeTask({
    api: recorded(calls), route: 'pick-place', request: pickPlaceRequest(frame),
    revision: state.revision, cancelEpoch: state.cancel_epoch, isCancelled: () => false,
    onDraft() {}, onPreview() {}}), /变化/);
  assert.deepEqual(calls, ['pick-place', 'state']);
  assert.deepEqual(await service.api('draft'), theirs.draft);
});

test('cancelling while the preview is shown never executes', async () => {
  const {frame, state} = await freshScene();
  const calls = [];
  let cancelled = false;
  await assert.rejects(automatic.runStrokeTask({
    api: recorded(calls), route: 'pick-place', request: pickPlaceRequest(frame),
    revision: state.revision, cancelEpoch: state.cancel_epoch, isCancelled: () => cancelled,
    onDraft() {}, onPreview() { cancelled = true; }}), /两点抓放已取消/);
  assert.deepEqual(calls, ['pick-place', 'state', 'draft', 'preview']);
  assert.equal((await service.api('state')).execution.state, 'ready');
});

test('a brush draft is saved, previewed and executed exactly once', async () => {
  const {frame, state} = await freshScene();
  const draft = await brushDraft(frame), calls = [];
  const {saved, plan} = await automatic.runDraftExecution({
    api: recorded(calls), draft, revision: state.revision, cancelEpoch: state.cancel_epoch,
    isCancelled: () => false, onSaved() {}, onPreview() {}});
  assert.deepEqual(calls, ['state', 'draft', 'preview', 'execute']);
  assert.equal(plan.revision, saved.revision);
  const done = await service.waitFor(['completed', 'error']);
  assert.equal(done.execution.state, 'completed', JSON.stringify(done.execution));
  assert.equal(done.arms.left.gripper_mm, 70, 'the release at the end of the stroke opened the gripper');
});

test('a brush stroke cancelled before it is saved sends nothing', async () => {
  const {frame, state} = await freshScene();
  const draft = await brushDraft(frame), calls = [];
  await assert.rejects(automatic.runDraftExecution({
    api: recorded(calls), draft, revision: state.revision, cancelEpoch: state.cancel_epoch,
    isCancelled: () => true, onSaved() {}, onPreview() {}}), /画线即执行已取消/);
  assert.deepEqual(calls, []);
});

test('a stop pressed after the draft is saved makes the planner refuse the preview', async () => {
  const {frame, state} = await freshScene();
  const draft = await brushDraft(frame), calls = [];
  await assert.rejects(automatic.runDraftExecution({
    api: recorded(calls), draft, revision: state.revision, cancelEpoch: state.cancel_epoch, isCancelled: () => false,
    onSaved: () => service.api('cancel', 'POST'), onPreview() {}}));
  assert.deepEqual(calls, ['state', 'draft', 'preview']);
  assert.notEqual((await service.api('state')).execution.state, 'running');
});

test('an execute refused after a stop is reported once and never retried', async () => {
  const {frame, state} = await freshScene();
  const draft = await brushDraft(frame), calls = [];
  await assert.rejects(automatic.runDraftExecution({
    api: recorded(calls), draft, revision: state.revision, cancelEpoch: state.cancel_epoch, isCancelled: () => false,
    onSaved() {}, onPreview: () => service.api('cancel', 'POST')}));
  assert.deepEqual(calls, ['state', 'draft', 'preview', 'execute']);
  assert.notEqual((await service.api('state')).execution.state, 'running');
});

/** The request the page sends for a grasp line (or, with one repeated pixel, a tapped release point). */
function graspRequest(frame, pixels, endpointAction) {
  return {arm: 'left', observation_id: frame.id, pixels, endpoint_action: endpointAction, inset_mm: 10,
          grip_effort: 1000, clearance_m: .12, speed_m_s: .8};
}

test('a grasp line across the block answers with what the page reports and runs with its pre-opening', async () => {
  const {frame, state} = await freshScene();
  const result = await service.api('grasp', 'POST', graspRequest(frame, ACROSS_BLOCK, 'grasp'));
  assert.equal(result.observation_id, frame.id);
  // The fields the page's status line reads (grasp_mode only comes with an arm model, which the simulation lacks).
  assert.ok(result.width_mm > 0 && result.width_mm <= 70, `width ${result.width_mm} mm`);
  assert.equal(typeof result.width_clamped, 'boolean');
  assert.equal(result.surface_xyz.length, 3);
  assert.ok(Number.isFinite(result.actual_inset_mm));
  assert.equal(typeof result.table_limited, 'boolean');
  assert.equal(result.arm.grasp_line, true);
  assert.ok(result.arm.gripper_events.some(e => Number(e.s) === 1 && e.opening_mm === 0), 'the line ends closing');
  const events = endpoint.withEndpointAction(result.arm.gripper_events, 'grasp', {graspLine: true});
  assert.deepEqual(events.filter(e => e.pregrasp_open).map(e => [e.s, e.opening_mm]), [[0, 70]]);
  const draft = {observation_id: frame.id, arms: {left: {...result.arm, gripper_events: events}}};
  await automatic.runDraftExecution({api: service.api, draft, revision: state.revision, cancelEpoch: state.cancel_epoch,
    isCancelled: () => false, onSaved() {}, onPreview() {}});
  const done = await service.waitFor(['completed', 'error']);
  assert.equal(done.execution.state, 'completed', JSON.stringify(done.execution));
});

test('a tap on the table is a release point: position only, opening at arrival', async () => {
  const {frame} = await freshScene();
  const result = await service.api('grasp', 'POST', graspRequest(frame, [TABLE, TABLE], 'release'));
  assert.equal(result.grasp_mode, 'position_only');
  assert.ok(result.release_clearance_mm > 0);
  assert.equal(result.arm.grasp_line, false);
  assert.ok(result.arm.gripper_events.some(e => Number(e.s) === 1 && e.opening_mm === 70 && e.wait_for_arrival));
});

test('the skill tool lists every catalogue entry except those posted elsewhere, and plans through the same pipeline', async () => {
  const {skills} = await service.api('skills');
  assert.deepEqual(skills.filter(skill => skill.call).map(skill => [skill.name, skill.call]), [['toss', 'POST /api/tasks']]);
  const listed = skills.filter(skill => !skill.call).map(skill => skill.name);
  assert.ok(listed.includes('grasp_line') && listed.includes('pick_place'), listed.join(', '));
  const {frame, state} = await freshScene();
  const calls = [];
  const {result} = await automatic.runStrokeTask({
    api: recorded(calls), route: 'skills/grasp_line', execute: false,
    request: {arm: 'left', observation_id: frame.id, pixels: ACROSS_BLOCK, clearance_m: .12, speed_m_s: .8},
    revision: state.revision, cancelEpoch: state.cancel_epoch, isCancelled: () => false, onDraft() {}, onPreview() {}});
  assert.deepEqual(calls, ['skills/grasp_line', 'state', 'draft']);
  assert.equal(result.skill, 'grasp_line');
  assert.equal(result.arm.grasp_line, true);
  assert.equal((await service.api('draft')).arms.left.grasp_line, true);
});

/** Save a brush draft and preview it the way the page's preview button does. */
async function previewBrush() {
  const {frame, state} = await freshScene();
  const saved = await service.api('draft', 'PUT', {...await brushDraft(frame), expected_revision: state.revision});
  const plan = await service.api('preview', 'POST', {expected_revision: saved.revision, expected_cancel_epoch: state.cancel_epoch});
  return {plan, cancelEpoch: state.cancel_epoch};
}

test('manual execute dispatches the previewed plan once', async () => {
  const {plan, cancelEpoch} = await previewBrush();
  const calls = [];
  const sent = await automatic.executePreparedPreview({api: recorded(calls), plan, cancelEpoch,
    onPreview() { throw new Error('a fresh preview is executed as it is'); }});
  assert.deepEqual(calls, ['state', 'execute']);
  assert.equal(sent.id, plan.id);
  assert.equal((await service.waitFor(['completed', 'error'])).execution.state, 'completed');
});

test('manual execute refuses a preview whose draft or settings another client changed', async () => {
  const {plan, cancelEpoch} = await previewBrush();
  const settings = await service.api('settings');
  await service.api('settings', 'PUT', {home_speed_m_s: settings.home_speed_m_s});
  const calls = [];
  await assert.rejects(automatic.executePreparedPreview({api: recorded(calls), plan, cancelEpoch, onPreview() {}}), /变化/);
  assert.deepEqual(calls, ['state']);
  assert.notEqual((await service.api('state')).execution.state, 'running');
});

test('manual execute plans again when the robot moved after the preview', async () => {
  // A finished run returns the arm home by itself; a slow stroke stopped half-way leaves it out in the workspace,
  // so that homing it after the preview is a real motion.
  const away = await freshScene();
  await automatic.runDraftExecution({api: service.api, draft: await brushDraft(away.frame, .02), revision: away.state.revision,
    cancelEpoch: away.state.cancel_epoch, isCancelled: () => false, onSaved() {}, onPreview() {}});
  await new Promise(resolve => setTimeout(resolve, 1500));
  await service.api('cancel', 'POST');
  assert.equal((await service.waitFor(['cancelled', 'completed', 'error'])).execution.state, 'cancelled');
  const {plan, cancelEpoch} = await previewBrush();
  await service.api('home', 'POST', {arms: ['left'], cancel_epoch: cancelEpoch});
  const homed = await service.waitFor(['completed', 'error']);
  assert.equal(homed.execution.state, 'completed', JSON.stringify(homed.execution));
  assert.equal(homed.observation_stale, true, 'the arm moved since the observation the preview used');
  const calls = [], shown = [];
  const sent = await automatic.executePreparedPreview({api: recorded(calls), plan, cancelEpoch,
    onPreview: fresh => { shown.push(fresh); }});
  assert.deepEqual(calls, ['state', 'preview', 'execute']);
  assert.deepEqual(shown.map(p => p.id), [sent.id]);
  assert.notEqual(sent.id, plan.id);
  assert.notEqual(sent.observation_id, plan.observation_id, 'the new plan stands on a refreshed observation');
  assert.equal((await service.waitFor(['completed', 'error'])).execution.state, 'completed');
});
