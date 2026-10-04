/* Figure (a): closed-loop execution through URAI (paper Figure 1a and Figure 2b).
   The execution agent reads an observation, selects one tool and fills its arguments; URAI grounds the
   request (pixels -> 3D), plans a timed path and executes the whole multi-phase motion locally (nominal
   50 Hz); feedback and a new observation return to the agent. Two cycles: a reusable PnP call
   (tic-tac-toe) and a task-specific tool written by the programming agent (unscrew a bottle cap). In each,
   the photo's pose trail (earlier arm poses within the call) wipes in during Execute; then the recording of
   the same run plays on from the photo's frame in the photo's place (clips.js) before the next cycle. */
import { Scene, ease, span, within, window01 } from './engine.js';
import { arrow, el, packet, stepNumber, svgRoot, text, toggle } from './svg.js';
import { agentCard, phaseTrack, photoSlot, uraiBox } from './parts.js';
import { SlotClips } from './clips.js';

const PHOTOS = [
  {
    label: 'Tic-tac-toe vs. a human', call: 'PnP(…)', kind: 'reusable', labelSide: 'left', wipe: 'ltr',
    plain: 'assets/img/teaser/tictactoe.webp', ghost: 'assets/img/teaser/tictactoe-ghost.webp',
    clip: { src: 'assets/video/teaser/tictactoe.mp4', duration: 18.867, speed: '3×' },
  },
  {
    label: 'Unscrew a bottle cap', call: 'bottle_cap_twist(…)', kind: 'task', labelSide: 'right', wipe: 'rtl',
    plain: 'assets/img/teaser/bottlecap.webp', ghost: 'assets/img/teaser/bottlecap-ghost.webp',
    clip: { src: 'assets/video/teaser/bottlecap.mp4', duration: 17.167, speed: '4×' },
  },
];
// A clip is the recording of the run its still was taken from, starting on the still's frame, sped up, with the
// waits while the agent decides cut (tools/build_assets.py); `duration` is the file's, in seconds. Within a cycle
// the pose trail wipes in over Execute (4.2-6.6 s). At CLIP_GATE the full trail is on screen and the scene waits
// until the clip can play; the trail and the still lift off the clip's first frame until CLIP_START, when the
// recording plays on. At its end the still returns over the last frame (CLIP_FADE), then CYCLE_TAIL to the next.
export const CLIP_GATE = 6.9;
export const CLIP_START = 7.5;
const CLIP_FADE = 0.6;
const CYCLE_TAIL = 0.4;
/** Per photo: when its cycle starts, when its clip ends (in cycle time) and the cycle's length, in seconds. */
export const CYCLES = [];
for (const photo of PHOTOS) {
  const prev = CYCLES[CYCLES.length - 1];
  const clipEnd = CLIP_START + photo.clip.duration;
  CYCLES.push({ at: prev ? prev.at + prev.length : 0, clipEnd, length: clipEnd + CLIP_FADE + CYCLE_TAIL });
}
export const DURATION = CYCLES[1].at + CYCLES[1].length;
const inCycle = (i, lines) => lines.map(([t, line]) => [CYCLES[i].at + t, line]);
const PNP_PHASES = ['approach', 'grasp', 'transfer', 'place', 'retreat'];
const TASK_PHASE = ['bottle_cap_twist(…): one complete, timed motion'];
const ARIA = 'Closed-loop execution: the execution agent sends one tool call to URAI, which grounds, plans and '
  + 'executes a complete motion locally, then returns feedback and a new observation to the agent.';

export const STATUS = [
  ...inCycle(0, [
    [0, 'The execution agent reads the current observation.'],
    [1.0, 'It selects one tool from the collection L and fills its arguments (arm, pixels, …): PnP(…).'],
    [2.6, 'URAI grounds the request: selected pixels are back-projected to 3D points from calibrated depth.'],
    [3.4, 'It plans a timed path (path, speed profile, gripper events) and checks it against the robot’s constraints.'],
    [4.2, 'The backend executes the complete motion locally at a nominal 50 Hz: approach, grasp, transfer, place, retreat. No model call in between.'],
    [6.8, 'Feedback and a new observation return to the agent, which decides the next call.'],
    [CLIP_START + 2, 'The recording of this run plays on from the photo’s frame (3×; waits while the agent decides are cut): one PnP call per move against the human.'],
  ]),
  ...inCycle(1, [
    [0, 'Tools come validated and frozen from the programming agent; this one is task-specific: bottle_cap_twist(…).'],
    [1.0, 'The agent selects it from the current observation and fills its arguments.'],
    [2.6, 'URAI grounds the pixels in 3D …'],
    [3.4, '… plans a timed path …'],
    [4.2, '… and runs the whole motion locally before returning control.'],
    [6.8, 'Feedback and a new observation return to the agent for its next decision.'],
    [CLIP_START + 2, 'The recording plays on from the photo’s frame (4×; waits while the agent decides are cut): the cap comes off, the bottle is set upright, the arms return home.'],
  ]),
];
export const STATIC_STATUS = 'Execution agent → tool call (arm, pixels, …) → URAI: Ground (pixels → 3D), Plan (timed path), '
  + 'Execute (local, 50 Hz) → feedback + new observation → next decision. Tools are written, validated and frozen by the programming agent.';

function loopArrows(svg, g) {
  const call = arrow(svg, g.call, { cls: 'blue' });
  const feedback = arrow(svg, g.feedback, { cls: 'blue' });
  const failures = arrow(svg, g.failures, { cls: 'warm dashed thin' });
  const frozen = arrow(svg, g.frozen, { cls: 'warm' });
  return { call, feedback, failures, frozen };
}

function buildWide(stage, clips) {
  const svg = svgRoot(stage, 1200, 470, ARIA);
  const defs = el('defs', {}, svg);
  const card = { pad: 20, titleY: 36, titleSize: 19, membersY: 54, chipSize: 15, chipH: 30, descY: 112, descSize: 15 };
  const exec = agentCard(svg, {
    ...card, x: 0, y: 0, w: 440, h: 150, cls: 'exec', title: 'Execution agent', titleCls: 't-blue',
    members: [{ label: 'GPT-6' }, { label: 'Claude' }, { label: 'Human', person: true }],
    desc: ['reads the new observation,', 'then picks the next tool call'],
  });
  const prog = agentCard(svg, {
    ...card, x: 470, y: 0, w: 290, h: 150, cls: 'prog', title: 'Programming agent', titleCls: 't-warm',
    members: [{ label: 'GPT-6' }, { label: 'Claude' }], desc: ['writes, validates,', 'and freezes tools'],
  });
  const arrows = loopArrows(svg, {
    call: [[56, 152], [56, 248]], feedback: [[270, 248], [270, 152]],
    failures: [[500, 248], [500, 152]], frozen: [[650, 152], [650, 248]],
  });
  stepNumber(svg, 84, 186, 1);
  text(svg, 102, 191, 'tool call', { size: 16, cls: 't-blue t-bold' });
  text(svg, 102, 212, 'arm, pixels, …', { size: 14, cls: 't-muted' });
  stepNumber(svg, 298, 186, 3);
  text(svg, 316, 191, 'feedback', { size: 16, cls: 't-blue t-bold' });
  text(svg, 316, 212, '+ new observation', { size: 14, cls: 't-muted' });
  text(svg, 512, 205, 'failures', { size: 14, cls: 't-warm' });
  text(svg, 662, 194, ['validated,', 'frozen tools'], { size: 14, cls: 't-warm', lh: 1.35 });

  const urai = uraiBox(svg, {
    x: 0, y: 250, w: 760, h: 214, pad: 24, markSize: 46, markY: 64, chipSize: 15, chipH: 30,
    tagline: ['Universal Robot–', 'Agent Interface'], tagY: 100, tagSize: 16, divider: 'v', dividerAt: 232,
    rightX: 256, toolsY: 278, toolsSize: 17, stepAt: [268, 378], stageX: 284, stageY: 332, stageH: 92,
    stageW: 134, stageGap: 30, stagePad: 14, nameY: 28, noteY: 50, glyphY: 62, glyphH: 18, nameSize: 17,
    noteSize: 14, caption: 'timed path · orientation · gripper', captionY: 448,
  });

  const slots = PHOTOS.map((photo, i) => photoSlot(svg, defs, { x: 800, y: i * 196, w: 400, h: 179, labelSize: 14, labelH: 28 }, photo));
  clips.attach(stage, slots.map((slot) => slot.frame));
  const trunk = arrow(svg, [[760, 378], [781, 378]], { head: 0, cls: 'blue' });
  const branches = [89.5, 285.5].map((cy) => arrow(svg, [[780, 378], [780, cy], [799, cy]], { cls: 'blue', r: 10 }));
  const captions = [
    text(svg, 800, 398, 'phases of one PnP call', { size: 12.5, cls: 't-dim' }),
    text(svg, 800, 398, 'one task-specific call', { size: 12.5, cls: 't-dim' }),
  ];
  const tracks = {
    pnp: phaseTrack(svg, { x: 800, y: 406, w: 400, h: 32, size: 13 }, PNP_PHASES),
    task: phaseTrack(svg, { x: 800, y: 406, w: 400, h: 32, size: 13, task: true }, TASK_PHASE),
  };
  return makeDraw({ svg, exec, prog, arrows, urai, slots, clips, trunk, branches, captions, tracks, singleSlot: false });
}

function buildTall(stage, clips) {
  const svg = svgRoot(stage, 360, 756, ARIA);
  const defs = el('defs', {}, svg);
  const card = { pad: 14, titleY: 28, titleSize: 15, membersY: 50, chipSize: 12.5, descY: 76, descSize: 12.5 };
  const exec = agentCard(svg, {
    ...card, x: 0, y: 0, w: 174, h: 150, cls: 'exec', title: 'Execution agent', titleCls: 't-blue',
    membersText: 'GPT-6 · Claude · Human', desc: ['reads the new', 'observation, then', 'picks the next', 'tool call'],
  });
  const prog = agentCard(svg, {
    ...card, x: 186, y: 0, w: 174, h: 150, cls: 'prog', title: 'Programming agent', titleCls: 't-warm',
    membersText: 'GPT-6 · Claude', desc: ['writes, validates,', 'and freezes tools'],
  });
  const arrows = loopArrows(svg, {
    call: [[22, 152], [22, 234]], feedback: [[160, 234], [160, 152]],
    failures: [[198, 234], [198, 152]], frozen: [[338, 152], [338, 234]],
  });
  stepNumber(svg, 40, 167, 1, 9);
  text(svg, 54, 172, 'tool call', { size: 13.5, cls: 't-blue t-bold' });
  text(svg, 54, 189, 'arm, pixels, …', { size: 12.5, cls: 't-muted' });
  stepNumber(svg, 140, 205, 3, 9);
  text(svg, 126, 210, 'feedback', { size: 13.5, cls: 't-blue t-bold', anchor: 'end' });
  text(svg, 150, 227, '+ new observation', { size: 12.5, cls: 't-muted', anchor: 'end' });
  text(svg, 208, 176, 'failures', { size: 12.5, cls: 't-warm' });
  text(svg, 328, 206, ['validated,', 'frozen tools'], { size: 12.5, cls: 't-warm', anchor: 'end', lh: 1.35 });

  const urai = uraiBox(svg, {
    x: 0, y: 236, w: 360, h: 268, pad: 14, markSize: 34, markY: 44, chipSize: 13, chipH: 26,
    tagline: 'Universal Robot–Agent Interface', tagY: 66, tagSize: 12.5, divider: 'h', dividerAt: 316,
    rightX: 14, toolsY: 326, toolsSize: 14, stepAt: [22, 375], stepR: 9, stageX: 10, stageY: 390, stageH: 86,
    stageW: 104, stageGap: 14, stagePad: 10, nameY: 24, noteY: 43, glyphY: 55, glyphH: 16, nameSize: 15,
    noteSize: 12, caption: 'timed path · orientation · gripper', captionY: 494,
  });
  text(svg, 36, 379, 'inside one tool call', { size: 12.5, cls: 't-muted' });

  const slots = PHOTOS.map((photo) => photoSlot(svg, defs, { x: 0, y: 532, w: 360, h: 161, labelSize: 12.5, labelH: 25 }, photo));
  clips.attach(stage, slots.map((slot) => slot.frame));
  const branches = [arrow(svg, [[298, 505], [298, 531]], { cls: 'blue' })];
  const captions = [
    text(svg, 0, 714, 'phases of one PnP call', { size: 12, cls: 't-dim' }),
    text(svg, 0, 714, 'one task-specific call', { size: 12, cls: 't-dim' }),
  ];
  const tracks = {
    pnp: phaseTrack(svg, { x: 0, y: 722, w: 360, h: 28, size: 11.5, gap: 5 }, PNP_PHASES),
    task: phaseTrack(svg, { x: 0, y: 722, w: 360, h: 28, size: 11.5, task: true }, ['bottle_cap_twist(…): one complete motion']),
  };
  return makeDraw({ svg, exec, prog, arrows, urai, slots, clips, trunk: null, branches, captions, tracks, singleSlot: true });
}

function makeDraw(R) {
  const packets = {
    call: packet(R.svg),
    feedback: packet(R.svg),
    frozen: packet(R.svg, 'warm'),
    exec: packet(R.svg),
  };
  const { ground, plan, execute } = R.urai.stages;
  const setStage = (stage, on) => {
    toggle(stage.g, 'is-active', on);
    toggle(stage.box, 'is-active', on);
  };

  function drawFinal() {
    [R.exec.rect, R.prog.rect].forEach((n) => toggle(n, 'is-active', false));
    Object.values(R.arrows).forEach((a) => toggle(a.g, 'is-active', false));
    Object.values(R.urai.tools).forEach((c) => toggle(c, 'is-active', false));
    Object.values(packets).forEach((p) => p.node.setAttribute('opacity', 0));
    [ground, plan, execute].forEach((s) => setStage(s, false));
    ground.glyph.set(1);
    plan.glyph.set(1);
    execute.glyph.set(null, 1);
    R.slots.forEach((slot, i) => slot.set({
      reveal: 1, ghostOpacity: 1, dim: 0, active: false, opacity: R.singleSlot && i > 0 ? 0 : 1,
    }));
    R.branches.forEach((b) => toggle(b.g, 'is-active', false));
    if (R.trunk) toggle(R.trunk.g, 'is-active', false);
    R.captions[0].setAttribute('opacity', 1);
    R.captions[1].setAttribute('opacity', 0);
    R.tracks.pnp.set(1, 1);
    R.tracks.task.set(0, 0);
  }

  return (t, final) => {
    if (final) {
      drawFinal();
      return;
    }
    const cycle = t < CYCLES[1].at ? 0 : 1;
    const { at, clipEnd, length } = CYCLES[cycle];
    const tau = t - at;
    const task = cycle === 1;

    toggle(R.exec.rect, 'is-active', within(tau, 0, 1.8) || within(tau, 7.7, length));
    toggle(R.prog.rect, 'is-active', task && within(tau, 0, 1.0));
    toggle(R.urai.tools.PnP, 'is-active', !task && within(tau, 1.0, 7.8));
    toggle(R.urai.tools.T, 'is-active', task && within(tau, 0.6, 7.8));

    toggle(R.arrows.call.g, 'is-active', within(tau, 1.8, 2.7));
    toggle(R.arrows.feedback.g, 'is-active', within(tau, 6.8, 7.8));
    toggle(R.arrows.frozen.g, 'is-active', task && within(tau, 0, 1.0));
    R.arrows.failures.path.setAttribute('stroke-dashoffset', (-t * 8).toFixed(1));

    packets.call.at(R.arrows.call.path, ease(tau, 1.8, 2.6), window01(tau, 1.75, 2.7, 0.12));
    packets.feedback.at(R.arrows.feedback.path, ease(tau, 6.8, 7.7), window01(tau, 6.75, 7.8, 0.12));
    packets.frozen.at(R.arrows.frozen.path, ease(tau, 0.1, 0.9), task ? window01(tau, 0.05, 1.0, 0.12) : 0);

    const executing = within(tau, 4.2, 6.8);
    const branch = R.branches[R.singleSlot ? 0 : cycle];
    R.branches.forEach((b) => toggle(b.g, 'is-active', b === branch && executing));
    if (R.trunk) toggle(R.trunk.g, 'is-active', executing);
    packets.exec.at(branch.path, ease(tau, 4.2, 4.9), window01(tau, 4.15, 5.0, 0.12));

    setStage(ground, within(tau, 2.6, 3.4));
    ground.glyph.set(ease(tau, 2.65, 3.3));
    setStage(plan, within(tau, 3.4, 4.2));
    plan.glyph.set(ease(tau, 3.45, 4.1), 1 - 0.6 * ease(tau, 7.8, 8.6));
    setStage(execute, executing);
    execute.glyph.set(executing ? (tau - 4.2) * 24 : null, span(tau, 4.2, 6.6) * (1 - ease(tau, 7.8, 8.6)));

    const progress = span(tau, 4.2, 6.6);
    R.slots.forEach((slot, i) => {
      const current = i === cycle;
      let opacity = 1;
      if (R.singleSlot) opacity = current ? ease(tau, 0, 0.6) : 1 - ease(tau, 0, 0.6);
      // The trail lifts off the clip's first frame (the still itself); the still returns over its last frame.
      const recording = current && R.clips.live;
      slot.set({
        reveal: current ? ease(tau, 4.2, 6.6) : 0,
        ghostOpacity: current ? 1 - ease(tau, CLIP_GATE, CLIP_START) : 0,
        still: recording ? 1 - ease(tau, CLIP_GATE, CLIP_START) + ease(tau, clipEnd, clipEnd + CLIP_FADE) : 1,
        speed: recording ? window01(tau, CLIP_START - 0.3, clipEnd + CLIP_FADE, 0.3) : 0,
        dim: current ? (within(tau, 4.2, clipEnd + CLIP_FADE) ? 0 : 0.2) : 0.5,
        active: current && executing,
        opacity,
      });
    });
    R.captions[0].setAttribute('opacity', task ? 0 : 1);
    R.captions[1].setAttribute('opacity', task ? 1 : 0);
    const fade = 1 - 0.45 * ease(tau, 7.8, 8.6);
    R.tracks.pnp.set(task ? 0 : progress, task ? 0 : fade);
    R.tracks.task.set(task ? progress : 0, task ? fade : 0);
  };
}

export function mountLoop(figure) {
  const clips = new SlotClips(PHOTOS.map((photo, i) => ({
    src: photo.clip.src, poster: photo.plain, duration: photo.clip.duration,
    gate: CYCLES[i].at + CLIP_GATE, start: CYCLES[i].at + CLIP_START, fade: CLIP_FADE,
  })));
  return new Scene(figure, {
    id: 'loop',
    duration: DURATION,
    finalTime: CYCLES[1].at - 0.01,
    status: STATUS,
    staticStatus: STATIC_STATUS,
    media: clips,
    layouts: [
      { name: 'wide', minWidth: 940, build: (stage) => buildWide(stage, clips) },
      { name: 'tall', minWidth: 0, build: (stage) => buildTall(stage, clips) },
    ],
  }).mount();
}
