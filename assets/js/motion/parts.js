/* Building blocks shared by the method figures: agent cards, the URAI box with its Ground -> Plan ->
   Execute stages and their glyphs, and a photo slot whose ghost trail is wiped in during execution and whose
   still can give way to the recording that continues it. */
import { clamp, ease } from './engine.js';
import {
  card, chip, chipRow, el, personIcon, roundedClip, stepNumber, text, textWidth, toggle,
} from './svg.js';

const personGlyph = (g, x, y, size) => personIcon(g, x, y + 1, size - 2, 'var(--text-sub)');

/**
 * A frontier-model role card. members: [{label, person?}] drawn as chips, or `membersText` as one line.
 * Returns {g, rect}.
 */
export function agentCard(parent, o) {
  const g = el('g', {}, parent);
  const rect = card(g, { x: o.x, y: o.y, w: o.w, h: o.h, r: o.r ?? 14, cls: o.cls });
  text(g, o.x + o.pad, o.y + o.titleY, o.title, { size: o.titleSize, cls: `${o.titleCls} t-bold` });
  if (o.members) {
    chipRow(g, o.x + o.pad, o.y + o.membersY, o.members.map((m) => ({
      label: m.label, size: o.chipSize, h: o.chipH, cls: 'member', weight: 600, padX: 9,
      icon: m.person ? personGlyph : null,
    })), 8);
  } else {
    text(g, o.x + o.pad, o.y + o.membersY, o.membersText, { size: o.chipSize, cls: 't-sub t-bold' });
  }
  text(g, o.x + o.pad, o.y + o.descY, o.desc, { size: o.descSize, cls: 't-muted', lh: 1.3 });
  return { g, rect };
}

/* ------------------------------------------------------------------ stage glyphs */

/** Ground: a pixel on the image plane becomes a 3D point. */
function groundGlyph(parent, x, y, w, h) {
  const g = el('g', {}, parent);
  const pw = h * 1.25;
  el('rect', { x, y: y + 1, width: pw, height: h - 2, rx: 2, class: 's-glyph' }, g);
  const px = x + pw * 0.62;
  const py = y + h * 0.45;
  el('path', { d: `M${px - 4} ${py} H${px + 4} M${px} ${py - 4} V${py + 4}`, class: 's-glyph' }, g);
  const ox = x + w - h * 0.75;
  const oy = y + h * 0.7;
  const s = h * 0.62;
  el('path', { d: `M${ox} ${oy} V${oy - s} M${ox} ${oy} H${ox + s * 0.95} M${ox} ${oy} L${ox - s * 0.55} ${oy + s * 0.4}`, class: 's-glyph' }, g);
  const link = el('path', { d: `M${x + pw + 4} ${y + h / 2} H${ox - s * 0.7}`, class: 's-glyph', 'stroke-dasharray': '2 4' }, g);
  const dot = el('circle', { r: 2.8, class: 's-glyph fill' }, g);
  return {
    set(u) {
      const k = clamp(u);
      dot.setAttribute('cx', (px + (ox - px) * k).toFixed(1));
      dot.setAttribute('cy', (py + (oy - py) * k).toFixed(1));
      link.setAttribute('opacity', k > 0 && k < 1 ? 1 : 0.5);
    },
  };
}

/** Plan: a path whose time stamps bunch up where the motion is slow (approach and placement). */
function planGlyph(parent, x, y, w, h) {
  const g = el('g', {}, parent);
  const x0 = x + 2;
  const x1 = x + w - 2;
  const d = `M${x0} ${y + h - 2} C${x0 + (x1 - x0) * 0.2} ${y - h * 0.5}, ${x0 + (x1 - x0) * 0.8} ${y - h * 0.5}, ${x1} ${y + h - 2}`;
  const path = el('path', { d, class: 's-glyph' }, g);
  const len = path.getTotalLength();
  path.setAttribute('stroke-dasharray', `${len} ${len}`);
  // Equal time steps of a slow-fast-slow speed profile: dense at both ends, sparse in the middle.
  const stamps = [0, 0.04, 0.1, 0.19, 0.31, 0.5, 0.69, 0.81, 0.9, 0.96, 1];
  const dots = stamps.map((s) => {
    const p = path.getPointAtLength(s * len);
    return { s, node: el('circle', { cx: p.x.toFixed(1), cy: p.y.toFixed(1), r: 2.2, class: 's-glyph fill' }, g) };
  });
  return {
    set(u, opacity = 1) {
      const k = clamp(u);
      path.setAttribute('stroke-dashoffset', ((1 - k) * len).toFixed(1));
      for (const dot of dots) dot.node.setAttribute('opacity', dot.s <= k + 1e-6 ? opacity : 0);
      g.setAttribute('opacity', opacity);
    },
  };
}

/** Execute: references streamed at a fixed rate, and the motion's progress. */
function executeGlyph(parent, x, y, w, h) {
  const g = el('g', {}, parent);
  const n = 12;
  const gap = 3;
  const tw = (w - gap * (n - 1)) / n;
  const ticks = Array.from({ length: n }, (_, i) => el('rect', {
    x: (x + i * (tw + gap)).toFixed(1), y, width: tw.toFixed(1), height: h - 7, rx: 1, class: 's-tick',
  }, g));
  el('rect', { x, y: y + h - 3, width: w, height: 3, rx: 1.5, class: 's-tick' }, g);
  const fill = el('rect', { x, y: y + h - 3, width: 0, height: 3, rx: 1.5, class: 's-tick is-on' }, g);
  return {
    set(phase, progress) {
      const lit = phase === null ? -1 : Math.floor(phase) % n;
      ticks.forEach((tick, i) => tick.classList.toggle('is-on', lit >= 0 && (i === lit || i === (lit + n - 1) % n)));
      fill.setAttribute('width', (w * clamp(progress)).toFixed(1));
    },
  };
}

/* ------------------------------------------------------------------ URAI box */

/**
 * The shared interface: wordmark, GUI | API, tool collection L and the per-call stages.
 * o: {x, y, w, h, markSize, tagline (lines), tagSize, chipSize, chipH, toolsY, stageY, stageH, stageW, stageGap,
 *     stageX, nameSize, noteSize, divider: 'v' | 'h', dividerAt, rightX, stepAt: [cx, cy] | null, caption}
 * Returns refs {rect, tools: {PnP, PushButton, more, T}, stages: {ground, plan, execute}}.
 */
export function uraiBox(parent, o) {
  const g = el('g', {}, parent);
  const rect = card(g, { x: o.x, y: o.y, w: o.w, h: o.h, r: 16, cls: 'urai' });
  const markW = textWidth('URAI', o.markSize, { weight: 700 });
  text(g, o.x + o.pad, o.y + o.markY, 'URAI', { size: o.markSize, cls: 't-blue t-mark' });
  chip(g, {
    x: o.x + o.pad + markW + 12, y: o.y + o.markY - o.markSize * 0.72 - 2, label: 'GUI | API',
    size: o.chipSize - 1, h: o.chipH - 2, cls: 'neutral', weight: 600,
  });
  text(g, o.x + o.pad, o.y + o.tagY, o.tagline, { size: o.tagSize, cls: 't-muted', lh: 1.3 });
  if (o.divider === 'v') {
    el('line', { x1: o.dividerAt, y1: o.y + 16, x2: o.dividerAt, y2: o.y + o.h - 16, class: 's-divider' }, g);
  } else {
    el('line', { x1: o.x + 12, y1: o.dividerAt, x2: o.x + o.w - 12, y2: o.dividerAt, class: 's-divider' }, g);
  }
  const toolsLabelW = textWidth('Tools L', o.toolsSize, { weight: 600 });
  text(g, o.rightX, o.toolsY + o.chipH / 2 + o.toolsSize * 0.36,
    [[['Tools ', {}], ['L', { italic: true }]]], { size: o.toolsSize, cls: 't-title' });
  const row = chipRow(g, o.rightX + toolsLabelW + 12, o.toolsY, [
    { label: 'PnP', size: o.chipSize, h: o.chipH },
    { label: 'PushButton', size: o.chipSize, h: o.chipH },
    { label: '…', size: o.chipSize, h: o.chipH, cls: 'neutral' },
    { label: 'T(τ)', size: o.chipSize, h: o.chipH, cls: 'task' },
  ], 7);
  const [PnP, PushButton, more, T] = row.chips;
  if (o.stepAt) stepNumber(g, o.stepAt[0], o.stepAt[1], 2, o.stepR ?? 11);

  const names = [['Ground', 'pixels → 3D'], ['Plan', 'timed path'], ['Execute', 'local, 50 Hz']];
  const stages = {};
  names.forEach(([name, note], i) => {
    const bx = o.stageX + i * (o.stageW + o.stageGap);
    const sg = el('g', { class: 's-stage-group' }, g);
    const box = el('rect', { x: bx, y: o.stageY, width: o.stageW, height: o.stageH, rx: 9, class: 's-stage' }, sg);
    text(sg, bx + o.stagePad, o.stageY + o.nameY, name, { size: o.nameSize, cls: 't-title' });
    text(sg, bx + o.stagePad, o.stageY + o.noteY, note, { size: o.noteSize, cls: 't-muted' });
    const gx = bx + o.stagePad;
    const gw = o.stageW - 2 * o.stagePad;
    const gy = o.stageY + o.glyphY;
    const glyph = i === 0 ? groundGlyph(sg, gx, gy, gw, o.glyphH)
      : i === 1 ? planGlyph(sg, gx, gy, gw, o.glyphH) : executeGlyph(sg, gx, gy, gw, o.glyphH);
    if (i > 0) {
      const ax = bx - o.stageGap;
      const ay = o.stageY + o.stageH / 2;
      el('path', { d: `M${ax + 3} ${ay} H${bx - 4}`, class: 's-glyph', 'stroke-width': 1.6 }, g);
      el('polygon', { points: `${bx - 2},${ay} ${bx - 8},${ay - 4} ${bx - 8},${ay + 4}`, class: 's-glyph fill' }, g);
    }
    stages[['ground', 'plan', 'execute'][i]] = { g: sg, box, glyph };
  });
  if (o.caption) {
    text(g, o.stageX + (3 * o.stageW + 2 * o.stageGap) / 2, o.captionY, o.caption,
      { size: o.noteSize - 0.5, cls: 't-dim', anchor: 'middle' });
  }
  return { g, rect, tools: { PnP, PushButton, more, T }, stages };
}

/* ------------------------------------------------------------------ photo slot */

/**
 * A demonstration frame and its ghost-trail composite (earlier arm poses within one tool call). The ghost is
 * wiped in along the direction of the motion. With `photo.clip`, the still can fade out over the recording that
 * continues it (laid under the SVG in the frame's box, see clips.js) while a chip gives the recording's speed.
 * Returns {g, frame, set({reveal, ghostOpacity, still, speed, dim, active, opacity})}.
 */
export function photoSlot(parent, defs, o, photo) {
  const g = el('g', {}, parent);
  const clip = roundedClip(defs, o.x, o.y, o.w, o.h, 8);
  const body = el('g', { 'clip-path': clip.url }, g);
  const still = el('image', {
    href: photo.plain, x: o.x, y: o.y, width: o.w, height: o.h, preserveAspectRatio: 'xMidYMid slice', class: 'no-fade',
  }, body);
  const wipe = roundedClip(defs, o.x, o.y, 0, o.h, 0);
  const ghostWrap = el('g', { 'clip-path': wipe.url }, body);
  // Opacities of the still and the trail are eased per frame and must not lag the recording under them.
  const ghost = el('image', {
    href: photo.ghost, x: o.x, y: o.y, width: o.w, height: o.h, preserveAspectRatio: 'xMidYMid slice', class: 'no-fade',
  }, ghostWrap);
  const shade = el('rect', { x: o.x, y: o.y, width: o.w, height: o.h, fill: '#000', opacity: 0, class: 'no-fade' }, body);
  const frame = el('rect', { x: o.x, y: o.y, width: o.w, height: o.h, rx: 8, class: 's-photo-frame' }, g);
  chip(g, {
    x: photo.labelSide === 'right' ? o.x + o.w - 10 : o.x + 10, anchor: photo.labelSide === 'right' ? 'end' : 'start',
    y: o.y + 10, label: photo.label, size: o.labelSize, h: o.labelH, cls: 's-photo-label', weight: 600,
  });
  chip(g, {
    x: o.x + o.w - 10, anchor: 'end', y: o.y + o.h - o.labelH - 10, label: photo.call, size: o.labelSize - 0.5,
    h: o.labelH, cls: photo.kind === 'task' ? 'task' : '', mono: true, weight: 500,
  });
  // The speed of the recording, in the top corner the label leaves free.
  const speed = photo.clip ? chip(g, {
    x: photo.labelSide === 'right' ? o.x + 10 : o.x + o.w - 10, anchor: photo.labelSide === 'right' ? 'start' : 'end',
    y: o.y + 10, label: photo.clip.speed, size: o.labelSize - 0.5, h: o.labelH, cls: 's-photo-label speed', mono: true,
    weight: 500,
  }) : null;
  return {
    g,
    frame,
    set({ reveal = 0, ghostOpacity = 1, still: stillOpacity = 1, speed: speedOpacity = 0, dim = 0, active = false,
      opacity = 1 }) {
      const k = clamp(reveal);
      const ww = o.w * k;
      wipe.rect.setAttribute('width', ww.toFixed(1));
      wipe.rect.setAttribute('x', (photo.wipe === 'rtl' ? o.x + o.w - ww : o.x).toFixed(1));
      ghost.setAttribute('opacity', ghostOpacity.toFixed(3));
      still.setAttribute('opacity', clamp(stillOpacity).toFixed(3));
      speed?.g.setAttribute('opacity', clamp(speedOpacity).toFixed(3));
      shade.setAttribute('opacity', clamp(dim).toFixed(3));
      toggle(frame, 'is-active', active);
      g.setAttribute('opacity', clamp(opacity).toFixed(3));
    },
  };
}

/** The five phases of one PnP call, or one bar for a task-specific tool. */
export function phaseTrack(parent, o, phases) {
  const g = el('g', {}, parent);
  const n = phases.length;
  const gap = o.gap ?? 6;
  const pw = (o.w - gap * (n - 1)) / n;
  const items = phases.map((name, i) => {
    const pg = el('g', { class: `s-phase ${o.task ? 'task' : ''}`.trim() }, g);
    el('rect', { x: o.x + i * (pw + gap), y: o.y, width: pw, height: o.h, rx: 6 }, pg);
    text(pg, o.x + i * (pw + gap) + pw / 2, o.y + o.h / 2 + o.size * 0.36, name, { size: o.size, anchor: 'middle', cls: '' });
    return pg;
  });
  return {
    g,
    set(progress, opacity = 1) {
      items.forEach((item, i) => item.classList.toggle('is-on', progress > 0 && progress >= i / n));
      g.setAttribute('opacity', clamp(opacity).toFixed(3));
    },
  };
}

export { ease };
