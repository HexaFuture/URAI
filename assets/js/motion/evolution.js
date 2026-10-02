/* Figure (b): tool programming and persistent evolution between episodes (paper Figure 2a).
   Failures and natural-language guidance reach the programming agent, which proposes a candidate
   phi_{k+1} = (L, C, P). Validation (contracts, regression, cost) either sends it back or lets it be frozen;
   accepted tools persist in URAI for later episodes. Foundation-model weights theta never change. */
import { Scene, ease, within, window01 } from './engine.js';
import {
  arrow, card, checkMark, chipRow, el, lockIcon, packet, personIcon, svgRoot, text, textWidth, toggle,
} from './svg.js';

const DURATION = 14;
const ARIA = 'Tool evolution: failures and human guidance reach the programming agent, which proposes a candidate '
  + 'revision of the tool collection, tool code and execution strategy; validated revisions are frozen and kept '
  + 'in URAI, while foundation-model weights stay fixed.';

export const STATUS = [
  [0, 'Development episodes surface failure cases; a person can also steer in natural language.'],
  [1.3, 'The programming agent writes or fixes a tool: its argument contract and code.'],
  [3.1, 'The candidate version changes the tool collection L, the tool code C or the execution strategy P.'],
  [4.1, 'Validation checks contracts, regressions and cost. A candidate that fails a check goes back for revision.'],
  [5.9, 'The programming agent revises the candidate.'],
  [8.5, 'The revised version passes every check …'],
  [10.1, '… and is frozen before evaluation.'],
  [10.8, 'Accepted tools persist in URAI across episodes; recurrent failures start the next round.'],
  [12.0, 'Foundation-model weights θ stay fixed: adaptation happens in φ = (L, C, P), never in θ.'],
];
export const STATIC_STATUS = 'Failures and human guidance → programming agent → candidate φ = (L, C, P) → validate '
  + '(contracts, regression, cost) → freeze → accepted tools in URAI for later episodes. Foundation-model weights θ stay fixed.';

const phi = (sub) => [['φ', { italic: true }], [sub, { sub: true }]];

function versionChip(parent, x, y, sub, size, h) {
  const g = el('g', { class: 's-chip version' }, parent);
  const w = Math.ceil(textWidth('φ', size) + textWidth(sub, size * 0.7) + 22);
  el('rect', { x, y, width: w, height: h, rx: 7 }, g);
  text(g, x + 11, y + h / 2 + size * 0.36, [phi(sub)], { size, cls: '' });
  return { g, w };
}

function thetaBadge(parent, xr, y, size) {
  const label = 'θ fixed';
  const w = textWidth(label, size, { weight: 600 }) + size + 22;
  const g = el('g', { class: 's-badge' }, parent);
  el('rect', { x: xr - w, y, width: w, height: size + 12, rx: 6 }, g);
  lockIcon(g, xr - w + 7, y + 4, size + 2);
  text(g, xr - w + size + 14, y + size * 0.36 + (size + 12) / 2, label, { size, cls: '' });
  return g;
}

/** Contracts / regression / cost, each pending, passed or failed. */
function checkItems(parent, items, size) {
  return items.map(([label, x, y]) => {
    const g = el('g', {}, parent);
    const pending = el('circle', { cx: x + size / 2, cy: y - size * 0.32, r: size * 0.42, class: 's-pending' }, g);
    const ok = checkMark(g, x, y - size * 0.8, size, true);
    const no = checkMark(g, x + size * 0.1, y - size * 0.72, size * 0.8, false);
    text(g, x + size + 8, y, label, { size, cls: 't-body' });
    return {
      set(state) {
        pending.setAttribute('opacity', state === null ? 1 : 0);
        ok.setAttribute('opacity', state === true ? 1 : 0);
        no.setAttribute('opacity', state === false ? 1 : 0);
      },
    };
  });
}

/** The three parts of phi: highlightable rows with a letter badge. */
function lcpRows(parent, x, y, w, rowH, sizes) {
  const rows = [['L', 'Tool collection', 'PnP, PushButton, T(τ)'], ['C', 'Tool code', 'executable definition'],
    ['P', 'Execution strategy', 'control and recovery']];
  return rows.map(([letter, name, desc], i) => {
    const ry = y + i * rowH;
    const hl = el('rect', { x: x - 8, y: ry, width: w + 16, height: rowH - 4, rx: 7, class: 's-row-hl' }, parent);
    const badge = el('g', { class: 's-letter' }, parent);
    const b = sizes.badge;
    el('rect', { x, y: ry + (rowH - 4 - b) / 2, width: b, height: b, rx: 6 }, badge);
    text(badge, x + b / 2, ry + (rowH - 4) / 2 + sizes.name * 0.38, letter, { size: sizes.name, anchor: 'middle', cls: '' });
    text(parent, x + b + 12, ry + sizes.nameY, name, { size: sizes.name, cls: 't-title' });
    text(parent, x + b + 12, ry + sizes.descY, desc, { size: sizes.desc, cls: 't-muted' });
    return { hl, badge };
  });
}

/** Three short bars that grow like code being typed. */
function codeLines(parent, x, y, widths, gap) {
  const lines = widths.map((w, i) => ({
    w, node: el('rect', { x, y: y + i * gap, width: 0, height: 4, rx: 2, class: `s-codeline ${i === 1 ? 'add' : ''}`.trim() }, parent),
  }));
  return {
    set(u) {
      lines.forEach((line, i) => {
        const k = Math.max(0, Math.min(1, u * lines.length - i));
        line.node.setAttribute('width', (line.w * k).toFixed(1));
      });
    },
  };
}

function buildWide(stage) {
  const svg = svgRoot(stage, 1200, 424, ARIA);
  const R = { svg };
  // human guidance, development episodes, programming agent
  R.bubble = card(svg, { x: 228, y: 0, w: 262, h: 46, r: 10 });
  R.bubble.setAttribute('class', 's-bubble');
  personIcon(svg, 244, 12, 20, 'var(--warm)');
  text(svg, 272, 29, 'human guidance in natural language', { size: 13, cls: 't-sub' });
  R.guide = arrow(svg, [[350, 47], [350, 73]], { cls: 'warm' });
  R.dev = card(svg, { x: 0, y: 96, w: 190, h: 124, r: 12 });
  text(svg, 18, 128, ['Development', 'episodes'], { size: 17, cls: 't-title', lh: 1.25 });
  text(svg, 18, 180, ['task outcome', 'failure cases'], { size: 14, cls: 't-muted', lh: 1.35 });
  R.failBadge = el('g', {}, svg);
  el('circle', { cx: 182, cy: 104, r: 13, fill: '#2a0f0b', stroke: 'var(--bad)', 'stroke-width': 1.5 }, R.failBadge);
  checkMark(R.failBadge, 176, 98, 12, false);
  R.prog = card(svg, { x: 228, y: 74, w: 244, h: 168, r: 14, cls: 'prog' });
  thetaBadge(svg, 462, 61, 13);
  text(svg, 246, 114, 'Programming agent', { size: 18, cls: 't-warm t-bold' });
  chipRow(svg, 246, 128, [{ label: 'GPT-6', size: 14, h: 28, cls: 'member', weight: 600 },
    { label: 'Claude', size: 14, h: 28, cls: 'member', weight: 600 }], 8);
  text(svg, 246, 182, ['writes or fixes a tool:', 'contract + code'], { size: 14, cls: 't-muted', lh: 1.3 });
  R.code = codeLines(svg, 246, 214, [150, 110, 180], 8);
  R.toProg = arrow(svg, [[190, 158], [227, 158]], { cls: 'warm' });

  // candidate, validation, freeze
  R.cand = card(svg, { x: 508, y: 60, w: 300, h: 196, r: 14, cls: 'cand' });
  text(svg, 526, 90, [[['Candidate ', {}], ...phi('k+1')]], { size: 18, cls: 't-warm t-bold' });
  R.rows = lcpRows(svg, 528, 104, 260, 48, { badge: 32, name: 15.5, nameY: 21, desc: 13, descY: 39 });
  R.toCand = arrow(svg, [[472, 158], [507, 158]], { cls: 'warm' });
  R.valid = card(svg, { x: 844, y: 88, w: 150, h: 140, r: 12, cls: 'valid' });
  checkMark(svg, 864, 106, 16, true);
  text(svg, 888, 121, 'Validate', { size: 17, cls: 't-good t-bold' });
  R.checks = checkItems(svg, [['contracts', 864, 158], ['regression', 864, 184], ['cost', 864, 210]], 14);
  R.toValid = arrow(svg, [[808, 158], [843, 158]], { cls: 'warm' });
  R.freeze = card(svg, { x: 1030, y: 88, w: 170, h: 140, r: 12, cls: 'freeze' });
  text(svg, 1048, 121, [[['Freeze ', {}], ...phi('k+1')]], { size: 17, cls: 't-blue t-bold' });
  R.lock = lockIcon(svg, 1150, 140, 30);
  text(svg, 1048, 192, 'held-out tasks', { size: 14, cls: 't-body' });
  text(svg, 1048, 212, 'fixed budget', { size: 14, cls: 't-muted' });
  R.toFreeze = arrow(svg, [[994, 158], [1029, 158]], { cls: 'warm' });

  // rejection, acceptance, persistence
  R.reject = arrow(svg, [[900, 229], [900, 272], [440, 272], [440, 244]], { cls: 'warm dashed thin bad' });
  R.rejectLabel = text(svg, 670, 291, 'fails a check: back to the programming agent', { size: 13, cls: 't-bad', anchor: 'middle' });
  R.accept = arrow(svg, [[1115, 229], [1115, 335]], { cls: 'warm dashed' });
  text(svg, 1103, 290, 'accepted tools', { size: 13.5, cls: 't-warm', anchor: 'end' });
  R.shelf = card(svg, { x: 480, y: 336, w: 720, h: 80, r: 14, cls: 'urai' });
  text(svg, 502, 371, 'URAI', { size: 26, cls: 't-blue t-mark' });
  text(svg, 502, 398, 'frozen tools, used by the execution agent in later episodes', { size: 13.5, cls: 't-muted' });
  R.versions = [versionChip(svg, 920, 361, 'k−1', 16, 30), versionChip(svg, 1004, 361, 'k', 16, 30)];
  R.newVersion = versionChip(svg, 1080, 361, 'k+1', 16, 30);
  R.weights = card(svg, { x: 0, y: 336, w: 440, h: 80, r: 14, cls: 'weights' });
  R.weightLock = lockIcon(svg, 20, 358, 28);
  text(svg, 62, 370, 'Foundation-model weights θ', { size: 15, cls: 't-title' });
  text(svg, 62, 394, 'fixed: adaptation happens in φ = (L, C, P), never in θ', { size: 13, cls: 't-muted' });
  R.recur = arrow(svg, [[480, 376], [462, 376], [462, 312], [95, 312], [95, 221]], { cls: 'warm dashed thin' });
  text(svg, 110, 305, 'recurrent failures', { size: 13.5, cls: 't-warm' });
  return makeDraw(R);
}

function buildTall(stage) {
  const svg = svgRoot(stage, 360, 904, ARIA);
  const R = { svg };
  R.dev = card(svg, { x: 22, y: 0, w: 150, h: 92, r: 12 });
  text(svg, 36, 26, ['Development', 'episodes'], { size: 14.5, cls: 't-title', lh: 1.2 });
  text(svg, 36, 66, ['task outcome,', 'failure cases'], { size: 12, cls: 't-muted', lh: 1.3 });
  R.failBadge = el('g', {}, svg);
  el('circle', { cx: 164, cy: 8, r: 11, fill: '#2a0f0b', stroke: 'var(--bad)', 'stroke-width': 1.5 }, R.failBadge);
  checkMark(R.failBadge, 159, 3, 10, false);
  R.bubble = card(svg, { x: 188, y: 0, w: 150, h: 92, r: 12 });
  R.bubble.setAttribute('class', 's-bubble');
  personIcon(svg, 202, 14, 18, 'var(--warm)');
  text(svg, 226, 28, 'human', { size: 14, cls: 't-sub t-bold' });
  text(svg, 202, 56, ['guidance in', 'natural language'], { size: 12, cls: 't-muted', lh: 1.3 });
  R.toProg = arrow(svg, [[97, 93], [97, 129]], { cls: 'warm' });
  R.guide = arrow(svg, [[263, 93], [263, 129]], { cls: 'warm' });

  R.prog = card(svg, { x: 22, y: 130, w: 316, h: 132, r: 14, cls: 'prog' });
  thetaBadge(svg, 330, 117, 12);
  text(svg, 38, 164, 'Programming agent', { size: 16, cls: 't-warm t-bold' });
  chipRow(svg, 38, 176, [{ label: 'GPT-6', size: 13, h: 26, cls: 'member', weight: 600 },
    { label: 'Claude', size: 13, h: 26, cls: 'member', weight: 600 }], 7);
  text(svg, 38, 224, 'writes or fixes a tool: contract + code', { size: 12.5, cls: 't-muted' });
  R.code = codeLines(svg, 38, 236, [140, 100, 170], 8);
  R.toCand = arrow(svg, [[180, 263], [180, 289]], { cls: 'warm' });

  R.cand = card(svg, { x: 22, y: 290, w: 316, h: 178, r: 14, cls: 'cand' });
  text(svg, 38, 318, [[['Candidate ', {}], ...phi('k+1')]], { size: 16, cls: 't-warm t-bold' });
  R.rows = lcpRows(svg, 40, 330, 280, 44, { badge: 28, name: 14, nameY: 18, desc: 12, descY: 34 });
  R.toValid = arrow(svg, [[180, 469], [180, 495]], { cls: 'warm' });

  R.valid = card(svg, { x: 22, y: 496, w: 316, h: 86, r: 12, cls: 'valid' });
  checkMark(svg, 38, 508, 14, true);
  text(svg, 60, 521, 'Validate', { size: 15, cls: 't-good t-bold' });
  R.checks = checkItems(svg, [['contracts', 38, 562], ['regression', 140, 562], ['cost', 250, 562]], 12.5);
  R.reject = arrow(svg, [[339, 539], [352, 539], [352, 196], [339, 196]], { cls: 'warm dashed thin bad', head: 7, r: 6 });
  R.rejectLabel = text(svg, 346, 286, '', { size: 1, cls: 't-bad' });
  R.toFreeze = arrow(svg, [[180, 583], [180, 609]], { cls: 'warm' });

  R.freeze = card(svg, { x: 22, y: 610, w: 316, h: 72, r: 12, cls: 'freeze' });
  text(svg, 38, 638, [[['Freeze ', {}], ...phi('k+1')]], { size: 15, cls: 't-blue t-bold' });
  text(svg, 38, 664, 'held-out tasks · fixed budget', { size: 12.5, cls: 't-muted' });
  R.lock = lockIcon(svg, 296, 626, 26);
  R.accept = arrow(svg, [[180, 683], [180, 715]], { cls: 'warm dashed' });
  text(svg, 192, 704, 'accepted tools', { size: 12, cls: 't-warm' });

  R.shelf = card(svg, { x: 22, y: 716, w: 316, h: 92, r: 14, cls: 'urai' });
  text(svg, 38, 745, 'URAI', { size: 19, cls: 't-blue t-mark' });
  text(svg, 92, 744, 'frozen tools for later episodes', { size: 12, cls: 't-muted' });
  R.versions = [versionChip(svg, 38, 762, 'k−1', 14, 28), versionChip(svg, 108, 762, 'k', 14, 28)];
  R.newVersion = versionChip(svg, 166, 762, 'k+1', 14, 28);
  R.weights = card(svg, { x: 22, y: 826, w: 316, h: 76, r: 14, cls: 'weights' });
  R.weightLock = lockIcon(svg, 36, 842, 24);
  text(svg, 72, 852, 'Foundation-model weights θ', { size: 14, cls: 't-title' });
  text(svg, 72, 872, ['fixed; adaptation happens in', 'φ = (L, C, P), never in θ'], { size: 12, cls: 't-muted', lh: 1.3 });
  R.recur = arrow(svg, [[21, 762], [9, 762], [9, 46], [21, 46]], { cls: 'warm dashed thin', head: 7, r: 6 });
  text(svg, 28, 709, 'recurrent failures', { size: 12, cls: 't-warm' });
  return makeDraw(R);
}

function makeDraw(R) {
  const svg = R.svg;
  const p = {
    dev: packet(svg, 'warm'), guide: packet(svg, 'warm'), cand: packet(svg, 'warm'), valid: packet(svg, 'warm'),
    freeze: packet(svg, 'warm'), accept: packet(svg, 'warm'), reject: packet(svg, 'bad'),
  };
  const setLock = (lock, open) => {
    lock.shackle.style.transform = `translateY(${(-open * lock.size * 0.18).toFixed(2)}px)`;
  };
  const popIn = (node, u) => {
    node.setAttribute('opacity', u.toFixed(3));
  };

  function drawFinal() {
    [R.dev, R.prog, R.cand, R.freeze, R.shelf, R.bubble].forEach((n) => toggle(n, 'is-active', false));
    toggle(R.valid, 'is-active', false);
    toggle(R.valid, 'is-rejected', false);
    Object.values(p).forEach((pk) => pk.node.setAttribute('opacity', 0));
    [R.guide, R.toProg, R.toCand, R.toValid, R.toFreeze, R.accept, R.reject, R.recur].forEach((a) => toggle(a.g, 'is-active', false));
    R.reject.g.setAttribute('opacity', 0.35);
    R.rejectLabel.setAttribute('opacity', 0.6);
    R.failBadge.setAttribute('opacity', 1);
    R.code.set(1);
    R.rows.forEach((row) => {
      toggle(row.hl, 'is-active', false);
      toggle(row.badge, 'is-active', false);
    });
    R.checks.forEach((c) => c.set(true));
    setLock(R.lock, 0);
    popIn(R.newVersion.g, 1);
    toggle(R.newVersion.g, 'is-active', false);
  }

  return (t, final) => {
    if (final) {
      drawFinal();
      return;
    }
    const second = t >= 5.9;
    toggle(R.dev, 'is-active', within(t, 0, 1.3));
    toggle(R.bubble, 'is-active', within(t, 0.3, 1.3));
    popIn(R.failBadge, ease(t, 0.15, 0.5) * (1 - 0.6 * ease(t, 12.6, 13.4)));
    toggle(R.prog, 'is-active', within(t, 1.2, 2.8) || within(t, 5.8, 7.2));
    toggle(R.cand, 'is-active', within(t, 3.0, 3.8) || within(t, 7.4, 8.2));
    toggle(R.valid, 'is-active', within(t, 4.0, 4.7) || within(t, 8.4, 9.8));
    toggle(R.valid, 'is-rejected', within(t, 4.7, 5.9));
    toggle(R.freeze, 'is-active', within(t, 10.0, 11.0));
    toggle(R.shelf, 'is-active', within(t, 11.4, 13.0));

    // packets between the stages
    p.dev.at(R.toProg.path, ease(t, 0.7, 1.25), window01(t, 0.65, 1.3, 0.1));
    p.guide.at(R.guide.path, ease(t, 0.6, 1.15), window01(t, 0.55, 1.2, 0.1));
    const toCand = within(t, 2.6, 3.2) ? ease(t, 2.7, 3.1) : ease(t, 7.0, 7.4);
    p.cand.at(R.toCand.path, toCand, window01(t, 2.65, 3.15, 0.08) + window01(t, 6.95, 7.45, 0.08));
    const toValid = t < 6 ? ease(t, 3.7, 4.05) : ease(t, 8.1, 8.45);
    p.valid.at(R.toValid.path, toValid, window01(t, 3.65, 4.1, 0.08) + window01(t, 8.05, 8.5, 0.08));
    p.freeze.at(R.toFreeze.path, ease(t, 9.7, 10.05), window01(t, 9.65, 10.1, 0.08));
    p.accept.at(R.accept.path, ease(t, 10.8, 11.5), window01(t, 10.75, 11.55, 0.1));
    p.reject.at(R.reject.path, ease(t, 5.0, 5.85), window01(t, 4.95, 5.9, 0.1));

    toggle(R.guide.g, 'is-active', within(t, 0.55, 1.25));
    toggle(R.toProg.g, 'is-active', within(t, 0.65, 1.3));
    toggle(R.toCand.g, 'is-active', within(t, 2.65, 3.15) || within(t, 6.95, 7.45));
    toggle(R.toValid.g, 'is-active', within(t, 3.65, 4.1) || within(t, 8.05, 8.5));
    toggle(R.toFreeze.g, 'is-active', within(t, 9.65, 10.1));
    toggle(R.accept.g, 'is-active', within(t, 10.75, 11.6));
    toggle(R.reject.g, 'is-active', within(t, 4.7, 5.95));
    R.reject.g.setAttribute('opacity', (0.25 + 0.75 * window01(t, 4.6, 6.0, 0.2)).toFixed(3));
    R.rejectLabel.setAttribute('opacity', (window01(t, 4.6, 6.0, 0.2)).toFixed(3));
    R.recur.path.setAttribute('stroke-dashoffset', (-t * 8).toFixed(1));
    toggle(R.recur.g, 'is-active', within(t, 12.2, 13.6));

    // the programming agent types, the candidate's parts light up
    R.code.set(second ? ease(t, 5.9, 7.0) : ease(t, 1.4, 2.6));
    const firstEdit = within(t, 3.0, 3.8);
    const secondEdit = within(t, 7.4, 8.2);
    const [L, C] = R.rows;
    [[L, secondEdit], [C, firstEdit || secondEdit], [R.rows[2], false]].forEach(([row, on]) => {
      toggle(row.hl, 'is-active', on);
      toggle(row.badge, 'is-active', on);
    });

    // validation: the first candidate fails regression, the revision passes all three checks
    const marks = second
      ? [t >= 8.7 ? true : null, t >= 9.0 ? true : null, t >= 9.3 ? true : null]
      : [t >= 4.3 ? true : null, t >= 4.7 ? false : null, null];
    R.checks.forEach((c, i) => c.set(marks[i]));

    setLock(R.lock, 1 - ease(t, 10.2, 10.6));
    popIn(R.newVersion.g, ease(t, 11.3, 11.7));
    toggle(R.newVersion.g, 'is-active', within(t, 11.3, 12.6));
    toggle(R.weights, 'is-active', within(t, 12.0, 13.4));
  };
}

export function mountEvolution(figure) {
  return new Scene(figure, {
    id: 'evolution',
    duration: DURATION,
    finalTime: DURATION - 0.01,
    status: STATUS,
    staticStatus: STATIC_STATUS,
    layouts: [
      { name: 'wide', minWidth: 940, build: buildWide },
      { name: 'tall', minWidth: 0, build: buildTall },
    ],
  }).mount();
}
