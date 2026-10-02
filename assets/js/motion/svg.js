/* Small SVG drawing layer shared by the method figures: elements, measured text, cards, chips,
   arrows with separate heads (so a class can recolour both), packets that ride a path, and icons. */

const NS = 'http://www.w3.org/2000/svg';

export function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== undefined && value !== null && value !== false) node.setAttribute(key, value);
  }
  if (parent) parent.appendChild(node);
  return node;
}

export function svgRoot(stage, width, height, label) {
  const svg = el('svg', {
    viewBox: `0 0 ${width} ${height}`, class: 'scene-svg', role: 'img', 'aria-label': label,
    preserveAspectRatio: 'xMidYMin meet',
  });
  stage.appendChild(svg);
  return svg;
}

let measureCtx = null;
const rootStyle = () => window.getComputedStyle(document.documentElement);

/** Rendered width of one line in user units, measured with the page's own font stacks. */
export function textWidth(str, size, { weight = 400, mono = false } = {}) {
  measureCtx ??= document.createElement('canvas').getContext('2d');
  const family = rootStyle().getPropertyValue(mono ? '--mono' : '--site-font').trim();
  measureCtx.font = `${weight} ${size}px ${family}`;
  return measureCtx.measureText(str).width;
}

/**
 * Text with one or more lines. `content` is a string, an array of lines, or an array of lines whose items
 * are runs [str, {cls, sub, italic}]. Returns the <text> element.
 */
export function text(parent, x, y, content, { size = 15, cls = 't-body', anchor = 'start', lh = 1.35 } = {}) {
  const node = el('text', { x, y, 'font-size': size, class: cls, 'text-anchor': anchor }, parent);
  const lines = Array.isArray(content) ? content : [content];
  lines.forEach((line, i) => {
    const runs = Array.isArray(line) ? line : [[line, {}]];
    runs.forEach(([str, opt = {}], j) => {
      const span = el('tspan', { class: opt.cls }, node);
      if (j === 0) {
        span.setAttribute('x', x);
        if (i > 0) span.setAttribute('dy', `${lh}em`);
      }
      if (opt.italic) span.setAttribute('font-style', 'italic');
      if (opt.sub) {
        span.setAttribute('font-size', size * 0.7);
        span.setAttribute('baseline-shift', 'sub');
      }
      span.textContent = str;
    });
  });
  return node;
}

export function card(parent, { x, y, w, h, r = 12, cls = '' }) {
  return el('rect', { x, y, width: w, height: h, rx: r, class: `s-card ${cls}`.trim() }, parent);
}

/** A rounded label. `x` is the left edge unless anchor = 'middle' | 'end'. Returns {g, x, y, w, h}. */
export function chip(parent, { x, y, label, size = 14, h = 28, padX = 10, cls = '', mono = false, weight = 500,
  anchor = 'start', icon = null }) {
  const iconW = icon ? size + 6 : 0;
  const w = Math.ceil(textWidth(label, size, { weight, mono }) + 2 * padX + iconW);
  const left = anchor === 'middle' ? x - w / 2 : anchor === 'end' ? x - w : x;
  const g = el('g', { class: `s-chip ${cls}`.trim() }, parent);
  el('rect', { x: left, y, width: w, height: h, rx: Math.min(7, h / 2) }, g);
  if (icon) icon(g, left + padX, y + (h - size) / 2, size);
  text(g, left + padX + iconW, y + h / 2 + size * 0.36, label, { size, cls: mono ? 't-mono' : '' })
    .setAttribute('font-weight', weight);
  return { g, x: left, y, w, h };
}

/** Lay chips out left to right; returns the chips and the x where the row ends. */
export function chipRow(parent, x, y, items, gap = 8) {
  const out = [];
  let cx = x;
  for (const item of items) {
    const c = chip(parent, { ...item, x: cx, y });
    out.push(c);
    cx += c.w + gap;
  }
  return { chips: out, end: cx - gap };
}

/** Path through points with rounded corners of radius r. */
export function roundedPath(points, r = 12) {
  let d = `M${points[0][0]} ${points[0][1]}`;
  for (let i = 1; i < points.length - 1; i += 1) {
    const [px, py] = points[i - 1];
    const [cx, cy] = points[i];
    const [nx, ny] = points[i + 1];
    const l1 = Math.hypot(cx - px, cy - py);
    const l2 = Math.hypot(nx - cx, ny - cy);
    const rr = Math.min(r, l1 / 2, l2 / 2);
    const ax = cx - ((cx - px) / l1) * rr;
    const ay = cy - ((cy - py) / l1) * rr;
    const bx = cx + ((nx - cx) / l2) * rr;
    const by = cy + ((ny - cy) / l2) * rr;
    d += ` L${ax.toFixed(1)} ${ay.toFixed(1)} Q${cx} ${cy} ${bx.toFixed(1)} ${by.toFixed(1)}`;
  }
  const [lx, ly] = points[points.length - 1];
  return `${d} L${lx} ${ly}`;
}

/**
 * An arrow along `points` (or an explicit `d` ending in direction `dir`), with its head drawn as a polygon
 * so that a class on the group recolours both. The head tip sits on the last point.
 */
export function arrow(parent, points, { cls = '', head = 9, r = 12, d = null, dir = null } = {}) {
  const g = el('g', { class: `s-arrow ${cls}`.trim() }, parent);
  const last = points[points.length - 1];
  const prev = dir ? [last[0] - dir[0], last[1] - dir[1]] : points[points.length - 2];
  const len = Math.hypot(last[0] - prev[0], last[1] - prev[1]);
  const ux = (last[0] - prev[0]) / len;
  const uy = (last[1] - prev[1]) / len;
  const base = [last[0] - ux * head, last[1] - uy * head];
  const trimmed = points.slice(0, -1).concat([[last[0] - ux * (head - 1), last[1] - uy * (head - 1)]]);
  const path = el('path', { d: d ?? roundedPath(trimmed, r) }, g);
  if (head) {
    const w = head * 0.62;
    el('polygon', {
      points: `${last[0]},${last[1]} ${base[0] - uy * w},${base[1] + ux * w} ${base[0] + uy * w},${base[1] - ux * w}`,
    }, g);
  }
  return { g, path };
}

/** A glowing dot that can be placed at a fraction of a path's length. */
export function packet(parent, cls = '', r = 6) {
  const node = el('circle', { r, class: `s-packet ${cls}`.trim(), opacity: 0 }, parent);
  return {
    node,
    at(path, u, opacity = 1) {
      if (opacity <= 0) {
        node.setAttribute('opacity', 0);
        return;
      }
      path.__len ??= path.getTotalLength();
      const p = path.getPointAtLength(Math.max(0, Math.min(1, u)) * path.__len);
      node.setAttribute('cx', p.x.toFixed(1));
      node.setAttribute('cy', p.y.toFixed(1));
      node.setAttribute('opacity', opacity.toFixed(3));
    },
  };
}

/** Numbered step marker. */
export function stepNumber(parent, cx, cy, n, r = 11) {
  const g = el('g', { class: 's-num' }, parent);
  el('circle', { cx, cy, r }, g);
  text(g, cx, cy + r * 0.38, String(n), { size: r * 1.1, anchor: 'middle', cls: '' });
  return g;
}

export function toggle(node, cls, on) {
  (node.g ?? node).classList.toggle(cls, Boolean(on));
}

export function setOpacity(node, value) {
  (node.g ?? node).setAttribute('opacity', Math.max(0, Math.min(1, value)).toFixed(3));
}

/* ------------------------------------------------------------------ icons (drawn in a size x size box) */
export function personIcon(parent, x, y, size, color = 'currentColor') {
  const g = el('g', { class: 's-icon-person', fill: color }, parent);
  el('circle', { cx: x + size / 2, cy: y + size * 0.3, r: size * 0.2 }, g);
  el('path', {
    d: `M${x + size * 0.12} ${y + size} Q${x + size * 0.12} ${y + size * 0.56} ${x + size / 2} ${y + size * 0.56}`
      + ` Q${x + size * 0.88} ${y + size * 0.56} ${x + size * 0.88} ${y + size} Z`,
  }, g);
  return g;
}

export function lockIcon(parent, x, y, size, { open = 0 } = {}) {
  const g = el('g', { class: 's-icon', style: 'color: var(--text-sub)' }, parent);
  const bw = size * 0.72;
  const bx = x + (size - bw) / 2;
  el('rect', { x: bx, y: y + size * 0.45, width: bw, height: size * 0.5, rx: size * 0.08 }, g);
  const shackle = el('path', {
    d: `M${bx + bw * 0.22} ${y + size * 0.45} V${y + size * 0.28} a${bw * 0.28} ${bw * 0.28} 0 0 1 ${bw * 0.56} 0 V${y + size * 0.45}`,
  }, g);
  shackle.style.transform = `translateY(${-open * size * 0.18}px)`;
  return { g, shackle, size };
}

export function checkMark(parent, x, y, size, ok = true) {
  const g = el('g', { class: `s-check ${ok ? 'ok' : 'no'}` }, parent);
  if (ok) el('path', { d: `M${x} ${y + size * 0.55} L${x + size * 0.38} ${y + size * 0.9} L${x + size} ${y + size * 0.12}` }, g);
  else el('path', { d: `M${x + size * 0.1} ${y + size * 0.1} L${x + size * 0.9} ${y + size * 0.9} M${x + size * 0.9} ${y + size * 0.1} L${x + size * 0.1} ${y + size * 0.9}` }, g);
  return g;
}

/** Clip path of a rounded rectangle; returns its url(). */
let clipSeq = 0;
export function roundedClip(defs, x, y, w, h, r) {
  clipSeq += 1;
  const id = `clip-${clipSeq}`;
  const clip = el('clipPath', { id }, defs);
  const rect = el('rect', { x, y, width: w, height: h, rx: r }, clip);
  return { url: `url(#${id})`, rect, id };
}
