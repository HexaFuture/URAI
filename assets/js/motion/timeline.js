/* Figure (c): what one model call controls (paper Figure 1b). One schematic episode per paradigm, drawn
   in HTML so the rows reflow: dots are model calls, bars robot motion. A shared playhead sweeps the three
   rows; each dot lights as it is reached and the call counter counts it.
   Direct end-effector control: nine equal steps. Code as Policies: one call writes the program, then code
   runs. Code as Policy with URAI: four tool executions of different lengths (FIG1_TOOL_DURATIONS). */
import { Scene, clamp, span } from './engine.js';

const DURATION = 9;
const SWEEP = [0.3, 7.3];
const TOOL_DURATIONS = [1.9, 0.7, 1.4, 0.8];   // relative lengths of the four tool executions, as in the paper
const PROGRAM_WRITE = 0.12;                    // share of the row taken by writing the program (schematic)

function segmentsFor(kind) {
  if (kind === 'program') return [[0, 1]];
  const weights = kind === 'steps' ? Array(9).fill(1) : TOOL_DURATIONS;
  const total = weights.reduce((a, b) => a + b, 0);
  let x = 0;
  return weights.map((w) => {
    const seg = [x / total, (x + w) / total];
    x += w;
    return seg;
  });
}

function h(tag, cls, parent, style = {}) {
  const node = document.createElement(tag);
  node.className = cls;
  Object.assign(node.style, style);
  parent.appendChild(node);
  return node;
}

/** Build the dots and bars of one row inside its [data-track]; returns draw(x, final). */
function buildRow(mode) {
  const kind = mode.dataset.mode;
  const track = mode.querySelector('[data-track]');
  const counter = mode.querySelector('[data-count]');
  track.replaceChildren();
  const ours = kind === 'tools';
  const segs = segmentsFor(kind).map(([a, b], i, all) => {
    const seg = h('div', 'tl-seg', track, { left: `${a * 100}%`, width: `${(b - a) * 100}%` });
    const dot = h('span', 'tl-dot', seg);
    const ring = h('span', 'tl-ring', seg);
    let code = null;
    let barLeft = 'calc(14px)';
    if (kind === 'program') {
      code = h('span', 'tl-code', seg);
      code.textContent = '{ }';
      barLeft = 'calc(58px)';
    }
    const bar = h('span', `tl-bar${ours ? ' ours' : ''}`, seg, { left: barLeft, right: i + 1 < all.length ? '5px' : '0' });
    const fill = h('span', 'tl-fill', bar);
    return { a, b, dot, ring, code, fill };
  });
  const head = h('span', 'tl-head', track);
  return (x, final) => {
    let calls = 0;
    for (const s of segs) {
      const reached = final || x >= s.a;
      if (reached) calls += 1;
      s.dot.classList.toggle('is-on', reached);
      const ping = final ? 0 : clamp(1 - (x - s.a) / 0.05);
      s.ring.style.opacity = reached && !final ? (ping * 0.9).toFixed(3) : '0';
      s.ring.style.transform = `scale(${(1 + (1 - ping) * 1.6).toFixed(3)})`;
      let motion;
      if (s.code) {
        s.code.classList.toggle('is-on', !final && x >= 0 && x < PROGRAM_WRITE);
        motion = final ? 1 : clamp((x - PROGRAM_WRITE) / (1 - PROGRAM_WRITE));
      } else {
        motion = final ? 1 : clamp((x - s.a) / (s.b - s.a));
      }
      s.fill.style.transform = `scaleX(${motion.toFixed(4)})`;
    }
    counter.value = String(calls);
    counter.textContent = String(calls);
    head.style.left = `${(x * 100).toFixed(3)}%`;
    head.style.opacity = final || x <= 0 || x >= 1 ? '0' : '1';
  };
}

export function mountTimeline(figure) {
  const modes = [...figure.querySelectorAll('[data-mode]')];
  return new Scene(figure, {
    id: 'modes',
    duration: DURATION,
    finalTime: DURATION - 0.01,
    keepStage: true,
    layouts: [{
      name: 'rows',
      minWidth: 0,
      build() {
        const rows = modes.map(buildRow);
        return (t, final) => {
          const x = span(t, SWEEP[0], SWEEP[1]);
          rows.forEach((draw) => draw(x, final));
        };
      },
    }],
  }).mount();
}
