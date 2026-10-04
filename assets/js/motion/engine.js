/* Scene engine for the animated method figures.

   One requestAnimationFrame clock drives every playing figure. A figure plays only while it is on screen,
   loops, and can be paused or replayed. Each figure picks a layout from its own width (wide or tall) and
   rebuilds when it crosses a breakpoint. Under prefers-reduced-motion a figure renders its final state once
   and stays still. `?freeze=<seconds>` renders every figure at that time, for screenshots. A figure can carry
   recordings (`media`, see clips.js): while one plays, the figure's clock follows the recording's. */

export const clamp = (v, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
export const lerp = (a, b, u) => a + (b - a) * u;
/** Smootherstep: zero velocity and acceleration at both ends. */
export const smooth = (u) => {
  const x = clamp(u);
  return x * x * x * (x * (x * 6 - 15) + 10);
};
/** Progress of t through [a, b], clamped to [0, 1]. */
export const span = (t, a, b) => clamp((t - a) / (b - a));
/** Eased progress of t through [a, b]. */
export const ease = (t, a, b) => smooth(span(t, a, b));
/** True while a <= t < b. */
export const within = (t, a, b) => t >= a && t < b;
/** 0 -> 1 -> 0 over [a, b], with eased edges of length `edge`. */
export const window01 = (t, a, b, edge = 0.25) => Math.min(ease(t, a, a + edge), 1 - ease(t, b - edge, b));

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const query = new URLSearchParams(window.location.search);
const FREEZE = query.has('freeze') ? Number(query.get('freeze')) : null;

/** Mounted scenes by id; exposed as window.URAI_SCENES for automated checks. */
export const scenes = new Map();
window.URAI_SCENES = scenes;

const playing = new Set();
let clockRunning = false;

function ensureClock() {
  if (clockRunning || playing.size === 0) return;
  clockRunning = true;
  let last = null;
  const tick = (now) => {
    if (playing.size === 0) {
      clockRunning = false;
      return;
    }
    const dt = last === null ? 0 : Math.min(0.1, (now - last) / 1000);
    last = now;
    for (const scene of playing) scene.advance(dt);
    window.requestAnimationFrame(tick);
  };
  window.requestAnimationFrame(tick);
}

document.addEventListener('visibilitychange', () => {
  for (const scene of scenes.values()) scene.sync();
});

export class Scene {
  /**
   * @param {HTMLElement} figure Root with [data-scene-stage]; optional [data-scene-status],
   *   [data-scene-progress] > span, [data-scene-toggle], [data-scene-replay].
   * @param {{id: string, duration: number, finalTime: number, staticStatus?: string,
   *   status?: Array<[number, string]>,
   *   layouts: Array<{name: string, minWidth: number, build: (stage: HTMLElement) => (t: number, final: boolean) => void}>}} spec
   *   `layouts` is ordered from the widest; the first whose minWidth fits the stage is used. With
   *   `keepStage`, the stage's own markup is kept and the layout only binds to it. `media`
   *   ({update(t, {running, active}), advance(t, dt)}) learns every rendered time and whether the clock runs,
   *   and gives the next time while the clock runs.
   */
  constructor(figure, spec) {
    this.figure = figure;
    this.spec = spec;
    this.stage = figure.querySelector('[data-scene-stage]');
    this.statusEl = figure.querySelector('[data-scene-status]');
    this.progressEl = figure.querySelector('[data-scene-progress] span');
    this.toggleBtn = figure.querySelector('[data-scene-toggle]');
    this.replayBtn = figure.querySelector('[data-scene-replay]');
    this.t = 0;
    this.layout = null;
    this.draw = null;
    this.visible = false;
    this.userPaused = false;
    this.running = false;
    this.isStatic = reducedMotion.matches || FREEZE !== null;
    this.statusText = '';
    scenes.set(spec.id, this);
  }

  mount() {
    this.relayout();
    new ResizeObserver(() => this.relayout()).observe(this.stage);
    reducedMotion.addEventListener('change', () => {
      this.isStatic = reducedMotion.matches || FREEZE !== null;
      this.applyStatic();
    });
    this.toggleBtn?.addEventListener('click', () => {
      this.userPaused = !this.userPaused;
      this.sync();
    });
    this.replayBtn?.addEventListener('click', () => this.replay());
    new IntersectionObserver(([entry]) => {
      this.visible = entry.isIntersecting;
      this.sync();
    }, { threshold: 0.25 }).observe(this.figure);
    this.applyStatic();
    return this;
  }

  applyStatic() {
    this.figure.classList.toggle('is-static', this.isStatic);
    if (this.isStatic) {
      playing.delete(this);
      this.running = false;
      if (FREEZE !== null) this.render(Math.min(FREEZE, this.spec.duration - 1e-3));
      else this.render(this.spec.finalTime, true);
    } else {
      this.render(this.t);
      this.sync();
    }
  }

  relayout() {
    const width = this.stage.clientWidth;
    if (!width) return;
    const layouts = this.spec.layouts;
    const layout = layouts.find((l) => width >= l.minWidth) ?? layouts[layouts.length - 1];
    if (layout === this.layout) return;
    this.layout = layout;
    this.figure.dataset.layout = layout.name;
    if (!this.spec.keepStage) this.stage.replaceChildren();
    this.draw = layout.build(this.stage);
    if (this.isStatic && FREEZE === null) this.render(this.spec.finalTime, true);
    else this.render(this.t);
  }

  render(t, final = false) {
    this.t = t;
    this.spec.media?.update(t, { running: this.running, active: !this.isStatic });
    this.draw?.(t, final);
    if (this.progressEl) this.progressEl.style.transform = `scaleX(${(t / this.spec.duration).toFixed(4)})`;
    if (!this.statusEl) return;
    let text = this.spec.staticStatus ?? '';
    if (!final && this.spec.status) {
      for (const [at, line] of this.spec.status) if (t >= at) text = line;
    }
    if (text !== this.statusText) {
      this.statusText = text;
      this.statusEl.textContent = text;
    }
  }

  advance(dt) {
    const next = this.spec.media ? this.spec.media.advance(this.t, dt) : this.t + dt;
    this.render(next % this.spec.duration);
  }

  /** Start or stop the clock from visibility, the user's pause and the page's visibility. */
  sync() {
    const run = !this.isStatic && this.visible && !this.userPaused && document.visibilityState === 'visible';
    this.running = run;
    this.spec.media?.update(this.t, { running: run, active: !this.isStatic });
    if (run) playing.add(this);
    else playing.delete(this);
    if (this.toggleBtn) {
      this.toggleBtn.setAttribute('aria-pressed', String(this.userPaused));
      this.toggleBtn.setAttribute('aria-label', this.userPaused ? 'Play animation' : 'Pause animation');
    }
    ensureClock();
  }

  replay() {
    this.userPaused = false;
    this.render(0);
    this.sync();
  }

  /** Render one frame and hold it (used by automated checks and screenshots). */
  seek(t) {
    this.userPaused = true;
    this.sync();
    this.render(t);
  }
}
