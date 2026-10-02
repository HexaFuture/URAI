/* URAI project page: header state, section tracking, figure scenes, chart reveal, BibTeX copy and the
   hexagon cursor. Everything degrades to a readable static page without JavaScript. */
import { mountEvolution } from './motion/evolution.js';
import { mountLoop } from './motion/loop.js';
import { mountTimeline } from './motion/timeline.js';

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

function initHeader() {
  const header = document.querySelector('.site-header');
  if (!header) return;
  header.querySelectorAll('.site-nav .nav-label').forEach((label) => {
    label.dataset.label = label.textContent.trim();
    const frame = document.createElement('span');
    frame.className = 'nav-window';
    label.before(frame);
    frame.append(label);
  });
  const refresh = () => header.classList.toggle('is-scrolled', window.scrollY > 24);
  refresh();
  window.addEventListener('scroll', refresh, { passive: true });

  // The link of the section that crosses the middle of the viewport is current; above Method none is.
  const links = new Map([...header.querySelectorAll('.site-nav a[href^="#"]')]
    .map((a) => [a.getAttribute('href').slice(1), a]));
  const setCurrent = (id) => links.forEach((a, key) => {
    if (key === id) a.setAttribute('aria-current', 'true');
    else a.removeAttribute('aria-current');
  });
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) if (entry.isIntersecting) setCurrent(entry.target.id);
  }, { rootMargin: '-45% 0px -50% 0px' });
  document.querySelectorAll('main > section[id]').forEach((section) => observer.observe(section));
}

function initFigures() {
  const mounts = [['fig-loop', mountLoop], ['fig-evolution', mountEvolution], ['fig-modes', mountTimeline]];
  for (const [id, mount] of mounts) {
    const figure = document.getElementById(id);
    if (figure) mount(figure);
  }
}

function initReveal() {
  const targets = document.querySelectorAll('.reveal');
  if (reducedMotion.matches || !('IntersectionObserver' in window)) {
    targets.forEach((t) => t.classList.add('is-visible'));
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add('is-visible');
      observer.unobserve(entry.target);
    }
  }, { threshold: 0.3 });
  targets.forEach((t) => observer.observe(t));
}

async function copyText(textValue) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(textValue);
    return;
  }
  const area = document.createElement('textarea');
  area.value = textValue;
  area.setAttribute('readonly', '');
  area.style.position = 'fixed';
  area.style.opacity = '0';
  document.body.append(area);
  area.select();
  document.execCommand('copy');
  area.remove();
}

function initCopy() {
  document.querySelectorAll('[data-copy]').forEach((button) => {
    const source = document.getElementById(button.dataset.copy);
    const label = button.querySelector('[data-copy-label]');
    let timer = 0;
    button.addEventListener('click', async () => {
      try {
        const lines = source.querySelectorAll('.bib-line');
        await copyText(lines.length ? [...lines].map((l) => l.textContent).join('\n') : source.textContent.trim());
        button.dataset.state = 'done';
        label.textContent = 'Copied';
      } catch {
        button.dataset.state = 'error';
        label.textContent = 'Select and copy';
      }
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        delete button.dataset.state;
        label.textContent = 'Copy';
      }, 2000);
    });
  });
}

/* A hexagon with two blue eyes follows fine pointers; a click sends out two hexagonal ripples. */
const HEXAGON = 'M24 8.5 L37.4 16.25 L37.4 31.75 L24 39.5 L10.6 31.75 L10.6 16.25 Z';
const INTERACTIVE = 'a[href], button, summary, [role="button"], label';
const NATIVE = 'input, textarea, select, pre, [contenteditable]';

function initCursor() {
  const fine = window.matchMedia('(any-hover: hover) and (any-pointer: fine)');
  const forced = window.matchMedia('(forced-colors: active)');
  if (!fine.matches || forced.matches || !window.PointerEvent) return;
  const root = document.documentElement;
  const cursor = document.createElement('div');
  cursor.className = 'hex-cursor';
  cursor.hidden = true;
  cursor.setAttribute('aria-hidden', 'true');
  cursor.innerHTML = `<svg viewBox="0 0 48 48"><g class="body"><path class="halo" d="${HEXAGON}"/>`
    + `<path class="shell" d="${HEXAGON}"/><g class="eyes"><rect x="17.5" y="19.5" width="4" height="9" rx="2"/>`
    + '<rect x="26.5" y="19.5" width="4" height="9" rx="2"/></g></g></svg>';
  const ripple = document.createElement('div');
  ripple.className = 'hex-ripple';
  ripple.hidden = true;
  ripple.setAttribute('aria-hidden', 'true');
  ripple.innerHTML = `<svg viewBox="0 0 48 48"><path d="${HEXAGON}"/><path d="${HEXAGON}"/></svg>`;
  document.body.append(ripple, cursor);
  const rings = ripple.querySelectorAll('path');

  const hide = () => {
    cursor.hidden = true;
    root.classList.remove('hex-cursor-on');
  };
  window.addEventListener('pointermove', (event) => {
    if (event.pointerType !== 'mouse') {
      hide();
      return;
    }
    const target = event.target instanceof Element ? event.target : null;
    if (target?.closest(NATIVE)) {
      hide();
      return;
    }
    cursor.style.transform = `translate3d(${event.clientX}px, ${event.clientY}px, 0)`;
    if (cursor.dataset.state !== 'press') cursor.dataset.state = target?.closest(INTERACTIVE) ? 'hover' : '';
    cursor.hidden = false;
    root.classList.add('hex-cursor-on');
  }, { passive: true });
  window.addEventListener('pointerdown', (event) => {
    if (event.pointerType !== 'mouse' || cursor.hidden) return;
    cursor.dataset.state = 'press';
    if (reducedMotion.matches || !ripple.animate) return;
    ripple.style.transform = `translate3d(${event.clientX}px, ${event.clientY}px, 0)`;
    ripple.hidden = false;
    rings.forEach((ring, i) => ring.animate(
      [{ opacity: 0.85, transform: 'scale(.9)' }, { opacity: 0, transform: 'scale(2.6)' }],
      { duration: 720, delay: i * 140, easing: 'cubic-bezier(.22, 1, .36, 1)', fill: 'both' },
    ));
    window.setTimeout(() => { ripple.hidden = true; }, 900);
  });
  window.addEventListener('pointerup', () => {
    if (cursor.dataset.state === 'press') cursor.dataset.state = '';
  });
  document.documentElement.addEventListener('pointerleave', hide);
  window.addEventListener('blur', hide);
}

initHeader();
initFigures();
initReveal();
initCopy();
initCursor();
