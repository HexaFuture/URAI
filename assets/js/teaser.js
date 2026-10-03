/* Teaser: each still is a frame of the robot recording, with the arm's earlier poses traced over it. After a
   short hold the recording of the same run plays on from that frame: a clip's first frame is its still, so the
   crossfade only lifts the pose trail. A clip that ends fades back to its still; once both have ended the hold
   starts again, so the whole teaser loops. It runs while on screen and the tab is visible, and the button
   pauses it. Under prefers-reduced-motion (and ?freeze, used for screenshots) the stills stay and no video is
   fetched. The crossfade itself is the opacity transition on .photo-video in site.css. */

const HOLD_MS = 3000; // stills on screen before the clips start, including the fade back from the clips

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const FREEZE = new URLSearchParams(window.location.search).has('freeze');

export function mountTeaser(section) {
  const clips = [...section.querySelectorAll('.photo-video')]
    .map((video) => ({ video, figure: video.closest('.photo'), done: false }));
  const toggle = section.querySelector('[data-teaser-toggle]');
  /** Exposed as window.URAI_TEASER for automated checks. */
  const teaser = { phase: 'still', cycles: 0, visible: false, userPaused: false, isStatic: true };
  window.URAI_TEASER = teaser;
  let timer = 0;
  let fetched = false;

  const running = () => !teaser.isStatic && teaser.visible && !teaser.userPaused
    && document.visibilityState === 'visible';

  function play(clip) {
    clip.video.play().catch((error) => {
      // Autoplay refused (a power-saving or data-saving mode): keep the stills. AbortError only means pause() won.
      if (error.name === 'NotAllowedError') applyStatic(true);
    });
  }

  function startClips() {
    teaser.phase = 'video';
    teaser.cycles += 1;
    for (const clip of clips) {
      clip.done = false;
      clip.video.currentTime = 0; // the frame of the still
      play(clip);
    }
  }

  /** Start or stop from visibility, the user's pause and the page's visibility. */
  function sync() {
    window.clearTimeout(timer);
    toggle.setAttribute('aria-pressed', String(teaser.userPaused));
    toggle.setAttribute('aria-label', teaser.userPaused ? 'Play videos' : 'Pause videos');
    if (!running()) {
      clips.forEach((clip) => clip.video.pause());
      return;
    }
    if (teaser.phase === 'video') clips.filter((clip) => !clip.done).forEach(play);
    else timer = window.setTimeout(startClips, HOLD_MS);
  }

  function applyStatic(refused = false) {
    teaser.isStatic = refused || reducedMotion.matches || FREEZE;
    section.classList.toggle('is-static', teaser.isStatic);
    toggle.hidden = teaser.isStatic;
    if (teaser.isStatic) {
      teaser.phase = 'still';
      clips.forEach((clip) => clip.figure.classList.remove('is-video'));
    } else if (!fetched) {
      fetched = true;
      clips.forEach((clip) => {
        clip.video.preload = 'auto';
        clip.video.load();
      });
    }
    sync();
  }

  for (const clip of clips) {
    clip.video.addEventListener('playing', () => clip.figure.classList.add('is-video'));
    clip.video.addEventListener('ended', () => {
      clip.done = true;
      clip.figure.classList.remove('is-video');
      if (clips.every((c) => c.done)) {
        teaser.phase = 'still';
        sync();
      }
    });
  }
  toggle.addEventListener('click', () => {
    teaser.userPaused = !teaser.userPaused;
    sync();
  });
  reducedMotion.addEventListener('change', () => applyStatic());
  document.addEventListener('visibilitychange', sync);
  new IntersectionObserver(([entry]) => {
    teaser.visible = entry.isIntersecting;
    sync();
  }, { threshold: 0.25 }).observe(section);
  applyStatic();
}
