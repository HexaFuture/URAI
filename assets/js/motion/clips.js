/* Recordings that continue the photo slots of a scene figure.

   A clip is the recording of the run a slot's still was taken from, starting on that still's frame. It plays in
   the slot's place: an HTML <video> laid under the figure's SVG in the box of the slot's frame, so the frame, the
   chips and the shade stay on top and the slot only has to fade its still out over it. While a clip plays, the
   scene's clock follows the video's own clock, so the figure and the recording cannot drift apart. At a clip's
   gate the scene waits until the clip's first frame is decoded, so the still never fades onto an empty slot.
   Clips are fetched once the scene first runs; a static scene (reduced motion, ?freeze) never fetches them. If
   the browser refuses to autoplay (a power-saving mode), the stills stay and the scene skips the recordings. */

const LEAD = 0.3; // s a clip lies under its still before the gate, so the video layer exists before the fade
const CUE = 0.05; // s a held clip may differ from the scene time before it is sought

export class SlotClips {
  /**
   * @param {Array<{src: string, poster: string, gate: number, start: number, duration: number, fade: number}>} specs
   *   Scene times in seconds. The scene waits at `gate` until the clip can play; the clip plays over
   *   [start, start + duration) and its last frame stays for `fade` while the still returns over it.
   */
  constructor(specs) {
    this.clips = specs.map((spec) => {
      const box = document.createElement('div');
      box.className = 'slot-clip';
      box.setAttribute('aria-hidden', 'true');
      const video = document.createElement('video');
      video.defaultMuted = true;
      video.muted = true;
      video.playsInline = true;
      video.preload = 'auto';
      video.poster = spec.poster;
      box.append(video);
      return { ...spec, end: spec.start + spec.duration, box, video, frame: null };
    });
    this.stage = null;
    this.fetched = false;
    this.refused = false;
    /** The clips take part: the scene animates and autoplay was not refused. */
    this.live = false;
    this.resizer = new ResizeObserver(() => this.place());
  }

  /** Lay the clips under their slots' frames (SVG rects, in the order of the specs) once a layout is built. */
  attach(stage, frames) {
    if (stage !== this.stage) {
      this.resizer.disconnect();
      this.resizer.observe(stage);
      this.stage = stage;
    }
    this.clips.forEach((clip, i) => {
      clip.frame = frames[i];
      stage.prepend(clip.box);
    });
    this.place();
  }

  /** Give each clip its frame's box, in pixels from the stage, with the frame's corner radius. */
  place() {
    const origin = this.stage.getBoundingClientRect();
    for (const clip of this.clips) {
      const box = clip.frame.getBoundingClientRect();
      const svg = clip.frame.ownerSVGElement;
      const scale = svg.getBoundingClientRect().width / svg.viewBox.baseVal.width;
      Object.assign(clip.box.style, {
        left: `${box.left - origin.left}px`,
        top: `${box.top - origin.top}px`,
        width: `${box.width}px`,
        height: `${box.height}px`,
        borderRadius: `${clip.frame.rx.baseVal.value * scale}px`,
      });
    }
  }

  /** Show, play, hold or cue every clip for scene time t; called on each render and when the scene starts or stops. */
  update(t, { running, active }) {
    this.live = active && !this.refused;
    if (this.live && running && !this.fetched) {
      this.fetched = true;
      this.clips.forEach((clip) => { clip.video.src = clip.src; });
    }
    for (const clip of this.clips) {
      const shown = this.live && this.fetched && t >= clip.gate - LEAD && t < clip.end + clip.fade;
      clip.box.classList.toggle('is-shown', shown);
      if (!this.fetched) continue;
      const { video } = clip;
      if (this.live && running && t >= clip.start && t < clip.end) {
        if (video.paused && !video.ended) this.play(clip);
        continue;
      }
      if (!video.paused) video.pause();
      // Held (the pause button, a seek) or between passes: the frame for t, or the first frame for the next pass.
      const at = shown ? Math.min(Math.max(t - clip.start, 0), clip.duration) : 0;
      if (Math.abs(video.currentTime - at) > CUE) video.currentTime = at;
    }
  }

  play(clip) {
    clip.video.play().catch((error) => {
      if (error.name !== 'NotAllowedError') return; // AbortError: a pause() came first
      this.refused = true;
      this.live = false;
      this.clips.forEach((c) => c.box.classList.remove('is-shown'));
    });
  }

  /** The scene's next time: held at a gate until the clip's first frame is ready, then the playing clip's clock. */
  advance(t, dt) {
    const next = t + dt;
    const clip = this.clips.find((c) => next >= c.gate && t < c.end);
    if (!clip) return next;
    if (!this.live) return next >= clip.start ? Math.max(next, clip.end) : next;
    const { video } = clip;
    if (t < clip.start) {
      const ready = video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !video.seeking && video.currentTime < CUE;
      return ready ? next : Math.max(t, clip.gate);
    }
    if (video.ended) return clip.end;
    // Starting, seeking or buffering: the figure waits for the recording.
    if (video.paused || video.seeking || video.readyState < HTMLMediaElement.HAVE_FUTURE_DATA) return t;
    return Math.max(t, clip.start + video.currentTime);
  }
}
