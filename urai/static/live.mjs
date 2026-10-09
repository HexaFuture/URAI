/** MJPEG receiver: fetch() streams the multipart body, only the newest part is decoded (createImageBitmap, off the
 * main thread) and painted on the next animation frame. The observation frame stays the geometry source. */
export const LIVE_LABEL = '实时 RGB · 几何用落笔时的观测';
export const SNAPSHOT_LABEL = '观测快照';
export const LIVE_STATUS_LABELS = {
  connecting: '实时 RGB · 正在连接相机流…',
  live: LIVE_LABEL,
  error: '实时画面连接失败，正在重试…',
  stopped: SNAPSHOT_LABEL,
};
export const BOUNDARY = 'urai-frame';
const STATS_WINDOW_MS = 1000;
const HEADER_END = new Uint8Array([13, 10, 13, 10]);

/** Heading text for a painter state; painted fps and the latency estimate are appended once known. */
export function liveStatusLabel(state, stats) {
  if (state !== 'live' || !stats) return LIVE_STATUS_LABELS[state];
  return `实时 RGB · ${stats.fps.toFixed(0)} fps · 时延约 ${Math.round(stats.latencyMs)} ms · 几何用落笔时的观测`;
}

function indexOf(haystack, needle, from) {
  const last = haystack.length - needle.length;
  outer: for (let i = from; i <= last; i++) {
    for (let j = 0; j < needle.length; j++) if (haystack[i + j] !== needle[j]) continue outer;
    return i;
  }
  return -1;
}

/** Incremental multipart/x-mixed-replace parser: push() returns every complete part received so far. */
export class MultipartParser {
  constructor(boundary = BOUNDARY) {
    this.marker = new TextEncoder().encode(`--${boundary}\r\n`);
    this.buffer = new Uint8Array(0);
    this.decoder = new TextDecoder();
  }

  push(chunk) {
    if (this.buffer.length === 0) this.buffer = chunk;
    else {
      const merged = new Uint8Array(this.buffer.length + chunk.length);
      merged.set(this.buffer);
      merged.set(chunk, this.buffer.length);
      this.buffer = merged;
    }
    const parts = [];
    for (;;) {
      const start = indexOf(this.buffer, this.marker, 0);
      if (start < 0) {
        // Bytes before a marker belong to nothing; keep a tail that could be the head of a split marker.
        this.buffer = this.buffer.subarray(Math.max(0, this.buffer.length - this.marker.length + 1));
        return parts;
      }
      const headerStart = start + this.marker.length;
      const headerEnd = indexOf(this.buffer, HEADER_END, headerStart);
      if (headerEnd < 0) { this.buffer = this.buffer.subarray(start); return parts; }
      const headers = {};
      for (const line of this.decoder.decode(this.buffer.subarray(headerStart, headerEnd)).split('\r\n')) {
        const colon = line.indexOf(':');
        headers[line.slice(0, colon).trim().toLowerCase()] = line.slice(colon + 1).trim();
      }
      if (!('content-length' in headers)) throw new Error('multipart part without Content-Length');
      const length = Number(headers['content-length']);
      const bodyStart = headerEnd + HEADER_END.length;
      if (this.buffer.length < bodyStart + length) { this.buffer = this.buffer.subarray(start); return parts; }
      parts.push({headers, bytes: this.buffer.slice(bodyStart, bodyStart + length)});
      this.buffer = this.buffer.subarray(bodyStart + length);
    }
  }
}

export class LivePainter {
  constructor({url = '/api/live.mjpg', retryMs = 1000, statsIntervalMs = 1000, paint, status = () => {},
               fetchStream = (target, signal) => fetch(target, {signal, cache: 'no-store'}),
               decode = blob => createImageBitmap(blob),
               requestFrame = callback => requestAnimationFrame(callback),
               cancelFrame = id => cancelAnimationFrame(id),
               now = () => performance.now(),
               schedule = (fn, ms) => setTimeout(fn, ms),
               unschedule = id => clearTimeout(id)} = {}) {
    Object.assign(this, {url, retryMs, statsIntervalMs, paint, status, fetchStream, decode,
                         requestFrame, cancelFrame, now, schedule, unschedule});
    this.running = false;
    this.ready = false;
    this.bitmap = null;
    this.pending = null;
    this.decoding = false;
    this.dirty = false;
    this.controller = null;
    this.generation = 0;
    this.frameId = null;
    this.retryId = null;
    this.latest = null;
    this.received = [];
    this.painted = [];
    this.dropped = 0;
    this.lastStatsAt = -Infinity;
    this.tick = this.tick.bind(this);
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.received = [];
    this.painted = [];
    this.dropped = 0;
    this.latest = null;
    this.connect();
  }

  stop() {
    if (!this.running) return;
    this.running = false;
    this.generation++;  // reads and decodes still in flight for the old stream are discarded
    this.disconnect();
    if (this.frameId !== null) { this.cancelFrame(this.frameId); this.frameId = null; }
    if (this.retryId !== null) { this.unschedule(this.retryId); this.retryId = null; }
    this.pending = null;
    this.dirty = false;
    this.ready = false;
    this.swapBitmap(null);
    this.status('stopped');
  }

  /** Open a fresh multipart stream; the last picture stays on screen until the new stream paints. */
  connect() {
    this.disconnect();
    const generation = ++this.generation;
    const controller = new AbortController();
    this.controller = controller;
    this.pending = null;
    this.ready = false;
    this.status('connecting');
    this.consume(generation, controller.signal);
  }

  /** Aborting the request closes the connection, so the server drops this subscription. */
  disconnect() {
    const controller = this.controller;
    this.controller = null;
    if (controller) controller.abort();
  }

  async consume(generation, signal) {
    try {
      const response = await this.fetchStream(`${this.url}?t=${Date.now()}`, signal);
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      const reader = response.body.getReader();
      const parser = new MultipartParser();
      for (;;) {
        const {value, done} = await reader.read();
        if (generation !== this.generation) return;
        if (done) break;
        const arrivedAt = this.now();
        for (const part of parser.push(value)) this.receive(part, arrivedAt);
      }
    } catch {
      // A broken stream reconnects exactly like a finished one.
    }
    if (generation === this.generation) this.failed();
  }

  failed() {
    this.disconnect();
    this.status('error');
    this.retryId = this.schedule(() => { this.retryId = null; if (this.running) this.connect(); }, this.retryMs);
  }

  /** Keep only the newest undecoded part: a decoder that cannot keep up drops pictures instead of queueing them. */
  receive(part, arrivedAt) {
    this.received.push(arrivedAt);
    if (this.pending) this.dropped++;
    this.pending = {part, arrivedAt};
    if (!this.decoding) this.decodeNext();
  }

  async decodeNext() {
    this.decoding = true;
    try {
      while (this.pending && this.running) {
        const {part, arrivedAt} = this.pending;
        this.pending = null;
        const generation = this.generation;
        const bitmap = await this.decode(new Blob([part.bytes], {type: 'image/jpeg'}));
        if (generation !== this.generation) { bitmap.close(); return; }
        this.swapBitmap(bitmap);
        this.latest = {arrivedAt, index: Number(part.headers['x-frame-index']),
                       serverAgeMs: Number(part.headers['x-age-ms'])};
        this.dirty = true;
        if (this.frameId === null) this.frameId = this.requestFrame(this.tick);
      }
    } finally {
      this.decoding = false;
    }
  }

  swapBitmap(next) {
    const previous = this.bitmap;
    this.bitmap = next;
    if (previous) previous.close();
  }

  tick() {
    this.frameId = null;
    if (!this.running || !this.dirty) return;
    this.dirty = false;
    const t = this.now();
    this.painted.push(t);
    this.latest.paintedAt = t;
    if (!this.ready || t - this.lastStatsAt >= this.statsIntervalMs) {
      this.ready = true;
      this.lastStatsAt = t;
      this.status('live', this.stats(t));
    }
    this.paint(this.bitmap);
  }

  /** Painted and received rates over the last second, dropped parts, and the latency estimate: the server's
   * camera-arrival-to-send age plus this client's arrival-to-paint time (no shared clock needed). */
  stats(t = this.now()) {
    for (const list of [this.painted, this.received]) while (list.length && t - list[0] > STATS_WINDOW_MS) list.shift();
    const {serverAgeMs, arrivedAt, paintedAt} = this.latest;
    const clientMs = (paintedAt ?? t) - arrivedAt;
    return {fps: this.painted.length, receivedFps: this.received.length, dropped: this.dropped,
            serverAgeMs, clientMs, latencyMs: serverAgeMs + clientMs};
  }

  /** The newest decoded live frame, or null while stopped or before the first frame arrives. */
  current() {
    return this.running && this.bitmap ? this.bitmap : null;
  }
}
