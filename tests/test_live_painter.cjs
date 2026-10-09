// The MJPEG receiver of the live view: node --test tests/test_live_painter.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const {startService, freePort} = require('./support/urai_server.cjs');

const encoder = new TextEncoder();

function part(index, ageMs, payload) {
  const head = encoder.encode(`--urai-frame\r\nContent-Type: image/jpeg\r\nContent-Length: ${payload.length}\r\n`
                              + `X-Frame-Index: ${index}\r\nX-Age-Ms: ${ageMs}\r\n\r\n`);
  const bytes = new Uint8Array(head.length + payload.length + 2);
  bytes.set(head); bytes.set(payload, head.length); bytes.set([13, 10], head.length + payload.length);
  return bytes;
}

function jpeg(fill, length = 40) { return new Uint8Array(length).fill(fill); }

let service;
test.before(async () => { service = await startService(); });
test.after(async () => { await service.stop(); });

test('the multipart parser yields complete parts however the bytes are chunked', async () => {
  const {MultipartParser} = await import('../urai/static/live.mjs');
  const a = part(7, 12.5, jpeg(1, 30)), b = part(8, 3, jpeg(2, 50));
  const whole = new Uint8Array(a.length + b.length); whole.set(a); whole.set(b, a.length);
  const inOne = new MultipartParser().push(whole);
  assert.equal(inOne.length, 2);
  assert.deepEqual(inOne[0].headers, {'content-type': 'image/jpeg', 'content-length': '30', 'x-frame-index': '7', 'x-age-ms': '12.5'});
  assert.deepEqual(Array.from(inOne[0].bytes), Array.from(jpeg(1, 30)));
  assert.deepEqual(Array.from(inOne[1].bytes), Array.from(jpeg(2, 50)));
  const byteWise = new MultipartParser();
  const trickle = [];
  for (let i = 0; i < whole.length; i++) trickle.push(...byteWise.push(whole.slice(i, i + 1)));
  assert.equal(trickle.length, 2);
  assert.deepEqual(trickle.map(p => p.headers['x-frame-index']), ['7', '8']);
  assert.deepEqual(Array.from(trickle[1].bytes), Array.from(jpeg(2, 50)));
  const split = new MultipartParser();
  assert.equal(split.push(whole.slice(0, 5)).length, 0, 'a marker split across chunks is held back');
  assert.equal(split.push(whole.slice(5, a.length + 3)).length, 1);
  assert.equal(split.push(whole.slice(a.length + 3)).length, 1);
  assert.throws(() => new MultipartParser().push(encoder.encode('--urai-frame\r\nContent-Type: image/jpeg\r\n\r\nxx')),
                /Content-Length/);
});

test('the service stream parses into JPEG parts carrying the headers the painter reads', async () => {
  const {MultipartParser, BOUNDARY} = await import('../urai/static/live.mjs');
  const controller = new AbortController();
  const response = await fetch(`${service.base}/api/live.mjpg?t=${Date.now()}`, {signal: controller.signal, cache: 'no-store'});
  assert.equal(response.ok, true);
  assert.equal(response.headers.get('content-type'), `multipart/x-mixed-replace; boundary=${BOUNDARY}`);
  const reader = response.body.getReader(), parser = new MultipartParser(), parts = [];
  while (parts.length < 3) {
    const {value, done} = await reader.read();
    assert.equal(done, false, 'the stream ended before three pictures arrived');
    parts.push(...parser.push(value));
  }
  controller.abort();
  const indices = parts.map(p => Number(p.headers['x-frame-index']));
  assert.ok(indices.every((index, i) => Number.isInteger(index) && (i === 0 || index > indices[i - 1])), String(indices));
  for (const {headers, bytes} of parts) {
    assert.equal(headers['content-type'], 'image/jpeg');
    assert.ok(Number.isFinite(Number(headers['x-age-ms'])));
    assert.deepEqual([bytes[0], bytes[1]], [0xff, 0xd8], 'JPEG start-of-image marker');
    assert.deepEqual([bytes.at(-2), bytes.at(-1)], [0xff, 0xd9], 'JPEG end-of-image marker');
  }
});

/** Collect painter states until `done(states)` holds. */
function watch(options) {
  return import('../urai/static/live.mjs').then(({LivePainter}) => {
    const states = [];
    let wake = () => {};
    const painter = new LivePainter({retryMs: 50, paint() { throw new Error('nothing is decoded here'); },
      status: state => { states.push(state); wake(); }, ...options});
    const until = done => new Promise(resolve => {
      wake = () => { if (done(states)) resolve(states); };
      wake();
    });
    return {painter, states, until};
  });
}

test('an unreachable stream is reported and retried until the painter stops', async () => {
  const port = await freePort();   // nothing listens here
  const {painter, states, until} = await watch({url: `http://127.0.0.1:${port}/api/live.mjpg`});
  painter.start();
  await until(s => s.filter(state => state === 'error').length >= 2);
  assert.deepEqual(states.slice(0, 4), ['connecting', 'error', 'connecting', 'error']);
  painter.stop();
  assert.equal(states.at(-1), 'stopped');
  const count = states.length;
  await new Promise(resolve => setTimeout(resolve, 200));
  assert.equal(states.length, count, 'stop cancels the pending retry');
  assert.equal(painter.current(), null);
});

test('a stream the service refuses is retried like a broken one', async () => {
  const {painter, states, until} = await watch({url: `${service.base}/api/no-such-stream.mjpg`});
  painter.start();
  await until(s => s.includes('error') && s.lastIndexOf('connecting') > s.indexOf('error'));
  painter.stop();
  assert.deepEqual(states.slice(0, 3), ['connecting', 'error', 'connecting']);
  assert.equal(states.at(-1), 'stopped');
});

test('the status label carries the painted rate and latency once frames flow', async () => {
  const {liveStatusLabel, LIVE_STATUS_LABELS} = await import('../urai/static/live.mjs');
  assert.equal(liveStatusLabel('connecting'), LIVE_STATUS_LABELS.connecting);
  assert.equal(liveStatusLabel('live'), '实时 RGB · 几何用落笔时的观测');
  assert.equal(liveStatusLabel('live', {fps: 29.6, latencyMs: 84.4}), '实时 RGB · 30 fps · 时延约 84 ms · 几何用落笔时的观测');
  assert.equal(liveStatusLabel('stopped', {fps: 30, latencyMs: 80}), '观测快照');
});
