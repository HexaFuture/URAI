// Runs the real URAI service (simulation backend) on a free loopback port for one test file.
// The page's own HTTP client (urai/static/api.mjs) talks to it, so the tests exercise the real routes.
const {spawn} = require('node:child_process');
const net = require('node:net');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..', '..');
const PYTHON = path.join(ROOT, '.venv', 'bin', 'python');

function freePort() {
  return new Promise((resolve, reject) => {
    const probe = net.createServer();
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const {port} = probe.address();
      probe.close(() => resolve(port));
    });
  });
}

const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

/** Start `python -m urai.app --port <free port>` and wait until GET /api/state answers 200. */
async function startService({startupTimeoutMs = 60000} = {}) {
  const port = await freePort();
  const child = spawn(PYTHON, ['-m', 'urai.app', '--port', String(port)], {cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe']});
  let output = '';
  child.stdout.on('data', chunk => { output += chunk; });
  child.stderr.on('data', chunk => { output += chunk; });
  const exited = new Promise(resolve => child.once('exit', resolve));
  const base = `http://127.0.0.1:${port}`;
  const stop = async () => {
    if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
    await exited;
  };
  const deadline = Date.now() + startupTimeoutMs;
  for (;;) {
    if (child.exitCode !== null || child.signalCode !== null) throw new Error(`service exited during start-up:\n${output}`);
    const answered = await fetch(`${base}/api/state`).then(response => response.ok, () => false);
    if (answered) break;
    if (Date.now() > deadline) {
      await stop();
      throw new Error(`service did not answer /api/state within ${startupTimeoutMs} ms:\n${output}`);
    }
    await pause(200);
  }
  const {apiClient} = await import('../../urai/static/api.mjs');
  const api = apiClient(base);
  return {
    base, api, stop, output: () => output,
    /** Poll /api/state until the execution reaches one of `states`. */
    async waitFor(states, timeoutMs = 60000) {
      const until = Date.now() + timeoutMs;
      for (;;) {
        const state = await api('state');
        if (states.includes(state.execution.state)) return state;
        if (Date.now() > until) throw new Error(`execution stayed ${JSON.stringify(state.execution)}`);
        await pause(100);
      }
    },
  };
}

module.exports = {startService, freePort};
