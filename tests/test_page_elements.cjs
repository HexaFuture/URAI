// Static consistency of the page and its script: node --test tests/test_page_elements.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const app = fs.readFileSync(path.join(__dirname, '../urai/static/app.js'), 'utf8');
const page = fs.readFileSync(path.join(__dirname, '../urai/static/index.html'), 'utf8');

test('nothing is looked up that the page does not have', () => {
  // Removing a control takes two edits: the element and every `$('that-id')`. A lookup left behind returns
  // null, and the first `.disabled = ...` on it stops the whole script with "Cannot set properties of null".
  const wanted = [...app.matchAll(/\$\('([^']+)'\)/g)].map(m => m[1]);
  const have = new Set([...page.matchAll(/id="([^"]+)"/g)].map(m => m[1]));
  const missing = [...new Set(wanted)].filter(id => !have.has(id)).sort();
  assert.deepEqual(missing, [], `app.js looks up elements the page does not have: ${missing.join(', ')}`);
  assert.ok(wanted.length > 80, 'the scan must actually find the lookups');
});

test('the tool buttons on the page and in the script are the same set', () => {
  // A tool is named in three places - the toolbar, the click handler and the pressed-state loop.
  const inPage = [...page.matchAll(/id="([a-z-]+)-tool"/g)].map(m => m[1]).sort();
  const handler = app.match(/for\(const tool of \[([^\]]+)\]\)/);
  const pressed = app.match(/for\(const t of \[([^\]]+)\]\)/);
  assert.ok(handler && pressed, 'cannot find the tool loops');
  const names = text => text.split(',').map(s => s.trim().replace(/'/g, '')).sort();
  assert.deepEqual(names(handler[1]), inPage);
  assert.deepEqual(names(pressed[1]), inPage);
});

test('a stroke drawn while the robot moves is queued before anything can touch the draft', () => {
  // The queued branch must come first in the pointer-down handler and must not save, observe or invalidate.
  const handler = app.slice(app.indexOf("$('main-view').addEventListener('pointerdown'"),
                            app.indexOf("$('main-view').addEventListener('pointermove'"));
  assert.ok(handler.indexOf('if(running()||pendingStrokes.length)') < handler.indexOf('commitMotionInputs'));
  const queued = handler.slice(0, handler.indexOf('const p=pointEvent'));
  assert.doesNotMatch(queued, /api\(|freezeObservation\(|invalidate\(/);
});
