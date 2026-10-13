const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const js = name => fs.readFileSync(path.join(__dirname, '../vibetrack/viewers/web/js', name), 'utf8');

function load(names = [], saved = null, withPills = false) {
  const store = {vt_colors_: saved && JSON.stringify(saved)};
  const context = vm.createContext({
    window: {VT_DATA: names.map(name => ({name}))},
    document: {},
    localStorage: {getItem: k => store[k] ?? null, setItem: (k, v) => { store[k] = v; }},
  });
  vm.runInContext(js('core.js'), context);
  if (withPills) vm.runInContext(js('pills.js') + ';globalThis.expColors = expColors;', context);
  return context;
}

function pickMany(context, n) {
  const used = [];
  for (let i = 0; i < n; i++) used.push(context.pickDistinctColor(used));
  return used;
}

const dist = (ctx, a, b) => Math.hypot(...ctx.hexToOklab(a).map((v, i) => v - ctx.hexToOklab(b)[i]));

// The first ten picks are unchanged from the palette-only picker.
const ctx = load();
const picks = pickMany(ctx, 40);
assert.deepEqual(picks.slice(0, 10), [
  '#58a6ff', '#f0883e', '#3fb950', '#d2a8ff', '#a5d6ff',
  '#ff7b72', '#e3b341', '#79c0ff', '#56d364', '#f78166',
]);

// Beyond the palette every color is new and clearly separated from the rest
// (the palette's own closest pair is ~0.026 apart in OKLab).
assert.equal(new Set(picks).size, picks.length, 'colors must not repeat');
for (let i = 10; i < picks.length; i++) {
  const nearest = Math.min(...picks.slice(0, i).map(c => dist(ctx, picks[i], c)));
  assert.ok(nearest > 0.05, `pick ${i + 1} ${picks[i]} is only ${nearest.toFixed(3)} from an earlier color`);
}
console.log('PASS: palette order kept, 40 distinct colors');

// User-chosen colors are avoided too.
const custom = ['#58a6ff', '#f0883e', '#3fb950', '#d2a8ff', '#a5d6ff', '#ff7b72', '#e3b341', '#79c0ff', '#56d364', '#f78166', '#123456'];
assert.ok(!custom.includes(ctx.pickDistinctColor(custom)));

// Colors saved by the old picker (blue repeated after the 10th) are repaired;
// colors that were already unique stay put.
const names = Array.from({length: 30}, (_, i) => `run${i}`);
const oldSaved = Object.fromEntries(names.map((n, i) => [n, i < 10 ? picks[i] : '#58a6ff']));
const repaired = load(names, oldSaved, true).expColors;
assert.equal(new Set(Object.values(repaired)).size, 30, 'saved duplicates must be re-picked');
names.slice(0, 10).forEach((n, i) => assert.equal(repaired[n], picks[i]));

// A new (unsaved, listed first) experiment must not steal a saved color.
const mixed = load(['fresh', 'older'], {older: '#58a6ff'}, true).expColors;
assert.equal(mixed.older, '#58a6ff');
assert.notEqual(mixed.fresh, '#58a6ff');

// Without saved colors, assignment is identical to plain picking.
const plain = load(names, null, true).expColors;
assert.deepEqual(names.map(n => plain[n]), pickMany(load(), 30));
console.log('PASS: saved duplicates repaired, saved choices preserved');
