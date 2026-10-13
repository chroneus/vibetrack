const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = process.argv[2] || path.join(__dirname, '../vibetrack/viewers/web/js/media.js');
let callback, canvas;
const probes = [];
const grid = {style: {setProperty() {}}, innerHTML: ''};
const imagesRoot = {innerHTML: ''};
const context = vm.createContext({
  DATA: [], expColors: {}, localStorage: {getItem() { return null; }},
  thumbUrl: (p, w) => `/thumb?path=${p}&w=${w}`,
  Image: class { constructor() { probes.push(this); } },
  document: {
    querySelectorAll() { return []; },
    getElementById(id) { return {'video-grid': grid, 'images-container': imagesRoot}[id] || null; },
    createElement(tag) {
      assert.equal(tag, 'canvas');
      canvas = {getContext() { return {drawImage() {}}; }, toDataURL() { return 'data:image/jpeg;base64,x'; }};
      return canvas;
    },
  },
  IntersectionObserver: class {
    constructor(cb) { callback = cb; }
    observe() {}
    disconnect() {}
  },
});
vm.runInContext(fs.readFileSync(source, 'utf8'), context);
function video() {
  return {
    dataset: {}, style: {}, listeners: {}, loads: 0, pauses: 0,
    videoWidth: 4320, videoHeight: 5760, currentTime: 3,
    addEventListener(n, cb) { this.listeners[n] = cb; },
    hasAttribute(n) { return Object.hasOwn(this, n); },
    removeAttribute(n) { delete this[n]; },
    load() { this.loads++; }, pause() { this.pauses++; },
  };
}
const v = video();
context._vidEnsurePoster(v);
assert.equal(v.loads, 0, 'registering a poster must not load the video');
assert.equal(v.currentTime, 3, 'poster must not force a seek');
v.listeners.loadeddata();
assert.equal(canvas.width, 360);
assert.equal(canvas.height, 480);
context._vidObserve(v, '/clip.mp4');
assert.equal(v.src, undefined, 'offscreen clips must not have a source');
callback([{target:v, isIntersecting:true}]);
assert.equal(v.src, '/clip.mp4');
assert.equal(v.preload, 'metadata');
callback([{target:v, isIntersecting:false}]);
assert.equal(v.src, undefined, 'offscreen decoder must be released');
assert.equal(v.pauses, 1);
assert.equal(v.loads, 1);
assert.equal(v.style.aspectRatio, '4320 / 5760');
context.buildVideo();
grid.innerHTML = 'preserve video cards';
context.buildVideo();
assert.equal(grid.innerHTML, 'preserve video cards', 'unchanged video data must preserve players');
console.log('PASS: bounded posters, no forced decode/seek, viewport loading, decoder release, stable refresh');

// With a server poster, visible tiles fetch nothing until play.
const p = video();
p.clientWidth = 300;
p.dataset.videoPath = '/runs/a/media/clip/0.mp4';
context._vidObserve(p, '/clip.mp4');
callback([{target:p, isIntersecting:true}]);
assert.equal(p.src, '/clip.mp4');
assert.equal(p.preload, 'none', 'poster tiles must not preload video data');
assert.equal(probes.length, 1);
assert.equal(probes[0].src, '/thumb?path=/runs/a/media/clip/0.mp4&w=300');
Object.assign(probes[0], {naturalWidth: 640, naturalHeight: 360});
probes[0].onload();
assert.equal(p.poster, probes[0].src);
assert.equal(p.style.aspectRatio, '640 / 360');
callback([{target:p, isIntersecting:false}]);
callback([{target:p, isIntersecting:true}]);
assert.equal(probes.length, 1, 'poster is fetched once per tile');
assert.equal(p.preload, 'none');

// No poster available (e.g. no ffmpeg): fall back to decoding metadata.
const f = video();
f.dataset.videoPath = '/runs/a/media/clip/1.mp4';
context._vidObserve(f, '/clip1.mp4');
callback([{target:f, isIntersecting:true}]);
const loadsBefore = f.loads;
probes.at(-1).onerror();
assert.equal(f.preload, 'metadata');
assert.equal(f.loads, loadsBefore + 1);
callback([{target:f, isIntersecting:false}]);
callback([{target:f, isIntersecting:true}]);
assert.equal(f.preload, 'metadata', 'failed poster is not retried on every scroll');
console.log('PASS: server posters with preload=none, metadata fallback');

context.buildImages();
imagesRoot.innerHTML = 'preserve image grid';
context.buildImages();
assert.equal(imagesRoot.innerHTML, 'preserve image grid', 'unchanged image data must not rebuild the grid');
console.log('PASS: image grid survives refreshes with unchanged images');

const times = [0, .04, .12, .14];
assert.equal(context._vidFrameIndex(times, .119), 1);
assert.equal(context._vidFrameIndex(times, .12), 2);
assert.equal(context._vidFrameIndex(times, 1), 3);
for (let i = 0; i < times.length; i++) {
  assert.equal(context._vidFrameIndex(times, context._vidFrameTime(times, i, .18)), i);
}
console.log('PASS: variable-rate frame lookup and seek targets');
