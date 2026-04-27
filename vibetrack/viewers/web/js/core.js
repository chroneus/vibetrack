// core.js — globals, utilities, drag-sort, apiUrl, fullscreen overlay, tab management.
// All other module files (charts.js, pills.js, media.js, settings.js, main.js) rely on
// the globals declared here.

const DATA = window.VT_DATA || [];
const COLORS = ['#58a6ff', '#f78166', '#3fb950', '#d2a8ff', '#f0883e', '#79c0ff', '#56d364', '#e3b341', '#ff7b72', '#a5d6ff'];
const _delIcon = `<svg width="11" height="13" viewBox="0 0 11 13" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"><path d="M1 3h9"/><path d="M3.5 3V2a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v1"/><path d="M2 3l.5 8a1 1 0 0 0 1 1h4a1 1 0 0 0 1-1L9 3"/><path d="M4.5 5.5v4"/><path d="M6.5 5.5v4"/></svg>`;
const _delIconLg = `<svg width="15" height="17" viewBox="0 0 11 13" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"><path d="M1 3h9"/><path d="M3.5 3V2a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v1"/><path d="M2 3l.5 8a1 1 0 0 0 1 1h4a1 1 0 0 0 1-1L9 3"/><path d="M4.5 5.5v4"/><path d="M6.5 5.5v4"/></svg>`;

// Prefix project-scoped API endpoints. `/api/data`, `/api/config`, `/api/rename`
// are scoped per-project when window.VT_PROJECT is set. Endpoints that encode the
// project directly (`/api/project/${p}`, `/api/experiment`, `/api/move-logdir`,
// `/media`) stay unscoped.
function apiUrl(base) {
  return window.VT_PROJECT ? `${base}/${window.VT_PROJECT}` : base;
}

function escapeHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function mediaUrl(p) { return '/media?path=' + encodeURIComponent(p); }
function emaSmooth(vals, w) { if (!vals.length) return []; let n = 0, d = 0; return vals.map(v => { n = w * n + (1 - w) * v; d = w * d + (1 - w); return n / d; }); }
function parseHexColor(color) {
  if (!/^#[0-9a-fA-F]{6}$/.test(color)) return null;
  return {
    r: parseInt(color.slice(1, 3), 16),
    g: parseInt(color.slice(3, 5), 16),
    b: parseInt(color.slice(5, 7), 16),
  };
}
function colorDistance(a, b) {
  const ca = parseHexColor(a), cb = parseHexColor(b);
  if (!ca || !cb) return a === b ? 0 : 1e9;
  const dr = ca.r - cb.r, dg = ca.g - cb.g, db = ca.b - cb.b;
  return dr * dr + dg * dg + db * db;
}
function pickDistinctColor(usedColors) {
  const normalized = usedColors
    .filter(c => /^#[0-9a-fA-F]{6}$/.test(c))
    .map(c => c.toLowerCase());
  const unused = COLORS.filter(c => !normalized.includes(c.toLowerCase()));
  const candidates = unused.length ? unused : COLORS;
  if (!normalized.length) return candidates[0] || COLORS[0];
  let best = candidates[0] || COLORS[0];
  let bestScore = -1;
  candidates.forEach(candidate => {
    const score = normalized.reduce((minScore, used) => Math.min(minScore, colorDistance(candidate, used)), Infinity);
    if (score > bestScore) {
      bestScore = score;
      best = candidate;
    }
  });
  return best;
}
function withOpacity(color, opacity) {
  const alpha = Math.max(0, Math.min(255, Math.round(opacity * 255))).toString(16).padStart(2, '0');
  return /^#[0-9a-fA-F]{6}$/.test(color) ? color + alpha : color;
}
function emaSmoothXY(steps, vals, w) {
  if (!vals.length) return [];
  const out = [];
  let n = 0, d = 0;
  for (let i = 0; i < vals.length; i++) {
    const v = vals[i];
    if (v === null || v === undefined || (typeof v === 'number' && isNaN(v))) {
      out.push({ x: steps[i], y: null }); continue;
    }
    n = w * n + (1 - w) * v; d = w * d + (1 - w);
    out.push({ x: steps[i], y: n / d });
  }
  return out;
}
function humanSize(b) { if (!b) return ''; const u = ['B', 'KB', 'MB', 'GB']; let i = 0; while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; } return b.toFixed(i ? 1 : 0) + ' ' + u[i]; }
function formatVal(v) {
  if (v === null || v === undefined || (typeof v === 'number' && isNaN(v))) return 'NaN';
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs >= 1e6 || (abs > 0 && abs < 1e-4)) return v.toExponential(3);
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 1) return parseFloat(v.toPrecision(5)).toString();
  return parseFloat(v.toPrecision(4)).toString();
}

// ── Drag-and-drop sort ───────────────────────────────────────
function saveDragOrder(container, key) {
  const ids = [...container.children].map(el => el.dataset.dragId).filter(Boolean);
  try { localStorage.setItem('vt_order_' + key, JSON.stringify(ids)); } catch (e) {}
}

function restoreDragOrder(container, key) {
  try {
    const saved = JSON.parse(localStorage.getItem('vt_order_' + key));
    if (!Array.isArray(saved) || !saved.length) return;
    const byId = new Map();
    [...container.children].forEach(el => { if (el.dataset.dragId) byId.set(el.dataset.dragId, el); });
    saved.forEach(id => { const el = byId.get(id); if (el) { container.appendChild(el); byId.delete(id); } });
    byId.forEach(el => container.appendChild(el));
  } catch (e) {}
}

function enableDragSort(container, storageKey) {
  if (!container || !container.children.length) return;
  if (storageKey) restoreDragOrder(container, storageKey);
  [...container.children].forEach(el => { el.draggable = true; el.classList.add('draggable'); });
  if (container._dragSortEnabled) return;
  container._dragSortEnabled = true;
  container._dragSortKey = storageKey;
  let dragEl = null;
  container.addEventListener('dragstart', e => {
    dragEl = e.target.closest('[draggable]');
    if (dragEl) { dragEl.classList.add('dragging'); e.dataTransfer.effectAllowed = 'move'; }
  });
  container.addEventListener('dragend', () => {
    if (dragEl) dragEl.classList.remove('dragging');
    container.querySelectorAll('.drag-over').forEach(x => x.classList.remove('drag-over'));
    dragEl = null;
  });
  container.addEventListener('dragover', e => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    container.querySelectorAll('.drag-over').forEach(x => x.classList.remove('drag-over'));
    const target = e.target.closest('[draggable]');
    if (target && target !== dragEl && target.parentNode === container) target.classList.add('drag-over');
  });
  container.addEventListener('drop', e => {
    e.preventDefault();
    container.querySelectorAll('.drag-over').forEach(x => x.classList.remove('drag-over'));
    const target = e.target.closest('[draggable]');
    if (!dragEl || !target || target === dragEl || target.parentNode !== container) return;
    const children = [...container.children];
    if (children.indexOf(dragEl) < children.indexOf(target)) target.after(dragEl);
    else target.before(dragEl);
    if (container._dragSortKey) saveDragOrder(container, container._dragSortKey);
  });
}

// ── Fullscreen overlay ───────────────────────────────────────
function openFullscreen(title, buildFn) {
  const ov = document.createElement('div'); ov.className = 'fs-overlay';
  ov.innerHTML = `<div class="fs-header"><h2></h2><div class="fs-header-right"><button class="fs-close">Close · Esc</button></div></div><div class="fs-body"></div>`;
  ov.querySelector('.fs-header h2').textContent = title;
  document.body.appendChild(ov);
  const body = ov.querySelector('.fs-body');
  buildFn(body, ov);
  const close = () => { ov.remove(); };
  ov.querySelector('.fs-close').addEventListener('click', close);
  const onKey = e => { if (e.key === 'Escape') { close(); document.removeEventListener('keydown', onKey); } };
  document.addEventListener('keydown', onKey);
}

function openVideoFullscreen(title, src) {
  const ov = document.createElement('div'); ov.className = 'fs-overlay';
  ov.innerHTML = `<div class="fs-header"><h2></h2><div class="fs-header-right"><button class="fs-close">Close · Esc</button></div></div><div class="fs-body"></div>`;
  ov.querySelector('.fs-header h2').textContent = title;
  const body = ov.querySelector('.fs-body');
  const vid = document.createElement('video');
  vid.controls = true; vid.autoplay = true; vid.src = src;
  body.appendChild(vid);
  document.body.appendChild(ov);
  const close = () => { vid.pause(); ov.remove(); };
  ov.querySelector('.fs-close').addEventListener('click', close);
  const onKey = e => { if (e.key === 'Escape') { close(); document.removeEventListener('keydown', onKey); } };
  document.addEventListener('keydown', onKey);
}

function openImageGalleryFullscreen(title, entries) {
  const steps = _imgUniqueSteps(entries);
  if (!steps.length) return;
  const byExp = {};
  entries.forEach(e => { if (!byExp[e.exp]) byExp[e.exp] = {}; byExp[e.exp][e.step] = e.path; });
  const expNames = Object.keys(byExp);

  const ov = document.createElement('div'); ov.className = 'fs-overlay';
  const _zoomOut = `<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><circle cx="6" cy="6" r="4"/><line x1="9" y1="9" x2="13" y2="13"/><line x1="4" y1="6" x2="8" y2="6"/></svg>`;
  const _zoomIn = `<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><circle cx="6" cy="6" r="4"/><line x1="9" y1="9" x2="13" y2="13"/><line x1="6" y1="4" x2="6" y2="8"/><line x1="4" y1="6" x2="8" y2="6"/></svg>`;
  ov.innerHTML = `<div class="fs-header"><h2></h2><div style="display:flex;gap:8px;align-items:center;"><button class="fs-zoom-btn" data-delta="-20" title="Zoom out">${_zoomOut}</button><button class="fs-zoom-btn" data-delta="20" title="Zoom in">${_zoomIn}</button><button class="fs-close">Close · Esc</button></div></div><div class="fs-anim-bar" style="display:flex;align-items:center;gap:10px;padding:8px 16px;background:var(--bg2);border-bottom:1px solid var(--border);"><button class="fs-play-btn">&#9654;</button><input type="range" class="fs-step-slider" min="0" max="${steps.length - 1}" value="0" style="flex:1;accent-color:var(--accent);"><span class="fs-step-label" style="font-size:0.85em;color:var(--muted);white-space:nowrap;"></span></div><div class="fs-body fs-img-body" style="display:flex;flex-wrap:wrap;gap:12px;justify-content:center;overflow-y:auto;"></div>`;
  ov.querySelector('.fs-header h2').textContent = title;
  const body = ov.querySelector('.fs-body');
  const slider = ov.querySelector('.fs-step-slider');
  const stepLabel = ov.querySelector('.fs-step-label');
  const playBtn = ov.querySelector('.fs-play-btn');

  let zoom = 80;
  const imgs = {};
  expNames.forEach(exp => {
    const cell = document.createElement('div');
    cell.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:4px;';
    const img = document.createElement('img'); img.className = 'fs-img-zoom'; img.style.width = zoom + '%';
    const lbl = document.createElement('div'); lbl.textContent = exp;
    lbl.style.cssText = 'font-size:0.8em;color:' + (expColors[exp] || 'var(--muted)') + ';';
    cell.append(img, lbl); body.appendChild(cell); imgs[exp] = img;
  });

  ov.querySelectorAll('.fs-zoom-btn').forEach(btn => btn.addEventListener('click', () => {
    zoom = Math.min(Math.max(zoom + parseInt(btn.dataset.delta), 20), 400);
    Object.values(imgs).forEach(i => i.style.width = zoom + '%');
  }));

  function renderFrame(idx) {
    const step = steps[idx];
    stepLabel.textContent = 'Step ' + step + ' / ' + steps[steps.length - 1];
    slider.value = idx;
    expNames.forEach(exp => {
      const path = byExp[exp][step];
      imgs[exp].style.display = path ? '' : 'none';
      if (path) imgs[exp].src = mediaUrl(path);
    });
  }

  let playing = false, timer = null;
  function stopAnim() { playing = false; playBtn.innerHTML = '&#9654;'; if (timer) { clearInterval(timer); timer = null; } }
  function startAnim() {
    playing = true; playBtn.innerHTML = '&#9646;&#9646;';
    const fps = parseInt((document.getElementById('cfg-image-fps') || {}).value) || 4;
    timer = setInterval(() => { let i = parseInt(slider.value) + 1; if (i >= steps.length) i = 0; renderFrame(i); }, 1000 / fps);
  }
  playBtn.addEventListener('click', () => { if (playing) stopAnim(); else startAnim(); });
  slider.addEventListener('input', () => { stopAnim(); renderFrame(parseInt(slider.value)); });
  renderFrame(0);
  document.body.appendChild(ov);
  const close = () => { stopAnim(); ov.remove(); document.removeEventListener('keydown', onKey); };
  ov.querySelector('.fs-close').addEventListener('click', close);
  const onKey = e => { if (e.key === 'Escape') close(); };
  document.addEventListener('keydown', onKey);
}

function openImageFullscreen(title, src) {
  const ov = document.createElement('div'); ov.className = 'fs-overlay';
  const zoomOut = `<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><circle cx="6" cy="6" r="4"/><line x1="9" y1="9" x2="13" y2="13"/><line x1="4" y1="6" x2="8" y2="6"/></svg>`;
  const zoomIn = `<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><circle cx="6" cy="6" r="4"/><line x1="9" y1="9" x2="13" y2="13"/><line x1="6" y1="4" x2="6" y2="8"/><line x1="4" y1="6" x2="8" y2="6"/></svg>`;
  ov.innerHTML = `<div class="fs-header"><h2></h2><div style="display:flex;gap:8px;align-items:center;"><button class="fs-zoom-btn" data-delta="-20" title="Zoom out">${zoomOut}</button><button class="fs-zoom-btn" data-delta="20" title="Zoom in">${zoomIn}</button><button class="fs-close">Close · Esc</button></div></div><div class="fs-body fs-img-body"></div>`;
  ov.querySelector('.fs-header h2').textContent = title;
  const body = ov.querySelector('.fs-body');
  const img = document.createElement('img'); img.src = src; img.className = 'fs-img-zoom';
  body.appendChild(img);
  document.body.appendChild(ov);
  let zoom = 100;
  ov.querySelectorAll('.fs-zoom-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      zoom = Math.min(Math.max(zoom + parseInt(btn.dataset.delta), 20), 400);
      img.style.width = zoom + '%';
    });
  });
  const close = () => { ov.remove(); };
  ov.querySelector('.fs-close').addEventListener('click', close);
  const onKey = e => { if (e.key === 'Escape') { close(); document.removeEventListener('keydown', onKey); } };
  document.addEventListener('keydown', onKey);
}

function addExpandBtn(parent, title, buildFn) {
  const btn = document.createElement('button'); btn.className = 'expand-btn';
  btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="2,5 2,2 5,2"/><polyline points="7,2 10,2 10,5"/><polyline points="10,7 10,10 7,10"/><polyline points="5,10 2,10 2,7"/></svg>';
  btn.title = 'Fullscreen';
  btn.addEventListener('click', e => { e.stopPropagation(); openFullscreen(title, buildFn); });
  parent.appendChild(btn);
}

// ── Tab management ───────────────────────────────────────────
let _activeTab = null;
function switchTab(id) {
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  const btn = document.querySelector(`.tabs button[data-tab="${id}"]`);
  if (btn) btn.classList.add('active');
  document.getElementById('tab-' + id).classList.add('active');
  if (_activeTab === 'images' && id !== 'images') _imgClearSelection();
  if (_activeTab === 'video' && id !== 'video') _vidClearSelection();
  _activeTab = id;
}

function buildTabs() {
  const el = document.getElementById('main-tabs');
  // Remember currently active tab
  const prev = _activeTab || (el.querySelector('button.active') || {}).dataset?.tab;
  el.innerHTML = '';
  const defs = [
    { id: 'scalars', label: 'Scalars', has: DATA.some(d => (d.tags || []).length > 0) },
    { id: 'images', label: 'Images', has: DATA.some(d => (d.image_tags || []).length > 0) },
    { id: 'audio', label: 'Audio', has: DATA.some(d => (d.audio_tags || []).length > 0) },
    { id: 'video', label: 'Video', has: DATA.some(d => (d.video_tags || []).length > 0) },
    { id: 'artifacts', label: 'Artifacts', has: DATA.some(d => (d.artifact_tags || []).length > 0) },
    { id: 'text', label: 'Text', has: DATA.some(d => (d.text_tags || []).length > 0) },
    { id: 'histograms', label: 'Histograms', has: DATA.some(d => (d.histogram_tags || []).length > 0) },
    { id: 'system', label: 'System', has: DATA.some(d => (d.system_tags || []).length > 0) },
    { id: 'settings', label: 'Settings', has: true },
  ];
  const available = defs.filter(d => d.has);
  available.forEach(d => {
    const btn = document.createElement('button');
    btn.dataset.tab = d.id; btn.textContent = d.label;
    btn.addEventListener('click', () => switchTab(d.id));
    el.appendChild(btn);
  });
  // Restore previous tab if still available, otherwise pick first
  const target = available.find(d => d.id === prev) ? prev : (available[0] || {}).id;
  if (target) switchTab(target);
}
