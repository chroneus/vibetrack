// runs.js — Runs tab: one result card per run (status, headline metrics,
// params that differ between runs, latest video/image) and the
// ?run=NAME&tab=TAB deep links that `vibetrack open` and run reports print.

const _runsCards = new Map();   // run name -> { card, mediaSig }
const _runsFocus = new Set();   // runs named in ?run= or focused via the card button
let _runsAutoplayDone = false;

const _RUN_STATUS_CLASS = {
  finished: 'st-ok', running: 'st-run', failed: 'st-bad', crashed: 'st-bad',
  interrupted: 'st-warn', unknown: 'st-unknown',
};

function _runsFmtDuration(s) {
  if (s === null || s === undefined || s < 0) return '';
  s = Math.round(s);
  if (s < 60) return s + 's';
  let m = Math.floor(s / 60); s %= 60;
  if (m < 60) return m + 'm' + String(s).padStart(2, '0') + 's';
  let h = Math.floor(m / 60); m %= 60;
  if (h < 24) return h + 'h' + String(m).padStart(2, '0') + 'm';
  const d = Math.floor(h / 24); h %= 24;
  return d + 'd' + String(h).padStart(2, '0') + 'h';
}

function _runsFmtAge(ts) {
  if (!ts) return '';
  const d = Date.now() / 1000 - ts;
  if (d < 60) return 'just now';
  if (d < 3600) return Math.floor(d / 60) + 'm ago';
  if (d < 86400) return Math.floor(d / 3600) + 'h ago';
  return Math.floor(d / 86400) + 'd ago';
}

function _runsFlatten(obj, prefix, out) {
  Object.entries(obj || {}).forEach(([k, v]) => {
    const key = prefix ? prefix + '.' + k : k;
    if (v && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length) _runsFlatten(v, key, out);
    else out[key] = v;
  });
  return out;
}

// Keys whose value differs between runs (missing counts as different).
function _runsVaryingKeys() {
  const flat = DATA.map(d => _runsFlatten(d.hparams, '', {}));
  const keys = new Set(); flat.forEach(f => Object.keys(f).forEach(k => keys.add(k)));
  if (DATA.length < 2) return [...keys].sort();
  return [...keys].filter(k => new Set(flat.map(f => k in f ? JSON.stringify(f[k]) : '\u0000')).size > 1).sort();
}

function _runsValueText(v) {
  if (v === null || v === undefined) return '-';
  if (typeof v === 'number') return formatVal(v);
  const s = typeof v === 'string' ? v : JSON.stringify(v);
  return s.length > 40 ? s.slice(0, 37) + '...' : s;
}

// Latest entry per tag for one media kind: [{tag, step, path, count}]
function _runsLatest(byTag) {
  return Object.entries(byTag || {}).map(([tag, rows]) => {
    if (!rows || !rows.length) return null;
    const last = rows.reduce((a, b) => (b.step >= a.step ? b : a));
    return { tag, step: last.step, path: last.path, count: rows.length };
  }).filter(Boolean).sort((a, b) => a.tag.localeCompare(b.tag));
}

function _runsMediaItems(exp) {
  const videos = _runsLatest(exp.video_data).map(v => ({ ...v, kind: 'video' }));
  const images = _runsLatest(exp.images).map(v => ({ ...v, kind: 'image' }));
  const items = videos.length ? videos.slice(0, 2) : images.slice(0, 2);
  if (videos.length && images.length && items.length < 2) items.push(images[0]);
  return items;
}

// Own lazy-load observer: the Video tab disconnects its shared one on rebuild.
let _runsViewportObserver = null;
function _runsObserve(video, url) {
  video.preload = 'none';
  video.dataset.videoUrl = url;
  if (typeof IntersectionObserver === 'undefined') { _vidAttach(video); return; }
  if (!_runsViewportObserver) {
    _runsViewportObserver = new IntersectionObserver(entries => {
      entries.forEach(({ target, isIntersecting }) => {
        if (isIntersecting && !target.hasAttribute('src')) _vidAttach(target);
        else if (!isIntersecting && target.hasAttribute('src') && target.paused) _vidRelease(target);
      });
    }, { rootMargin: '400px 0px' });
  }
  _runsViewportObserver.observe(video);
}
function _runsUnobserve(video) {
  if (_runsViewportObserver) _runsViewportObserver.unobserve(video);
}

function _runsBuildMedia(el, exp, autoplay) {
  el.querySelectorAll('video').forEach(v => { _runsUnobserve(v); _vidRelease(v); });
  el.innerHTML = '';
  const items = _runsMediaItems(exp);
  if (!items.length) { el.style.display = 'none'; return; }
  el.style.display = '';
  items.forEach(item => {
    const fig = document.createElement('figure'); fig.className = 'run-media-item';
    const url = mediaUrl(item.path);
    const isGif = /\.gif$/i.test(item.path);
    if (item.kind === 'video' && !isGif) {
      const v = document.createElement('video');
      v.controls = true; v.playsInline = true; v.muted = true; v.preload = 'none';
      v.dataset.videoPath = item.path;
      _vidEnsurePoster(v);
      if (autoplay) { v.autoplay = true; v.preload = 'auto'; v.src = url; }
      else _runsObserve(v, url);
      fig.appendChild(v);
      addExpandBtn(fig, `${exp.name} / ${item.tag} - step ${item.step}`, body => {
        el.querySelectorAll('video').forEach(x => x.pause());
        const full = document.createElement('video');
        full.controls = true; full.autoplay = true; full.playsInline = true; full.src = url;
        body.appendChild(full);
      });
    } else {
      const img = document.createElement('img');
      img.loading = 'lazy'; img.alt = item.tag;
      img.src = isGif ? url : thumbUrl(item.path, 480);
      img.addEventListener('click', () => openImageFullscreen(`${exp.name} / ${item.tag} - step ${item.step}`, url));
      fig.appendChild(img);
    }
    const cap = document.createElement('figcaption');
    cap.textContent = `${item.kind} ${item.tag} - step ${item.step}` + (item.count > 1 ? ` (${item.count} logged)` : '');
    fig.appendChild(cap);
    el.appendChild(fig);
  });
}

function _runsCreateCard(name) {
  const card = document.createElement('div');
  card.className = 'run-card'; card.dataset.run = name;
  card.innerHTML = `
    <div class="run-card-head">
      <span class="dot"></span><span class="run-name"></span><span class="run-status"></span>
      <span class="run-meta"></span>
      <span class="run-actions">
        <button type="button" data-act="focus" title="Show only this run in charts">Focus</button>
        <button type="button" data-act="scalars" title="Open charts for this run">Charts</button>
      </span>
    </div>
    <div class="run-error"></div>
    <div class="run-metrics"></div>
    <div class="run-params"></div>
    <div class="run-media"></div>
    <div class="run-foot"></div>`;
  card.querySelector('.run-name').textContent = name;
  card.querySelector('[data-act="focus"]').addEventListener('click', () => {
    if (_runsFocus.size === 1 && _runsFocus.has(name)) focusRuns([]); else focusRuns([name]);
  });
  card.querySelector('[data-act="scalars"]').addEventListener('click', () => { focusRuns([name]); switchTab('scalars'); });
  return { card, mediaSig: null };
}

function _runsUpdateCard(entry, exp, varying) {
  const { card } = entry;
  const run = exp.run || {};
  const status = run.status || 'unknown';
  card.querySelector('.dot').style.background = expColors[exp.name] || 'var(--muted)';
  const st = card.querySelector('.run-status');
  st.className = 'run-status ' + (_RUN_STATUS_CLASS[status] || 'st-unknown');
  st.textContent = status === 'running' && run.stale ? 'running?' : status;
  const meta = [];
  if (run.started_at) meta.push(_runsFmtAge(run.started_at));
  if (run.duration_s !== null && run.duration_s !== undefined) meta.push(_runsFmtDuration(run.duration_s));
  if (run.last_step !== null && run.last_step !== undefined) meta.push('step ' + run.last_step);
  card.querySelector('.run-meta').textContent = meta.join(' · ');

  const err = card.querySelector('.run-error');
  err.textContent = run.error || '';
  err.style.display = run.error ? '' : 'none';

  const metrics = card.querySelector('.run-metrics');
  metrics.innerHTML = '';
  (run.headline || []).forEach(h => {
    const m = document.createElement('span'); m.className = 'run-metric';
    const best = (h.best !== null && h.best !== undefined && h.best !== h.last)
      ? `<span class="best">best ${escapeHtml(formatVal(h.best))} @${escapeHtml(String(h.best_step))}</span>` : '';
    const goal = h.goal === 'max' ? ' ↑' : (h.goal === 'min' ? ' ↓' : '');
    m.innerHTML = `<span class="tag">${escapeHtml(h.tag)}${goal}</span><span class="val">${escapeHtml(formatVal(h.last))}</span>${best}`;
    metrics.appendChild(m);
  });
  metrics.style.display = metrics.children.length ? '' : 'none';

  const params = card.querySelector('.run-params');
  params.innerHTML = '';
  const flat = _runsFlatten(exp.hparams, '', {});
  varying.filter(k => k in flat).slice(0, 10).forEach(k => {
    const chip = document.createElement('span'); chip.className = 'run-param';
    chip.textContent = k + '=' + _runsValueText(flat[k]);
    chip.title = k + ' = ' + JSON.stringify(flat[k]);
    params.appendChild(chip);
  });
  params.style.display = params.children.length ? '' : 'none';

  const foot = [];
  if (run.git && run.git.commit) foot.push('git ' + run.git.commit.slice(0, 8) + (run.git.dirty ? '+dirty' : ''));
  if (run.command) foot.push(run.command);
  const footEl = card.querySelector('.run-foot');
  footEl.textContent = foot.join(' · ');
  footEl.title = run.command || '';
  footEl.style.display = foot.length ? '' : 'none';

  const items = _runsMediaItems(exp);
  const sig = JSON.stringify(items.map(i => [i.kind, i.tag, i.step, i.path]));
  if (sig !== entry.mediaSig) {
    entry.mediaSig = sig;
    const autoplay = !_runsAutoplayDone && _runsFocus.has(exp.name) && _activeTab === 'runs';
    _runsBuildMedia(card.querySelector('.run-media'), exp, autoplay);
  }
  card.classList.toggle('run-focused', _runsFocus.has(exp.name));
}

function buildRuns() {
  const root = document.getElementById('runs-list');
  if (!root) return;
  if (!DATA.length) { root.innerHTML = '<div class="empty-msg">No runs yet.</div>'; _runsCards.clear(); return; }
  const empty = root.querySelector('.empty-msg'); if (empty) empty.remove();
  const varying = _runsVaryingKeys();
  const seen = new Set();
  DATA.forEach((exp, i) => {
    seen.add(exp.name);
    let entry = _runsCards.get(exp.name);
    if (!entry) { entry = _runsCreateCard(exp.name); _runsCards.set(exp.name, entry); }
    _runsUpdateCard(entry, exp, varying);
    if (root.children[i] !== entry.card) root.insertBefore(entry.card, root.children[i] || null);
  });
  [..._runsCards.keys()].forEach(name => {
    if (seen.has(name)) return;
    const { card } = _runsCards.get(name);
    card.querySelectorAll('video').forEach(v => { _runsUnobserve(v); _vidRelease(v); });
    card.remove(); _runsCards.delete(name);
  });
  const summary = document.getElementById('runs-summary');
  if (summary) {
    const counts = {};
    DATA.forEach(d => { const s = (d.run || {}).status || 'unknown'; counts[s] = (counts[s] || 0) + 1; });
    summary.textContent = DATA.length + ' runs: ' + Object.entries(counts).map(([s, n]) => n + ' ' + s).join(', ')
      + (_runsFocus.size ? ' · focused: ' + [..._runsFocus].join(', ') : '');
  }
}

// Show only *names* in charts (empty list = show all runs again).
function focusRuns(names) {
  const known = new Set(getExpNames());
  const chosen = names.filter(n => known.has(n));
  _runsFocus.clear(); chosen.forEach(n => _runsFocus.add(n));
  activePills.clear();
  (chosen.length ? chosen : [...known]).forEach(n => activePills.add(n));
  document.querySelectorAll('.exp-pill').forEach(p => p.classList.toggle('active', activePills.has(p.dataset.name)));
  _runsCards.forEach(({ card }, name) => card.classList.toggle('run-focused', _runsFocus.has(name)));
  buildCharts(); buildHistograms(); buildSystem();
  buildRuns();
}

// ── Deep links: ?run=A&run=B&tab=runs ────────────────────────
function applyUrlFocus() {
  let params;
  try { params = new URLSearchParams(window.location.search); } catch (e) { return; }
  const names = params.getAll('run').filter(Boolean);
  if (!names.length) return;
  const known = new Set(getExpNames());
  const chosen = names.filter(n => known.has(n));
  if (!chosen.length) return;
  chosen.forEach(n => _runsFocus.add(n));
  activePills.clear(); chosen.forEach(n => activePills.add(n));
}

function applyUrlTab() {
  let params;
  try { params = new URLSearchParams(window.location.search); } catch (e) { return; }
  const tab = params.get('tab');
  if (tab && document.querySelector(`.tabs button[data-tab="${CSS.escape(tab)}"]`)) switchTab(tab);
  if (_activeTab === 'runs' && _runsFocus.size) {
    // Rebuild the focused cards' media so their latest video autoplays once.
    _runsCards.forEach((entry, name) => { if (_runsFocus.has(name)) entry.mediaSig = null; });
    buildRuns();
    _runsAutoplayDone = true;
    const first = _runsCards.get([..._runsFocus][0]);
    if (first) first.card.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }
  _runsAutoplayDone = true;
}
