// settings.js — config load/save, auto-refresh polling, project-info log-dir editor.
// Owns globals: currentConfig, refreshTimer.

let currentConfig = {};

function loadSettings() {
  return fetch(apiUrl('/api/config'))
    .then(r => r.json())
    .then(cfg => { currentConfig = cfg; applyConfig(cfg); })
    .catch(() => { });
}

function applyConfig(cfg) {
  const w = cfg.web || {};
  document.body.classList.remove('light', 'orange');
  const theme = w.theme || 'light';
  if (theme === 'light' || theme === 'orange') document.body.classList.add(theme);
  document.getElementById('cfg-theme').value = w.theme || 'light';
  document.getElementById('cfg-smoothing').value = cfg.smoothing || 'ema';
  const sw = cfg.smooth_weight ?? 0.6;
  document.getElementById('cfg-smooth-weight').value = sw;
  document.getElementById('cfg-smooth-weight-val').textContent = Number(sw).toFixed(1);
  document.getElementById('cfg-refresh').value = w.auto_refresh ?? 5;
  setAutoRefresh(w.auto_refresh ?? 5);
  document.getElementById('cfg-image-fps').value = String(w.image_play_fps || 4);
  const rawOpacity = w.raw_scalar_opacity ?? 0.17;
  document.getElementById('cfg-raw-opacity').value = rawOpacity;
  document.getElementById('cfg-raw-opacity-val').textContent = Number(rawOpacity).toFixed(2);
  document.getElementById('cfg-x-axis').value = w.x_axis_mode || 'step';
  const sysInterval = cfg.system_metrics_interval ?? (cfg.check_resources_before_run ? 'once' : 'none');
  document.getElementById('cfg-sys-interval').value = String(sysInterval);
}

function saveSettings() {
  const c = Object.assign({}, currentConfig);
  c.smoothing = document.getElementById('cfg-smoothing').value;
  c.smooth_weight = parseFloat(document.getElementById('cfg-smooth-weight').value);
  c.web = c.web || {}; c.web.theme = document.getElementById('cfg-theme').value;
  c.web.auto_refresh = parseInt(document.getElementById('cfg-refresh').value) || 0;
  c.web.image_play_fps = parseInt(document.getElementById('cfg-image-fps').value) || 4;
  c.web.raw_scalar_opacity = parseFloat(document.getElementById('cfg-raw-opacity').value);
  c.web.x_axis_mode = document.getElementById('cfg-x-axis').value;
  const iv = document.getElementById('cfg-sys-interval').value;
  c.system_metrics_interval = iv === 'none' ? null : isNaN(iv) ? iv : parseInt(iv);
  delete c.check_resources_before_run;
  currentConfig = c;
  fetch(apiUrl('/api/config'), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(c) });
  applyConfig(c); buildCharts(); buildSystem();
}

let refreshTimer = null;
let _lastDataText = null;

function _userIsBusy() {
  if (document.querySelector('.exp-rename-input')) return true;
  const ae = document.activeElement;
  if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA' || ae.isContentEditable)) return true;
  for (const m of document.querySelectorAll('audio, video')) {
    if (!m.paused && !m.ended) return true;
  }
  if (_imgAnimPlaying() || _imgCompareSelection.length !== 0) return true;
  const sel = window.getSelection && window.getSelection();
  if (sel && sel.rangeCount && !sel.isCollapsed && sel.toString().length > 2) return true;
  return false;
}

function setAutoRefresh(s) {
  if (refreshTimer) clearInterval(refreshTimer);
  if (s <= 0) return;
  refreshTimer = setInterval(() => {
    fetch(apiUrl('/api/data')).then(r => r.text()).then(txt => {
      if (txt === _lastDataText) return;
      if (_userIsBusy()) return;
      _lastDataText = txt;
      const d = JSON.parse(txt);
      const sx = window.scrollX, sy = window.scrollY;
      DATA.length = 0; DATA.push(...d);
      buildTabs(); buildPills(); buildCharts(); buildSystem();
      buildImages(); buildAudio(); buildVideo(); buildArtifacts(); buildText(); buildHistograms();
      window.scrollTo(sx, sy);
    });
  }, s * 1000);
}

function _wireSettingsInputs() {
  document.getElementById('cfg-smoothing').addEventListener('change', saveSettings);
  document.getElementById('cfg-smooth-weight').addEventListener('input', () => { document.getElementById('cfg-smooth-weight-val').textContent = Number(document.getElementById('cfg-smooth-weight').value).toFixed(1); saveSettings(); });
  document.getElementById('cfg-raw-opacity').addEventListener('input', () => { document.getElementById('cfg-raw-opacity-val').textContent = Number(document.getElementById('cfg-raw-opacity').value).toFixed(2); saveSettings(); });
  document.getElementById('cfg-theme').addEventListener('change', saveSettings);
  document.getElementById('cfg-refresh').addEventListener('change', saveSettings);
  document.getElementById('cfg-image-fps').addEventListener('change', saveSettings);
  document.getElementById('cfg-sys-interval').addEventListener('change', saveSettings);
  document.getElementById('cfg-x-axis').addEventListener('change', saveSettings);
}

// ── Project Info ────────────────────────────────────────────
function buildProjectInfo() {
  // Show parent (project) directories, not individual timestamped run dirs
  const parentDirs = [...new Set(DATA.map(e => {
    const d = e.log_dir; if (!d) return null;
    const sep = d.includes('/') ? '/' : '\\';
    return d.substring(0, d.lastIndexOf(sep));
  }).filter(Boolean))];
  if (!parentDirs.length) return;
  const card = document.getElementById('project-info-card');
  const content = document.getElementById('project-info-content');
  card.style.display = '';
  content.innerHTML = '';
  parentDirs.forEach(d => {
    const row = document.createElement('div');
    row.className = 'settings-row';
    row.innerHTML = `<label style="min-width:auto;flex:none;">Log directory</label>`;
    const input = document.createElement('input');
    input.type = 'text'; input.value = d;
    input.style.cssText = 'flex:1;font-size:0.82em;font-family:monospace;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:4px 8px;';
    input.dataset.original = d;
    input.addEventListener('keydown', e => { if (e.key === 'Enter') input.blur(); if (e.key === 'Escape') { input.value = input.dataset.original; input.blur(); } });
    input.addEventListener('blur', () => {
      const newDir = input.value.trim();
      const oldDir = input.dataset.original;
      if (!newDir || newDir === oldDir) { input.value = oldDir; return; }
      if (!confirm('Move log directory?\n\nFrom: ' + oldDir + '\nTo: ' + newDir)) { input.value = oldDir; return; }
      input.disabled = true;
      fetch('/api/move-logdir', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ old: oldDir, new: newDir }),
      })
      .then(r => r.json())
      .then(res => {
        input.disabled = false;
        if (res.ok) {
          input.dataset.original = newDir;
          // Update all run log_dirs that live under the moved parent
          DATA.forEach(e => {
            if (e.log_dir && (e.log_dir === oldDir || e.log_dir.startsWith(oldDir + '/') || e.log_dir.startsWith(oldDir + '\\'))) {
              e.log_dir = newDir + e.log_dir.substring(oldDir.length);
            }
          });
        } else { alert('Move failed: ' + (res.error || 'unknown')); input.value = oldDir; }
      })
      .catch(() => { input.disabled = false; alert('Move failed'); input.value = oldDir; });
    });
    row.appendChild(input);
    content.appendChild(row);
  });
}
