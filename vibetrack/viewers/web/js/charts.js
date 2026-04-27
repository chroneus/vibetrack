// charts.js — Chart.js helpers + scalars, histograms, system charts.
// Exposes: themeColors, yTickCallback, makeLineChart, makeBarChart, buildCharts,
// buildHistograms, buildSystem, hideChart, restoreChart.
// Owns globals: _hiddenScalars, _hiddenCharts, _hoveredChartTag.

// ── Resolve CSS var colors for Chart.js ──────────────────────
function themeColors() {
  const s = getComputedStyle(document.body);
  return {
    muted: s.getPropertyValue('--muted').trim(),
    text: s.getPropertyValue('--text').trim(),
    grid: s.getPropertyValue('--grid').trim(),
  };
}

// ── Hidden-charts state ──────────────────────────────────────
const _hiddenScalars = new Set();
const _hiddenChartsKey = 'vt_hidden_charts_' + (window.VT_PROJECT || '');
const _hiddenCharts = new Set((() => { try { return JSON.parse(localStorage.getItem(_hiddenChartsKey)) || []; } catch { return []; } })());
function _saveHiddenCharts() { try { localStorage.setItem(_hiddenChartsKey, JSON.stringify([..._hiddenCharts])); } catch {} }
function hideChart(tag) { if (!tag) return; _hiddenCharts.add(tag); _saveHiddenCharts(); buildCharts(); }
function restoreChart(tag) { _hiddenCharts.delete(tag); _saveHiddenCharts(); buildCharts(); }
let _hoveredChartTag = null;

// ── Chart helpers ────────────────────────────────────────────
function yTickCallback(v, index, ticks) {
  if (v === null || isNaN(v)) return '';
  const range = this.max - this.min;
  if (range === 0 || ticks.length <= 1) return formatVal(v);
  const step = range / (ticks.length - 1);
  const abs = Math.abs(v);
  if (abs >= 1e6 || (abs > 0 && abs < 1e-4)) return v.toExponential(2);
  const decPlaces = step > 0 ? Math.max(0, -Math.floor(Math.log10(Math.abs(step)))) + 1 : 4;
  return parseFloat(v.toFixed(Math.min(decPlaces, 8))).toString();
}

function makeLineChart(ctx, datasets, opts) {
  const o = opts || {};
  const tc = themeColors();
  const isTime = _xAxisMode() === 'wall_time';
  const xTickCb = isTime
    ? v => fmtDuration(v)
    : v => Number.isInteger(v) ? v : null;
  const xStepSize = isTime ? undefined : 1;
  return new Chart(ctx, {
    type: 'line', data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      scales: {
        x: { type: 'linear', title: { display: true, text: isTime ? 'time' : 'step', color: tc.muted }, ticks: { color: tc.muted, maxTicksLimit: o.xTicks || 8, autoSkip: true, stepSize: xStepSize, callback: xTickCb }, grid: { color: tc.grid } },
        y: { beginAtZero: false, ticks: { color: tc.muted, maxTicksLimit: o.yTicks || 7, autoSkip: true, callback: yTickCallback }, grid: { color: tc.grid } },
      },
      plugins: {
        legend: {
          labels: { color: tc.text, font: { size: 11 }, filter: i => !i.text.startsWith('_') },
          onClick: (e, i, l) => {
            Chart.defaults.plugins.legend.onClick.call(l, e, i, l);
            const ds = l.chart.data.datasets;
            const clicked = ds[i.datasetIndex];
            if (!clicked) return;
            const name = clicked.label.startsWith('_raw_') ? clicked.label.slice(5) : clicked.label;
            const pairLabels = ['_raw_' + name, name];
            const isHidden = l.chart.getDatasetMeta(i.datasetIndex).hidden;
            ds.forEach((d, idx) => {
              if (idx === i.datasetIndex) return;
              if (pairLabels.includes(d.label)) l.chart.getDatasetMeta(idx).hidden = isHidden;
            });
            if (isHidden) _hiddenScalars.add(name); else _hiddenScalars.delete(name);
            l.chart.update();
          }
        },
        tooltip: {
          mode: 'nearest', axis: 'x', intersect: false,
          animation: { duration: 150 },
          filter: item => !item.dataset.label.startsWith('_'),
          callbacks: {
            title: items => {
              if (!items.length) return '';
              return isTime ? fmtDuration(items[0].parsed.x) : 'step ' + items[0].parsed.x;
            },
            label: item => {
              const y = item.parsed.y;
              if (y === null || y === undefined || isNaN(y)) return item.dataset.label + ': NaN';
              return item.dataset.label + ': ' + formatVal(y);
            }
          }
        }
      },
    },
  });
}

function makeBarChart(ctx, labels, datasets) {
  const tc = themeColors();
  return new Chart(ctx, {
    type: 'bar', data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      scales: { x: { ticks: { color: tc.muted, maxTicksLimit: 20 }, grid: { color: tc.grid } }, y: { ticks: { color: tc.muted, maxTicksLimit: 7, autoSkip: true, callback: yTickCallback }, grid: { color: tc.grid } } },
      plugins: { legend: { labels: { color: tc.text, font: { size: 11 } } } }
    }
  });
}

function fmtDuration(secs) {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  return h > 0 ? `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`
                : `${m}:${String(s).padStart(2,'0')}`;
}

function _xAxisMode() { return currentConfig.web?.x_axis_mode || 'step'; }

function _getXVals(s, baseTime) {
  if (_xAxisMode() === 'wall_time' && s.wall_times) {
    return s.wall_times.map(t => t - baseTime);
  }
  return s.steps;
}

function _computeBaseTime(tag, dataArr, accessor) {
  let min = Infinity;
  dataArr.forEach(exp => {
    const s = accessor(exp, tag); if (!s || !s.wall_times) return;
    for (const t of s.wall_times) { if (t < min) min = t; }
  });
  return min === Infinity ? 0 : min;
}

function buildScalarDatasets(tag, sw) {
  const sel = [...activePills]; const ds = [];
  const rawOpacity = currentConfig.web?.raw_scalar_opacity ?? 0.17;
  const baseTime = _computeBaseTime(tag, DATA, (e, t) => e.scalars[t]);
  DATA.forEach(exp => {
    if (!sel.includes(exp.name)) return;
    const s = exp.scalars[tag]; if (!s) return;
    const col = expColors[exp.name];
    const xVals = _getXVals(s, baseTime);
    const raw = xVals.map((st, j) => ({ x: st, y: s.values[j] }));
    const pr = raw.length <= 3 ? 4 : 0;

    // Compute global min/max for extrema markers
    const validYs = s.values.filter(v => v !== null && v !== undefined && !isNaN(v));
    const minY = validYs.length >= 3 ? Math.min(...validYs) : null;
    const maxY = validYs.length >= 3 ? Math.max(...validYs) : null;
    const hasExt = minY !== null && maxY !== null && minY !== maxY;
    const extRadius = v => hasExt && (v === minY || v === maxY) ? 5 : pr;
    const extBorder = v => hasExt && (v === minY || v === maxY) ? '#fff8' : 'transparent';
    const extBorderW = v => hasExt && (v === minY || v === maxY) ? 1.5 : 0;

    const hiddenFlag = _hiddenScalars.has(exp.name);
    if (sw > 0) {
      ds.push({
        label: '_raw_' + exp.name, data: raw, borderColor: withOpacity(col, rawOpacity), spanGaps: false,
        pointRadius: s.values.map(v => extRadius(v)), pointBackgroundColor: col,
        pointBorderColor: s.values.map(v => extBorder(v)), pointBorderWidth: s.values.map(v => extBorderW(v)),
        borderWidth: 1, tension: 0, hidden: hiddenFlag
      });
      const sm = emaSmoothXY(xVals, s.values, sw);
      ds.push({ label: exp.name, data: sm, borderColor: col, spanGaps: false, pointRadius: 0, backgroundColor: col, borderWidth: 2, tension: 0, hidden: hiddenFlag });
    } else {
      ds.push({
        label: exp.name, data: raw, borderColor: col, spanGaps: false,
        pointRadius: s.values.map(v => extRadius(v)), pointBackgroundColor: col,
        pointBorderColor: s.values.map(v => extBorder(v)), pointBorderWidth: s.values.map(v => extBorderW(v)),
        backgroundColor: col, borderWidth: 2, tension: 0, hidden: hiddenFlag
      });
    }
  });
  return ds;
}

function buildSystemDatasets(tag, sw) {
  const ds = [];
  const baseTime = _computeBaseTime(tag, DATA, (e, t) => (e.system_scalars || {})[t]);
  DATA.forEach(exp => {
    const s = (exp.system_scalars || {})[tag]; if (!s) return;
    const col = expColors[exp.name];
    const fillCol = col + '33';
    const xVals = _getXVals(s, baseTime);
    const pr = xVals.length <= 3 ? 4 : 0;
    if (sw > 0) {
      const sm = emaSmoothXY(xVals, s.values, sw);
      ds.push({ label: exp.name, data: sm, borderColor: col, spanGaps: false, pointRadius: 0, backgroundColor: fillCol, borderWidth: 2, tension: 0, fill: true });
    } else {
      ds.push({ label: exp.name, data: xVals.map((st, j) => ({ x: st, y: s.values[j] })), borderColor: col, spanGaps: false, pointRadius: pr, backgroundColor: fillCol, borderWidth: 2, tension: 0, fill: true });
    }
  });
  return ds;
}

function buildHistDatasets(tag) {
  const ds = []; let labels = [];
  DATA.forEach(exp => {
    const entries = (exp.histogram_data || {})[tag]; if (!entries || !entries.length) return;
    const last = entries[entries.length - 1];
    if (!labels.length && last.bins.length > 1) labels = last.bins.slice(0, -1).map((b, j) => ((b + last.bins[j + 1]) / 2).toFixed(2));
    const col = expColors[exp.name] || pickDistinctColor(Object.values(expColors));
    ds.push({ label: exp.name, data: last.counts, backgroundColor: withOpacity(col, 0.5), borderColor: col, borderWidth: 1 });
  });
  return { labels, datasets: ds };
}

// ── Scalars ──────────────────────────────────────────────────
function getConfigSmooth() { return currentConfig.smooth_weight ?? 0.6; }

function addMinimizeBtn(card, tag) {
  const btn = document.createElement('button');
  btn.className = 'minimize-btn';
  btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><line x1="2" y1="6" x2="10" y2="6"/></svg>';
  btn.title = 'Hide chart (press _ while hovering)';
  btn.addEventListener('click', e => { e.stopPropagation(); hideChart(tag); });
  card.appendChild(btn);
}

function buildHiddenChartsTray(hiddenTags) {
  const tray = document.getElementById('hidden-charts-tray');
  const list = document.getElementById('hidden-charts-list');
  const count = document.getElementById('hidden-charts-count');
  list.innerHTML = '';
  if (!hiddenTags.length) { tray.style.display = 'none'; return; }
  tray.style.display = '';
  count.textContent = hiddenTags.length;
  hiddenTags.forEach(tag => {
    const chip = document.createElement('span');
    chip.className = 'hidden-chip';
    chip.title = 'Restore ' + tag;
    chip.innerHTML = `<span class="restore-icon">↺</span><span></span>`;
    chip.querySelector('span:last-child').textContent = tag;
    chip.addEventListener('click', () => restoreChart(tag));
    list.appendChild(chip);
  });
}

function buildCharts() {
  const sw = getConfigSmooth();
  const c = document.getElementById('charts'); c.innerHTML = '';
  if (window._charts) window._charts.forEach(x => x.destroy()); window._charts = [];
  const allTags = [...new Set(DATA.flatMap(d => d.tags || []))].sort();
  const hiddenTags = allTags.filter(t => _hiddenCharts.has(t));
  const tags = allTags.filter(t => !_hiddenCharts.has(t));
  tags.forEach(tag => {
    const card = document.createElement('div'); card.className = 'card';
    card.dataset.dragId = tag;
    card.innerHTML = `<h2>${escapeHtml(tag)}</h2><canvas></canvas>`;
    card.addEventListener('mouseenter', () => { _hoveredChartTag = tag; });
    card.addEventListener('mouseleave', () => { if (_hoveredChartTag === tag) _hoveredChartTag = null; });
    c.appendChild(card);
    addMinimizeBtn(card, tag);
    addExpandBtn(card, tag, (body, ov) => {
      const ctrl = document.createElement('label');
      ctrl.style.cssText = 'display:flex;align-items:center;gap:6px;font-size:0.85em;color:var(--muted);';
      ctrl.innerHTML = `Smoothing <input type="range" min="0" max="0.99" step="0.01" value="${sw}" style="width:120px;accent-color:var(--accent);"> <span class="smooth-val">${sw}</span>`;
      ov.querySelector('.fs-header-right').insertBefore(ctrl, ov.querySelector('.fs-close'));
      const cv = document.createElement('canvas');
      body.appendChild(cv);
      const fsTicks = { xTicks: 20, yTicks: 15 };
      let chart = makeLineChart(cv.getContext('2d'), buildScalarDatasets(tag, sw), fsTicks);
      const sl = ctrl.querySelector('input[type=range]'), vl = ctrl.querySelector('.smooth-val');
      sl.addEventListener('input', () => { vl.textContent = sl.value; chart.destroy(); chart = makeLineChart(cv.getContext('2d'), buildScalarDatasets(tag, parseFloat(sl.value)), fsTicks); });
    });
    window._charts.push(makeLineChart(card.querySelector('canvas').getContext('2d'), buildScalarDatasets(tag, sw)));
  });
  if (!allTags.length) c.innerHTML = '<div class="empty-msg">No scalars logged.</div>';
  else if (tags.length) enableDragSort(c, 'scalars');
  buildHiddenChartsTray(hiddenTags);
}

// ── Histograms ───────────────────────────────────────────────
function buildHistograms() {
  const c = document.getElementById('histogram-charts'); c.innerHTML = '';
  if (window._hc) window._hc.forEach(x => x.destroy()); window._hc = [];
  const tags = [...new Set(DATA.flatMap(d => d.histogram_tags || []))].sort();
  tags.forEach(tag => {
    const card = document.createElement('div'); card.className = 'card';
    card.dataset.dragId = tag;
    card.innerHTML = `<h2>${escapeHtml(tag)}</h2><canvas></canvas>`;
    c.appendChild(card);
    const hd = buildHistDatasets(tag);
    addExpandBtn(card, tag, body => {
      const cv = document.createElement('canvas'); body.appendChild(cv);
      const hd2 = buildHistDatasets(tag);
      makeBarChart(cv.getContext('2d'), hd2.labels, hd2.datasets);
    });
    window._hc.push(makeBarChart(card.querySelector('canvas').getContext('2d'), hd.labels, hd.datasets));
  });
  if (!tags.length) c.innerHTML = '<div class="empty-msg">No histograms logged.</div>';
  else enableDragSort(c, 'histograms');
}

// ── System ───────────────────────────────────────────────────
function lastVal(tag) {
  for (let i = DATA.length - 1; i >= 0; i--) {
    const s = (DATA[i].system_scalars || {})[tag];
    if (s && s.values.length) return s.values[s.values.length - 1];
  }
  return null;
}

function buildSystem() {
  const c = document.getElementById('system-charts'); c.innerHTML = '';
  if (window._sc) window._sc.forEach(x => x.destroy()); window._sc = [];

  const hasSys = DATA.some(d => (d.system_tags || []).length > 0);
  if (!hasSys) { c.innerHTML = '<div class="empty-msg">No system metrics logged.</div>'; return; }

  // Alerts
  DATA.forEach(exp => {
    ((exp.text_data || {})['system/alerts'] || []).forEach(e => {
      e.value.split('\n').forEach(line => {
        if (!line.trim()) return;
        const d = document.createElement('div');
        d.style.cssText = 'background:#3d1114;border:1px solid #f85149;border-radius:8px;padding:10px 16px;margin-bottom:8px;color:#ff7b72;font-weight:600;';
        d.textContent = line; c.appendChild(d);
      });
    });
  });

  const grid = document.createElement('div'); grid.className = 'grid'; c.appendChild(grid);

  function maybeChart(card, ...tags) {
    const datasets = tags.flatMap(tag => buildSystemDatasets(tag, 0));
    if (!datasets.some(d => d.data.length > 1)) return;
    const wrap = document.createElement('div'); wrap.style.cssText = 'height:120px;margin-top:12px;';
    const cv = document.createElement('canvas'); wrap.appendChild(cv); card.appendChild(wrap);
    window._sc.push(makeLineChart(cv.getContext('2d'), datasets));
  }

  // CPU
  const cpuCount = lastVal('system/cpu_count');
  const cpuPct = lastVal('system/cpu_percent');
  const load1 = lastVal('system/cpu_load_1m');
  const load5 = lastVal('system/cpu_load_5m');
  const load15 = lastVal('system/cpu_load_15m');
  if (cpuPct !== null || load1 !== null) {
    const card = document.createElement('div'); card.className = 'card';
    let html = '<h2>CPU</h2>';
    if (cpuCount) html += `<div style="font-size:1.6em;font-weight:700;margin:8px 0;">${cpuCount} cores</div>`;
    if (cpuPct !== null) html += `<div style="color:var(--muted);font-size:0.9em;">Usage: ${cpuPct.toFixed(0)}%</div>`;
    if (load1 !== null) html += `<div style="color:var(--muted);font-size:0.9em;">Load: ${load1.toFixed(2)} ${(load5||0).toFixed(2)} ${(load15||0).toFixed(2)} (1m 5m 15m)</div>`;
    card.innerHTML = html;
    maybeChart(card, 'system/cpu_percent');
    grid.appendChild(card);
  }

  // Memory
  const memTotal = lastVal('system/memory_total_gb');
  const memAvail = lastVal('system/memory_available_gb');
  const memPct = lastVal('system/memory_used_percent');
  const memUsed = lastVal('system/memory_used_gb');
  if (memTotal) {
    const barColor = memPct > 90 ? '#f85149' : memPct > 70 ? '#e3b341' : '#3fb950';
    const card = document.createElement('div'); card.className = 'card';
    card.innerHTML = `<h2>Memory</h2>
      <div style="font-size:1.6em;font-weight:700;margin:8px 0;">${memAvail.toFixed(1)}G free <span style="color:var(--muted);font-size:0.5em;">/ ${memTotal.toFixed(1)}G</span></div>
      <div style="background:var(--border);border-radius:4px;height:8px;margin:8px 0;"><div style="background:${barColor};height:100%;border-radius:4px;width:${memPct.toFixed(0)}%;"></div></div>
      <div style="color:var(--muted);font-size:0.85em;">${memPct.toFixed(0)}% used (${memUsed.toFixed(1)}G)</div>`;
    maybeChart(card, 'system/memory_used_percent');
    grid.appendChild(card);
  }

  // Disk
  const diskTotal = lastVal('system/disk_total_gb');
  const diskFree = lastVal('system/disk_free_gb');
  const diskPct = lastVal('system/disk_used_percent');
  const diskUsed = lastVal('system/disk_used_gb');
  if (diskTotal) {
    const barColor = diskPct > 95 ? '#f85149' : diskPct > 85 ? '#e3b341' : '#3fb950';
    const card = document.createElement('div'); card.className = 'card';
    card.innerHTML = `<h2>Disk</h2>
      <div style="font-size:1.6em;font-weight:700;margin:8px 0;">${diskFree.toFixed(1)}G free <span style="color:var(--muted);font-size:0.5em;">/ ${diskTotal.toFixed(0)}G</span></div>
      <div style="background:var(--border);border-radius:4px;height:8px;margin:8px 0;"><div style="background:${barColor};height:100%;border-radius:4px;width:${diskPct.toFixed(0)}%;"></div></div>
      <div style="color:var(--muted);font-size:0.85em;">${diskPct.toFixed(0)}% used (${diskUsed.toFixed(1)}G)</div>`;
    maybeChart(card, 'system/disk_used_percent');
    grid.appendChild(card);
  }

  // GPUs
  let gi = 0;
  while (lastVal(`gpu/${gi}/utilization_percent`) !== null) {
    const util = lastVal(`gpu/${gi}/utilization_percent`);
    const memUsedGpu = lastVal(`gpu/${gi}/memory_used_gb`) || 0;
    const memTotalGpu = lastVal(`gpu/${gi}/memory_total_gb`) || 1;
    const temp = lastVal(`gpu/${gi}/temperature_c`) || 0;
    const barColor = util > 90 ? '#f85149' : util > 70 ? '#e3b341' : '#3fb950';
    const gMemPct = (memUsedGpu / memTotalGpu * 100).toFixed(0);
    const memColor = gMemPct > 90 ? '#f85149' : gMemPct > 70 ? '#e3b341' : '#58a6ff';
    const card = document.createElement('div'); card.className = 'card';
    card.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <h2 style="margin:0; font-size:1.15em; display:flex; align-items:center; gap:8px; color:var(--text); letter-spacing:-0.2px;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.8"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>
          GPU ${gi}
        </h2>
        <div style="background:var(--border); padding:4px 10px; border-radius:16px; font-size:0.75em; font-weight:700; color:var(--text); display:flex; align-items:center; gap:4px; box-shadow:inset 0 1px 2px rgba(0,0,0,0.1);">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"></path></svg>
          ${temp.toFixed(0)}°C
        </div>
      </div>
      <div style="margin-top:20px; display:flex; flex-direction:column; gap:16px;">
        <div>
          <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px;">
            <span style="color:var(--muted); font-size:0.75em; text-transform:uppercase; letter-spacing:0.8px; font-weight:600;">Core Utilization</span>
            <span style="font-size:1.4em; font-weight:800; line-height:1; letter-spacing:-0.5px;">${util.toFixed(0)}<span style="font-size:0.6em; color:var(--muted); margin-left:2px;">%</span></span>
          </div>
          <div style="background:var(--border); border-radius:8px; height:8px; overflow:hidden; box-shadow:inset 0 1px 3px rgba(0,0,0,0.05);">
            <div style="background:${barColor}; height:100%; width:${util}%; border-radius:8px; box-shadow:0 0 10px ${barColor}80; transition:width 0.4s ease-out;"></div>
          </div>
        </div>
        <div>
          <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px;">
            <span style="color:var(--muted); font-size:0.75em; text-transform:uppercase; letter-spacing:0.8px; font-weight:600;">Memory Usage</span>
            <span style="font-size:1.1em; font-weight:700; line-height:1; letter-spacing:-0.2px;">${(memUsedGpu*1024).toFixed(0)} <span style="font-size:0.8em; color:var(--muted); font-weight:500;">/ ${(memTotalGpu*1024).toFixed(0)} MiB</span></span>
          </div>
          <div style="background:var(--border); border-radius:8px; height:8px; overflow:hidden; box-shadow:inset 0 1px 3px rgba(0,0,0,0.05);">
            <div style="background:${memColor}; height:100%; width:${gMemPct}%; border-radius:8px; box-shadow:0 0 10px ${memColor}80; transition:width 0.4s ease-out;"></div>
          </div>
        </div>
      </div>`;
    maybeChart(card, `gpu/${gi}/utilization_percent`, `gpu/${gi}/memory_used_gb`);
    grid.appendChild(card);
    gi++;
  }
}
