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
  const compact = !!o.compact;
  const tickFont = compact ? { size: 10 } : undefined;
  const xTickCb = isTime
    ? v => fmtDuration(v)
    : v => Number.isInteger(v) ? v : null;
  const xStepSize = isTime ? undefined : 1;
  return new Chart(ctx, {
    type: 'line', data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      layout: compact ? { padding: { top: 2, right: 4, bottom: 0, left: 0 } } : undefined,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      scales: {
        x: {
          type: 'linear',
          title: { display: o.xTitle !== false && !compact, text: isTime ? 'time' : 'step', color: tc.muted },
          ticks: { color: tc.muted, maxTicksLimit: o.xTicks || (compact ? 4 : 8), autoSkip: true, stepSize: xStepSize, callback: xTickCb, font: tickFont },
          grid: { color: tc.grid },
        },
        y: {
          beginAtZero: o.yBeginAtZero ?? false,
          min: o.yMin,
          max: o.yMax,
          suggestedMin: o.ySuggestedMin,
          suggestedMax: o.ySuggestedMax,
          ticks: { color: tc.muted, maxTicksLimit: o.yTicks || (compact ? 4 : 7), autoSkip: true, callback: yTickCallback, font: tickFont },
          grid: { color: tc.grid },
        },
      },
      plugins: {
        legend: {
          display: o.legend !== false,
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

    // Compute global min/max for extrema markers. When several points share
    // the extremum value (plateau), highlight the one closest to either
    // edge of the series so the marker hugs the border rather than stamping
    // every point on the plateau.
    const validYs = s.values.filter(v => v !== null && v !== undefined && !isNaN(v));
    const minY = validYs.length >= 3 ? Math.min(...validYs) : null;
    const maxY = validYs.length >= 3 ? Math.max(...validYs) : null;
    const hasExt = minY !== null && maxY !== null && minY !== maxY;
    const _borderIdx = target => {
      const last = s.values.length - 1;
      let best = -1; let bestDist = Infinity;
      for (let i = 0; i <= last; i++) {
        if (s.values[i] !== target) continue;
        const d = Math.min(i, last - i);
        if (d < bestDist) { best = i; bestDist = d; }
      }
      return best;
    };
    const minIdx = hasExt ? _borderIdx(minY) : -1;
    const maxIdx = hasExt ? _borderIdx(maxY) : -1;
    const isExt = j => j === minIdx || j === maxIdx;
    const extRadius = (v, j) => isExt(j) ? 5 : pr;
    const extBorder = (v, j) => isExt(j) ? '#fff8' : 'transparent';
    const extBorderW = (v, j) => isExt(j) ? 1.5 : 0;

    const hiddenFlag = _hiddenScalars.has(exp.name);
    if (sw > 0) {
      ds.push({
        label: '_raw_' + exp.name, data: raw, borderColor: withOpacity(col, rawOpacity), spanGaps: false,
        pointRadius: s.values.map((v, j) => extRadius(v, j)), pointBackgroundColor: col,
        pointBorderColor: s.values.map((v, j) => extBorder(v, j)), pointBorderWidth: s.values.map((v, j) => extBorderW(v, j)),
        borderWidth: 1, tension: 0, hidden: hiddenFlag
      });
      const sm = emaSmoothXY(xVals, s.values, sw);
      ds.push({ label: exp.name, data: sm, borderColor: col, spanGaps: false, pointRadius: 0, backgroundColor: col, borderWidth: 2, tension: 0, hidden: hiddenFlag });
    } else {
      ds.push({
        label: exp.name, data: raw, borderColor: col, spanGaps: false,
        pointRadius: s.values.map((v, j) => extRadius(v, j)), pointBackgroundColor: col,
        pointBorderColor: s.values.map((v, j) => extBorder(v, j)), pointBorderWidth: s.values.map((v, j) => extBorderW(v, j)),
        backgroundColor: col, borderWidth: 2, tension: 0, hidden: hiddenFlag
      });
    }
  });
  return ds;
}

function buildSystemDatasets(tag, sw) {
  const o = arguments[2] || {};
  const ds = [];
  const baseTime = _computeBaseTime(tag, DATA, (e, t) => (e.system_scalars || {})[t]);
  DATA.forEach(exp => {
    const s = (exp.system_scalars || {})[tag]; if (!s) return;
    const col = o.color || expColors[exp.name];
    const fillCol = withOpacity(col, o.fillOpacity ?? 0.08);
    const xVals = _getXVals(s, baseTime);
    const pr = o.pointRadius ?? (xVals.length <= 3 ? 3 : 0);
    const labelBase = o.label || tag;
    const label = DATA.length > 1 ? `${exp.name} · ${labelBase}` : labelBase;
    const common = {
      label,
      borderColor: col,
      spanGaps: false,
      pointRadius: pr,
      backgroundColor: fillCol,
      borderWidth: o.borderWidth || 2,
      tension: 0,
      fill: o.fill ?? false,
      borderDash: o.borderDash,
    };
    if (sw > 0) {
      const sm = emaSmoothXY(xVals, s.values, sw);
      ds.push({ ...common, data: sm, pointRadius: 0 });
    } else {
      ds.push({ ...common, data: xVals.map((st, j) => ({ x: st, y: s.values[j] })) });
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
  const hasPR = DATA.some(d => (d.pr_curve_tags || []).length > 0);
  const hasFigures = DATA.some(d => (d.figure_tags || []).length > 0);
  if (!allTags.length && !hasPR && !hasFigures) {
    c.innerHTML = '<div class="empty-msg">No scalars logged.</div>';
  } else if (tags.length) {
    enableDragSort(c, 'scalars');
  }
  buildHiddenChartsTray(hiddenTags);
  // PR curves and matplotlib figures share the #charts grid with the line
  // plots — they're all "scalars-tab content" conceptually.
  buildPRCurves();
  buildFigures();
}

// ── PR curves (rendered inside the Scalars tab) ──────────────
function _makePRChart(ctx, datasets) {
  const tc = themeColors();
  return new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: 'nearest', intersect: false },
      scales: {
        x: {
          type: 'linear', min: 0, max: 1,
          title: { display: true, text: 'recall', color: tc.muted },
          ticks: { color: tc.muted, callback: v => Number(v).toFixed(2) },
          grid: { color: tc.grid },
        },
        y: {
          min: 0, max: 1,
          title: { display: true, text: 'precision', color: tc.muted },
          ticks: { color: tc.muted, callback: v => Number(v).toFixed(2) },
          grid: { color: tc.grid },
        },
      },
      plugins: {
        legend: { labels: { color: tc.text, font: { size: 11 } } },
        tooltip: {
          mode: 'nearest', intersect: false,
          callbacks: {
            label: item => `${item.dataset.label}: P=${item.parsed.y.toFixed(3)} R=${item.parsed.x.toFixed(3)}`,
          },
        },
      },
    },
  });
}

function _prCardForTag(tag) {
  // Collect every (experiment, step) curve under this PR tag.
  const series = [];
  DATA.forEach(exp => {
    ((exp.pr_curves || {})[tag] || []).forEach(item => {
      const expName = exp.name;
      series.push({ exp: expName, step: item.step, path: item.path });
    });
  });
  if (!series.length) return null;
  const card = document.createElement('div');
  card.className = 'card pr-card';
  card.dataset.dragId = 'pr:' + tag;
  card.innerHTML = `<h2>${escapeHtml(tag)} <span class="pr-badge">PR</span></h2><canvas></canvas>`;
  // Fetch artifact JSONs in parallel, then render once.
  Promise.all(series.map(s => fetch(mediaUrl(s.path))
    .then(r => r.ok ? r.json() : null)
    .then(payload => ({ ...s, payload }))
    .catch(() => ({ ...s, payload: null }))
  )).then(rows => {
    const datasets = rows
      .filter(r => r.payload && Array.isArray(r.payload.points))
      .map(r => {
        const col = expColors[r.exp] || pickDistinctColor(Object.values(expColors));
        // Sort by recall ascending so the line draws cleanly.
        const pts = [...r.payload.points]
          .map(p => ({ x: Number(p.recall), y: Number(p.precision) }))
          .sort((a, b) => a.x - b.x);
        return {
          label: `${r.exp} · step ${r.step}`,
          data: pts,
          borderColor: col,
          backgroundColor: withOpacity(col, 0.15),
          tension: 0.1,
          pointRadius: 2.5,
          borderWidth: 2,
          showLine: true,
          fill: false,
        };
      });
    if (!datasets.length) {
      card.innerHTML += '<div class="empty-msg">PR data unavailable</div>';
      return;
    }
    const ctx = card.querySelector('canvas').getContext('2d');
    if (!window._prCharts) window._prCharts = [];
    window._prCharts.push(_makePRChart(ctx, datasets));
  });
  return card;
}

function buildPRCurves() {
  if (window._prCharts) { window._prCharts.forEach(x => x.destroy()); window._prCharts = []; }
  const grid = document.getElementById('charts');
  const tags = [...new Set(DATA.flatMap(d => d.pr_curve_tags || []))].sort();
  tags.forEach(tag => {
    const card = _prCardForTag(tag);
    if (card) grid.appendChild(card);
  });
}

// ── Figures (matplotlib charts rendered server-side; live on Scalars) ──
function _figureCardForTag(tag) {
  // One card per tag with a step slider when there's more than one entry.
  // Figures from multiple experiments stack vertically inside the card.
  const expEntries = [];
  DATA.forEach(exp => {
    const entries = (exp.figures || {})[tag] || [];
    if (!entries.length) return;
    expEntries.push({
      exp: exp.name,
      entries: [...entries].sort((a, b) => a.step - b.step),
    });
  });
  if (!expEntries.length) return null;

  const card = document.createElement('div');
  card.className = 'card figure-card';
  card.dataset.dragId = 'fig:' + tag;
  const heading = document.createElement('h2');
  heading.innerHTML = `${escapeHtml(tag)} <span class="pr-badge">fig</span>`;
  card.appendChild(heading);

  // Compute a shared step axis so the slider lines up across experiments.
  const allSteps = [...new Set(expEntries.flatMap(g => g.entries.map(e => e.step)))]
    .sort((a, b) => a - b);

  const stack = document.createElement('div');
  card.appendChild(stack);

  function _imageFor(group, stepIdx) {
    // Pick the entry at-or-before the requested step so missing steps fall
    // back to the most recent figure for that experiment.
    const targetStep = allSteps[stepIdx];
    let chosen = group.entries[0];
    for (const e of group.entries) {
      if (e.step <= targetStep) chosen = e;
    }
    return chosen;
  }

  function render(stepIdx) {
    stack.innerHTML = '';
    expEntries.forEach(group => {
      const e = _imageFor(group, stepIdx);
      const wrap = document.createElement('div');
      wrap.style.marginBottom = '8px';
      const lbl = document.createElement('div');
      lbl.className = 'figure-controls';
      lbl.innerHTML = `<span style="color:${expColors[group.exp] || 'var(--muted)'}">${escapeHtml(group.exp)}</span>` +
        `<span>step ${e.step}</span>` +
        `<a href="${mediaUrl(e.path)}" download>PNG</a>`;
      const img = document.createElement('img');
      img.className = 'figure-img';
      img.alt = `${tag} · ${group.exp} step ${e.step}`;
      img.src = mediaUrl(e.path);
      wrap.append(lbl, img);
      stack.appendChild(wrap);
    });
  }

  if (allSteps.length > 1) {
    const ctrl = document.createElement('div'); ctrl.className = 'figure-controls';
    const slider = document.createElement('input'); slider.type = 'range';
    slider.min = 0; slider.max = allSteps.length - 1; slider.value = allSteps.length - 1;
    const stepLbl = document.createElement('span');
    stepLbl.textContent = `step ${allSteps[allSteps.length - 1]}`;
    ctrl.append(slider, stepLbl);
    card.appendChild(ctrl);
    slider.addEventListener('input', () => {
      const i = parseInt(slider.value, 10);
      stepLbl.textContent = `step ${allSteps[i]}`;
      render(i);
    });
    render(allSteps.length - 1);
  } else {
    render(0);
  }
  return card;
}

function buildFigures() {
  const grid = document.getElementById('charts');
  const tags = [...new Set(DATA.flatMap(d => d.figure_tags || []))].sort();
  tags.forEach(tag => {
    const card = _figureCardForTag(tag);
    if (card) grid.appendChild(card);
  });
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
  c.classList.add('system-grid');
  if (window._sc) window._sc.forEach(x => x.destroy()); window._sc = [];

  const hasSys = DATA.some(d => (d.system_tags || []).length > 0);
  if (!hasSys) { c.innerHTML = '<div class="empty-msg">No system metrics logged.</div>'; return; }

  // Alerts
  DATA.forEach(exp => {
    ((exp.text_data || {})['system/alerts'] || []).forEach(e => {
      e.value.split('\n').forEach(line => {
        if (!line.trim()) return;
        const d = document.createElement('div');
        d.className = 'system-alert';
        d.textContent = line; c.appendChild(d);
      });
    });
  });

  function pctColor(pct, warn, danger, calm) {
    if (pct > danger) return '#f85149';
    if (pct > warn) return '#e3b341';
    return calm || '#3fb950';
  }

  function clampPct(v) {
    if (!Number.isFinite(v)) return 0;
    return Math.max(0, Math.min(100, v));
  }

  function metricRow(label, value, pct, color) {
    const safePct = clampPct(pct);
    return `
      <div class="system-metric" style="--meter-color:${color};--meter-value:${safePct}%;">
        <div class="system-metric-head">
          <span>${label}</span>
          <strong>${value}</strong>
        </div>
        <div class="system-meter"><span></span></div>
      </div>`;
  }

  function maybeChart(card, specs, opts) {
    const chartSpecs = Array.isArray(specs) ? specs : [specs];
    const datasets = chartSpecs.flatMap(spec => buildSystemDatasets(spec.tag, 0, spec));
    if (!datasets.some(d => d.data.length > 1)) return;
    const wrap = document.createElement('div'); wrap.className = 'system-chart-wrap';
    const cv = document.createElement('canvas'); wrap.appendChild(cv); card.appendChild(wrap);
    window._sc.push(makeLineChart(cv.getContext('2d'), datasets, {
      compact: true,
      legend: false,
      xTicks: 4,
      yTicks: 4,
      yMin: opts?.percent ? 0 : undefined,
      yMax: opts?.percent ? 100 : undefined,
    }));
  }

  // CPU
  const cpuCount = lastVal('system/cpu_count');
  const cpuPct = lastVal('system/cpu_percent');
  const load1 = lastVal('system/cpu_load_1m');
  const load5 = lastVal('system/cpu_load_5m');
  const load15 = lastVal('system/cpu_load_15m');
  if (cpuPct !== null || load1 !== null) {
    const usageColor = pctColor(cpuPct || 0, 70, 90);
    const card = document.createElement('div'); card.className = 'card system-card';
    card.innerHTML = `
      <div class="system-card-head">
        <h2>CPU</h2>
        ${cpuCount ? `<span class="system-chip">${cpuCount} cores</span>` : ''}
      </div>
      <div class="system-metrics">
        ${cpuPct !== null ? metricRow('Usage', `${cpuPct.toFixed(0)}%`, cpuPct, usageColor) : ''}
        ${load1 !== null ? `<div class="system-kv"><span>Load</span><strong>${load1.toFixed(2)} ${(load5 || 0).toFixed(2)} ${(load15 || 0).toFixed(2)}</strong></div>` : ''}
      </div>`;
    if (cpuPct !== null) maybeChart(card, [{ tag: 'system/cpu_percent', label: 'CPU', color: usageColor, pointRadius: 0 }], { percent: true });
    else maybeChart(card, [{ tag: 'system/cpu_load_normalized', label: 'load/core', color: '#58a6ff', pointRadius: 0 }]);
    c.appendChild(card);
  }

  // Memory
  const memTotal = lastVal('system/memory_total_gb');
  const memAvail = lastVal('system/memory_available_gb');
  const memPct = lastVal('system/memory_used_percent');
  const memUsed = lastVal('system/memory_used_gb');
  if (memTotal) {
    const barColor = pctColor(memPct, 70, 90);
    const card = document.createElement('div'); card.className = 'card system-card';
    card.innerHTML = `
      <div class="system-card-head">
        <h2>Memory</h2>
        <span class="system-chip">${memAvail.toFixed(1)}G free</span>
      </div>
      <div class="system-metrics">
        ${metricRow('Used', `${memPct.toFixed(0)}% (${memUsed.toFixed(1)}G / ${memTotal.toFixed(1)}G)`, memPct, barColor)}
      </div>`;
    maybeChart(card, [{ tag: 'system/memory_used_percent', label: 'Memory', color: barColor, pointRadius: 0 }], { percent: true });
    c.appendChild(card);
  }

  // Disk
  const diskTotal = lastVal('system/disk_total_gb');
  const diskFree = lastVal('system/disk_free_gb');
  const diskPct = lastVal('system/disk_used_percent');
  const diskUsed = lastVal('system/disk_used_gb');
  if (diskTotal) {
    const barColor = pctColor(diskPct, 85, 95);
    const card = document.createElement('div'); card.className = 'card system-card';
    card.innerHTML = `
      <div class="system-card-head">
        <h2>Disk</h2>
        <span class="system-chip">${diskFree.toFixed(1)}G free</span>
      </div>
      <div class="system-metrics">
        ${metricRow('Used', `${diskPct.toFixed(0)}% (${diskUsed.toFixed(1)}G / ${diskTotal.toFixed(0)}G)`, diskPct, barColor)}
      </div>`;
    maybeChart(card, [{ tag: 'system/disk_used_percent', label: 'Disk', color: barColor, pointRadius: 0 }], { percent: true });
    c.appendChild(card);
  }

  // GPUs
  let gi = 0;
  while (lastVal(`gpu/${gi}/utilization_percent`) !== null) {
    const util = lastVal(`gpu/${gi}/utilization_percent`);
    const memUsedGpu = lastVal(`gpu/${gi}/memory_used_gb`) || 0;
    const memTotalGpu = lastVal(`gpu/${gi}/memory_total_gb`) || 1;
    const temp = lastVal(`gpu/${gi}/temperature_c`) || 0;
    const barColor = pctColor(util, 70, 90);
    const gMemPct = lastVal(`gpu/${gi}/memory_used_percent`) ?? (memUsedGpu / memTotalGpu * 100);
    const memColor = pctColor(gMemPct, 70, 90, '#58a6ff');
    const card = document.createElement('div'); card.className = 'card system-card';
    card.innerHTML = `
      <div class="system-card-head">
        <h2>GPU ${gi}</h2>
        <span class="system-chip">${temp.toFixed(0)}°C</span>
      </div>
      <div class="system-metrics">
        ${metricRow('Core', `${util.toFixed(0)}%`, util, barColor)}
        ${metricRow('VRAM', `${gMemPct.toFixed(0)}% (${(memUsedGpu * 1024).toFixed(0)} / ${(memTotalGpu * 1024).toFixed(0)} MiB)`, gMemPct, memColor)}
      </div>`;
    maybeChart(card, [
      { tag: `gpu/${gi}/utilization_percent`, label: 'Core', color: barColor, pointRadius: 0 },
      { tag: `gpu/${gi}/memory_used_percent`, label: 'VRAM', color: memColor, pointRadius: 0 },
    ], { percent: true });
    c.appendChild(card);
    gi++;
  }
}
