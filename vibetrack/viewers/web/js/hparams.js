// hparams.js - HParams tab: nested inspection, comparison table, and result plots.
// Exposes: buildHParams, flattenHParams, bestScalar, buildHParamRows.
// Uses globals from core.js/charts.js/pills.js: DATA, escapeHtml, formatVal,
// themeColors, yTickCallback, expColors, withOpacity.

let _hpMetricTag = null;
let _hpObjective = null;
let _hpObjectiveTouched = false;
const _HP_VIZ_LIMIT = 12;

function _isPlainObject(v) {
  return !!v && typeof v === 'object' && !Array.isArray(v);
}

function _hasHParams(run) {
  return !!(run && run.hparams && Object.keys(run.hparams).length);
}

function flattenHParams(obj, prefix) {
  const out = {};
  const base = prefix || '';
  if (!_isPlainObject(obj)) return out;
  Object.keys(obj).sort().forEach(key => {
    const path = base ? `${base}.${key}` : key;
    const value = obj[key];
    if (_isPlainObject(value)) {
      Object.assign(out, flattenHParams(value, path));
    } else {
      out[path] = value;
    }
  });
  return out;
}

function _hparamDisplayValue(value) {
  if (value === undefined) return '';
  if (value === null) return 'null';
  if (typeof value === 'number') return Number.isFinite(value) ? formatVal(value) : String(value);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'string') return value;
  try { return JSON.stringify(value); } catch { return String(value); }
}

function _hparamCanonicalValue(value) {
  if (value === undefined) return '__missing__';
  if (value === null) return 'null';
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : String(value);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'string') return value;
  try { return JSON.stringify(value); } catch { return String(value); }
}

function _allHParamKeys(rows) {
  return [...new Set(rows.flatMap(row => Object.keys(row.flat || {})))].sort();
}

function _allScalarTags(dataArr) {
  return [...new Set((dataArr || DATA).flatMap(run => run.tags || []))].sort();
}

function _defaultObjective(tag) {
  return /loss|error|wer|cer|perplexity/i.test(tag || '') ? 'min' : 'max';
}

function bestScalar(run, tag, objective) {
  const s = run && run.scalars ? run.scalars[tag] : null;
  const values = s && Array.isArray(s.values) ? s.values : [];
  const steps = s && Array.isArray(s.steps) ? s.steps : [];
  let bestValue = null;
  let bestStep = null;
  for (let i = 0; i < values.length; i++) {
    const value = Number(values[i]);
    if (!Number.isFinite(value)) continue;
    if (
      bestValue === null ||
      (objective === 'min' ? value < bestValue : value > bestValue)
    ) {
      bestValue = value;
      bestStep = steps[i] !== undefined ? steps[i] : i;
    }
  }
  return { value: bestValue, step: bestStep };
}

function buildHParamRows(dataArr, metricTag, objective) {
  const data = dataArr || DATA;
  const tag = metricTag || (_allScalarTags(data)[0] || '');
  const obj = objective || _defaultObjective(tag);
  const rows = data.filter(_hasHParams).map(run => ({
    run,
    flat: flattenHParams(run.hparams || {}),
    result: tag ? bestScalar(run, tag, obj) : { value: null, step: null },
  }));
  rows.sort((a, b) => {
    const av = a.result.value;
    const bv = b.result.value;
    const aOk = av !== null && Number.isFinite(av);
    const bOk = bv !== null && Number.isFinite(bv);
    if (aOk && bOk) return obj === 'min' ? av - bv : bv - av;
    if (aOk) return -1;
    if (bOk) return 1;
    return String(a.run.name || '').localeCompare(String(b.run.name || ''));
  });
  return rows;
}

function _ensureHParamSelection(tags) {
  if (!tags.length) {
    _hpMetricTag = '';
    _hpObjective = 'max';
    return;
  }
  if (!_hpMetricTag || !tags.includes(_hpMetricTag)) {
    _hpMetricTag = tags[0];
    _hpObjective = _defaultObjective(_hpMetricTag);
    _hpObjectiveTouched = false;
  }
  if (!_hpObjective) _hpObjective = _defaultObjective(_hpMetricTag);
}

function _buildHParamControls(root, tags) {
  root.innerHTML = '';
  if (!tags.length) {
    root.innerHTML = '<div class="empty-msg">Log a scalar to rank hparams against a result.</div>';
    return;
  }

  const metricWrap = document.createElement('label');
  metricWrap.textContent = 'Result scalar';
  const select = document.createElement('select');
  select.id = 'hparams-metric';
  tags.forEach(tag => {
    const opt = document.createElement('option');
    opt.value = tag;
    opt.textContent = tag;
    select.appendChild(opt);
  });
  select.value = _hpMetricTag;
  select.addEventListener('change', () => {
    _hpMetricTag = select.value;
    _hpObjective = _defaultObjective(_hpMetricTag);
    _hpObjectiveTouched = false;
    buildHParams();
  });
  metricWrap.appendChild(select);

  const objective = document.createElement('div');
  objective.className = 'hparam-objective';
  const maxBtn = document.createElement('button');
  const minBtn = document.createElement('button');
  maxBtn.type = 'button';
  minBtn.type = 'button';
  maxBtn.textContent = 'Max';
  minBtn.textContent = 'Min';
  function syncObjective() {
    maxBtn.classList.toggle('active', _hpObjective === 'max');
    minBtn.classList.toggle('active', _hpObjective === 'min');
  }
  maxBtn.addEventListener('click', () => {
    _hpObjective = 'max';
    _hpObjectiveTouched = true;
    buildHParams();
  });
  minBtn.addEventListener('click', () => {
    _hpObjective = 'min';
    _hpObjectiveTouched = true;
    buildHParams();
  });
  syncObjective();
  objective.append(maxBtn, minBtn);

  root.append(metricWrap, objective);
}

function _varyingKeySet(rows, keys) {
  const out = new Set();
  keys.forEach(key => {
    const vals = new Set(rows.map(row => _hparamCanonicalValue(row.flat[key])));
    if (vals.size > 1) out.add(key);
  });
  return out;
}

function _buildHParamTable(root, rows, keys, varyingKeys) {
  root.innerHTML = '';
  if (!rows.length) {
    root.innerHTML = '<div class="empty-msg">No hparams logged.</div>';
    return;
  }
  const table = document.createElement('table');
  table.className = 'hparams-table';
  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  ['Experiment', 'Best ' + (_hpMetricTag || 'result'), 'Step'].concat(keys).forEach(label => {
    const th = document.createElement('th');
    th.textContent = label;
    hr.appendChild(th);
  });
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  rows.forEach((row, idx) => {
    const tr = document.createElement('tr');
    if (idx === 0 && row.result.value !== null) tr.classList.add('best');

    const name = document.createElement('td');
    name.className = 'hparam-run';
    name.style.color = expColors[row.run.name] || 'inherit';
    name.textContent = row.run.name || 'run';
    tr.appendChild(name);

    const value = document.createElement('td');
    value.className = 'hparam-result';
    value.textContent = row.result.value === null ? '' : formatVal(row.result.value);
    tr.appendChild(value);

    const step = document.createElement('td');
    step.textContent = row.result.step === null || row.result.step === undefined ? '' : String(row.result.step);
    tr.appendChild(step);

    keys.forEach(key => {
      const td = document.createElement('td');
      const hasKey = Object.prototype.hasOwnProperty.call(row.flat, key);
      if (!hasKey) td.classList.add('missing');
      if (varyingKeys.has(key)) td.classList.add('diff');
      td.textContent = hasKey ? _hparamDisplayValue(row.flat[key]) : '';
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  root.appendChild(table);
}

function _renderHParamNode(parent, value) {
  if (_isPlainObject(value)) {
    const ul = document.createElement('ul');
    Object.keys(value).sort().forEach(key => {
      const li = document.createElement('li');
      const keyEl = document.createElement('span');
      keyEl.className = 'hparam-tree-key';
      keyEl.textContent = key;
      li.appendChild(keyEl);
      if (_isPlainObject(value[key])) {
        _renderHParamNode(li, value[key]);
      } else {
        const val = document.createElement('span');
        val.className = 'hparam-tree-value';
        val.textContent = _hparamDisplayValue(value[key]);
        li.appendChild(val);
      }
      ul.appendChild(li);
    });
    parent.appendChild(ul);
    return;
  }
  const val = document.createElement('span');
  val.className = 'hparam-tree-value';
  val.textContent = _hparamDisplayValue(value);
  parent.appendChild(val);
}

function _buildHParamTrees(root, rows) {
  root.innerHTML = '';
  rows.forEach(row => {
    const details = document.createElement('details');
    details.className = 'hparams-tree';
    details.open = rows.length <= 4;
    const summary = document.createElement('summary');
    summary.style.color = expColors[row.run.name] || 'var(--text)';
    summary.textContent = row.run.name || 'run';
    details.appendChild(summary);
    _renderHParamNode(details, row.run.hparams || {});
    root.appendChild(details);
  });
}

function _numericHParam(values) {
  return values.length > 0 && values.every(v => typeof v === 'number' && Number.isFinite(v));
}

function _chartOptions(key, labels) {
  const tc = themeColors();
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: 'nearest', intersect: true },
    scales: {
      x: {
        type: labels ? 'category' : 'linear',
        labels,
        title: { display: true, text: key, color: tc.muted },
        ticks: { color: tc.muted, maxTicksLimit: 8 },
        grid: { color: tc.grid },
      },
      y: {
        title: { display: true, text: _hpMetricTag || 'result', color: tc.muted },
        ticks: { color: tc.muted, maxTicksLimit: 6, callback: yTickCallback },
        grid: { color: tc.grid },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: item => {
            const raw = item.raw || {};
            const y = raw.y === null || raw.y === undefined ? 'NaN' : formatVal(raw.y);
            return `${raw.run || item.dataset.label}: ${key}=${raw.hpLabel}; ${_hpMetricTag}=${y}; step ${raw.step}`;
          },
        },
      },
    },
  };
}

function _buildHParamCharts(root, rows, varyingKeys) {
  root.innerHTML = '';
  if (window._hpCharts) window._hpCharts.forEach(chart => chart.destroy());
  window._hpCharts = [];

  const keys = [...varyingKeys].filter(key => rows.some(row => row.result.value !== null && Object.prototype.hasOwnProperty.call(row.flat, key))).slice(0, _HP_VIZ_LIMIT);
  if (!keys.length) {
    root.innerHTML = '<div class="empty-msg">No varying hparams with scalar results to visualize.</div>';
    return;
  }

  keys.forEach(key => {
    const points = rows
      .filter(row => row.result.value !== null && Object.prototype.hasOwnProperty.call(row.flat, key))
      .map(row => ({
        run: row.run.name,
        value: row.flat[key],
        hpLabel: _hparamDisplayValue(row.flat[key]),
        y: row.result.value,
        step: row.result.step,
      }));
    const distinct = new Set(points.map(p => _hparamCanonicalValue(p.value)));
    if (distinct.size < 2) return;
    const numeric = _numericHParam(points.map(p => p.value));
    const labels = numeric ? null : [...new Set(points.map(p => p.hpLabel))].sort();
    const datasets = points.map(p => {
      const color = expColors[p.run] || pickDistinctColor(Object.values(expColors));
      return {
        label: p.run,
        data: [{
          x: numeric ? Number(p.value) : p.hpLabel,
          y: p.y,
          run: p.run,
          hpLabel: p.hpLabel,
          step: p.step,
        }],
        borderColor: color,
        backgroundColor: withOpacity(color, 0.7),
        pointRadius: 4,
        pointHoverRadius: 6,
      };
    });

    const card = document.createElement('div');
    card.className = 'hparam-viz-card';
    card.dataset.dragId = key;
    const title = document.createElement('h3');
    title.textContent = key;
    const wrap = document.createElement('div');
    wrap.className = 'hparam-chart-wrap';
    const canvas = document.createElement('canvas');
    wrap.appendChild(canvas);
    card.append(title, wrap);
    root.appendChild(card);
    window._hpCharts.push(new Chart(canvas.getContext('2d'), {
      type: 'scatter',
      data: { datasets },
      options: _chartOptions(key, labels),
    }));
  });

  if (!root.children.length) {
    root.innerHTML = '<div class="empty-msg">No varying hparams with scalar results to visualize.</div>';
  } else {
    enableDragSort(root, 'hparams');
  }
}

function buildHParams() {
  const tab = document.getElementById('tab-hparams');
  if (!tab) return;

  const controls = document.getElementById('hparams-controls');
  const summary = document.getElementById('hparams-summary');
  const table = document.getElementById('hparams-table-wrap');
  const trees = document.getElementById('hparams-trees');
  const viz = document.getElementById('hparams-viz');
  const runs = DATA.filter(_hasHParams);

  if (!runs.length) {
    controls.innerHTML = '';
    summary.innerHTML = '';
    table.innerHTML = '<div class="empty-msg">No hparams logged.</div>';
    trees.innerHTML = '';
    viz.innerHTML = '';
    return;
  }

  const tags = _allScalarTags(DATA);
  _ensureHParamSelection(tags);
  _buildHParamControls(controls, tags);

  const rows = buildHParamRows(DATA, _hpMetricTag, _hpObjective);
  const keys = _allHParamKeys(rows);
  const varyingKeys = _varyingKeySet(rows, keys);
  const validResults = rows.filter(row => row.result.value !== null).length;

  summary.innerHTML =
    `<span>${runs.length} experiments</span>` +
    `<span>${keys.length} hparams</span>` +
    `<span>${varyingKeys.size} varying</span>` +
    `<span>${validResults} ranked by ${escapeHtml(_hpObjective)} ${escapeHtml(_hpMetricTag || 'result')}</span>`;

  _buildHParamTable(table, rows, keys, varyingKeys);
  _buildHParamTrees(trees, rows);
  _buildHParamCharts(viz, rows, varyingKeys);
}

if (typeof window !== 'undefined') {
  window.flattenHParams = flattenHParams;
  window.bestScalar = bestScalar;
  window.buildHParamRows = buildHParamRows;
  window.buildHParams = buildHParams;
}
