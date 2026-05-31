// media.js — images, audio, video, artifacts, text tabs.
// Owns globals: _imgAnimTimers, _imgCompareSelection.

// ── Images ───────────────────────────────────────────────────
const _imgAnimTimers = {};
function _imgAnimPlaying() { return Object.values(_imgAnimTimers).some(Boolean); }
const _imgCompareSelection = []; // { exp, tag, step, path, cellEl, sliderEl }
const _imgTimelineMarkerCallbacks = []; // functions to refresh markers

function _imgUpdateCompareToolbar() {
  let bar = document.getElementById('compare-toolbar');
  if (_imgCompareSelection.length < 1) {
    if (bar) bar.style.display = 'none';
    return;
  }
  if (!bar) {
    bar = document.createElement('div');
    bar.id = 'compare-toolbar';
    bar.className = 'compare-toolbar';
    document.body.appendChild(bar);
  }
  bar.style.display = '';
  bar.innerHTML = '';
  const lbl = document.createElement('span');
  lbl.textContent = _imgCompareSelection.length + ' image' + (_imgCompareSelection.length > 1 ? 's' : '') + ' selected';
  // Show tag info for cross-tag awareness
  const tags = [...new Set(_imgCompareSelection.map(s => s.tag))];
  if (tags.length > 1) {
    const tagInfo = document.createElement('span');
    tagInfo.className = 'cmp-tag-info';
    tagInfo.textContent = '(' + tags.length + ' tags)';
    lbl.appendChild(tagInfo);
  }
  const clearBtn = document.createElement('button');
  clearBtn.textContent = 'Clear';
  clearBtn.addEventListener('click', _imgClearSelection);
  const cmpBtn = document.createElement('button');
  cmpBtn.textContent = 'Compare';
  cmpBtn.className = 'cmp-btn-primary';
  cmpBtn.disabled = _imgCompareSelection.length < 2;
  cmpBtn.addEventListener('click', _imgOpenCompare);
  bar.append(lbl, clearBtn, cmpBtn);
}

function _imgClearSelection() {
  const seen = new Set();
  _imgCompareSelection.forEach(s => {
    if (!s.cellEl || seen.has(s.cellEl)) return;
    seen.add(s.cellEl);
    s.cellEl.classList.remove('img-selected');
    const cb = s.cellEl.querySelector('.img-select-cb');
    if (cb) cb.checked = false;
    s.cellEl.querySelectorAll('.img-pin-badge').forEach(b => b.remove());
    if (s.cellEl._unpinFn) s.cellEl._unpinFn();
  });
  _imgCompareSelection.length = 0;
  _imgUpdateCompareToolbar();
  _imgRefreshAllTimelineMarkers();
}

function _imgRefreshAllTimelineMarkers() {
  _imgTimelineMarkerCallbacks.forEach(fn => fn());
}

function _imgToggleSelect(exp, tag, step, path, cellEl, checked) {
  const stepIdx = _imgCompareSelection.findIndex(s => s.cellEl === cellEl && s.step === step);
  if (checked) {
    if (stepIdx === -1 && _imgCompareSelection.length < 6) {
      _imgCompareSelection.push({ exp, tag, step, path, cellEl });
      cellEl.classList.add('img-selected');
      _imgUpdateCellBadges(cellEl);
    } else if (stepIdx === -1) {
      const cb = cellEl.querySelector('.img-select-cb');
      if (cb) cb.checked = false;
      return;
    }
  } else {
    if (stepIdx !== -1) {
      _imgCompareSelection.splice(stepIdx, 1);
      _imgUpdateCellBadges(cellEl);
      const remaining = _imgCompareSelection.filter(s => s.cellEl === cellEl);
      if (!remaining.length) {
        cellEl.classList.remove('img-selected');
        if (cellEl._unpinFn) cellEl._unpinFn();
      }
    }
  }
  _imgUpdateCompareToolbar();
  _imgRefreshAllTimelineMarkers();
}

function _imgUpdateCellBadges(cellEl) {
  cellEl.querySelectorAll('.img-pin-badge').forEach(b => b.remove());
  const pins = _imgCompareSelection.filter(s => s.cellEl === cellEl);
  if (!pins.length) return;
  const badge = document.createElement('div');
  badge.className = 'img-pin-badge';
  badge.textContent = pins.map(p => 'step ' + p.step).join(', ');
  cellEl.appendChild(badge);
}

function _imgSyncCellCheckbox(cellEl, steps, sliderIdx) {
  const cb = cellEl.querySelector('.img-select-cb');
  if (!cb) return;
  const step = steps[sliderIdx];
  cb.checked = _imgCompareSelection.some(s => s.cellEl === cellEl && s.step === step);
}

function _imgOpenCompare() {
  const sel = _imgCompareSelection.slice();
  if (sel.length < 2) return;
  const pairs = [];
  for (let i = 0; i < sel.length; i++)
    for (let j = i + 1; j < sel.length; j++)
      pairs.push([sel[i], sel[j]]);

  const ov = document.createElement('div'); ov.className = 'fs-overlay';
  ov.innerHTML = '<div class="fs-header"><h2>Image Comparison</h2><div class="fs-header-right">' +
    '<div class="cmp-mode-bar">' +
    '<button data-mode="toggle" class="active">Toggle</button>' +
    '<button data-mode="slider">Slider</button>' +
    '<button data-mode="magnifier">Magnifier</button>' +
    '<button data-mode="blend">Blend</button></div>' +
    '<div class="cmp-zoom-bar" title="Zoom (click % to reset)">' +
    '<button class="cmp-zoom-out" title="Zoom out">\u2212</button>' +
    '<span class="cmp-zoom-val" title="Reset zoom">100%</span>' +
    '<button class="cmp-zoom-in" title="Zoom in">+</button></div>' +
    '<button class="fs-close">Close \u00b7 Esc</button></div></div>' +
    '<div class="fs-body compare-body"></div>';
  document.body.appendChild(ov);

  let zoom = 1;
  const zoomVal = ov.querySelector('.cmp-zoom-val');
  const ZOOM_MIN = 0.25, ZOOM_MAX = 5;
  const ZOOM_PRESETS = [0.5, 1, 1.5, 2, 3];
  const applyZoom = () => {
    ov.style.setProperty('--cmp-zoom', zoom);
    zoomVal.textContent = Math.round(zoom * 100) + '%';
  };
  ov.querySelector('.cmp-zoom-in').addEventListener('click', () => { zoom = Math.min(ZOOM_MAX, +(zoom * 1.25).toFixed(3)); applyZoom(); });
  ov.querySelector('.cmp-zoom-out').addEventListener('click', () => { zoom = Math.max(ZOOM_MIN, +(zoom / 1.25).toFixed(3)); applyZoom(); });
  zoomVal.title = 'Click to cycle preset zoom';
  zoomVal.addEventListener('click', () => {
    const next = ZOOM_PRESETS.find(p => p > zoom + 0.01);
    zoom = next ?? ZOOM_PRESETS[0];
    applyZoom();
  });
  applyZoom();

  const body = ov.querySelector('.fs-body');
  let activePair = 0;
  let activeMode = 'toggle';
  let toggleSide = 0; // 0 = first image, 1 = second

  // Mode switcher
  ov.querySelectorAll('.cmp-mode-bar button').forEach(btn => {
    btn.addEventListener('click', () => {
      ov.querySelectorAll('.cmp-mode-bar button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeMode = btn.dataset.mode;
      toggleSide = 0;
      renderPair(activePair);
    });
  });

  // Pair navigation
  let pairNav = null;
  if (pairs.length > 1) {
    pairNav = document.createElement('div');
    pairNav.className = 'compare-pair-nav';
    pairs.forEach((pair, pi) => {
      const btn = document.createElement('button');
      const crossTag = pair[0].tag !== pair[1].tag;
      const crossStep = pair[0].step !== pair[1].step;
      const sameExp = pair[0].exp === pair[1].exp;
      let labelA, labelB;
      if (crossTag) {
        labelA = _truncExp(pair[0].tag) + ':' + pair[0].step;
        labelB = _truncExp(pair[1].tag) + ':' + pair[1].step;
      } else if (sameExp && crossStep) {
        labelA = 'step ' + pair[0].step;
        labelB = 'step ' + pair[1].step;
      } else {
        labelA = _truncExp(pair[0].exp) + (crossStep ? ':' + pair[0].step : '');
        labelB = _truncExp(pair[1].exp) + (crossStep ? ':' + pair[1].step : '');
      }
      btn.textContent = labelA + ' vs ' + labelB;
      btn.addEventListener('click', () => { toggleSide = 0; renderPair(pi); });
      pairNav.appendChild(btn);
    });
    body.appendChild(pairNav);
  }

  const widgetWrap = document.createElement('div');
  widgetWrap.className = 'compare-widget-wrap';
  body.appendChild(widgetWrap);

  const labels = document.createElement('div');
  labels.className = 'compare-labels';
  body.appendChild(labels);

  function makeLabels(a, b) {
    labels.innerHTML = '';
    const lblA = document.createElement('span'); lblA.className = 'cmp-label';
    lblA.innerHTML = '<span class="exp">' + escapeHtml(a.exp) + '</span> / ' + escapeHtml(a.tag) + ' \u2014 step ' + a.step;
    const lblB = document.createElement('span'); lblB.className = 'cmp-label';
    lblB.innerHTML = '<span class="exp">' + escapeHtml(b.exp) + '</span> / ' + escapeHtml(b.tag) + ' \u2014 step ' + b.step;
    labels.append(lblA, lblB);
  }

  function renderSlider(a, b) {
    widgetWrap.innerHTML = '';
    const slider = document.createElement('img-comparison-slider');
    const imgA = document.createElement('img'); imgA.slot = 'first'; imgA.src = mediaUrl(a.path);
    const imgB = document.createElement('img'); imgB.slot = 'second'; imgB.src = mediaUrl(b.path);
    slider.append(imgA, imgB);
    widgetWrap.appendChild(slider);
    makeLabels(a, b);
  }

  function renderToggle(a, b) {
    widgetWrap.innerHTML = '';
    const wrap = document.createElement('div'); wrap.className = 'cmp-toggle-wrap';
    const img = document.createElement('img');
    const cur = toggleSide === 0 ? a : b;
    img.src = mediaUrl(cur.path);
    const indicator = document.createElement('div'); indicator.className = 'cmp-toggle-indicator';
    indicator.textContent = (toggleSide + 1) + ' / 2';
    const leftArr = document.createElement('button'); leftArr.className = 'cmp-toggle-arrow left'; leftArr.innerHTML = '&#9664;';
    const rightArr = document.createElement('button'); rightArr.className = 'cmp-toggle-arrow right'; rightArr.innerHTML = '&#9654;';
    leftArr.style.visibility = toggleSide === 0 ? 'hidden' : '';
    rightArr.style.visibility = toggleSide === 1 ? 'hidden' : '';
    leftArr.addEventListener('click', () => { if (toggleSide > 0) { toggleSide = 0; renderToggle(a, b); } });
    rightArr.addEventListener('click', () => { if (toggleSide < 1) { toggleSide = 1; renderToggle(a, b); } });
    wrap.append(indicator, leftArr, img, rightArr);
    widgetWrap.appendChild(wrap);
    // Highlight active label
    labels.innerHTML = '';
    const lblA = document.createElement('span'); lblA.className = 'cmp-label';
    lblA.style.opacity = toggleSide === 0 ? '1' : '0.4';
    lblA.innerHTML = '<span class="exp">' + escapeHtml(a.exp) + '</span> / ' + escapeHtml(a.tag) + ' \u2014 step ' + a.step;
    const lblB = document.createElement('span'); lblB.className = 'cmp-label';
    lblB.style.opacity = toggleSide === 1 ? '1' : '0.4';
    lblB.innerHTML = '<span class="exp">' + escapeHtml(b.exp) + '</span> / ' + escapeHtml(b.tag) + ' \u2014 step ' + b.step;
    labels.append(lblA, lblB);
  }

  function renderMagnifier(a, b) {
    widgetWrap.innerHTML = '';
    const wrap = document.createElement('div'); wrap.className = 'cmp-mag-wrap';
    const zoom = 3;
    const cells = [a, b].map(item => {
      const cell = document.createElement('div'); cell.className = 'cmp-mag-cell';
      const img = document.createElement('img'); img.src = mediaUrl(item.path);
      const lens = document.createElement('div'); lens.className = 'cmp-mag-lens';
      cell.append(img, lens);
      wrap.appendChild(cell);
      return { cell, img, lens };
    });
    widgetWrap.appendChild(wrap);

    function syncLens(srcIdx, e) {
      const src = cells[srcIdx];
      const rect = src.img.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const px = x / rect.width;
      const py = y / rect.height;
      cells.forEach(c => {
        const r = c.img.getBoundingClientRect();
        const lx = px * r.width;
        const ly = py * r.height;
        c.lens.style.display = 'block';
        c.lens.style.left = (lx - 80) + 'px';
        c.lens.style.top = (ly - 80) + 'px';
        c.lens.style.backgroundImage = 'url(' + c.img.src + ')';
        c.lens.style.backgroundSize = (r.width * zoom) + 'px ' + (r.height * zoom) + 'px';
        c.lens.style.backgroundPosition = (-lx * zoom + 80) + 'px ' + (-ly * zoom + 80) + 'px';
      });
    }

    function hideLens() { cells.forEach(c => { c.lens.style.display = 'none'; }); }

    cells.forEach((c, i) => {
      c.cell.addEventListener('mousemove', e => syncLens(i, e));
      c.cell.addEventListener('mouseleave', hideLens);
    });
    makeLabels(a, b);
  }

  let blendOpacity = 0.5;

  function renderBlend(a, b) {
    widgetWrap.innerHTML = '';
    const wrap = document.createElement('div'); wrap.className = 'cmp-blend-wrap';
    const canvas = document.createElement('div'); canvas.className = 'blend-canvas';
    const imgA = document.createElement('img'); imgA.src = mediaUrl(a.path);
    const imgB = document.createElement('img'); imgB.src = mediaUrl(b.path); imgB.className = 'blend-top';
    imgB.style.opacity = blendOpacity;
    canvas.append(imgA, imgB);
    wrap.appendChild(canvas);
    widgetWrap.appendChild(wrap);

    const ctrl = document.createElement('div'); ctrl.className = 'cmp-blend-controls';
    const lblA = document.createElement('label'); lblA.textContent = _truncExp(a.exp);
    const slider = document.createElement('input'); slider.type = 'range';
    slider.min = 0; slider.max = 100; slider.value = Math.round(blendOpacity * 100);
    const valSpan = document.createElement('span'); valSpan.className = 'blend-val';
    valSpan.textContent = Math.round(blendOpacity * 100) + '%';
    const lblB = document.createElement('label'); lblB.textContent = _truncExp(b.exp);
    slider.addEventListener('input', () => {
      blendOpacity = parseInt(slider.value) / 100;
      imgB.style.opacity = blendOpacity;
      valSpan.textContent = Math.round(blendOpacity * 100) + '%';
    });
    ctrl.append(lblA, slider, valSpan, lblB);
    body.appendChild(ctrl);

    makeLabels(a, b);
  }

  function renderPair(pi) {
    activePair = pi;
    const [a, b] = pairs[pi];
    // Remove blend controls from previous render
    const oldCtrl = body.querySelector('.cmp-blend-controls');
    if (oldCtrl) oldCtrl.remove();
    if (activeMode === 'slider') renderSlider(a, b);
    else if (activeMode === 'toggle') renderToggle(a, b);
    else if (activeMode === 'magnifier') renderMagnifier(a, b);
    else if (activeMode === 'blend') renderBlend(a, b);
    if (pairNav) pairNav.querySelectorAll('button').forEach((btn, i) => btn.classList.toggle('active', i === pi));
  }

  renderPair(0);

  // Keyboard
  const close = () => { ov.remove(); document.removeEventListener('keydown', onKey); };
  ov.querySelector('.fs-close').addEventListener('click', close);
  const onKey = e => {
    if (e.key === 'Escape') { close(); return; }
    if (activeMode === 'toggle') {
      if (e.key === 'ArrowLeft' && toggleSide > 0) { toggleSide = 0; renderPair(activePair); }
      if (e.key === 'ArrowRight' && toggleSide < 1) { toggleSide = 1; renderPair(activePair); }
    } else if (pairs.length > 1) {
      if (e.key === 'ArrowLeft' && activePair > 0) renderPair(activePair - 1);
      if (e.key === 'ArrowRight' && activePair < pairs.length - 1) renderPair(activePair + 1);
    }
  };
  document.addEventListener('keydown', onKey);
}

function _truncExp(name) {
  return name.length > 20 ? name.slice(0, 18) + '\u2026' : name;
}

function _imgCollectByTag() {
  // Returns { tag: [ { exp, step, path }, ... ] } sorted by step
  const byTag = {};
  DATA.forEach(exp => (exp.image_tags || []).forEach(tag => {
    if (!byTag[tag]) byTag[tag] = [];
    (exp.images[tag] || []).forEach(img => {
      byTag[tag].push({ exp: exp.name, step: img.step, path: img.path });
    });
  }));
  // Sort each tag's images by step
  Object.values(byTag).forEach(arr => arr.sort((a, b) => a.step - b.step));
  return byTag;
}

function _imgUniqueSteps(entries) {
  return [...new Set(entries.map(e => e.step))].sort((a, b) => a - b);
}

function _imgBuildAnimate(container, entries, tag) {
  const steps = _imgUniqueSteps(entries);
  if (!steps.length) return;

  // Group by experiment
  const byExp = {};
  entries.forEach(e => {
    if (!byExp[e.exp]) byExp[e.exp] = {};
    byExp[e.exp][e.step] = e.path;
  });
  const expNames = Object.keys(byExp);

  const wrap = document.createElement('div'); wrap.className = 'img-animate';

  // Controls — hide play/slider entirely for single-step tags
  const ctrl = document.createElement('div'); ctrl.className = 'img-animate-controls';
  const playBtn = document.createElement('button'); playBtn.innerHTML = '&#9654;'; playBtn.title = 'Play';
  const stepLabel = document.createElement('span'); stepLabel.className = 'step-label';
  const sliderWrap = document.createElement('div'); sliderWrap.className = 'timeline-slider-wrap';
  const slider = document.createElement('input'); slider.type = 'range';
  slider.min = 0; slider.max = Math.max(0, steps.length - 1); slider.value = 0;
  const markerTrack = document.createElement('div'); markerTrack.className = 'timeline-markers';
  sliderWrap.append(slider, markerTrack);
  if (steps.length > 1) {
    ctrl.append(playBtn, sliderWrap, stepLabel);
    wrap.appendChild(ctrl);
  }

  // Timeline marker refresh for this tag
  function refreshMarkers() {
    markerTrack.innerHTML = '';
    const selected = _imgCompareSelection.filter(s => s.tag === tag);
    if (!selected.length || steps.length < 2) return;
    selected.forEach(s => {
      const stepIdx = steps.indexOf(s.step);
      if (stepIdx === -1) return;
      const pct = (stepIdx / (steps.length - 1)) * 100;
      const marker = document.createElement('div');
      marker.className = 'timeline-marker';
      marker.style.left = pct + '%';
      marker.title = s.exp + ' — step ' + s.step;
      markerTrack.appendChild(marker);
    });
  }
  _imgTimelineMarkerCallbacks.push(refreshMarkers);

  // Viewport
  const viewport = document.createElement('div'); viewport.className = 'img-animate-viewport';
  const cells = {};
  const cellEls = {};
  const placeholders = {};
  expNames.forEach(exp => {
    const cell = document.createElement('div'); cell.className = 'animate-cell';
    const img = document.createElement('img'); img.alt = exp;
    const placeholder = document.createElement('div'); placeholder.className = 'img-placeholder';
    placeholder.style.display = 'none';
    const phMsg = document.createElement('span'); phMsg.className = 'img-placeholder-msg';
    placeholder.appendChild(phMsg);
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.className = 'img-select-cb';
    cb.addEventListener('click', e => e.stopPropagation());
    cb.addEventListener('change', () => {
      const step = steps[parseInt(slider.value)];
      const path = byExp[exp][step];
      if (!path) { cb.checked = false; return; }
      const alreadyPinned = _imgCompareSelection.some(s => s.cellEl === cell && s.step === step);
      if (alreadyPinned) {
        _imgToggleSelect(exp, tag, step, path, cell, false);
      } else {
        _imgToggleSelect(exp, tag, step, path, cell, true);
      }
      _imgSyncCellCheckbox(cell, steps, parseInt(slider.value));
    });
    const lbl = document.createElement('div'); lbl.className = 'cell-label';
    lbl.innerHTML = `<span class="exp" style="color:${expColors[exp] || 'var(--muted)'}">${escapeHtml(exp)}</span>`;
    cell.append(cb, img, placeholder, lbl);
    viewport.appendChild(cell);
    cells[exp] = img;
    cellEls[exp] = cell;
    placeholders[exp] = placeholder;
    cell._unpinFn = () => renderFrame(parseInt(slider.value));
    img.style.cursor = 'pointer';
    img.addEventListener('click', () => {
      const step = steps[parseInt(slider.value)];
      const path = byExp[exp][step];
      if (path) openImageFullscreen(`${exp} / ${escapeHtml(tag)} — step ${step}`, mediaUrl(path));
    });
  });
  wrap.appendChild(viewport);
  container.appendChild(wrap);

  let playing = false;
  const timerKey = tag;

  function renderFrame(idx) {
    const step = steps[idx];
    stepLabel.textContent = 'Step ' + step + ' / ' + steps[steps.length - 1];
    slider.value = idx;
    expNames.forEach(exp => {
      _imgSyncCellCheckbox(cellEls[exp], steps, idx);
      const path = byExp[exp][step];
      if (path) {
        cells[exp].src = mediaUrl(path);
        cells[exp].style.display = '';
        placeholders[exp].style.display = 'none';
      } else {
        cells[exp].removeAttribute('src');
        cells[exp].style.display = 'none';
        placeholders[exp].style.display = '';
        const msg = placeholders[exp].querySelector('.img-placeholder-msg');
        if (msg) msg.textContent = 'no frame @ step ' + step;
      }
    });
  }

  function stopAnim() {
    playing = false;
    playBtn.innerHTML = '&#9654;';
    if (_imgAnimTimers[timerKey]) { clearInterval(_imgAnimTimers[timerKey]); _imgAnimTimers[timerKey] = null; }
  }

  function startAnim() {
    playing = true;
    playBtn.innerHTML = '&#9646;&#9646;';
    const fps = parseInt(document.getElementById('cfg-image-fps').value) || 4;
    _imgAnimTimers[timerKey] = setInterval(() => {
      let idx = parseInt(slider.value) + 1;
      if (idx >= steps.length) idx = 0;
      renderFrame(idx);
    }, 1000 / fps);
  }

  playBtn.addEventListener('click', () => {
    if (playing) stopAnim(); else startAnim();
  });

  slider.addEventListener('input', () => {
    stopAnim();
    renderFrame(parseInt(slider.value));
  });

  // Click on marker track to jump to that step
  markerTrack.addEventListener('click', e => {
    if (steps.length < 2) return;
    const rect = markerTrack.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    const idx = Math.round(pct * (steps.length - 1));
    stopAnim();
    renderFrame(Math.max(0, Math.min(idx, steps.length - 1)));
  });

  renderFrame(0);
  refreshMarkers();
}

function buildImages() {
  // Clear selection, animation timers, and marker callbacks
  _imgCompareSelection.length = 0;
  _imgUpdateCompareToolbar();
  _imgTimelineMarkerCallbacks.length = 0;
  Object.keys(_imgAnimTimers).forEach(k => {
    if (_imgAnimTimers[k]) clearInterval(_imgAnimTimers[k]);
    delete _imgAnimTimers[k];
  });

  const root = document.getElementById('images-container'); root.innerHTML = '';
  const byTag = _imgCollectByTag();
  const tags = Object.keys(byTag).sort();

  if (!tags.length) {
    root.innerHTML = '<div class="empty-msg">No images logged.</div>';
    return;
  }

  tags.forEach(tag => {
    const entries = byTag[tag];
    const section = document.createElement('div'); section.className = 'img-tag-section';

    // Header
    const header = document.createElement('div'); header.className = 'img-tag-header';
    const h3 = document.createElement('h3'); h3.textContent = tag;
    const expandBtn = document.createElement('button'); expandBtn.className = 'expand-btn';
    expandBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="2,5 2,2 5,2"/><polyline points="7,2 10,2 10,5"/><polyline points="10,7 10,10 7,10"/><polyline points="5,10 2,10 2,7"/></svg>';
    expandBtn.title = 'Fullscreen';
    expandBtn.addEventListener('click', () => openImageGalleryFullscreen(tag, entries));
    header.append(h3, expandBtn);
    section.appendChild(header);

    // Body
    const body = document.createElement('div'); body.className = 'img-tag-body';
    section.appendChild(body);
    root.appendChild(section);

    _imgBuildAnimate(body, entries, tag);
  });
}

// ── Audio ────────────────────────────────────────────────────
function buildAudio() {
  const c = document.getElementById('audio-list'); c.innerHTML = ''; let n = 0;
  DATA.forEach(exp => (exp.audio_tags || []).forEach(tag => (exp.audio_data[tag] || []).forEach(a => {
    n++; const d = document.createElement('div'); d.className = 'media-card'; d.style.marginBottom = '12px';
    d.dataset.dragId = exp.name + '/' + tag + '/' + a.step;
    d.innerHTML = `<div class="label"><span class="exp" style="color:${expColors[exp.name] || 'var(--muted)'}">${escapeHtml(exp.name)}</span> / <span class="tag">${escapeHtml(tag)}</span> — step ${a.step}</div><audio controls src="${mediaUrl(a.path)}" style="width:100%;padding:0 12px 8px"></audio>`;
    c.appendChild(d);
  })));
  if (!n) c.innerHTML = '<div class="empty-msg">No audio logged.</div>';
  else enableDragSort(c, 'audio');
}

// ── Video ────────────────────────────────────────────────────
const _vidCompareSelection = []; // { exp, tag, step, path, cardEl }

function _vidUpdateCompareToolbar() {
  let bar = document.getElementById('compare-toolbar');
  // If image selection is active, let it own the toolbar
  if (_imgCompareSelection.length >= 2) return;
  if (_vidCompareSelection.length < 2) {
    if (bar) bar.style.display = 'none';
    return;
  }
  if (!bar) {
    bar = document.createElement('div');
    bar.id = 'compare-toolbar';
    bar.className = 'compare-toolbar';
    document.body.appendChild(bar);
  }
  bar.style.display = '';
  bar.innerHTML = '';
  const lbl = document.createElement('span');
  lbl.textContent = _vidCompareSelection.length + ' videos selected';
  const clearBtn = document.createElement('button');
  clearBtn.textContent = 'Clear';
  clearBtn.addEventListener('click', _vidClearSelection);
  const cmpBtn = document.createElement('button');
  cmpBtn.textContent = 'Compare';
  cmpBtn.className = 'cmp-btn-primary';
  cmpBtn.addEventListener('click', _vidOpenCompare);
  bar.append(lbl, clearBtn, cmpBtn);
}

function _vidClearSelection() {
  _vidCompareSelection.forEach(s => {
    if (s.cardEl) s.cardEl.classList.remove('vid-selected');
    const cb = s.cardEl && s.cardEl.querySelector('.vid-select-cb');
    if (cb) cb.checked = false;
    const txt = s.cardEl && s.cardEl.querySelector('.vid-select-text');
    if (txt) txt.textContent = 'Compare';
  });
  _vidCompareSelection.length = 0;
  _vidUpdateCompareToolbar();
}

function _vidToggleSelect(exp, tag, step, path, cardEl, checked) {
  const idx = _vidCompareSelection.findIndex(s => s.cardEl === cardEl);
  if (checked) {
    if (idx === -1 && _vidCompareSelection.length < 6) {
      _vidCompareSelection.push({ exp, tag, step, path, cardEl });
      cardEl.classList.add('vid-selected');
      const txt = cardEl.querySelector('.vid-select-text');
      if (txt) txt.textContent = 'Selected';
    } else if (idx === -1) {
      const cb = cardEl.querySelector('.vid-select-cb');
      if (cb) cb.checked = false;
      const txt = cardEl.querySelector('.vid-select-text');
      if (txt) txt.textContent = 'Compare';
      return;
    }
  } else if (idx !== -1) {
    _vidCompareSelection.splice(idx, 1);
    cardEl.classList.remove('vid-selected');
    const txt = cardEl.querySelector('.vid-select-text');
    if (txt) txt.textContent = 'Compare';
  }
  _vidUpdateCompareToolbar();
}

// Generate a first-frame poster by drawing an actual decoded frame to canvas.
// Avoids the black placeholder some browsers show with metadata-only preload.
function _vidEnsurePoster(videoEl) {
  if (!videoEl) return;
  const onLoaded = () => {
    if (videoEl._posterDone) return;
    const w = videoEl.videoWidth, h = videoEl.videoHeight;
    if (!w || !h) return;
    try {
      const c = document.createElement('canvas');
      c.width = w; c.height = h;
      c.getContext('2d').drawImage(videoEl, 0, 0, w, h);
      videoEl.poster = c.toDataURL('image/jpeg', 0.7);
      videoEl._posterDone = true;
    } catch (e) { /* CORS or codec issue — leave default */ }
  };
  videoEl.addEventListener('loadeddata', onLoaded);
  videoEl.addEventListener('seeked', onLoaded);
  videoEl.addEventListener('loadedmetadata', () => {
    if (videoEl.readyState >= 2) {
      onLoaded();
      return;
    }
    try {
      const target = Math.min(0.001, Math.max(0, (videoEl.duration || 0) - 0.001));
      videoEl.currentTime = isFinite(target) ? target : 0;
    } catch (e) {
      videoEl.load();
    }
  });
  try { videoEl.load(); } catch (e) {}
}

function buildVideo() {
  _vidClearSelection();
  const g = document.getElementById('video-grid'); g.innerHTML = ''; let n = 0;
  DATA.forEach(exp => (exp.video_tags || []).forEach(tag => (exp.video_data[tag] || []).forEach(v => {
    n++;
    const d = document.createElement('div'); d.className = 'media-card';
    d.dataset.dragId = exp.name + '/' + tag + '/' + v.step;
    const url = mediaUrl(v.path);
    d.innerHTML = `<video controls playsinline src="${url}#t=0.001" preload="auto"></video>`
      + `<div class="label video-label"><span><span class="exp" style="color:${expColors[exp.name] || 'var(--muted)'}">${escapeHtml(exp.name)}</span>`
      + ` / <span class="tag">${escapeHtml(tag)}</span> — step ${v.step}</span>`
      + '<label class="vid-select-label"><input type="checkbox" class="vid-select-cb"><span class="vid-select-text">Compare</span></label></div>';
    const cb = d.querySelector('.vid-select-cb');
    cb.addEventListener('click', e => e.stopPropagation());
    cb.addEventListener('change', () => _vidToggleSelect(exp.name, tag, v.step, v.path, d, cb.checked));
    _vidEnsurePoster(d.querySelector('video'));
    g.appendChild(d);
  })));
  if (!n) g.innerHTML = '<div class="empty-msg">No video logged.</div>';
  else enableDragSort(g, 'video');
}

function _vidOpenCompare() {
  const sel = _vidCompareSelection.slice();
  if (sel.length < 2) return;

  const ov = document.createElement('div'); ov.className = 'fs-overlay';
  ov.innerHTML = '<div class="fs-header"><h2>Video Comparison · ' + sel.length + ' clips</h2>'
    + '<div class="fs-header-right"><button class="fs-close">Close · Esc</button></div></div>'
    + '<div class="fs-body compare-body" style="padding:0">'
    + '<div class="vid-cmp-grid"></div>'
    + '<div class="vid-cmp-ctrl"></div>'
    + '</div>';
  document.body.appendChild(ov);

  const grid = ov.querySelector('.vid-cmp-grid');
  const ctrl = ov.querySelector('.vid-cmp-ctrl');

  const videos = sel.map(s => {
    const cell = document.createElement('div'); cell.className = 'vid-cmp-cell';
    const v = document.createElement('video');
    v.src = mediaUrl(s.path);
    v.preload = 'auto';
    v.muted = true;
    v.playsInline = true;
    _vidEnsurePoster(v);
    const lbl = document.createElement('div'); lbl.className = 'cell-label';
    lbl.innerHTML = `<span class="exp" style="color:${expColors[s.exp] || 'var(--muted)'}">${escapeHtml(s.exp)}</span> / <span class="tag">${escapeHtml(s.tag)}</span> — step ${s.step}`;
    cell.append(v, lbl);
    grid.appendChild(cell);
    return v;
  });

  const playBtn = document.createElement('button'); playBtn.innerHTML = '&#9654;'; playBtn.title = 'Play / pause';
  const seek = document.createElement('input'); seek.type = 'range'; seek.min = '0'; seek.max = '1000'; seek.value = '0'; seek.step = '1';
  const timeLbl = document.createElement('span'); timeLbl.textContent = '0.00s / 0.00s';
  ctrl.append(playBtn, seek, timeLbl);

  let maxDur = 0;
  let scrubbing = false;
  let playing = false;

  function fmt(t) { return (isFinite(t) ? t : 0).toFixed(2) + 's'; }
  function refreshDur() {
    maxDur = videos.reduce((m, v) => Math.max(m, isFinite(v.duration) ? v.duration : 0), 0);
    timeLbl.textContent = fmt(videos[0].currentTime || 0) + ' / ' + fmt(maxDur);
  }
  videos.forEach(v => v.addEventListener('loadedmetadata', refreshDur));

  const master = videos[0];
  master.addEventListener('timeupdate', () => {
    if (scrubbing || !maxDur) return;
    const t = master.currentTime;
    seek.value = String(Math.round((t / maxDur) * 1000));
    timeLbl.textContent = fmt(t) + ' / ' + fmt(maxDur);
    videos.slice(1).forEach(v => {
      const target = Math.min(t, v.duration || t);
      if (Math.abs(v.currentTime - target) > 0.18) v.currentTime = target;
    });
  });
  master.addEventListener('ended', () => {
    videos.forEach(v => v.pause());
    playing = false; playBtn.innerHTML = '&#9654;';
  });

  playBtn.addEventListener('click', async () => {
    if (playing) {
      videos.forEach(v => v.pause());
      playing = false; playBtn.innerHTML = '&#9654;';
    } else {
      await Promise.all(videos.map(v => v.play().catch(() => {})));
      playing = true; playBtn.innerHTML = '&#9646;&#9646;';
    }
  });

  seek.addEventListener('input', () => {
    scrubbing = true;
    if (!maxDur) return;
    const t = (parseInt(seek.value) / 1000) * maxDur;
    videos.forEach(v => { v.currentTime = Math.min(t, v.duration || t); });
    timeLbl.textContent = fmt(t) + ' / ' + fmt(maxDur);
  });
  seek.addEventListener('change', () => { scrubbing = false; });

  const close = () => {
    videos.forEach(v => { try { v.pause(); v.src = ''; v.load(); } catch (e) {} });
    ov.remove();
    document.removeEventListener('keydown', onKey);
  };
  const onKey = e => {
    if (e.key === 'Escape') close();
    else if (e.key === ' ') { e.preventDefault(); playBtn.click(); }
  };
  ov.querySelector('.fs-close').addEventListener('click', close);
  document.addEventListener('keydown', onKey);
}

// ── Artifacts ────────────────────────────────────────────────
function buildArtifacts() {
  const tb = document.getElementById('artifacts-body'); tb.innerHTML = ''; let n = 0;
  DATA.forEach(exp => (exp.artifact_tags || []).forEach(tag => (exp.artifacts[tag] || []).forEach(a => {
    const m = a.metadata || {};
    // graphs and pr_curves have their own tabs; never list them here
    if (m.kind === 'graph' || m.kind === 'pr_curve' || m.format === 'dot') return;
    n++; const tr = document.createElement('tr');
    tr.dataset.dragId = exp.name + '/' + tag + '/' + a.step;
    tr.innerHTML = `<td style="color:${expColors[exp.name] || 'inherit'}">${escapeHtml(exp.name)}</td><td>${escapeHtml(tag)}</td><td>${a.step}</td><td>${escapeHtml(m.original_filename || '')}</td><td>${humanSize(m.file_size)}</td><td>${escapeHtml(m.mime_type || '')}</td><td><a href="${mediaUrl(a.path)}" download>Download</a></td>`;
    tb.appendChild(tr);
  })));
  if (!n) { const tr = document.createElement('tr'); tr.innerHTML = '<td colspan="7" class="empty-msg">No artifacts logged.</td>'; tb.appendChild(tr); }
  else enableDragSort(tb, 'artifacts');
}

// ── Graphs ───────────────────────────────────────────────────
let _graphViz = null;
function _graphRenderer() {
  if (typeof Viz === 'undefined') return null;
  if (!_graphViz) _graphViz = new Viz();
  return _graphViz;
}

function _graphFallback(container, _dot, message) {
  // We deliberately do NOT render the raw DOT source — the rendered PNG is
  // the canonical view. When neither a PNG nor an SVG renderer is available
  // we just show a short status message; the DOT itself is still
  // downloadable from the card header link.
  container.innerHTML = '';
  if (message) {
    const msg = document.createElement('div');
    msg.className = 'graph-render-msg';
    msg.textContent = message;
    container.appendChild(msg);
  }
}

function _renderGraphDot(container, dot) {
  const viz = _graphRenderer();
  if (!viz) {
    _graphFallback(container, dot, 'DOT renderer unavailable; showing source.');
    return;
  }
  container.innerHTML = '<div class="empty-msg">Rendering graph...</div>';
  viz.renderSVGElement(dot)
    .then(svg => {
      container.innerHTML = '';
      svg.classList.add('graph-svg');
      container.appendChild(svg);
    })
    .catch(err => {
      _graphViz = null;
      _graphFallback(container, dot, 'DOT render failed: ' + (err && err.message ? err.message : err));
    });
}

function _fetchAndRenderGraph(container, item) {
  fetch(mediaUrl(item.path))
    .then(r => r.ok ? r.text() : Promise.reject(new Error('HTTP ' + r.status)))
    .then(dot => _renderGraphDot(container, dot))
    .catch(err => _graphFallback(container, '', 'Unable to load graph: ' + err.message));
}

function buildGraphs() {
  // Renders the "Models" tab. Each entry now ships a server-rendered PNG
  // alongside the DOT source — prefer the PNG (instant, sharp, accurate
  // shape annotations) and fall back to client-side viz.js DOT rendering
  // only when no PNG is available.
  const c = document.getElementById('models-list') || document.getElementById('graphs-list');
  if (!c) return;
  c.innerHTML = '';
  const byTag = {};
  DATA.forEach(exp => {
    const tags = exp.model_tags || exp.graph_tags || [];
    const items = exp.models || exp.graphs || {};
    tags.forEach(tag => {
      if (!byTag[tag]) byTag[tag] = [];
      (items[tag] || []).forEach(g => byTag[tag].push({ exp: exp.name, tag, ...g }));
    });
  });
  const tags = Object.keys(byTag).sort();
  if (!tags.length) { c.innerHTML = '<div class="empty-msg">No models logged.</div>'; return; }
  tags.forEach(tag => {
    const entries = byTag[tag].sort((a, b) => a.step - b.step);
    const section = document.createElement('div'); section.className = 'graph-section';
    const header = document.createElement('div'); header.className = 'img-tag-header';
    const h3 = document.createElement('h3'); h3.textContent = tag;
    header.appendChild(h3); section.appendChild(header);
    const grid = document.createElement('div'); grid.className = 'graph-grid';
    entries.forEach(item => {
      const card = document.createElement('div'); card.className = 'graph-card';
      const title = document.createElement('div'); title.className = 'graph-card-title';
      const meta = item.metadata || {};
      // Defensive: older runs stored repr(model) here, which for raw DOT
      // input is the entire source ("'digraph foo {...}'"). Drop anything
      // that looks like DOT or a multi-line blob — only keep a short class
      // name like "ResNet" or "Sequential".
      let modelName = '';
      if (meta.model) {
        const raw = String(meta.model).split('(')[0].trim();
        const looksLikeDot = /\bdigraph\b|\bgraph\b\s*\{|\{|\n/.test(raw);
        if (!looksLikeDot && raw.length > 0 && raw.length < 80) modelName = raw;
      }
      title.innerHTML = `<span style="color:${expColors[item.exp] || 'var(--muted)'}">${escapeHtml(item.exp)}</span>` +
        (modelName ? `<span class="model-class">${escapeHtml(modelName)}</span>` : '') +
        `<span>step ${item.step}</span>` +
        `<a href="${mediaUrl(item.path)}" download>DOT</a>`;
      const body = document.createElement('div'); body.className = 'graph-body';
      card.append(title, body); grid.appendChild(card);
      const pngAbs = meta.rendered_png_abs;
      if (pngAbs) {
        const img = document.createElement('img');
        img.className = 'model-png';
        img.alt = `${item.tag} step ${item.step}`;
        img.src = mediaUrl(pngAbs);
        body.appendChild(img);
      } else {
        _fetchAndRenderGraph(body, item);
      }
    });
    section.appendChild(grid); c.appendChild(section);
  });
}

// ── Text ─────────────────────────────────────────────────────
function _txtBuildSlider(container, entries, tag) {
  const steps = [...new Set(entries.map(e => e.step))].sort((a, b) => a - b);
  if (!steps.length) return;

  const byExp = {};
  entries.forEach(e => { if (!byExp[e.exp]) byExp[e.exp] = {}; byExp[e.exp][e.step] = e.value; });
  const expNames = Object.keys(byExp);

  const wrap = document.createElement('div'); wrap.className = 'img-animate';

  const ctrl = document.createElement('div'); ctrl.className = 'img-animate-controls';
  const stepLabel = document.createElement('span'); stepLabel.className = 'step-label';
  const slider = document.createElement('input'); slider.type = 'range';
  slider.min = 0; slider.max = steps.length - 1; slider.value = 0;
  ctrl.append(slider, stepLabel);
  wrap.appendChild(ctrl);

  const viewport = document.createElement('div'); viewport.className = 'txt-slider-viewport';
  const cells = {};
  expNames.forEach(exp => {
    const cell = document.createElement('div'); cell.className = 'txt-slider-cell';
    const lbl = document.createElement('div'); lbl.className = 'cell-label';
    lbl.innerHTML = `<span class="exp" style="color:${expColors[exp] || 'var(--muted)'}">${escapeHtml(exp)}</span>`;
    const pre = document.createElement('pre');
    const copyBtn = document.createElement('button'); copyBtn.className = 'copy-btn'; copyBtn.title = 'Copy'; copyBtn.textContent = '\u2398';
    copyBtn.addEventListener('click', ev => { ev.stopPropagation(); navigator.clipboard.writeText(pre.textContent).then(() => { copyBtn.textContent = '\u2713'; setTimeout(() => { copyBtn.textContent = '\u2398'; }, 1500); }); });
    cell.append(lbl, pre, copyBtn);
    viewport.appendChild(cell);
    cells[exp] = pre;
  });
  wrap.appendChild(viewport);
  container.appendChild(wrap);

  function renderStep(idx) {
    const step = steps[idx];
    stepLabel.textContent = 'Step ' + step + ' / ' + steps[steps.length - 1];
    slider.value = idx;
    expNames.forEach(exp => {
      const val = byExp[exp][step];
      cells[exp].textContent = val !== undefined ? val : '';
      cells[exp].parentElement.style.display = val !== undefined ? '' : 'none';
    });
  }

  slider.addEventListener('input', () => renderStep(parseInt(slider.value)));
  renderStep(0);
}

function buildText() {
  const c = document.getElementById('text-list'); c.innerHTML = '';
  const sysPfx = ['system/', 'gpu/'];
  const byTag = {};
  DATA.forEach(exp => (exp.text_tags || []).filter(tag => !sysPfx.some(p => tag.startsWith(p))).forEach(tag => {
    if (!byTag[tag]) byTag[tag] = [];
    (exp.text_data[tag] || []).forEach(e => byTag[tag].push({ exp: exp.name, step: e.step, value: e.value }));
  }));
  const tags = Object.keys(byTag).sort();
  if (!tags.length) { c.innerHTML = '<div class="empty-msg">No text logged.</div>'; return; }
  tags.forEach(tag => {
    const entries = byTag[tag].sort((a, b) => a.step - b.step);
    const section = document.createElement('div'); section.className = 'img-tag-section';
    const header = document.createElement('div'); header.className = 'img-tag-header';
    const h3 = document.createElement('h3'); h3.textContent = tag;
    header.appendChild(h3); section.appendChild(header);
    const body = document.createElement('div'); body.className = 'img-tag-body';
    section.appendChild(body); c.appendChild(section);
    _txtBuildSlider(body, entries, tag);
  });
}
