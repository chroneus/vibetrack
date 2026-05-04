// embeddings.js — TensorBoard-projector-style 3-D point cloud viewer for
// `add_embedding` outputs.
//
// The server has already PCA-reduced each embedding to 3-D and (when the
// writer rendered a sprite atlas from `label_img`) shipped the URL of the
// companion PNG. Each card lazily mounts a Three.js scene with OrbitControls
// + Raycaster hover. When sprite_url is non-null, a custom ShaderMaterial
// renders each point as a billboarded thumbnail crop from the atlas.
//
// Mirrors the WebGL lifecycle pattern from meshes.js so we get the same
// safety net: webgl2/webgl/experimental-webgl fallback, antialias retry,
// IntersectionObserver lazy-init, forceContextLoss on dispose.

const _embInstances = new Set();
let _embLazyObserver = null;

function _embDisposeMaterial(material) {
  if (!material) return;
  if (Array.isArray(material)) { material.forEach(_embDisposeMaterial); return; }
  try { material.dispose(); } catch (_) {}
}

function _embLoseContext(context) {
  try {
    const lose = context && context.getExtension && context.getExtension('WEBGL_lose_context');
    if (lose) lose.loseContext();
  } catch (_) {}
}

function _embDisposeInstance(inst) {
  if (!inst || inst.disposed) return;
  inst.disposed = true;

  if (inst.raf) { try { cancelAnimationFrame(inst.raf); } catch (_) {} inst.raf = 0; }
  if (inst.resizeObserver) { try { inst.resizeObserver.disconnect(); } catch (_) {} }
  if (inst.onResize) { try { window.removeEventListener('resize', inst.onResize); } catch (_) {} }
  if (inst.controls) { try { inst.controls.dispose(); } catch (_) {} }
  if (inst.points && inst.points.geometry) {
    try { inst.points.geometry.dispose(); } catch (_) {}
  }
  if (inst.points) _embDisposeMaterial(inst.points.material);
  if (inst.atlasTexture) { try { inst.atlasTexture.dispose(); } catch (_) {} }

  if (inst.canvas) {
    try { inst.canvas.removeEventListener('mousemove', inst.onMouseMove); } catch (_) {}
    try { inst.canvas.removeEventListener('mouseleave', inst.onMouseLeave); } catch (_) {}
  }

  if (inst.renderer) {
    const canvas = inst.renderer.domElement;
    try { inst.renderer.forceContextLoss(); } catch (_) {}
    try { inst.renderer.dispose(); } catch (_) {}
    try { if (canvas && canvas.parentNode) canvas.parentNode.removeChild(canvas); } catch (_) {}
  } else if (inst.context) {
    _embLoseContext(inst.context);
  }
  if (inst.tooltip && inst.tooltip.parentNode) {
    try { inst.tooltip.parentNode.removeChild(inst.tooltip); } catch (_) {}
  }
  if (inst.container && inst.container.__embInstance === inst) {
    delete inst.container.__embInstance;
  }
  _embInstances.delete(inst);
}

function _embDisposeAll() {
  Array.from(_embInstances).forEach(_embDisposeInstance);
  if (_embLazyObserver) { try { _embLazyObserver.disconnect(); } catch (_) {} _embLazyObserver = null; }
}

function _embNotice(container, msg) {
  container.innerHTML = '';
  const d = document.createElement('div');
  d.className = 'embedding-render-msg';
  d.textContent = msg;
  container.appendChild(d);
}

function _embIsChrome() {
  const ua = navigator.userAgent || '';
  const brands = (navigator.userAgentData && navigator.userAgentData.brands) || [];
  const brandChrome = brands.some(b => /Google Chrome|Chromium/.test(b.brand));
  return brandChrome || (/Chrome\//.test(ua) && !/Edg\//.test(ua) && !/OPR\//.test(ua));
}

function _embWebGLErrorMessage(err) {
  const detail = err && err.message ? err.message : err;
  if (_embIsChrome()) {
    return 'WebGL is disabled or unavailable in Chrome. Enable "Use graphics acceleration when available" at chrome://settings/system, restart Chrome, then reload this page. (' + detail + ')';
  }
  return 'WebGL is unavailable in this browser/tab — cannot render embedding cloud. (' + detail + ')';
}

function _embCreateRenderer() {
  const baseAttrs = {
    alpha: false,
    depth: true,
    stencil: false,
    failIfMajorPerformanceCaveat: false,
    preserveDrawingBuffer: false,
    powerPreference: 'default',
  };
  const attempts = [{ antialias: true }, { antialias: false }];
  let lastErr = null;
  for (const attempt of attempts) {
    const attrs = { ...baseAttrs, ...attempt };
    const canvas = document.createElement('canvas');
    let context = null;
    try {
      context = canvas.getContext('webgl2', attrs) ||
        canvas.getContext('webgl', attrs) ||
        canvas.getContext('experimental-webgl', attrs);
    } catch (err) { lastErr = err; }
    if (!context) continue;
    try {
      return { renderer: new THREE.WebGLRenderer({ canvas, context, ...attrs }), context };
    } catch (err) {
      lastErr = err;
      _embLoseContext(context);
    }
  }
  throw lastErr || new Error('Browser did not provide a WebGL context');
}

// ── Color helpers ────────────────────────────────────────────────────────
//
// We support two metadata-driven coloring modes:
//   • categorical (≤ 20 distinct values) → fixed tab10-style palette
//   • numeric                            → viridis ramp on the value range
// Both yield per-vertex floats fed to THREE.PointsMaterial(vertexColors).

const _EMB_PALETTE = [
  '#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f', '#edc949',
  '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab',
  '#86bcb6', '#fabfd2', '#499894', '#fdb863', '#b15928', '#6a3d9a',
  '#cab2d6', '#33a02c', '#fb9a99', '#a6cee3',
];

const _EMB_VIRIDIS = [
  [0.267, 0.005, 0.329], [0.279, 0.175, 0.483], [0.230, 0.322, 0.546],
  [0.173, 0.443, 0.558], [0.128, 0.567, 0.551], [0.157, 0.685, 0.501],
  [0.369, 0.789, 0.382], [0.679, 0.864, 0.190], [0.993, 0.906, 0.144],
];

function _embHexToRGB(hex) {
  const m = /^#([\da-f]{6})$/i.exec(hex);
  if (!m) return [1, 1, 1];
  const v = parseInt(m[1], 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}

function _embViridisAt(t) {
  // t in [0, 1], picks-and-interpolates an RGB triple from the ramp.
  if (!isFinite(t)) return [0.5, 0.5, 0.5];
  const x = Math.max(0, Math.min(1, t)) * (_EMB_VIRIDIS.length - 1);
  const lo = Math.floor(x);
  const hi = Math.min(_EMB_VIRIDIS.length - 1, lo + 1);
  const f = x - lo;
  const a = _EMB_VIRIDIS[lo];
  const b = _EMB_VIRIDIS[hi];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function _embExtractColumn(rows, header, columnKey) {
  // rows can be a list of strings (single-column) or a list of lists.
  // Return a list of stringified cell values (or null on missing rows).
  if (!Array.isArray(rows)) return [];
  return rows.map(r => {
    if (r === null || r === undefined) return null;
    if (Array.isArray(r)) {
      if (header && header.length) {
        const idx = header.indexOf(columnKey);
        if (idx >= 0) return r[idx] === undefined ? null : String(r[idx]);
      }
      // No header — fall back to first column.
      return r[0] === undefined ? null : String(r[0]);
    }
    return String(r);
  });
}

function _embAvailableColumns(rows, header) {
  // Return the list of column labels usable in the dropdown. For a list of
  // strings (single-col), surfaces "label" as a synthetic name.
  if (!Array.isArray(rows) || rows.length === 0) return [];
  if (Array.isArray(header) && header.length) return header.slice();
  if (Array.isArray(rows[0])) {
    return rows[0].map((_, i) => 'col' + i);
  }
  return ['label'];
}

function _embBuildColors(values) {
  // Decide categorical vs numeric and produce a Float32Array (R,G,B) and a
  // legend hint for the tooltip. ≤ 20 unique values → tab10-style palette.
  const n = values.length;
  const out = new Float32Array(n * 3);
  if (!n) return { array: out, mode: 'none', legend: null };

  // Try numeric first (parseFloat on every cell except null/empty).
  const numeric = [];
  let allNumeric = true;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (v === null || v === undefined || v === '') { numeric.push(NaN); continue; }
    const f = Number(v);
    if (!Number.isFinite(f)) { allNumeric = false; break; }
    numeric.push(f);
  }

  if (allNumeric) {
    let lo = Infinity, hi = -Infinity;
    for (const f of numeric) { if (Number.isFinite(f)) { if (f < lo) lo = f; if (f > hi) hi = f; } }
    const span = hi - lo || 1;
    for (let i = 0; i < n; i++) {
      const t = Number.isFinite(numeric[i]) ? (numeric[i] - lo) / span : 0.5;
      const rgb = _embViridisAt(t);
      out[i * 3] = rgb[0]; out[i * 3 + 1] = rgb[1]; out[i * 3 + 2] = rgb[2];
    }
    return { array: out, mode: 'numeric', legend: { lo, hi } };
  }

  // Categorical: assign palette slots in first-seen order.
  const slot = new Map();
  let next = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i] === null ? '∅' : String(values[i]);
    if (!slot.has(v)) slot.set(v, next++);
    if (next > 64) break;  // Bail out of palette if cardinality is huge — fall through to viridis-by-hash below.
  }
  if (next <= _EMB_PALETTE.length) {
    for (let i = 0; i < n; i++) {
      const v = values[i] === null ? '∅' : String(values[i]);
      const rgb = _embHexToRGB(_EMB_PALETTE[slot.get(v) % _EMB_PALETTE.length]);
      out[i * 3] = rgb[0]; out[i * 3 + 1] = rgb[1]; out[i * 3 + 2] = rgb[2];
    }
    return { array: out, mode: 'categorical', legend: { count: next } };
  }
  // Too many categories — hash to viridis so each gets a stable color.
  for (let i = 0; i < n; i++) {
    const v = values[i] === null ? '∅' : String(values[i]);
    let h = 0;
    for (let k = 0; k < v.length; k++) h = (h * 31 + v.charCodeAt(k)) >>> 0;
    const rgb = _embViridisAt((h % 1024) / 1024);
    out[i * 3] = rgb[0]; out[i * 3 + 1] = rgb[1]; out[i * 3 + 2] = rgb[2];
  }
  return { array: out, mode: 'hashed', legend: { count: next } };
}

// ── Sprite-atlas shader ──────────────────────────────────────────────────
//
// Single draw call for thousands of billboarded textured points. The atlas
// is laid out left-to-right, top-to-bottom; per-vertex `spriteIndex` picks
// the cell. WebGL's gl_PointCoord uses Y-down, and Three.js loads PNG
// textures Y-up by default (we set flipY=false at load time so our UV math
// is consistent).

const _EMB_VERT_SHADER = `
attribute float spriteIndex;
uniform float pointSize;
uniform float pixelRatio;
uniform float attenuation;  // 0 = constant screen-pixel size, >0 = perspective attenuation
varying float vSpriteIndex;
void main() {
  vSpriteIndex = spriteIndex;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  // pointSize is in CSS pixels; multiply by pixelRatio to land in device pixels.
  // With attenuation=0 we get a constant on-screen size (best for thumbnails so
  // they don't explode to fill the canvas as you orbit). Set attenuation=1 to
  // get a mild perspective shrink at distance.
  float att = mix(1.0, attenuation / max(-mv.z, 0.0001), step(0.001, attenuation));
  gl_PointSize = pointSize * pixelRatio * att;
  gl_Position = projectionMatrix * mv;
}
`;

const _EMB_FRAG_SHADER = `
uniform sampler2D atlas;
uniform vec2 atlasGrid;
varying float vSpriteIndex;
void main() {
  float col = mod(vSpriteIndex, atlasGrid.x);
  float row = floor(vSpriteIndex / atlasGrid.x);
  vec2 cellOrigin = vec2(col, row) / atlasGrid;
  vec2 cellSize = 1.0 / atlasGrid;
  // gl_PointCoord is Y-down; texture is loaded with flipY=false so atlas
  // (0,0) is top-left. UV maps directly without flipping.
  vec2 uv = cellOrigin + gl_PointCoord * cellSize;
  vec4 c = texture2D(atlas, uv);
  if (c.a < 0.04) discard;
  gl_FragColor = c;
}
`;

function _embFrameCamera(camera, controls, points3d) {
  // Mean-center + sphere-fit. PCA already centered the cloud, but logged
  // embeddings can be re-projected with non-zero mean in pathological cases.
  let cx = 0, cy = 0, cz = 0, n = points3d.length;
  for (const p of points3d) { cx += p[0]; cy += p[1]; cz += p[2]; }
  if (n) { cx /= n; cy /= n; cz /= n; }
  let r2 = 0;
  for (const p of points3d) {
    const dx = p[0] - cx, dy = p[1] - cy, dz = p[2] - cz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > r2) r2 = d2;
  }
  const radius = Math.max(Math.sqrt(r2), 1e-3);
  const fovRad = camera.fov * Math.PI / 180;
  const dist = (radius / Math.tan(fovRad / 2)) * 1.4;
  // Default oblique 3/4 view — point clouds aren't axis-degenerate after PCA.
  const v = [1.0, 0.7, 1.0];
  const vlen = Math.hypot(v[0], v[1], v[2]);
  camera.position.set(cx + (v[0] / vlen) * dist, cy + (v[1] / vlen) * dist, cz + (v[2] / vlen) * dist);
  camera.lookAt(cx, cy, cz);
  controls.target.set(cx, cy, cz);
  controls.update();
  return { center: [cx, cy, cz], radius };
}

function _embRenderToolTip(inst, idx, clientX, clientY) {
  const tip = inst.tooltip;
  if (!tip) return;
  const meta = inst.metadataRows;
  const header = inst.metadataHeader;
  const sprite = inst.sprite;

  // Build the per-row table on demand to keep the DOM small.
  while (tip.firstChild) tip.removeChild(tip.firstChild);

  if (sprite && inst.atlasImage) {
    const tw = sprite.tile_w, th = sprite.tile_h, cols = sprite.cols;
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    const c = document.createElement('canvas');
    c.width = 64; c.height = 64;
    const ctx = c.getContext('2d');
    if (ctx) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(inst.atlasImage, col * tw, row * th, tw, th, 0, 0, 64, 64);
    }
    tip.appendChild(c);
  }

  const rowsLine = document.createElement('div');
  rowsLine.className = 'row';
  const idxLabel = document.createElement('span'); idxLabel.className = 'key'; idxLabel.textContent = '#';
  const idxVal = document.createElement('span'); idxVal.textContent = String(idx);
  rowsLine.append(idxLabel, idxVal);
  tip.appendChild(rowsLine);

  if (Array.isArray(meta) && idx < meta.length) {
    const cell = meta[idx];
    if (Array.isArray(cell)) {
      cell.forEach((v, i) => {
        const r = document.createElement('div'); r.className = 'row';
        const k = document.createElement('span'); k.className = 'key';
        k.textContent = (header && header[i]) ? header[i] : 'col' + i;
        const val = document.createElement('span'); val.textContent = v === null || v === undefined ? '' : String(v);
        r.append(k, val); tip.appendChild(r);
      });
    } else {
      const r = document.createElement('div'); r.className = 'row';
      const k = document.createElement('span'); k.className = 'key'; k.textContent = 'label';
      const val = document.createElement('span'); val.textContent = cell === null || cell === undefined ? '' : String(cell);
      r.append(k, val); tip.appendChild(r);
    }
  }

  // Position relative to the body (offset within its bounding rect).
  const rect = inst.container.getBoundingClientRect();
  const tipW = tip.offsetWidth || 220;
  const tipH = tip.offsetHeight || 80;
  let left = clientX - rect.left + 10;
  let top = clientY - rect.top + 10;
  if (left + tipW > rect.width) left = rect.width - tipW - 4;
  if (top + tipH > rect.height) top = rect.height - tipH - 4;
  tip.style.left = Math.max(0, left) + 'px';
  tip.style.top = Math.max(0, top) + 'px';
  tip.style.display = 'block';
}

function _embRenderEmbedding(container, item, controlsState) {
  if (container.__embInstance) return true;
  if (!window.THREE) {
    _embNotice(container, 'Three.js is not loaded — embeddings cannot be rendered.');
    return false;
  }
  const points3d = item.points3d || [];
  if (!points3d.length) {
    _embNotice(container, 'Embedding has no points after PCA.');
    return false;
  }

  const width = container.clientWidth || 480;
  const height = 400;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c1016);
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.001, 5000);

  let rendererInfo, renderer;
  try {
    rendererInfo = _embCreateRenderer();
    renderer = rendererInfo.renderer;
  } catch (err) {
    _embNotice(container, _embWebGLErrorMessage(err));
    return false;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  container.innerHTML = '';
  container.appendChild(renderer.domElement);

  const tooltip = document.createElement('div');
  tooltip.className = 'embedding-tooltip';
  container.appendChild(tooltip);

  const n = points3d.length;
  const positions = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const p = points3d[i];
    positions[i * 3] = p[0]; positions[i * 3 + 1] = p[1]; positions[i * 3 + 2] = p[2];
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  const wantsThumbnails = !!(item.sprite_url && item.sprite && controlsState.mode === 'thumbnail');
  let material;
  let atlasTexture = null;
  let atlasImage = null;

  if (wantsThumbnails) {
    const sprite = item.sprite;
    const indices = new Float32Array(n);
    if (Array.isArray(item.sampled_indices) && item.sampled_indices.length === n) {
      // Sub-sampled clouds: keep thumbnail indices aligned with the original
      // atlas slot the writer assigned.
      for (let i = 0; i < n; i++) indices[i] = item.sampled_indices[i];
    } else {
      for (let i = 0; i < n; i++) indices[i] = i;
    }
    geometry.setAttribute('spriteIndex', new THREE.BufferAttribute(indices, 1));

    const loader = new THREE.TextureLoader();
    atlasTexture = loader.load(item.sprite_url, (tex) => {
      // Once the image arrives, expose it for the tooltip's <canvas> crop.
      atlasImage = tex.image;
    });
    atlasTexture.flipY = false;
    atlasTexture.minFilter = THREE.LinearFilter;
    atlasTexture.magFilter = THREE.LinearFilter;
    atlasTexture.generateMipmaps = false;

    material = new THREE.ShaderMaterial({
      uniforms: {
        atlas: { value: atlasTexture },
        atlasGrid: { value: new THREE.Vector2(sprite.cols, sprite.rows) },
        pointSize: { value: controlsState.pointSize || 24 },
        pixelRatio: { value: Math.min(window.devicePixelRatio || 1, 2) },
        attenuation: { value: 0.0 },
      },
      vertexShader: _EMB_VERT_SHADER,
      fragmentShader: _EMB_FRAG_SHADER,
      transparent: true,
      depthWrite: true,
    });
  } else {
    // Color mode — categorical/numeric per metadata column.
    const rows = item.metadata_rows;
    const header = item.metadata_header;
    const col = controlsState.column;
    const colVals = col ? _embExtractColumn(rows, header, col) : null;
    // Use sizeAttenuation: false so `size` is in device pixels — matches the
    // shader-based thumbnail mode and makes the slider feel consistent across
    // both modes. Attenuation otherwise ties the on-screen size to the camera
    // distance, which makes points pop to canvas-filling when zoomed in.
    if (colVals) {
      const built = _embBuildColors(colVals);
      geometry.setAttribute('color', new THREE.BufferAttribute(built.array, 3));
      material = new THREE.PointsMaterial({
        vertexColors: true,
        size: controlsState.pointSize || 6,
        sizeAttenuation: false,
      });
    } else {
      material = new THREE.PointsMaterial({
        color: 0x9fc5e8,
        size: controlsState.pointSize || 6,
        sizeAttenuation: false,
      });
    }
  }

  const points = new THREE.Points(geometry, material);
  scene.add(points);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  const frame = _embFrameCamera(camera, controls, points3d);

  // Raycaster for hover-pick. Threshold is in world units; tune relative to
  // the cloud's bounding-sphere radius. A ~5% radius window roughly matches
  // the on-screen footprint of a 24-30 px point at the default camera distance.
  const raycaster = new THREE.Raycaster();
  raycaster.params.Points = { threshold: Math.max(frame.radius * 0.05, 0.02) };
  const mouseNDC = new THREE.Vector2(-2, -2);

  const inst = {
    container,
    canvas: renderer.domElement,
    controls,
    points,
    renderer,
    context: rendererInfo.context,
    atlasTexture,
    get atlasImage() { return atlasImage; },
    raf: 0,
    disposed: false,
    resizeObserver: null,
    onResize: null,
    metadataRows: item.metadata_rows,
    metadataHeader: item.metadata_header,
    sprite: item.sprite,
    tooltip,
    frame,
    onMouseMove: null,
    onMouseLeave: null,
  };
  _embInstances.add(inst);

  const onMouseMove = (ev) => {
    if (inst.disposed) return;
    const rect = renderer.domElement.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    mouseNDC.x = (x / rect.width) * 2 - 1;
    mouseNDC.y = -(y / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouseNDC, camera);
    const hits = raycaster.intersectObject(points);
    if (hits.length) {
      _embRenderToolTip(inst, hits[0].index, ev.clientX, ev.clientY);
    } else {
      tooltip.style.display = 'none';
    }
  };
  const onMouseLeave = () => { tooltip.style.display = 'none'; };
  renderer.domElement.addEventListener('mousemove', onMouseMove);
  renderer.domElement.addEventListener('mouseleave', onMouseLeave);
  inst.onMouseMove = onMouseMove;
  inst.onMouseLeave = onMouseLeave;

  function tick() {
    if (inst.disposed) return;
    controls.update();
    renderer.render(scene, camera);
    inst.raf = requestAnimationFrame(tick);
  }
  inst.raf = requestAnimationFrame(tick);

  const resize = () => {
    if (inst.disposed) return;
    const w = container.clientWidth || width;
    renderer.setSize(w, height);
    camera.aspect = w / height;
    camera.updateProjectionMatrix();
  };
  if ('ResizeObserver' in window) {
    const ro = new ResizeObserver(resize); ro.observe(container);
    inst.resizeObserver = ro;
  } else {
    inst.onResize = resize; window.addEventListener('resize', resize);
  }

  container.__embInstance = inst;
  return true;
}

// IntersectionObserver lazy-init mirrors the meshes pattern: WebGL contexts
// are precious in Chrome (~16/page), so don't allocate them for cards the
// user hasn't scrolled to yet.
function _scheduleLazyEmbedding(body, item, controlsState) {
  if (!('IntersectionObserver' in window)) {
    _embRenderEmbedding(body, item, controlsState);
    return;
  }
  if (!_embLazyObserver) {
    _embLazyObserver = new IntersectionObserver(entries => {
      entries.forEach(e => {
        const node = e.target;
        if (!e.isIntersecting || node.clientWidth < 8) {
          const inst = node.__embInstance;
          if (inst) _embDisposeInstance(inst);
          return;
        }
        const pending = node.__pendingEmbedding;
        if (!pending) return;
        _embRenderEmbedding(node, pending.item, pending.controlsState);
      });
    }, { rootMargin: '100px' });
  }
  body.__pendingEmbedding = { item, controlsState };
  _embLazyObserver.observe(body);
}

function _embRebuildCard(card, item, controlsState) {
  const body = card.querySelector('.embedding-body');
  if (!body) return;
  if (body.__embInstance) _embDisposeInstance(body.__embInstance);
  body.innerHTML = '';
  body.__pendingEmbedding = { item, controlsState };
  // If the body is currently visible, render immediately; otherwise let the
  // observer do it on next intersection.
  if (body.clientWidth >= 8) _embRenderEmbedding(body, item, controlsState);
}

function buildEmbeddings() {
  const root = document.getElementById('embeddings-list');
  if (!root) return;
  _embDisposeAll();
  root.innerHTML = '';

  const byTag = {};
  DATA.forEach(exp => {
    const tags = exp.embedding_tags || [];
    const items = exp.embeddings || {};
    tags.forEach(tag => {
      if (!byTag[tag]) byTag[tag] = [];
      (items[tag] || []).forEach(m => byTag[tag].push({ exp: exp.name, tag, ...m }));
    });
  });
  const tags = Object.keys(byTag).sort();
  if (!tags.length) {
    root.innerHTML = '<div class="empty-msg">No embeddings logged. Use <code>writer.add_embedding(mat, metadata=..., label_img=...)</code> to add one.</div>';
    return;
  }

  tags.forEach(tag => {
    const allEntries = byTag[tag].sort((a, b) => a.step - b.step);
    const section = document.createElement('div'); section.className = 'graph-section';
    const header = document.createElement('div'); header.className = 'img-tag-header';
    const h3 = document.createElement('h3'); h3.textContent = tag;
    header.appendChild(h3); section.appendChild(header);

    const grid = document.createElement('div'); grid.className = 'graph-grid';
    // Group by experiment so each card has its own step slider.
    const byExp = {};
    allEntries.forEach(e => { if (!byExp[e.exp]) byExp[e.exp] = []; byExp[e.exp].push(e); });

    Object.keys(byExp).sort().forEach(expName => {
      const entries = byExp[expName];
      const latest = entries[entries.length - 1];

      const card = document.createElement('div'); card.className = 'graph-card';
      const title = document.createElement('div'); title.className = 'graph-card-title';
      const dim = (latest.shape && latest.shape[1]) ? `${latest.shape[1]}-D` : '';
      const ptCount = `${latest.n_points}${latest.n_total > latest.n_points ? '/' + latest.n_total : ''} pts`;
      title.innerHTML = `<span style="color:${expColors[expName] || 'var(--muted)'}">${escapeHtml(expName)}</span>` +
        `<span class="step-label">step ${latest.step}</span>` +
        `<span class="mesh-stats">${dim} · ${ptCount}</span>` +
        `<a href="${mediaUrl(latest.path)}" download>JSON</a>`;
      const body = document.createElement('div'); body.className = 'graph-body embedding-body';
      const ctrls = document.createElement('div'); ctrls.className = 'embedding-controls';
      card.append(title, body, ctrls); grid.appendChild(card);

      // Initial control state — if the writer logged thumbnails, default to
      // thumbnail mode (the killer feature for image-embedding debugging).
      const hasSprite = !!latest.sprite_url;
      const cols = _embAvailableColumns(latest.metadata_rows, latest.metadata_header);
      const controlsState = {
        mode: hasSprite ? 'thumbnail' : 'color',
        column: cols[0] || null,
        pointSize: hasSprite ? 28 : 6,
        stepIdx: entries.length - 1,
      };

      // Mode toggle (only when sprite atlas is available).
      if (hasSprite) {
        const lbl = document.createElement('label');
        lbl.textContent = 'View';
        const sel = document.createElement('select');
        ['thumbnail', 'color'].forEach(m => {
          const o = document.createElement('option');
          o.value = m; o.textContent = m === 'thumbnail' ? 'Thumbnails' : 'Colored points';
          if (m === controlsState.mode) o.selected = true;
          sel.appendChild(o);
        });
        sel.addEventListener('change', () => {
          controlsState.mode = sel.value;
          _embRebuildCard(card, entries[controlsState.stepIdx], controlsState);
        });
        ctrls.append(lbl, sel);
      }

      // Color-by metadata column dropdown (only matters in 'color' mode).
      if (cols.length) {
        const lbl = document.createElement('label');
        lbl.textContent = 'Color by';
        const sel = document.createElement('select');
        cols.forEach(c => {
          const o = document.createElement('option');
          o.value = c; o.textContent = c;
          if (c === controlsState.column) o.selected = true;
          sel.appendChild(o);
        });
        sel.addEventListener('change', () => {
          controlsState.column = sel.value;
          if (controlsState.mode !== 'thumbnail') {
            _embRebuildCard(card, entries[controlsState.stepIdx], controlsState);
          }
        });
        ctrls.append(lbl, sel);
      }

      // Step slider (only when there are multiple steps for this tag).
      if (entries.length > 1) {
        const slider = document.createElement('input');
        slider.type = 'range'; slider.min = '0'; slider.max = String(entries.length - 1);
        slider.value = String(controlsState.stepIdx);
        slider.title = 'Step';
        const stepSpan = title.querySelector('.step-label');
        slider.addEventListener('input', () => {
          controlsState.stepIdx = parseInt(slider.value, 10);
          const entry = entries[controlsState.stepIdx];
          if (stepSpan) stepSpan.textContent = 'step ' + entry.step;
          _embRebuildCard(card, entry, controlsState);
        });
        ctrls.appendChild(slider);
      }

      // Point-size slider — applies to both modes.
      const sizeRange = document.createElement('input');
      sizeRange.type = 'range'; sizeRange.min = '2'; sizeRange.max = '60';
      sizeRange.value = String(controlsState.pointSize);
      sizeRange.title = 'Point size';
      sizeRange.style.maxWidth = '90px';
      sizeRange.addEventListener('input', () => {
        controlsState.pointSize = parseInt(sizeRange.value, 10);
        const inst = body.__embInstance;
        if (!inst) return;
        const mat = inst.points && inst.points.material;
        if (!mat) return;
        if (mat.uniforms && mat.uniforms.pointSize) {
          mat.uniforms.pointSize.value = controlsState.pointSize;
        } else if ('size' in mat) {
          mat.size = controlsState.pointSize;
        }
      });
      ctrls.appendChild(sizeRange);

      _scheduleLazyEmbedding(body, entries[controlsState.stepIdx], controlsState);
    });

    section.appendChild(grid); root.appendChild(section);
  });
}
