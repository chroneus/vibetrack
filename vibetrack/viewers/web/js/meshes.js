// meshes.js — render `add_mesh` outputs as interactive 3D viewers.
//
// Each (experiment, tag, step) entry mounts its own Three.js scene with
// OrbitControls. The mesh JSON payload has already been inlined into
// `exp.meshes[tag][i]` by the server serializer (vertices/colors/faces are
// flat arrays — drop the "shape" wrapper at write time).
//
// Three.js + OrbitControls are vendored under /static/vendor/ so this works
// offline. They're attached to the global THREE namespace.

const _meshInstances = new Set();  // renderers/scenes currently owned by mesh cards
let _meshLazyObserver = null;

function _meshDisposeMaterial(material) {
  if (!material) return;
  if (Array.isArray(material)) {
    material.forEach(_meshDisposeMaterial);
    return;
  }
  try { material.dispose(); } catch (_) {}
}

function _meshLoseContext(context) {
  try {
    const lose = context && context.getExtension && context.getExtension('WEBGL_lose_context');
    if (lose) lose.loseContext();
  } catch (_) {}
}

function _meshDisposeInstance(inst) {
  if (!inst || inst.disposed) return;
  inst.disposed = true;

  if (inst.raf) {
    try { cancelAnimationFrame(inst.raf); } catch (_) {}
    inst.raf = 0;
  }
  if (inst.resizeObserver) {
    try { inst.resizeObserver.disconnect(); } catch (_) {}
  }
  if (inst.onResize) {
    try { window.removeEventListener('resize', inst.onResize); } catch (_) {}
  }
  if (inst.controls) {
    try { inst.controls.dispose(); } catch (_) {}
  }
  if (inst.object) {
    try {
      if (inst.object.geometry) inst.object.geometry.dispose();
      _meshDisposeMaterial(inst.object.material);
    } catch (_) {}
  }
  if (inst.renderer) {
    const canvas = inst.renderer.domElement;
    // Chrome holds WebGL contexts longer than Firefox unless they are
    // explicitly lost, which can exhaust the per-tab context budget.
    try { inst.renderer.forceContextLoss(); } catch (_) {}
    try { inst.renderer.dispose(); } catch (_) {}
    try { if (canvas && canvas.parentNode) canvas.parentNode.removeChild(canvas); } catch (_) {}
  } else if (inst.context) {
    _meshLoseContext(inst.context);
  }

  if (inst.container && inst.container.__meshInstance === inst) {
    delete inst.container.__meshInstance;
  }
  _meshInstances.delete(inst);
}

function _meshDisposeAll() {
  Array.from(_meshInstances).forEach(_meshDisposeInstance);
  if (_meshLazyObserver) { try { _meshLazyObserver.disconnect(); } catch (_) {} _meshLazyObserver = null; }
}

function _flatXYZ(values) {
  // Mesh JSON ships values nested as [batch, N, 3]; flatten any depth into
  // a flat Float32-friendly array of XYZ triples.
  const out = [];
  function walk(v) {
    if (Array.isArray(v) && v.length && typeof v[0] === 'number') { out.push(...v); return; }
    if (Array.isArray(v)) { v.forEach(walk); return; }
  }
  walk(values);
  return out;
}

function _flatIndex(values) {
  // Faces ship as [batch, M, 3] integer triples; flatten to a Uint32 list.
  const out = [];
  function walk(v) {
    if (Array.isArray(v) && v.length && typeof v[0] === 'number') { out.push(...v); return; }
    if (Array.isArray(v)) { v.forEach(walk); return; }
  }
  walk(values);
  return out;
}

function _normColors(values) {
  // TB-style colors are 0–255 ints. Three.js wants 0–1 floats.
  const flat = _flatXYZ(values);
  const out = new Float32Array(flat.length);
  for (let i = 0; i < flat.length; i++) {
    const v = flat[i];
    out[i] = v > 1 ? v / 255 : v;
  }
  return out;
}

function _meshNotice(container, msg) {
  container.innerHTML = '';
  const d = document.createElement('div');
  d.className = 'mesh-render-msg';
  d.textContent = msg;
  container.appendChild(d);
}

function _meshIsChrome() {
  const ua = navigator.userAgent || '';
  const brands = (navigator.userAgentData && navigator.userAgentData.brands) || [];
  const brandChrome = brands.some(b => /Google Chrome|Chromium/.test(b.brand));
  return brandChrome || (/Chrome\//.test(ua) && !/Edg\//.test(ua) && !/OPR\//.test(ua));
}

function _meshWebGLErrorMessage(err) {
  const detail = err && err.message ? err.message : err;
  if (_meshIsChrome()) {
    return 'WebGL is disabled or unavailable in Chrome. Enable "Use graphics acceleration when available" at chrome://settings/system, restart Chrome, then reload this page. (' + detail + ')';
  }
  return 'WebGL is unavailable in this browser/tab — cannot render mesh. (' + detail + ')';
}

function _meshCreateRenderer() {
  const baseAttrs = {
    alpha: false,
    depth: true,
    stencil: false,
    failIfMajorPerformanceCaveat: false,
    preserveDrawingBuffer: false,
    powerPreference: 'default',
  };
  const attempts = [
    { antialias: true },
    { antialias: false },
  ];
  let lastErr = null;

  for (const attempt of attempts) {
    const attrs = { ...baseAttrs, ...attempt };
    const canvas = document.createElement('canvas');
    let context = null;
    try {
      context = canvas.getContext('webgl2', attrs) ||
        canvas.getContext('webgl', attrs) ||
        canvas.getContext('experimental-webgl', attrs);
    } catch (err) {
      lastErr = err;
    }
    if (!context) continue;

    try {
      return {
        renderer: new THREE.WebGLRenderer({ canvas, context, ...attrs }),
        context,
      };
    } catch (err) {
      lastErr = err;
      _meshLoseContext(context);
    }
  }

  throw lastErr || new Error('Browser did not provide a WebGL context');
}

function _renderMesh(container, entry) {
  if (container.__meshInstance) return true;
  if (!window.THREE) {
    _meshNotice(container, 'Three.js is not loaded — meshes cannot be rendered.');
    return false;
  }
  const verts = entry.vertices;
  if (!verts) {
    _meshNotice(container, 'Mesh has no vertices.');
    return false;
  }
  const positions = new Float32Array(_flatXYZ(verts));
  if (!positions.length) {
    _meshNotice(container, 'Mesh vertex array is empty.');
    return false;
  }

  const width = container.clientWidth || 480;
  const height = 360;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111418);

  const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 5000);

  // WebGL context creation can fail when the canvas lives in a hidden
  // subtree (display:none on the tab) or when the browser has hit its
  // per-page context limit. Catch and show a friendly message instead of
  // letting the whole tab crash.
  let rendererInfo;
  let renderer;
  try {
    rendererInfo = _meshCreateRenderer();
    renderer = rendererInfo.renderer;
  } catch (err) {
    _meshNotice(container, _meshWebGLErrorMessage(err));
    return false;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  container.innerHTML = '';
  container.appendChild(renderer.domElement);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  let hasColors = false;
  if (entry.colors) {
    const colors = _normColors(entry.colors);
    if (colors.length === positions.length) {
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      hasColors = true;
    }
  }

  let drawAsMesh = false;
  if (entry.faces) {
    const idx = _flatIndex(entry.faces);
    if (idx.length) {
      const indexArr = positions.length / 3 > 65535
        ? new Uint32Array(idx) : new Uint16Array(idx);
      geometry.setIndex(new THREE.BufferAttribute(indexArr, 1));
      drawAsMesh = true;
    }
  }
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  geometry.computeBoundingBox();

  let object;
  if (drawAsMesh) {
    const mat = new THREE.MeshLambertMaterial({
      vertexColors: hasColors,
      color: hasColors ? 0xffffff : 0xc8d0d8,
      side: THREE.DoubleSide,
      flatShading: true,
    });
    object = new THREE.Mesh(geometry, mat);
  } else {
    // Point cloud — no faces. Render as colored points so it's still
    // visible in 3D rather than a single dot.
    const mat = new THREE.PointsMaterial({
      vertexColors: hasColors,
      color: hasColors ? 0xffffff : 0x9fc5e8,
      size: 0.02,
      sizeAttenuation: true,
    });
    object = new THREE.Points(geometry, mat);
  }
  scene.add(object);

  // Two lights so plain (no-color) meshes are still legible from any angle.
  scene.add(new THREE.HemisphereLight(0xffffff, 0x222233, 0.7));
  const dir = new THREE.DirectionalLight(0xffffff, 0.7);
  dir.position.set(1, 2, 3);
  scene.add(dir);

  // Frame the camera. For "flat" meshes (e.g. height fields where Z is much
  // smaller than X/Y), the default oblique view shows the surface nearly
  // edge-on. Detect the smallest-extent axis and tilt the camera up so the
  // flat face is angled toward the viewer.
  const sphere = geometry.boundingSphere;
  const box = geometry.boundingBox;
  const cx = sphere ? sphere.center.x : 0;
  const cy = sphere ? sphere.center.y : 0;
  const cz = sphere ? sphere.center.z : 0;
  const radius = (sphere && sphere.radius > 0) ? sphere.radius : 1;
  const sx = box ? Math.max(box.max.x - box.min.x, 1e-6) : 1;
  const sy = box ? Math.max(box.max.y - box.min.y, 1e-6) : 1;
  const sz = box ? Math.max(box.max.z - box.min.z, 1e-6) : 1;
  const fovRad = camera.fov * Math.PI / 180;
  // Distance: fit bounding sphere in vertical FOV, with a small padding so
  // the surface doesn't kiss the edge of the viewport.
  const dist = (radius / Math.tan(fovRad / 2)) * 1.15;
  // Pick a view direction that is NOT aligned with the flattest axis.
  // We keep Y as Three.js up (default), so for a typical "ground" surface
  // (large X & Z, small Y) we rise above; for a height-field in the Z
  // direction (large X & Y, small Z) we mostly look along +Z; for other
  // cases an oblique default gives a 3D feel.
  let vx, vy, vz;
  const FLAT = 0.4;  // axis < 40% of next-largest is "flat"
  if (sz <= FLAT * Math.min(sx, sy)) {
    // Z-flat surface (vertices in X-Y plane). Look mostly down the Z axis
    // with a little tilt for parallax.
    vx = 0.35; vy = 0.45; vz = 1.0;
  } else if (sy <= FLAT * Math.min(sx, sz)) {
    // Y-flat surface (the typical "ground"). Rise above it.
    vx = 0.6; vy = 1.0; vz = 0.6;
  } else if (sx <= FLAT * Math.min(sy, sz)) {
    // X-flat surface. Look from the side.
    vx = 1.0; vy = 0.45; vz = 0.35;
  } else {
    // Default oblique 3/4 view.
    vx = 1.0; vy = 0.7; vz = 1.0;
  }
  const vlen = Math.hypot(vx, vy, vz);
  camera.position.set(
    cx + (vx / vlen) * dist,
    cy + (vy / vlen) * dist,
    cz + (vz / vlen) * dist,
  );
  camera.lookAt(cx, cy, cz);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(cx, cy, cz);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.update();

  const inst = {
    container,
    context: rendererInfo.context,
    controls,
    entry,
    object,
    renderer,
    resizeObserver: null,
    onResize: null,
    raf: 0,
    disposed: false,
  };
  _meshInstances.add(inst);

  function tick() {
    if (inst.disposed) return;
    controls.update();
    renderer.render(scene, camera);
    inst.raf = requestAnimationFrame(tick);
  }
  inst.raf = requestAnimationFrame(tick);

  // Resize handling — re-fit canvas when the card width changes.
  const resize = () => {
    if (inst.disposed) return;
    const w = container.clientWidth || width;
    renderer.setSize(w, height);
    camera.aspect = w / height;
    camera.updateProjectionMatrix();
  };
  if ('ResizeObserver' in window) {
    const ro = new ResizeObserver(resize);
    ro.observe(container);
    inst.resizeObserver = ro;
  } else {
    inst.onResize = resize;
    window.addEventListener('resize', resize);
  }
  container.__meshInstance = inst;
  return true;
}

function buildMeshes() {
  const root = document.getElementById('meshes-list');
  if (!root) return;
  _meshDisposeAll();
  root.innerHTML = '';

  const byTag = {};
  DATA.forEach(exp => {
    const tags = exp.mesh_tags || [];
    const items = exp.meshes || {};
    tags.forEach(tag => {
      if (!byTag[tag]) byTag[tag] = [];
      (items[tag] || []).forEach(m => byTag[tag].push({ exp: exp.name, tag, ...m }));
    });
  });
  const tags = Object.keys(byTag).sort();
  if (!tags.length) {
    root.innerHTML = '<div class="empty-msg">No meshes logged. Use <code>writer.add_mesh(tag, vertices, faces=...)</code> to add one.</div>';
    return;
  }

  tags.forEach(tag => {
    const entries = byTag[tag].sort((a, b) => a.step - b.step);
    const section = document.createElement('div'); section.className = 'graph-section';
    const header = document.createElement('div'); header.className = 'img-tag-header';
    const h3 = document.createElement('h3'); h3.textContent = tag;
    header.appendChild(h3); section.appendChild(header);

    const grid = document.createElement('div'); grid.className = 'graph-grid';
    entries.forEach(item => {
      const card = document.createElement('div'); card.className = 'graph-card mesh-card';
      const title = document.createElement('div'); title.className = 'graph-card-title';
      const nVerts = item.vertices ? _flatXYZ(item.vertices).length / 3 : 0;
      const nFaces = item.faces ? _flatIndex(item.faces).length / 3 : 0;
      title.innerHTML = `<span style="color:${expColors[item.exp] || 'var(--muted)'}">${escapeHtml(item.exp)}</span>` +
        `<span>step ${item.step}</span>` +
        `<span class="mesh-stats">${nVerts} verts · ${nFaces} faces</span>` +
        `<a href="${mediaUrl(item.path)}" download>JSON</a>`;
      const body = document.createElement('div'); body.className = 'graph-body mesh-body';
      card.append(title, body); grid.appendChild(card);
      // Defer the actual WebGL renderer creation until the body has a real
      // size — that is, once the Meshes tab is visible. Creating a context
      // inside a display:none subtree fails on some Chrome/driver combos
      // ("Error creating WebGL context"), and even when it succeeds it
      // wastes a slot from the browser's per-page WebGL context budget
      // (~16) for a card the user may never look at.
      _scheduleLazyMesh(body, item);
    });
    section.appendChild(grid); root.appendChild(section);
  });
}

// IntersectionObserver keeps WebGL contexts tied to visible mesh bodies.
// Chrome has a tighter context budget than Firefox, so off-screen cards are
// disposed and recreated if the user scrolls back.
function _scheduleLazyMesh(body, item) {
  if (!('IntersectionObserver' in window)) {
    // Old browser fallback — try synchronously and rely on the try/catch
    // inside _renderMesh.
    _renderMesh(body, item);
    return;
  }
  if (!_meshLazyObserver) {
    _meshLazyObserver = new IntersectionObserver(entries => {
      entries.forEach(e => {
        const node = e.target;
        // Some browsers consider display:none subtrees "intersecting" with
        // a 0-sized bounding rect at (0,0). Require a real width before we
        // create a WebGL context — otherwise context creation fails on
        // some Chrome/driver combinations.
        if (!e.isIntersecting || node.clientWidth < 8) {
          const inst = node.__meshInstance;
          if (inst) _meshDisposeInstance(inst);
          return;
        }
        const pending = node.__pendingMesh;
        if (!pending) return;
        _renderMesh(node, pending);
      });
    }, { rootMargin: '100px' });
  }
  body.__pendingMesh = item;
  _meshLazyObserver.observe(body);
}
