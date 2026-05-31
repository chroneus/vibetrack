// main.js — bootstrap: project switcher, keyboard shortcuts, init.
// Runs last; all build* functions have been registered by the prior modules.

function buildAll() {
  buildCharts();
  buildImages(); buildAudio(); buildVideo(); buildArtifacts();
  buildGraphs(); buildText(); buildHistograms(); buildHParams(); buildSystem();
  if (typeof buildMeshes === 'function') buildMeshes();
  if (typeof buildEmbeddings === 'function') buildEmbeddings();
}

// ── Project switcher ─────────────────────────────────────────
function _renderProjectMenu(projects) {
  window.VT_PROJECTS = Array.isArray(projects) ? projects : [];
  const projName = document.getElementById('project-name');
  const menu = document.getElementById('project-menu');
  menu.innerHTML = '';
  if (!(window.VT_PROJECTS && window.VT_PROJECTS.length > 0)) {
    projName.classList.remove('has-menu');
    return;
  }
  projName.textContent = window.VT_PROJECT;
  projName.classList.add('has-menu');
  window.VT_PROJECTS.forEach(p => {
    const a = document.createElement('a');
    a.href = '/' + p; a.textContent = p;
    if (p === window.VT_PROJECT) a.classList.add('active');
    menu.appendChild(a);
  });
}

function refreshProjectMenu() {
  if (!window.VT_PROJECT) return Promise.resolve();
  return fetch('/api/projects')
    .then(r => r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status)))
    .then(projects => _renderProjectMenu(projects))
    .catch(() => {});
}

function _setupProjectSwitcher() {
  if (window.VT_PROJECT) localStorage.setItem('vt_last_project', window.VT_PROJECT);
  if (!(window.VT_PROJECTS && window.VT_PROJECTS.length > 0)) return;

  _renderProjectMenu(window.VT_PROJECTS);
  const projName = document.getElementById('project-name');
  const menu = document.getElementById('project-menu');
  projName.addEventListener('click', e => { e.stopPropagation(); menu.classList.toggle('open'); });
  document.addEventListener('click', () => menu.classList.remove('open'));
  menu.addEventListener('click', e => e.stopPropagation());

  // Rename icon
  const renameIcon = document.createElement('span');
  renameIcon.className = 'project-rename-btn';
  renameIcon.title = 'Rename project';
  renameIcon.textContent = '✎';
  renameIcon.addEventListener('click', e => {
    e.stopPropagation();
    menu.classList.remove('open');
    startProjectRename();
  });
  projName.parentNode.insertBefore(renameIcon, menu);

  const delBtn = document.getElementById('delete-project-btn');
  document.getElementById('danger-zone-card').style.display = '';
  delBtn.addEventListener('click', () => {
    if (!confirm('Delete project "' + window.VT_PROJECT + '" and all its data? This cannot be undone.')) return;
    fetch('/api/project/' + window.VT_PROJECT, { method: 'DELETE' })
      .then(r => r.json())
      .then(res => { if (res.ok) { localStorage.removeItem('vt_last_project'); window.location = '/'; } else alert('Delete failed: ' + (res.error || 'unknown')); })
      .catch(() => alert('Delete failed'));
  });
}

function startProjectRename() {
  const projName = document.getElementById('project-name');
  const oldName = window.VT_PROJECT;
  const input = document.createElement('input');
  input.className = 'project-rename-input';
  input.value = oldName;
  const renameBtn = document.querySelector('.project-rename-btn');
  if (renameBtn) renameBtn.style.display = 'none';
  projName.replaceWith(input);
  input.focus();
  input.select();

  let committing = false;
  function commitProjectRename() {
    if (committing) return;
    committing = true;
    const newName = input.value.trim();
    if (!newName || newName === oldName) { committing = false; cancelProjectRename(); return; }
    if (newName.includes('/')) {
      const parts = newName.split('/', 2);
      if (!confirm(
        'Move all experiments from "' + oldName + '" into project "' + parts[0] +
        '" with base name "' + parts[1] + '"?\n\nThis cannot be undone.'
      )) { committing = false; cancelProjectRename(); return; }
    }
    fetch('/api/rename-project/' + encodeURIComponent(oldName), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_name: newName }),
    })
    .then(r => r.json())
    .then(res => {
      if (res.ok) {
        const target = res.new_project || res.target_project || newName.split('/')[0];
        localStorage.setItem('vt_last_project', target);
        window.location = '/' + encodeURIComponent(target);
      } else {
        alert('Rename failed: ' + (res.error || 'unknown'));
        cancelProjectRename();
      }
    })
    .catch(() => { alert('Rename failed'); cancelProjectRename(); });
  }

  function cancelProjectRename() {
    const span = document.createElement('span');
    span.id = 'project-name';
    span.textContent = oldName;
    span.classList.add('has-menu');
    input.replaceWith(span);
    if (renameBtn) renameBtn.style.display = '';
    span.addEventListener('click', e => {
      e.stopPropagation();
      document.getElementById('project-menu').classList.toggle('open');
    });
  }

  input.addEventListener('keydown', e => {
    e.stopPropagation();
    if (e.key === 'Enter') { e.preventDefault(); commitProjectRename(); }
    if (e.key === 'Escape') { e.preventDefault(); cancelProjectRename(); }
  });
  input.addEventListener('click', e => e.stopPropagation());
  input.addEventListener('blur', () => commitProjectRename());
}

// ── Hide-chart keyboard shortcut and tray toggle ─────────────
document.addEventListener('keydown', e => {
  if (e.key !== '_') return;
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
  if (_activeTab !== 'scalars') return;
  if (!_hoveredChartTag) return;
  e.preventDefault();
  hideChart(_hoveredChartTag);
});
document.getElementById('hidden-charts-toggle').addEventListener('click', () => {
  document.getElementById('hidden-charts-tray').classList.toggle('collapsed');
});

// ── Init ─────────────────────────────────────────────────────
_setupProjectSwitcher();
_wireSettingsInputs();
buildProjectInfo();
loadSettings().finally(() => { buildTabs(); buildPills(); buildAll(); });
