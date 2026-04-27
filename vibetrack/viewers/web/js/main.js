// main.js — bootstrap: project switcher, keyboard shortcuts, init.
// Runs last; all build* functions have been registered by the prior modules.

function buildAll() {
  buildCharts();
  buildImages(); buildAudio(); buildVideo(); buildArtifacts();
  buildText(); buildHistograms(); buildSystem();
}

// ── Project switcher ─────────────────────────────────────────
function _setupProjectSwitcher() {
  if (window.VT_PROJECT) localStorage.setItem('vt_last_project', window.VT_PROJECT);
  if (!(window.VT_PROJECTS && window.VT_PROJECTS.length > 0)) return;

  const projName = document.getElementById('project-name');
  projName.textContent = window.VT_PROJECT;
  projName.classList.add('has-menu');
  const menu = document.getElementById('project-menu');
  window.VT_PROJECTS.forEach(p => {
    const a = document.createElement('a');
    a.href = '/' + p; a.textContent = p;
    if (p === window.VT_PROJECT) a.classList.add('active');
    menu.appendChild(a);
  });
  projName.addEventListener('click', e => { e.stopPropagation(); menu.classList.toggle('open'); });
  document.addEventListener('click', () => menu.classList.remove('open'));
  menu.addEventListener('click', e => e.stopPropagation());
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
