// pills.js — experiment pills (toggle visibility, recolor, rename, delete).
// Owns globals: activePills, expColors.

const activePills = new Set(DATA.map(d => d.name));
const _colorKey = 'vt_colors_' + (window.VT_PROJECT || '');
const _savedColors = (() => { try { return JSON.parse(localStorage.getItem(_colorKey)) || {}; } catch { return {}; } })();
const expColors = {};
function _saveColors() { try { localStorage.setItem(_colorKey, JSON.stringify(expColors)); } catch {} }
DATA.forEach(d => { expColors[d.name] = _savedColors[d.name] || pickDistinctColor(Object.values(expColors)); });
function getExpNames() { return DATA.map(d => d.name); }
function syncExperimentState() {
  let changed = false;
  getExpNames().forEach(name => {
    if (!(name in expColors)) {
      expColors[name] = _savedColors[name] || pickDistinctColor(Object.values(expColors));
      changed = true;
    }
    activePills.add(name);
  });
  if (changed) _saveColors();
}

function buildPills() {
  if (document.querySelector('.exp-rename-input')) { syncExperimentState(); return; }
  syncExperimentState();
  const c = document.getElementById('exp-pills'); c.innerHTML = '';
  getExpNames().forEach((name, idx) => {
    const expRow = DATA[idx] || {};
    const expProject = expRow.project ? expRow.project : (window.VT_PROJECT || '');
    const color = expColors[name];
    const pill = document.createElement('div');
    pill.className = 'exp-pill' + (activePills.has(name) ? ' active' : ''); pill.dataset.name = name;
    const ci = document.createElement('input'); ci.type = 'color'; ci.value = color;
    ci.style.cssText = 'position:absolute;width:0;height:0;opacity:0;pointer-events:none;';
    function _applyColor(v) { expColors[name] = v; _saveColors(); pill.querySelector('.dot').style.background = v; buildCharts(); buildHistograms(); buildSystem(); }
    ci.addEventListener('input',  e => _applyColor(e.target.value));
    ci.addEventListener('change', e => _applyColor(e.target.value));
    const dot = document.createElement('span'); dot.className = 'dot'; dot.style.background = color; dot.title = 'Change color';
    dot.addEventListener('click', e => { e.stopPropagation(); ci.click(); });
    const nameSpan = document.createElement('span'); nameSpan.className = 'exp-name'; nameSpan.textContent = name;
    const renameBtn = document.createElement('span'); renameBtn.className = 'rename-btn'; renameBtn.title = 'Rename'; renameBtn.textContent = '\u270E';
    renameBtn.addEventListener('click', e => {
      e.stopPropagation();
      startRename(pill, nameSpan, name, idx, expProject);
    });
    const deleteBtn = document.createElement('span'); deleteBtn.className = 'delete-btn'; deleteBtn.title = 'Delete'; deleteBtn.innerHTML = _delIcon;
    deleteBtn.addEventListener('click', e => {
      e.stopPropagation();
      if (!confirm('Delete experiment "' + name + '"? This cannot be undone.')) return;
      fetch('/api/experiment', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, project: expProject }),
      })
      .then(r => r.json())
      .then(res => { if (res.ok) window.location.reload(); else alert('Delete failed: ' + (res.error || 'unknown')); })
      .catch(() => alert('Delete failed'));
    });
    pill.append(ci, dot, nameSpan, renameBtn, deleteBtn);
    pill.addEventListener('click', () => {
      if (activePills.has(name)) { activePills.delete(name); pill.classList.remove('active'); }
      else { activePills.add(name); pill.classList.add('active'); }
      buildCharts();
    });
    c.appendChild(pill);
  });
}

function startRename(pill, nameSpan, oldName, idx, expProject) {
  const input = document.createElement('input'); input.className = 'exp-rename-input';
  input.value = oldName;
  nameSpan.replaceWith(input);
  input.focus(); input.select();
  // Hide rename button while editing
  const renameBtn = pill.querySelector('.rename-btn');
  if (renameBtn) renameBtn.style.display = 'none';

  function commitRename() {
    const newName = input.value.trim();
    if (!newName || newName === oldName) {
      cancelRename();
      return;
    }
    fetch(apiUrl('/api/rename'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ old_name: oldName, new_name: newName, project: expProject }),
    })
    .then(r => r.json())
    .then(res => {
      if (res.ok) {
        window.location.reload();
      } else {
        alert('Rename failed: ' + (res.error || 'unknown error'));
        cancelRename();
      }
    })
    .catch(() => cancelRename());
  }

  function cancelRename() {
    const ns = document.createElement('span'); ns.className = 'exp-name'; ns.textContent = oldName;
    input.replaceWith(ns);
    if (renameBtn) renameBtn.style.display = '';
  }

  input.addEventListener('keydown', e => {
    e.stopPropagation();
    if (e.key === 'Enter') { e.preventDefault(); commitRename(); }
    if (e.key === 'Escape') { e.preventDefault(); cancelRename(); }
  });
  input.addEventListener('click', e => e.stopPropagation());
  input.addEventListener('blur', () => commitRename());
}
