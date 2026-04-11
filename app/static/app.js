/* ─── RawAccel Studio — Frontend Logic ─────────────────────────── */

const STYLE_ICONS = {
  linear: '📏', classic: '📈', natural: '🌿',
  power: '⚡', synchronous: '🔄', jump: '🦘'
};

const STYLE_DESCRIPTIONS = {
  linear: 'Constant acceleration — simple and predictable',
  classic: 'Power-curve acceleration — popular for FPS gaming',
  natural: 'Logarithmic feel — smooth and organic response',
  power: 'Exponential power curve — aggressive high-speed scaling',
  synchronous: 'Speed-synchronized output — 1:1 at sync point',
  jump: 'Step function — instant sensitivity jump at threshold'
};

let selectedFile = null;
let curveChart = null;
let lastResult = null;

/* ── Init ────────────────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
  checkModel();
  setupUpload();
  setupButtons();
});

/* ── Model status ────────────────────────────────────────────────── */

async function checkModel() {
  const chip = document.getElementById('nav-status');
  try {
    const r = await fetch('/api/model-status');
    const d = await r.json();
    if (d.loaded) {
      chip.textContent = 'v2 ready';
      chip.className = 'status-chip ready';
    }
  } catch {
    chip.textContent = 'offline';
    chip.className = 'status-chip offline';
  }
}

/* ── Upload handling ─────────────────────────────────────────────── */

function setupUpload() {
  const zone = document.getElementById('drop-zone');
  const input = document.getElementById('file-input');
  const btn = document.getElementById('upload-btn');

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  input.addEventListener('change', () => {
    if (input.files.length) handleFile(input.files[0]);
  });

  btn.addEventListener('click', uploadAndPredict);
}

function handleFile(file) {
  selectedFile = file;
  const zone = document.getElementById('drop-zone');
  zone.querySelector('.uz-text').textContent = file.name;
  zone.querySelector('.uz-sub').textContent = `${(file.size / 1024).toFixed(1)} KB • ready`;
  document.getElementById('upload-btn').disabled = false;
}

/* ── Predict ─────────────────────────────────────────────────────── */

async function uploadAndPredict() {
  if (!selectedFile) return;

  setPipeline(2);
  document.getElementById('upload-card').style.display = 'none';
  document.getElementById('loading').classList.add('show');
  document.getElementById('results').classList.remove('show');

  const dpi = parseInt(document.getElementById('dpi-input').value) || 800;
  const poll = parseInt(document.getElementById('poll-input').value) || 1000;

  const form = new FormData();
  form.append('file', selectedFile);
  form.append('dpi', dpi);
  form.append('poll_rate', poll);

  try {
    const r = await fetch('/api/predict', { method: 'POST', body: form });
    if (!r.ok) throw new Error(`Server error ${r.status}`);
    const data = await r.json();
    lastResult = data;

    // Get export JSON
    const exportR = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        params: data.params_dict,
        dpi: dpi,
        poll_rate: poll,
      }),
    });
    const exportData = await exportR.json();
    lastResult.settings_json = exportData.settings_json;

    document.getElementById('loading').classList.remove('show');
    setPipeline(3);
    renderResults(data);
  } catch (err) {
    document.getElementById('loading').classList.remove('show');
    document.getElementById('upload-card').style.display = 'block';
    setPipeline(1);
    showToast('Error: ' + err.message, 'error');
  }
}

/* ── Render results ──────────────────────────────────────────────── */

function renderResults(data) {
  const res = document.getElementById('results');

  // Session meta
  const meta = document.getElementById('session-meta');
  meta.innerHTML = `
    <div class="sm-item">File: <span>${selectedFile.name}</span></div>
    <div class="sm-item">Size: <span>${(selectedFile.size / 1024).toFixed(1)} KB</span></div>
    <div class="sm-item">DPI: <span>${document.getElementById('dpi-input').value}</span></div>
    <div class="sm-item">Poll: <span>${document.getElementById('poll-input').value} Hz</span></div>
  `;

  // Metrics
  const style = data.style || 'unknown';
  const conf = data.confidence || 0;
  document.getElementById('metric-style').textContent = style.toUpperCase();
  document.getElementById('metric-style-sub').textContent = STYLE_DESCRIPTIONS[style] || '';
  document.getElementById('metric-conf').textContent = (conf * 100).toFixed(1) + '%';
  document.getElementById('metric-points').textContent = data.feature_dim || '100';

  // Style display
  document.getElementById('style-icon').textContent = STYLE_ICONS[style] || '📈';
  document.getElementById('style-name').textContent = style.charAt(0).toUpperCase() + style.slice(1);
  document.getElementById('style-desc').textContent = STYLE_DESCRIPTIONS[style] || '';

  // Probability bars
  const probGrid = document.getElementById('prob-grid');
  probGrid.innerHTML = '';
  if (data.style_probs) {
    const sorted = Object.entries(data.style_probs)
      .sort(([,a], [,b]) => b - a);
    const topStyle = sorted[0][0];

    for (const [name, prob] of sorted) {
      const pct = (prob * 100).toFixed(1);
      const isTop = name === topStyle;
      probGrid.innerHTML += `
        <div class="prob-row">
          <span class="prob-name">${STYLE_ICONS[name] || ''} ${name}</span>
          <div class="prob-track">
            <div class="prob-fill ${isTop ? 'top' : ''}" style="width: ${pct}%"></div>
          </div>
          <span class="prob-pct">${pct}%</span>
        </div>
      `;
    }
  }

  // Chart
  renderChart(data);

  // Parameters
  const paramsGrid = document.getElementById('params-grid');
  paramsGrid.innerHTML = '';
  if (data.params_dict) {
    const paramLabels = {
      style: 'Curve Type', acceleration: 'Acceleration',
      exponent: 'Exponent', cap_output: 'Output Cap',
      offset: 'Input Offset', output_offset: 'Output Offset',
      decay_rate: 'Decay Rate', gamma: 'Gamma',
      motivity: 'Motivity', sync_speed: 'Sync Speed',
      scale: 'Scale', smooth: 'Smooth',
      jump_input: 'Jump Input', jump_output: 'Jump Output',
      gain: 'Gain Mode', sens_multiplier: 'Sensitivity',
      yx_ratio: 'Y/X Ratio'
    };

    for (const [key, val] of Object.entries(data.params_dict)) {
      const label = paramLabels[key] || key;
      const display = typeof val === 'number' ? val.toFixed(4) : String(val);
      paramsGrid.innerHTML += `
        <div class="param">
          <div class="p-name">${label}</div>
          <div class="p-val">${display}</div>
        </div>
      `;
    }
  }

  // JSON export
  if (lastResult.settings_json) {
    const formatted = typeof lastResult.settings_json === 'string'
      ? lastResult.settings_json
      : JSON.stringify(lastResult.settings_json, null, 2);
    document.getElementById('json-pre').textContent = formatted;
  }

  res.classList.add('show');
  res.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── Chart ───────────────────────────────────────────────────────── */

function renderChart(data) {
  const ctx = document.getElementById('curve-chart');
  if (curveChart) curveChart.destroy();

  if (!data.curve_preview) {
    // No preview — use basic display
    curveChart = new Chart(ctx, {
      type: 'line',
      data: { labels: ['0'], datasets: [{ label: 'No preview', data: [0] }] },
      options: { responsive: true }
    });
    return;
  }

  const speeds = data.curve_preview.input_speeds;
  const outputs = data.curve_preview.output_speeds;

  curveChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: speeds.map(s => s.toFixed(0)),
      datasets: [{
        label: `${data.style} curve`,
        data: outputs,
        borderColor: '#63b3ed',
        backgroundColor: 'rgba(99, 179, 237, 0.08)',
        fill: true,
        borderWidth: 2.5,
        pointRadius: 0,
        tension: 0.35,
      }, {
        label: '1:1 (no accel)',
        data: speeds.map(() => 1.0),
        borderColor: 'rgba(113, 128, 150, 0.3)',
        borderDash: [4, 4],
        borderWidth: 1,
        pointRadius: 0,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: '#718096', font: { family: 'Inter', size: 11 } }
        },
        tooltip: {
          backgroundColor: '#1a2232',
          titleColor: '#f7fafc',
          bodyColor: '#cbd5e0',
          borderColor: '#63b3ed',
          borderWidth: 1,
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Input Speed (counts/ms)', color: '#718096', font: { size: 11 } },
          ticks: { color: '#4a5568', maxTicksLimit: 8 },
          grid: { color: 'rgba(99,179,237,0.04)' },
        },
        y: {
          title: { display: true, text: 'Sens Multiplier', color: '#718096', font: { size: 11 } },
          ticks: { color: '#4a5568' },
          grid: { color: 'rgba(99,179,237,0.04)' },
        }
      }
    }
  });
}

/* ── Buttons ─────────────────────────────────────────────────────── */

function setupButtons() {
  document.getElementById('download-btn').addEventListener('click', () => {
    if (!lastResult?.settings_json) return;
    const text = typeof lastResult.settings_json === 'string'
      ? lastResult.settings_json
      : JSON.stringify(lastResult.settings_json, null, 2);
    const blob = new Blob([text], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'settings.json';
    a.click();
    showToast('Settings downloaded!', 'success');
  });

  document.getElementById('copy-btn').addEventListener('click', () => {
    if (!lastResult?.settings_json) return;
    const text = typeof lastResult.settings_json === 'string'
      ? lastResult.settings_json
      : JSON.stringify(lastResult.settings_json, null, 2);
    navigator.clipboard.writeText(text).then(() => showToast('Copied to clipboard!', 'success'));
  });

  document.getElementById('reset-btn').addEventListener('click', resetUI);
}

/* ── Pipeline ────────────────────────────────────────────────────── */

function setPipeline(step) {
  document.querySelectorAll('.pipe-step').forEach(el => {
    const s = parseInt(el.dataset.step);
    el.classList.remove('active', 'done');
    if (s < step) el.classList.add('done');
    else if (s === step) el.classList.add('active');
  });
}

/* ── Reset ───────────────────────────────────────────────────────── */

function resetUI() {
  selectedFile = null;
  lastResult = null;
  document.getElementById('file-input').value = '';
  document.getElementById('upload-btn').disabled = true;
  document.getElementById('results').classList.remove('show');
  document.getElementById('upload-card').style.display = 'block';
  const zone = document.getElementById('drop-zone');
  zone.querySelector('.uz-text').textContent = 'Drop your session JSON here';
  zone.querySelector('.uz-sub').textContent = 'or click to browse • from python -m collector.logger';
  setPipeline(1);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ── Toast ───────────────────────────────────────────────────────── */

function showToast(msg, type = 'success') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `toast ${type} show`;
  setTimeout(() => t.classList.remove('show'), 3000);
}
