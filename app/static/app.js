/* ─── RawAccel Studio — Frontend ──────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
    const fileInput  = document.getElementById('file-input');
    const uploadZone = document.getElementById('upload-zone');
    const predictBtn = document.getElementById('predict-btn');
    const dpiInput   = document.getElementById('dpi-input');
    const loading    = document.getElementById('loading');
    const results    = document.getElementById('results');
    const copyBtn    = document.getElementById('copy-btn');
    const downloadBtn= document.getElementById('download-btn');

    let selectedFile = null;
    let lastResult   = null;
    let chart        = null;

    // ── Check model status ─────────────────────────────────────────────
    async function checkModel() {
        try {
            const res = await fetch('/api/model-status');
            const data = await res.json();
            const el = document.getElementById('model-status');
            if (data.loaded) {
                el.textContent = '● Model Ready';
                el.className = 'nav-status ready';
            } else {
                el.textContent = '○ No Model';
                el.className = 'nav-status offline';
            }
        } catch {
            document.getElementById('model-status').textContent = '○ Offline';
        }
    }
    checkModel();

    // ── File upload ────────────────────────────────────────────────────
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            uploadZone.querySelector('.label').textContent = `✅ ${selectedFile.name}`;
            uploadZone.querySelector('.sublabel').textContent =
                `${(selectedFile.size / 1024).toFixed(1)} KB — ready to analyze`;
            predictBtn.disabled = false;
        }
    });

    // Drag & drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            selectedFile = e.dataTransfer.files[0];
            fileInput.files = e.dataTransfer.files;
            uploadZone.querySelector('.label').textContent = `✅ ${selectedFile.name}`;
            uploadZone.querySelector('.sublabel').textContent =
                `${(selectedFile.size / 1024).toFixed(1)} KB — ready to analyze`;
            predictBtn.disabled = false;
        }
    });

    // ── Predict ────────────────────────────────────────────────────────
    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        loading.classList.add('show');
        results.classList.remove('show');
        predictBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('dpi', dpiInput.value);

        try {
            const res = await fetch('/api/predict', { method: 'POST', body: formData });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Prediction failed');
            }
            lastResult = await res.json();
            renderResults(lastResult);
        } catch (err) {
            showToast(err.message, 'error');
        } finally {
            loading.classList.remove('show');
            predictBtn.disabled = false;
        }
    });

    // ── Render results ─────────────────────────────────────────────────
    function renderResults(data) {
        results.classList.add('show');

        // Style badge
        const badge = document.getElementById('style-badge');
        badge.textContent = `📐 ${data.prediction.style.toUpperCase()}`;

        // Confidence
        const conf = Math.round(data.prediction.confidence * 100);
        document.getElementById('confidence-fill').style.width = `${conf}%`;

        // Style probabilities
        const probBars = document.getElementById('prob-bars');
        probBars.innerHTML = '';
        const probs = Object.entries(data.prediction.style_probs)
            .sort((a, b) => b[1] - a[1]);

        for (const [style, prob] of probs) {
            const pct = Math.round(prob * 100);
            probBars.innerHTML += `
                <div class="prob-row">
                    <span class="prob-label">${style}</span>
                    <div class="prob-track">
                        <div class="prob-fill" style="width: ${pct}%"></div>
                    </div>
                    <span class="prob-val">${pct}%</span>
                </div>`;
        }

        // Chart
        renderChart(data.curve.speeds, data.curve.sensitivities);

        // Parameters
        const grid = document.getElementById('params-grid');
        const params = data.prediction.params;
        const displayParams = [
            ['Style', params.style],
            ['Acceleration', params.acceleration?.toFixed(4)],
            ['Exponent', params.exponent?.toFixed(4)],
            ['Cap Output', params.cap_output?.toFixed(4)],
            ['Offset', params.offset?.toFixed(4)],
            ['Sens Multiplier', params.sens_multiplier?.toFixed(4)],
            ['Decay Rate', params.decay_rate?.toFixed(4)],
            ['Scale', params.scale?.toFixed(4)],
        ];

        grid.innerHTML = displayParams.map(([name, val]) => `
            <div class="param-item">
                <div class="name">${name}</div>
                <div class="value">${val}</div>
            </div>
        `).join('');

        // JSON
        const jsonStr = JSON.stringify(data.settings_json, null, 2);
        document.getElementById('json-content').textContent = jsonStr;

        // Scroll to results
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ── Chart ──────────────────────────────────────────────────────────
    function renderChart(speeds, sensitivities) {
        const ctx = document.getElementById('curve-chart').getContext('2d');
        if (chart) chart.destroy();

        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: speeds.map(s => s.toFixed(1)),
                datasets: [{
                    label: 'Sensitivity Multiplier',
                    data: sensitivities,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.08)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: (items) => `Speed: ${items[0].label} counts/ms`,
                            label: (item) => `Sensitivity: ${item.raw.toFixed(3)}x`,
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Input Speed (counts/ms)', color: '#64748b' },
                        ticks: { color: '#64748b', maxTicksLimit: 10 },
                        grid: { color: 'rgba(56, 189, 248, 0.06)' },
                    },
                    y: {
                        title: { display: true, text: 'Sensitivity Multiplier', color: '#64748b' },
                        ticks: { color: '#64748b' },
                        grid: { color: 'rgba(56, 189, 248, 0.06)' },
                        beginAtZero: true,
                    }
                }
            }
        });
    }

    // ── Export ──────────────────────────────────────────────────────────
    copyBtn.addEventListener('click', () => {
        const json = document.getElementById('json-content').textContent;
        navigator.clipboard.writeText(json).then(() => {
            showToast('Copied to clipboard!', 'success');
        });
    });

    downloadBtn.addEventListener('click', () => {
        if (!lastResult) return;
        const json = JSON.stringify(lastResult.settings_json, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'settings.json';
        a.click();
        URL.revokeObjectURL(url);
        showToast('Downloaded settings.json', 'success');
    });

    // ── Toast ──────────────────────────────────────────────────────────
    function showToast(msg, type = 'success') {
        const toast = document.getElementById('toast');
        toast.textContent = msg;
        toast.className = `toast ${type} show`;
        setTimeout(() => toast.classList.remove('show'), 3000);
    }
});
