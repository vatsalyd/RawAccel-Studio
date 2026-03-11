/* =========================================================
   RawAccel Studio — Frontend Application
   ========================================================= */

(() => {
    'use strict';

    // ----------------------------------------------------------
    // State
    // ----------------------------------------------------------
    const state = {
        params: { k1: 0.002, a: 1.0, k2: 0.001, b: 1.2, v0: 400, sens_min: 0.2, sens_max: 6.0 },
        optimizedParams: null,
        chart: null,
        aimTask: null,
    };

    // ----------------------------------------------------------
    // DOM helpers
    // ----------------------------------------------------------
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // ----------------------------------------------------------
    // 1. Curve Visualizer
    // ----------------------------------------------------------

    function sensitivityFromSpeed(speed, params) {
        const v = Math.max(speed, 1e-6);
        const low = params.k1 * Math.pow(v, params.a);
        const high = params.k2 * Math.pow(v, params.b);
        let sens = v < params.v0 ? low : high;
        sens = Math.max(params.sens_min, Math.min(sens, params.sens_max));
        return sens;
    }

    function generateCurveData(params, count = 200, maxSpeed = 3000) {
        const speeds = [];
        const sens = [];
        for (let i = 0; i < count; i++) {
            const s = 1 + (maxSpeed - 1) * (i / (count - 1));
            speeds.push(Math.round(s));
            sens.push(sensitivityFromSpeed(s, params));
        }
        return { speeds, sens };
    }

    function initChart() {
        const ctx = $('#curve-chart').getContext('2d');
        const data = generateCurveData(state.params);

        state.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.speeds,
                datasets: [
                    {
                        label: 'Current Curve',
                        data: data.sens,
                        borderColor: '#00f0ff',
                        backgroundColor: 'rgba(0, 240, 255, 0.05)',
                        fill: true,
                        borderWidth: 2.5,
                        pointRadius: 0,
                        tension: 0.3,
                    },
                    {
                        label: 'Optimized',
                        data: [],
                        borderColor: '#a855f7',
                        backgroundColor: 'rgba(168, 85, 247, 0.05)',
                        fill: true,
                        borderWidth: 2.5,
                        pointRadius: 0,
                        tension: 0.3,
                        borderDash: [6, 4],
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        labels: {
                            color: '#8b8fa3',
                            font: { family: 'Outfit', size: 12 },
                            usePointStyle: true,
                            pointStyle: 'line',
                        },
                    },
                    tooltip: {
                        backgroundColor: 'rgba(10, 10, 18, 0.9)',
                        titleColor: '#e8eaed',
                        bodyColor: '#8b8fa3',
                        borderColor: 'rgba(0, 240, 255, 0.2)',
                        borderWidth: 1,
                        titleFont: { family: 'JetBrains Mono', size: 12 },
                        bodyFont: { family: 'JetBrains Mono', size: 11 },
                        callbacks: {
                            title: (items) => `Speed: ${items[0].label} counts/s`,
                            label: (item) => `${item.dataset.label}: ${item.raw.toFixed(4)}`,
                        },
                    },
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Mouse Speed (counts/s)', color: '#5a5e72', font: { family: 'Outfit', size: 12 } },
                        ticks: { color: '#5a5e72', font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 10 },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                    },
                    y: {
                        title: { display: true, text: 'Sensitivity Multiplier', color: '#5a5e72', font: { family: 'Outfit', size: 12 } },
                        ticks: { color: '#5a5e72', font: { family: 'JetBrains Mono', size: 10 } },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                    },
                },
            },
        });
    }

    function updateChart() {
        const data = generateCurveData(state.params);
        state.chart.data.labels = data.speeds;
        state.chart.data.datasets[0].data = data.sens;
        state.chart.update('none');
    }

    function updateChartOptimized(optimizedParams) {
        const data = generateCurveData(optimizedParams);
        state.chart.data.datasets[1].data = data.sens;
        state.chart.update();
    }

    // ----------------------------------------------------------
    // 2. Slider ↔ state binding
    // ----------------------------------------------------------

    function initSliders() {
        $$('input[data-param]').forEach((slider) => {
            const param = slider.dataset.param;

            // Display value
            const display = $(`#val-${param}`);
            const updateDisplay = () => {
                const val = parseFloat(slider.value);
                state.params[param] = val;
                display.textContent = val < 1 ? val.toFixed(4) : val < 10 ? val.toFixed(2) : Math.round(val);
                updateChart();
            };

            slider.addEventListener('input', updateDisplay);
        });
    }

    function setSliders(params) {
        for (const [key, val] of Object.entries(params)) {
            const slider = $(`#slider-${key}`);
            if (slider) {
                slider.value = val;
                const display = $(`#val-${key}`);
                if (display) display.textContent = val < 1 ? val.toFixed(4) : val < 10 ? val.toFixed(2) : Math.round(val);
                state.params[key] = val;
            }
        }
        updateChart();
    }

    // ----------------------------------------------------------
    // 3. Play style slider
    // ----------------------------------------------------------

    function initStyleSlider() {
        const slider = $('#input-style');
        const label = $('#style-label');
        slider.addEventListener('input', () => {
            const v = parseInt(slider.value);
            if (v < 35) label.textContent = 'Flicker';
            else if (v > 65) label.textContent = 'Tracker';
            else label.textContent = 'Balanced';
        });
    }

    function getPlayStyle() {
        const v = parseInt($('#input-style').value);
        if (v < 35) return 'flicker';
        if (v > 65) return 'tracker';
        return 'balanced';
    }

    // ----------------------------------------------------------
    // 4. Optimize button
    // ----------------------------------------------------------

    async function handleOptimize() {
        const btn = $('#btn-optimize');
        const loading = $('#curve-loading');

        btn.disabled = true;
        btn.textContent = '⏳ Optimizing...';
        loading.classList.add('active');

        const profile = {
            dpi: parseInt($('#input-dpi').value) || 800,
            sensitivity: parseFloat($('#input-sens').value) || 0.5,
            rank: $('#input-rank').value,
            agent_role: $('#input-role').value,
            play_style: getPlayStyle(),
            iterations: 200,
        };

        try {
            const res = await fetch('/api/optimize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(profile),
            });

            if (!res.ok) throw new Error(`API error: ${res.status}`);
            const result = await res.json();

            // Update sliders and chart with optimized params
            state.optimizedParams = result.recommended;
            setSliders(result.recommended);
            updateChartOptimized(result.recommended);

            // Show results section
            showResults(result);

            // Smooth scroll to results
            setTimeout(() => {
                $('#results').scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 300);
        } catch (err) {
            console.error('Optimization failed:', err);
            alert('Optimization failed. Make sure the server is running.');
        } finally {
            btn.disabled = false;
            btn.textContent = '⚡ Optimize My Curve';
            loading.classList.remove('active');
        }
    }

    // ----------------------------------------------------------
    // 5. Results display
    // ----------------------------------------------------------

    function showResults(result) {
        const section = $('#results');
        section.style.display = '';

        // Metrics cards
        const metrics = result.metrics;
        const baseline = result.baseline_metrics;
        const metricsHtml = [
            makeMetricCard('Hit Rate', (metrics.hit_rate * 100).toFixed(1) + '%', metrics.hit_rate > 0.7 ? 'good' : metrics.hit_rate > 0.4 ? 'warn' : 'bad',
                `Baseline: ${(baseline.hit_rate * 100).toFixed(1)}%`),
            makeMetricCard('Avg Time', metrics.avg_time.toFixed(3) + 's', metrics.avg_time < 0.35 ? 'good' : metrics.avg_time < 0.5 ? 'warn' : 'bad',
                `Baseline: ${baseline.avg_time.toFixed(3)}s`),
            makeMetricCard('Avg Error', metrics.avg_error.toFixed(1) + '°', metrics.avg_error < 5 ? 'good' : metrics.avg_error < 20 ? 'warn' : 'bad',
                `Baseline: ${baseline.avg_error.toFixed(1)}°`),
            makeMetricCard('Overshoot', (metrics.overshoot_rate * 100).toFixed(0) + '%', metrics.overshoot_rate < 0.3 ? 'good' : metrics.overshoot_rate < 0.6 ? 'warn' : 'bad',
                `Baseline: ${(baseline.overshoot_rate * 100).toFixed(0)}%`),
        ].join('');
        $('#results-metrics').innerHTML = metricsHtml;

        // Params table
        const params = result.recommended;
        const paramNames = {
            k1: 'k1 (low-speed gain)',
            a: 'a (low-speed exponent)',
            k2: 'k2 (high-speed gain)',
            b: 'b (high-speed exponent)',
            v0: 'v0 (breakpoint)',
            sens_min: 'sens_min (floor)',
            sens_max: 'sens_max (cap)',
        };
        let tbody = '';
        for (const [key, val] of Object.entries(params)) {
            tbody += `<tr>
                <td class="param-name">${paramNames[key] || key}</td>
                <td class="param-value">${typeof val === 'number' ? (val < 1 ? val.toFixed(6) : val < 10 ? val.toFixed(4) : val.toFixed(1)) : val}</td>
            </tr>`;
        }
        $('#results-params-body').innerHTML = tbody;

        // RawAccel export JSON
        const exportConfig = {
            Sensitivity: { x: 1.0, y: 1.0 },
            Acceleration: {
                mode: 'custom',
                customCurve: {
                    type: 'power',
                    segments: [
                        { speedRange: [0, params.v0 || 400], gain: params.k1 || 0.002, exponent: params.a || 1.0 },
                        { speedRange: [params.v0 || 400, 99999], gain: params.k2 || 0.001, exponent: params.b || 1.2 },
                    ],
                    cap: { min: params.sens_min || 0.2, max: params.sens_max || 6.0 },
                },
            },
        };
        $('#export-json').textContent = JSON.stringify(exportConfig, null, 2);
    }

    function makeMetricCard(label, value, status, subtitle) {
        return `<div class="metric-card">
            <div class="metric-value ${status}">${value}</div>
            <div class="metric-label">${label}</div>
            <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 4px;">${subtitle}</div>
        </div>`;
    }

    // ----------------------------------------------------------
    // 6. Aim Task Mini-Game
    // ----------------------------------------------------------

    class AimTask {
        constructor(canvas) {
            this.canvas = canvas;
            this.ctx = canvas.getContext('2d');
            this.width = canvas.width;
            this.height = canvas.height;
            this.running = false;
            this.target = null;
            this.hits = 0;
            this.misses = 0;
            this.totalClicks = 0;
            this.hitTimes = [];
            this.lastTargetTime = 0;
            this.mouseEvents = [];
            this.lastMousePos = { x: 0, y: 0 };
            this.trailPoints = [];

            this.canvas.addEventListener('click', (e) => this.onClick(e));
            this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));

            this.drawIdle();
        }

        drawIdle() {
            const { ctx, width, height } = this;
            ctx.fillStyle = '#080810';
            ctx.fillRect(0, 0, width, height);

            // Grid
            ctx.strokeStyle = 'rgba(0, 240, 255, 0.04)';
            ctx.lineWidth = 1;
            for (let x = 0; x < width; x += 50) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
            }
            for (let y = 0; y < height; y += 50) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
            }

            // Crosshair center
            const cx = width / 2, cy = height / 2;
            ctx.strokeStyle = 'rgba(0, 240, 255, 0.2)';
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(cx - 20, cy); ctx.lineTo(cx + 20, cy); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(cx, cy - 20); ctx.lineTo(cx, cy + 20); ctx.stroke();

            // Text
            ctx.fillStyle = '#5a5e72';
            ctx.font = '16px Outfit';
            ctx.textAlign = 'center';
            ctx.fillText('Press Start to begin the aim task', cx, cy + 60);
        }

        start() {
            this.running = true;
            this.hits = 0;
            this.misses = 0;
            this.totalClicks = 0;
            this.hitTimes = [];
            this.mouseEvents = [];
            this.trailPoints = [];
            this.spawnTarget();
            this.loop();
            this.updateStats();
        }

        stop() {
            this.running = false;
            this.target = null;
        }

        spawnTarget() {
            const margin = 40;
            const r = 12 + Math.random() * 14;
            this.target = {
                x: margin + Math.random() * (this.width - 2 * margin),
                y: margin + Math.random() * (this.height - 2 * margin),
                r: r,
                spawnTime: performance.now(),
                pulse: 0,
            };
            this.lastTargetTime = performance.now();
        }

        onClick(e) {
            if (!this.running || !this.target) return;

            const rect = this.canvas.getBoundingClientRect();
            const scaleX = this.width / rect.width;
            const scaleY = this.height / rect.height;
            const mx = (e.clientX - rect.left) * scaleX;
            const my = (e.clientY - rect.top) * scaleY;
            const dx = mx - this.target.x;
            const dy = my - this.target.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            this.totalClicks++;

            if (dist <= this.target.r + 4) {
                // Hit
                this.hits++;
                const elapsed = (performance.now() - this.target.spawnTime) / 1000;
                this.hitTimes.push(elapsed);
                this.spawnTarget();
            } else {
                this.misses++;
            }

            this.mouseEvents.push({
                type: 'click',
                x: mx, y: my,
                time: performance.now(),
                hit: dist <= this.target.r + 4,
                targetX: this.target.x,
                targetY: this.target.y,
            });

            this.updateStats();
        }

        onMouseMove(e) {
            if (!this.running) return;
            const rect = this.canvas.getBoundingClientRect();
            const scaleX = this.width / rect.width;
            const scaleY = this.height / rect.height;
            const mx = (e.clientX - rect.left) * scaleX;
            const my = (e.clientY - rect.top) * scaleY;

            const dx = mx - this.lastMousePos.x;
            const dy = my - this.lastMousePos.y;
            this.lastMousePos = { x: mx, y: my };

            this.trailPoints.push({ x: mx, y: my, t: performance.now() });
            if (this.trailPoints.length > 60) this.trailPoints.shift();

            this.mouseEvents.push({
                type: 'move',
                x: mx, y: my,
                dx, dy,
                time: performance.now(),
            });
        }

        updateStats() {
            $('#aim-hits').textContent = this.hits;
            $('#aim-misses').textContent = this.misses;
            const acc = this.totalClicks > 0 ? Math.round((this.hits / this.totalClicks) * 100) : 0;
            $('#aim-accuracy').textContent = acc + '%';
            if (this.hitTimes.length > 0) {
                const avg = this.hitTimes.reduce((a, b) => a + b, 0) / this.hitTimes.length;
                $('#aim-avg-time').textContent = avg.toFixed(3) + 's';
            }
        }

        loop() {
            if (!this.running) { this.drawIdle(); return; }

            const { ctx, width, height, target } = this;

            // Background
            ctx.fillStyle = '#080810';
            ctx.fillRect(0, 0, width, height);

            // Grid
            ctx.strokeStyle = 'rgba(0, 240, 255, 0.03)';
            ctx.lineWidth = 1;
            for (let x = 0; x < width; x += 50) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
            }
            for (let y = 0; y < height; y += 50) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
            }

            // Mouse trail
            if (this.trailPoints.length > 2) {
                const now = performance.now();
                ctx.beginPath();
                ctx.moveTo(this.trailPoints[0].x, this.trailPoints[0].y);
                for (let i = 1; i < this.trailPoints.length; i++) {
                    const age = (now - this.trailPoints[i].t) / 1000;
                    ctx.lineTo(this.trailPoints[i].x, this.trailPoints[i].y);
                }
                ctx.strokeStyle = 'rgba(0, 240, 255, 0.15)';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            // Target
            if (target) {
                target.pulse = (target.pulse + 0.05) % (Math.PI * 2);
                const pulseScale = 1 + 0.08 * Math.sin(target.pulse);
                const r = target.r * pulseScale;

                // Outer glow
                const gradient = ctx.createRadialGradient(target.x, target.y, r * 0.3, target.x, target.y, r * 2.5);
                gradient.addColorStop(0, 'rgba(255, 56, 96, 0.3)');
                gradient.addColorStop(1, 'rgba(255, 56, 96, 0)');
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(target.x, target.y, r * 2.5, 0, Math.PI * 2);
                ctx.fill();

                // Target circle
                ctx.fillStyle = '#ff3860';
                ctx.beginPath();
                ctx.arc(target.x, target.y, r, 0, Math.PI * 2);
                ctx.fill();

                // Inner ring
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(target.x, target.y, r * 0.5, 0, Math.PI * 2);
                ctx.stroke();

                // Center dot
                ctx.fillStyle = '#fff';
                ctx.beginPath();
                ctx.arc(target.x, target.y, 2, 0, Math.PI * 2);
                ctx.fill();
            }

            // Crosshair at mouse position
            const { x: mx, y: my } = this.lastMousePos;
            ctx.strokeStyle = 'rgba(0, 240, 255, 0.5)';
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(mx - 12, my); ctx.lineTo(mx - 4, my); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(mx + 4, my); ctx.lineTo(mx + 12, my); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(mx, my - 12); ctx.lineTo(mx, my - 4); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(mx, my + 4); ctx.lineTo(mx, my + 12); ctx.stroke();

            requestAnimationFrame(() => this.loop());
        }

        getSessionData() {
            return {
                events: this.mouseEvents,
                stats: {
                    hits: this.hits,
                    misses: this.misses,
                    accuracy: this.totalClicks > 0 ? this.hits / this.totalClicks : 0,
                    avg_hit_time: this.hitTimes.length > 0 ? this.hitTimes.reduce((a, b) => a + b, 0) / this.hitTimes.length : null,
                },
            };
        }
    }

    // ----------------------------------------------------------
    // 7. Copy to clipboard
    // ----------------------------------------------------------

    function initCopyButton() {
        $('#btn-copy-export').addEventListener('click', () => {
            const text = $('#export-json').textContent;
            navigator.clipboard.writeText(text).then(() => {
                const btn = $('#btn-copy-export');
                btn.textContent = '✅ Copied!';
                setTimeout(() => { btn.textContent = '📋 Copy'; }, 2000);
            });
        });
    }

    // ----------------------------------------------------------
    // 8. Save aim session
    // ----------------------------------------------------------

    async function saveAimSession() {
        if (!state.aimTask) return;
        const sessionData = state.aimTask.getSessionData();

        const profile = {
            dpi: parseInt($('#input-dpi').value) || 800,
            sensitivity: parseFloat($('#input-sens').value) || 0.5,
            rank: $('#input-rank').value,
            play_style: getPlayStyle(),
        };

        try {
            const res = await fetch('/api/record-session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    events: sessionData.events.slice(-5000),  // last 5k events to limit size
                    profile,
                    metadata: sessionData.stats,
                }),
            });
            if (!res.ok) throw new Error(`Save failed: ${res.status}`);
            const data = await res.json();
            alert(`Session saved! (${data.num_events} events → ${data.filename})`);
        } catch (err) {
            console.error('Save failed:', err);
            alert('Save failed. Is the server running?');
        }
    }

    // ----------------------------------------------------------
    // 9. Init
    // ----------------------------------------------------------

    function init() {
        initChart();
        initSliders();
        initStyleSlider();
        initCopyButton();

        // Optimize button
        $('#btn-optimize').addEventListener('click', handleOptimize);

        // Reset curve
        $('#btn-reset-curve').addEventListener('click', () => {
            setSliders({ k1: 0.002, a: 1.0, k2: 0.001, b: 1.2, v0: 400 });
            state.chart.data.datasets[1].data = [];
            state.chart.update();
        });

        // Aim task
        const aimCanvas = $('#aim-canvas');
        state.aimTask = new AimTask(aimCanvas);

        $('#btn-start-aim').addEventListener('click', () => {
            const btn = $('#btn-start-aim');
            if (state.aimTask.running) {
                state.aimTask.stop();
                btn.textContent = '▶ Start Task';
                $('#btn-save-aim').disabled = false;
            } else {
                state.aimTask.start();
                btn.textContent = '⏹ Stop Task';
                $('#btn-save-aim').disabled = true;
            }
        });

        $('#btn-save-aim').addEventListener('click', saveAimSession);

        // Smooth scroll for nav links
        $$('.nav-links a').forEach((link) => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(link.getAttribute('href'));
                if (target) target.scrollIntoView({ behavior: 'smooth' });
            });
        });
    }

    // Wait for DOM
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
