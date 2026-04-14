/* main.js — Fraud Detection Platform */

'use strict';

// ── Loading overlay ──────────────────────────────────────────────────
function showLoading(message) {
  const overlay = document.getElementById('loadingOverlay');
  const msg     = document.getElementById('loadingMessage');
  if (!overlay) return;
  if (message && msg) msg.textContent = message;
  overlay.classList.remove('d-none');
  overlay.style.display = 'flex';
}

function hideLoading() {
  const overlay = document.getElementById('loadingOverlay');
  if (overlay) overlay.classList.add('d-none');
}

// ── Bootstrap tooltip initialisation ────────────────────────────────
document.addEventListener('DOMContentLoaded', function () {
  // Tooltips
  const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
  tooltips.forEach(el => new bootstrap.Tooltip(el));

  // Auto-dismiss alerts after 6 seconds
  setTimeout(function () {
    document.querySelectorAll('.alert.alert-success, .alert.alert-info').forEach(el => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(el);
      if (bsAlert) bsAlert.close();
    });
  }, 6000);
});

// ── Fraud probability colour helper ─────────────────────────────────
function probToClass(prob) {
  if (prob >= 0.7) return 'text-danger';
  if (prob >= 0.3) return 'text-warning';
  return 'text-success';
}

// ── Confirm destructive actions ──────────────────────────────────────
function confirmAction(message) {
  return window.confirm(message || 'Are you sure?');
}

// ── Render a Plotly chart from JSON string ───────────────────────────
function renderPlotly(divId, jsonData) {
  if (!divId || !jsonData) return;
  const data = typeof jsonData === 'string' ? JSON.parse(jsonData) : jsonData;
  Plotly.newPlot(divId, data.data, data.layout, { responsive: true, displayModeBar: false });
}
