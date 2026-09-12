const API_BASE_URL = window.API_BASE_URL || "http://localhost:8000";
const state = { payload: null, charts: {} };

const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (character) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[character]));
const percent = (value) => `${(Number(value) * 100).toFixed(1)}%`;

function showAlert(message, kind = "error") {
  const alert = $("#alert");
  alert.className = `alert ${kind}`;
  alert.textContent = message;
  alert.hidden = false;
}

function hideAlert() {
  $("#alert").hidden = true;
}

function riskClass(risk) {
  return risk.toLowerCase().replace(" ", "-");
}

function renderKpis(summary) {
  const cards = [
    ["Total customers", summary.total_customers.toLocaleString(), "Portfolio size", "teal"],
    ["Predicted churn", summary.predicted_churn.toLocaleString(), "Customers to retain", "coral"],
    ["Predicted churn rate", percent(summary.predicted_churn_rate), "Portfolio exposure", "amber"],
    ["High-risk customers", summary.high_risk_customers.toLocaleString(), "Priority outreach", "red"],
    ["Average probability", percent(summary.average_churn_probability), "Model confidence signal", "teal"],
  ];
  $("#kpi-grid").innerHTML = cards.map(([label, value, note, color]) => `<article class="kpi-card ${color}"><span class="kpi-label">${label}</span><strong>${value}</strong><small>${note}</small></article>`).join("");
}

function renderInsights(payload) {
  const { results } = payload;
  const riskCounts = results.reduce((counts, row) => { counts[row.risk_level] = (counts[row.risk_level] || 0) + 1; return counts; }, {});
  const largestRisk = Object.entries(riskCounts).sort((a, b) => b[1] - a[1])[0];
  const insights = [`${largestRisk[0]} is the largest segment, with ${largestRisk[1].toLocaleString()} customers in the uploaded portfolio.`];
  const contractChart = payload.charts.churn_by_contract;
  if (contractChart.labels.length) insights.push(`${contractChart.labels[0]} customers have the highest predicted churn rate across contract groups.`);
  insights.push("Use the customer brief below to turn model signals into specific retention actions.");
  $("#business-insights").innerHTML = insights.map((item) => `<div class="business-insight"><span>↗</span><p>${escapeHtml(item)}</p></div>`).join("");
}

function chartOptions() {
  return { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: "#66736e", font: { family: "DM Sans" } } } }, scales: { x: { ticks: { color: "#66736e", font: { family: "DM Sans" } }, grid: { color: "#e4e5dc" } }, y: { ticks: { color: "#66736e", font: { family: "DM Sans" } }, grid: { color: "#e4e5dc" } } } };
}

function renderCharts(charts) {
  Object.values(state.charts).forEach((chart) => chart.destroy());
  state.charts.churn = new Chart($("#churn-chart"), { type: "doughnut", data: { labels: charts.churn_distribution.labels, datasets: [{ data: charts.churn_distribution.values, backgroundColor: ["#c9524e", "#236c63"], borderWidth: 0 }] }, options: { ...chartOptions(), cutout: "64%" } });
  state.charts.risk = new Chart($("#risk-chart"), { type: "bar", data: { labels: charts.risk_distribution.labels, datasets: [{ data: charts.risk_distribution.values, backgroundColor: ["#236c63", "#bd751d", "#c9524e"], borderRadius: 6, maxBarThickness: 46 }] }, options: { ...chartOptions(), plugins: { legend: { display: false } } } });
  state.charts.contract = new Chart($("#contract-chart"), { type: "bar", data: { labels: charts.churn_by_contract.labels, datasets: [{ data: charts.churn_by_contract.values, backgroundColor: "#236c63", borderRadius: 6, maxBarThickness: 54 }] }, options: { ...chartOptions(), plugins: { legend: { display: false } }, scales: { ...chartOptions().scales, y: { ...chartOptions().scales.y, ticks: { ...chartOptions().scales.y.ticks, callback: (value) => `${value}%` } } } } });
  state.charts.importance = new Chart($("#importance-chart"), { type: "bar", data: { labels: charts.feature_importance.labels, datasets: [{ data: charts.feature_importance.values, backgroundColor: "#bd751d", borderRadius: 5, barThickness: 14 }] }, options: { ...chartOptions(), indexAxis: "y", plugins: { legend: { display: false } } } });
}

function filteredRows() {
  const selected = [...$("#risk-filter").selectedOptions].map((option) => option.value);
  return state.payload.results.filter((row) => selected.includes(row.risk_level));
}

function renderTable() {
  const rows = filteredRows();
  $("#table-status").textContent = `${rows.length.toLocaleString()} of ${state.payload.results.length.toLocaleString()} customers shown`;
  $("#results-body").innerHTML = rows.map((row) => `<tr data-customer-index="${row.index}"><td>${escapeHtml(row.customer_id)}</td><td><span class="prediction ${row.churn_prediction.toLowerCase()}">${escapeHtml(row.churn_prediction)}</span></td><td>${percent(row.churn_probability)}</td><td><span class="risk-badge ${riskClass(row.risk_level)}">${escapeHtml(row.risk_level)}</span></td></tr>`).join("");
}

function renderCustomerOptions() {
  $("#customer-select").innerHTML = state.payload.results.map((row) => `<option value="${row.index}">${escapeHtml(row.customer_id)} &middot; ${escapeHtml(row.risk_level)}</option>`).join("");
}

function renderCustomer(index) {
  const row = state.payload.results.find((item) => item.index === Number(index));
  if (!row) return;
  $("#customer-detail").innerHTML = `<div class="customer-summary ${riskClass(row.risk_level)}"><div><span class="summary-kicker">Selected customer</span><strong>${escapeHtml(row.customer_id)}</strong></div><div class="summary-risk"><span class="risk-badge ${riskClass(row.risk_level)}">${escapeHtml(row.risk_level)}</span><span class="summary-probability">${percent(row.churn_probability)} churn probability</span></div></div><div class="detail-grid"><div class="detail-stat"><small>Prediction</small><strong>${escapeHtml(row.churn_prediction)}</strong></div><div class="detail-stat"><small>Probability</small><strong>${percent(row.churn_probability)}</strong></div><div class="detail-stat"><small>Risk level</small><strong>${escapeHtml(row.risk_level)}</strong></div></div><div class="explanation"><p class="eyebrow">Why is this customer at risk?</p><h3>Model explanation</h3><p>${escapeHtml(row.shap_explanation)}</p></div><div class="recommendation-heading"><p class="eyebrow">Recommended actions</p><h3>Make the next move count</h3></div><div class="recommendation-grid">${row.recommendations.map((recommendation, i) => `<article class="recommendation-card"><span>0${i + 1}</span><strong>Recommendation ${i + 1}</strong><p>${escapeHtml(recommendation)}</p></article>`).join("")}</div>`;
}

function downloadResults() {
  const headers = ["customer_id", "churn_prediction", "churn_probability", "risk_level"];
  const csv = [headers.join(","), ...state.payload.results.map((row) => headers.map((key) => JSON.stringify(row[key])).join(","))].join("\n");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  link.download = "churn_predictions.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

async function processFile(file) {
  hideAlert();
  $("#loading").hidden = false;
  $("#dashboard").hidden = true;
  const form = new FormData();
  form.append("file", file);
  try {
    const response = await fetch(`${API_BASE_URL}/predict/file`, { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The backend could not process this file.");
    state.payload = payload;
    $("#file-status").textContent = `${file.name} · ${Math.round(file.size / 1024)} KB · processed successfully`;
    renderKpis(payload.summary); renderInsights(payload); renderCharts(payload.charts); renderTable(); renderCustomerOptions(); renderCustomer(0);
    $("#empty-state").hidden = true;
    $("#dashboard").hidden = false;
  } catch (error) {
    showAlert(error.message || "The backend is unavailable. Check API_BASE_URL and try again.");
    $("#file-status").textContent = "Upload could not be processed";
  } finally {
    $("#loading").hidden = true;
  }
}

$("#csv-file").addEventListener("change", (event) => { const [file] = event.target.files; if (file) processFile(file); });
$("#risk-filter").addEventListener("change", renderTable);
$("#customer-select").addEventListener("change", (event) => renderCustomer(event.target.value));
$("#download-button").addEventListener("click", downloadResults);
