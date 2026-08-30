// Riskora AI State Management
let cachedMetrics = null;
let cachedSummary = null;
let cachedAuditLogs = [];
let currentPage = 'overview';

// Chart Instances
let overviewDonutChart = null;
let anPrChartInstance = null;
let anRocChartInstance = null;
let thCostChartInstance = null;

// Presets Definition
const PRESETS = {
  high_risk: {
    customer_id: "CUST_008412",
    product_category: "Apparel",
    order_amount: 2499,
    discount_percentage: 45,
    item_quantity: 2,
    pincode_tier: "Remote",
    user_past_return_rate: "0.8",
    payment_method: "COD"
  },
  medium_risk: {
    customer_id: "CUST_004120",
    product_category: "Footwear",
    order_amount: 1899,
    discount_percentage: 25,
    item_quantity: 1,
    pincode_tier: "Tier2_Urban",
    user_past_return_rate: "0.2",
    payment_method: "COD"
  },
  low_risk: {
    customer_id: "CUST_VIP_099",
    product_category: "Electronics",
    order_amount: 4500,
    discount_percentage: 5,
    item_quantity: 1,
    pincode_tier: "Tier1_Metro",
    user_past_return_rate: "0.0",
    payment_method: "UPI"
  }
};

// Application Initialization
document.addEventListener("DOMContentLoaded", async () => {
  await loadMetrics();
  await loadDashboardSummary();
  await refreshAuditLogs();
  triggerLiveScoring();
});

// Page Switcher
function switchPage(pageId) {
  currentPage = pageId;
  const sections = document.querySelectorAll(".page-section");
  sections.forEach(sec => sec.classList.add("hidden"));

  const targetSection = document.getElementById(`page-${pageId}`);
  if (targetSection) targetSection.classList.remove("hidden");

  // Update desktop navigation active states
  const navBtns = document.querySelectorAll(".nav-item");
  navBtns.forEach(btn => {
    btn.classList.remove("active");
    btn.classList.remove("bg-slate-100", "text-slate-900");
    btn.classList.add("text-slate-600");
  });

  const activeNav = document.getElementById(`nav-${pageId}`);
  if (activeNav) {
    activeNav.classList.add("active", "bg-slate-100", "text-slate-900");
    activeNav.classList.remove("text-slate-600");
  }

  // Update mobile navigation active states
  const mobBtns = document.querySelectorAll(".mob-nav-item");
  mobBtns.forEach(btn => {
    btn.classList.remove("active", "bg-slate-100", "text-slate-900");
    btn.classList.add("text-slate-600");
  });

  // Re-render charts on tab switch if needed
  if (pageId === "overview") {
    setTimeout(renderOverviewDonut, 50);
  } else if (pageId === "analytics") {
    setTimeout(renderAnalyticsCharts, 50);
  } else if (pageId === "threshold") {
    setTimeout(renderThresholdStudioChart, 50);
  } else if (pageId === "audit") {
    refreshAuditLogs();
  }
}

// Load Metrics from Backend
async function loadMetrics() {
  try {
    const res = await fetch("/api/v1/metrics");
    if (!res.ok) return;
    cachedMetrics = await res.json();

    const tStar = cachedMetrics.optimal_threshold;
    document.getElementById("header-t-star").innerText = `T* = ${tStar.toFixed(2)}`;
    document.getElementById("th-optimal-badge").innerText = `T* = ${tStar.toFixed(2)}`;

    populateAnalyticsView();
    updateStudioSimulation();
  } catch (err) {
    console.error("Failed to load metrics:", err);
  }
}

// Load Dashboard Summary
async function loadDashboardSummary() {
  try {
    const res = await fetch("/api/v1/dashboard-summary");
    if (!res.ok) return;
    cachedSummary = await res.json();

    document.getElementById("kpi-orders-count").innerText = cachedSummary.orders_scored.toLocaleString();
    document.getElementById("kpi-high-risk-count").innerText = cachedSummary.high_risk_orders.toLocaleString();
    document.getElementById("kpi-rtos-prevented").innerText = cachedSummary.estimated_rtos_prevented.toLocaleString();
    document.getElementById("kpi-savings-amount").innerText = `₹${cachedSummary.estimated_savings_inr.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;

    renderRecentChecksTable(cachedSummary.recent_risk_checks || []);
    renderOverviewDonut();
  } catch (err) {
    console.error("Failed to load summary:", err);
  }
}

// Render Recent Risk Checks in Overview
function renderRecentChecksTable(checks) {
  const tbody = document.getElementById("overview-recent-tbody");
  if (!checks || checks.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="5" class="py-6 text-center text-slate-400 text-xs">
          No orders scored yet. Try running an analysis in Live Scoring!
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = checks.map(c => {
    let badgeClass = "bg-emerald-50 text-emerald-700 border-emerald-200";
    if (c.risk_tier === "HIGH") badgeClass = "bg-red-50 text-red-700 border-red-200";
    else if (c.risk_tier === "MEDIUM") badgeClass = "bg-amber-50 text-amber-700 border-amber-200";

    return `
      <tr class="hover:bg-slate-50/80 transition-colors">
        <td class="py-2.5 px-2.5 font-mono font-medium text-slate-900">${c.order_id}</td>
        <td class="py-2.5 px-2 font-mono text-slate-700">₹${c.order_amount.toLocaleString()}</td>
        <td class="py-2.5 px-2 text-slate-600">${c.payment_method}</td>
        <td class="py-2.5 px-2 font-mono font-semibold text-slate-900">${(c.risk_score * 100).toFixed(1)}%</td>
        <td class="py-2.5 px-2.5 text-right">
          <span class="px-2 py-0.5 rounded text-[10px] font-bold border ${badgeClass}">${c.risk_tier}</span>
        </td>
      </tr>
    `;
  }).join("");
}

// Render Overview Donut Chart
function renderOverviewDonut() {
  const ctx = document.getElementById("overviewDonutChart")?.getContext("2d");
  if (!ctx) return;

  let low = cachedSummary?.low_risk_orders || 0;
  let med = cachedSummary?.medium_risk_orders || 0;
  let high = cachedSummary?.high_risk_orders || 0;

  // If no DB orders yet, use held-out test distribution
  if (low === 0 && med === 0 && high === 0 && cachedMetrics) {
    const cm = cachedMetrics.held_out_test_metrics.confusion_matrix;
    high = cm.tp + cm.fp;
    low = cm.tn;
    med = cm.fn;
  }

  document.getElementById("dist-low-val").innerText = low.toLocaleString();
  document.getElementById("dist-med-val").innerText = med.toLocaleString();
  document.getElementById("dist-high-val").innerText = high.toLocaleString();

  if (overviewDonutChart) overviewDonutChart.destroy();

  overviewDonutChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Low Risk', 'Medium Risk', 'High Risk'],
      datasets: [{
        data: [low || 1, med || 1, high || 1],
        backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
        borderWidth: 2,
        borderColor: '#ffffff'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      cutout: '72%'
    }
  });
}

// Load Demo Presets
function loadPreset(key) {
  const p = PRESETS[key];
  if (!p) return;

  document.getElementById("live-customer-id").value = p.customer_id;
  document.getElementById("live-category").value = p.product_category;
  document.getElementById("live-amount").value = p.order_amount;
  document.getElementById("live-discount").value = p.discount_percentage;
  document.getElementById("live-quantity").value = p.item_quantity;
  document.getElementById("live-pincode").value = p.pincode_tier;
  document.getElementById("live-past-return").value = p.user_past_return_rate;

  const radio = document.querySelector(`input[name="live_payment_method"][value="${p.payment_method}"]`);
  if (radio) radio.checked = true;

  triggerLiveScoring();
}

// Debounced Live Scoring
let liveScoringTimer = null;
function triggerLiveScoring() {
  clearTimeout(liveScoringTimer);
  liveScoringTimer = setTimeout(async () => {
    const spinner = document.getElementById("live-spinner");
    if (spinner) spinner.classList.remove("hidden");

    const payload = {
      order_id: "ORD_" + Math.floor(100000 + Math.random() * 900000),
      customer_id: document.getElementById("live-customer-id").value || "CUST_001",
      product_category: document.getElementById("live-category").value,
      order_amount: parseFloat(document.getElementById("live-amount").value) || 1500,
      discount_percentage: (parseFloat(document.getElementById("live-discount").value) || 0) / 100.0,
      item_quantity: parseInt(document.getElementById("live-quantity").value) || 1,
      pincode_tier: document.getElementById("live-pincode").value,
      user_past_return_rate: parseFloat(document.getElementById("live-past-return").value) || 0.2,
      user_past_orders_count: parseFloat(document.getElementById("live-past-return").value) > 0 ? 3 : 0,
      payment_method: document.querySelector('input[name="live_payment_method"]:checked')?.value || "COD",
      shipping_speed: "Standard",
      device_type: "Mobile_App"
    };

    try {
      const res = await fetch("/api/v1/score-order", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        const data = await res.json();
        updateLiveDecisionUI(data);
        loadDashboardSummary(); // Refresh summary in background
      }
    } catch (err) {
      console.error("Live scoring error:", err);
    } finally {
      if (spinner) spinner.classList.add("hidden");
    }
  }, 100);
}

// Update Live Scoring Result UI
function updateLiveDecisionUI(data) {
  const scorePercent = (data.risk_score * 100).toFixed(1);
  document.getElementById("live-score-val").innerText = `${scorePercent}%`;
  document.getElementById("live-action-title").innerText = data.action;
  document.getElementById("live-action-reason").innerText = data.action_reason;
  document.getElementById("live-savings-val").innerText = `+₹${data.potential_savings_inr.toFixed(2)}`;

  // Circular gauge calculation
  const maxDash = 238.76;
  const offset = maxDash - (maxDash * data.risk_score);
  const circle = document.getElementById("live-score-circle");
  circle.style.strokeDashoffset = offset;

  // Tier Badge & Payment Gating UI
  const tierBadge = document.getElementById("live-tier-badge");
  const actionTitle = document.getElementById("live-action-title");
  const codBox = document.getElementById("live-cod-box");
  const codBadge = document.getElementById("live-cod-badge");
  const upiBox = document.getElementById("live-upi-box");
  const upiBadge = document.getElementById("live-upi-badge");
  const gateCod = document.getElementById("gate-cod");
  const gateUpi = document.getElementById("gate-upi");

  if (data.risk_tier === "HIGH") {
    tierBadge.className = "px-3 py-1 rounded-full text-xs font-extrabold bg-red-100 text-red-800 border border-red-200";
    tierBadge.innerText = "HIGH RISK";
    circle.style.stroke = "#ef4444";
    actionTitle.className = "font-heading font-bold text-base text-red-600";

    codBox.classList.add("cod-disabled");
    codBadge.classList.remove("hidden");
    codBadge.innerHTML = `<span class="text-[10px] font-bold text-red-700 px-1.5 py-0.5 rounded bg-red-100 border border-red-200">⛔ COD Disabled</span>`;

    upiBox.classList.add("upi-recommended");
    upiBadge.classList.add("hidden");

    gateCod.className = "p-2 rounded bg-red-100 text-red-800 border border-red-200";
    gateCod.innerHTML = "COD: <span class='font-bold'>Disabled</span>";
    gateUpi.className = "p-2 rounded bg-emerald-100 text-emerald-800 border border-emerald-200";
    gateUpi.innerHTML = "UPI: <span class='font-bold'>Recommended</span>";

  } else if (data.risk_tier === "MEDIUM") {
    tierBadge.className = "px-3 py-1 rounded-full text-xs font-extrabold bg-amber-100 text-amber-800 border border-amber-200";
    tierBadge.innerText = "MEDIUM RISK";
    circle.style.stroke = "#f59e0b";
    actionTitle.className = "font-heading font-bold text-base text-amber-600";

    codBox.classList.remove("cod-disabled");
    codBadge.classList.add("hidden");

    upiBox.classList.add("upi-recommended");
    upiBadge.classList.remove("hidden");

    gateCod.className = "p-2 rounded bg-amber-100 text-amber-800 border border-amber-200";
    gateCod.innerHTML = "COD: <span class='font-bold'>Available</span>";
    gateUpi.className = "p-2 rounded bg-emerald-100 text-emerald-800 border border-emerald-200";
    gateUpi.innerHTML = "UPI: <span class='font-bold'>Recommended (Nudge)</span>";

  } else {
    tierBadge.className = "px-3 py-1 rounded-full text-xs font-extrabold bg-emerald-100 text-emerald-800 border border-emerald-200";
    tierBadge.innerText = "LOW RISK";
    circle.style.stroke = "#10b981";
    actionTitle.className = "font-heading font-bold text-base text-emerald-600";

    codBox.classList.remove("cod-disabled");
    codBadge.classList.add("hidden");

    upiBox.classList.remove("upi-recommended");
    upiBadge.classList.add("hidden");

    gateCod.className = "p-2 rounded bg-slate-100 text-slate-800 border border-slate-200";
    gateCod.innerHTML = "COD: <span class='font-bold'>Available</span>";
    gateUpi.className = "p-2 rounded bg-slate-100 text-slate-800 border border-slate-200";
    gateUpi.innerHTML = "UPI: <span class='font-bold'>Available</span>";
  }

  // Render Top 3 SHAP Drivers
  const driversContainer = document.getElementById("live-drivers-container");
  driversContainer.innerHTML = (data.top_drivers || []).map(d => {
    const isRisk = d.direction === "INCREASES_RISK";
    const barColor = isRisk ? "bg-red-500" : "bg-emerald-500";
    const badgeStyle = isRisk ? "bg-red-50 text-red-700 border-red-200" : "bg-emerald-50 text-emerald-700 border-emerald-200";
    const arrow = isRisk ? "▲" : "▼";

    return `
      <div class="p-3 rounded-lg bg-slate-50 border border-slate-100 space-y-1.5">
        <div class="flex items-center justify-between">
          <span class="font-semibold text-slate-800 text-xs">${d.display_name}</span>
          <span class="text-[10px] font-bold px-1.5 py-0.5 rounded border ${badgeStyle}">
            ${arrow} ${isRisk ? 'Increases Risk' : 'Reduces Risk'}
          </span>
        </div>
        <p class="text-[11px] text-slate-500 leading-tight">${d.explanation}</p>
        <div class="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden mt-1">
          <div class="${barColor} h-full driver-bar-fill rounded-full" style="width: ${Math.round(d.impact_score * 100)}%"></div>
        </div>
      </div>
    `;
  }).join("");
}

// Populate Analytics View with Held-out Test Metrics
function populateAnalyticsView() {
  if (!cachedMetrics) return;
  const testM = cachedMetrics.held_out_test_metrics;
  const baseM = cachedMetrics.baseline_comparison;

  document.getElementById("an-precision").innerText = testM.precision.toFixed(3);
  document.getElementById("an-recall").innerText = testM.recall.toFixed(3);
  document.getElementById("an-f1").innerText = testM.f1_score.toFixed(3);
  document.getElementById("an-prauc").innerText = testM.pr_auc.toFixed(3);
  document.getElementById("an-rocauc").innerText = testM.roc_auc.toFixed(3);

  // Comparison Table
  document.getElementById("an-comp-pr-lr").innerText = baseM.pr_auc.toFixed(3);
  document.getElementById("an-comp-pr-xgb").innerText = testM.pr_auc.toFixed(3);
  document.getElementById("an-comp-pr-lift").innerText = `${baseM.lift_over_baseline.pr_auc_delta >= 0 ? '+' : ''}${baseM.lift_over_baseline.pr_auc_delta.toFixed(3)}`;

  document.getElementById("an-comp-roc-lr").innerText = baseM.roc_auc.toFixed(3);
  document.getElementById("an-comp-roc-xgb").innerText = testM.roc_auc.toFixed(3);
  const rocLift = testM.roc_auc - baseM.roc_auc;
  document.getElementById("an-comp-roc-lift").innerText = `${rocLift >= 0 ? '+' : ''}${rocLift.toFixed(3)}`;

  document.getElementById("an-comp-prec-lr").innerText = baseM.precision.toFixed(3);
  document.getElementById("an-comp-prec-xgb").innerText = testM.precision.toFixed(3);
  const precLift = testM.precision - baseM.precision;
  document.getElementById("an-comp-prec-lift").innerText = `${precLift >= 0 ? '+' : ''}${precLift.toFixed(3)}`;

  document.getElementById("an-comp-rec-lr").innerText = baseM.recall.toFixed(3);
  document.getElementById("an-comp-rec-xgb").innerText = testM.recall.toFixed(3);
  const recLift = testM.recall - baseM.recall;
  document.getElementById("an-comp-rec-lift").innerText = `${recLift >= 0 ? '+' : ''}${recLift.toFixed(3)}`;

  document.getElementById("an-comp-profit-lr").innerText = `₹${baseM.net_profit_saved_inr.toLocaleString('en-IN')}`;
  document.getElementById("an-comp-profit-xgb").innerText = `₹${testM.net_profit_saved_inr.toLocaleString('en-IN')}`;
  const profitLift = baseM.lift_over_baseline.profit_saved_delta_inr;
  document.getElementById("an-comp-profit-lift").innerText = `${profitLift >= 0 ? '+' : ''}₹${profitLift.toLocaleString('en-IN')}`;

  // Confusion Matrix
  const cm = testM.confusion_matrix;
  document.getElementById("an-cm-tp").innerText = cm.tp.toLocaleString();
  document.getElementById("an-cm-fp").innerText = cm.fp.toLocaleString();
  document.getElementById("an-cm-fn").innerText = cm.fn.toLocaleString();
  document.getElementById("an-cm-tn").innerText = cm.tn.toLocaleString();
}

// Render PR and ROC Charts in Analytics
function renderAnalyticsCharts() {
  if (!cachedMetrics) return;

  // PR Curve
  const prCtx = document.getElementById("anPrCurveChart")?.getContext("2d");
  if (prCtx) {
    if (anPrChartInstance) anPrChartInstance.destroy();
    const prPoints = (cachedMetrics.pr_curve || []).map(p => ({ x: p.recall, y: p.precision }));
    anPrChartInstance = new Chart(prCtx, {
      type: 'line',
      data: {
        datasets: [{
          label: 'Calibrated XGBoost PR Curve',
          data: prPoints,
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.08)',
          borderWidth: 2,
          fill: true,
          pointRadius: 0,
          tension: 0.2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { type: 'linear', title: { display: true, text: 'Recall', font: { size: 10 } }, min: 0, max: 1 },
          y: { title: { display: true, text: 'Precision', font: { size: 10 } }, min: 0, max: 1 }
        }
      }
    });
  }

  // ROC Curve
  const rocCtx = document.getElementById("anRocCurveChart")?.getContext("2d");
  if (rocCtx) {
    if (anRocChartInstance) anRocChartInstance.destroy();
    const rocPoints = (cachedMetrics.roc_curve || []).map(p => ({ x: p.fpr, y: p.tpr }));
    anRocChartInstance = new Chart(rocCtx, {
      type: 'line',
      data: {
        datasets: [{
          label: 'Calibrated XGBoost ROC Curve',
          data: rocPoints,
          borderColor: '#0f172a',
          backgroundColor: 'rgba(15, 23, 42, 0.05)',
          borderWidth: 2,
          fill: true,
          pointRadius: 0,
          tension: 0.2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { type: 'linear', title: { display: true, text: 'False Positive Rate (1 - Specificity)', font: { size: 10 } }, min: 0, max: 1 },
          y: { title: { display: true, text: 'True Positive Rate (Sensitivity)', font: { size: 10 } }, min: 0, max: 1 }
        }
      }
    });
  }
}

// Update Threshold Studio ROI Simulation
function updateStudioSimulation() {
  const cFn = parseFloat(document.getElementById("th-slider-cfn").value);
  const cFp = parseFloat(document.getElementById("th-slider-cfp").value);
  const volume = parseFloat(document.getElementById("th-slider-vol").value);

  document.getElementById("th-cfn-val").innerText = `₹${cFn}`;
  document.getElementById("th-cfp-val").innerText = `₹${cFp}`;
  document.getElementById("th-vol-val").innerText = `${volume.toLocaleString()} orders`;

  if (!cachedMetrics) return;
  const cm = cachedMetrics.held_out_test_metrics.confusion_matrix;
  const totalTest = cm.tp + cm.fp + cm.fn + cm.tn;
  const scale = volume / totalTest;

  const scaledTp = Math.round(cm.tp * scale);
  const scaledFp = Math.round(cm.fp * scale);
  const scaledTn = Math.round(cm.tn * scale);

  const netSaved = (scaledTp * cFn) - (scaledFp * cFp);
  const friction = scaledFp * cFp;
  const roi = friction > 0 ? ((scaledTp * cFn) / friction).toFixed(1) : "10.0";
  const preservedPct = (((scaledTn + scaledFp) / volume) * 100).toFixed(1);

  document.getElementById("th-kpi-savings").innerText = `₹${(netSaved / 100000).toFixed(2)} Lakhs`;
  document.getElementById("th-kpi-rtos").innerText = scaledTp.toLocaleString();
  document.getElementById("th-kpi-preserved").innerText = `${preservedPct}%`;
  document.getElementById("th-kpi-roi").innerText = `${roi}x`;
}

function resetThresholdSliders() {
  document.getElementById("th-slider-cfn").value = 350;
  document.getElementById("th-slider-cfp").value = 175;
  document.getElementById("th-slider-vol").value = 25000;
  updateStudioSimulation();
}

// Render Cost Sweep Chart in Threshold Studio
function renderThresholdStudioChart() {
  const ctx = document.getElementById("thCostSweepChart")?.getContext("2d");
  if (!ctx || !cachedMetrics) return;

  if (thCostChartInstance) thCostChartInstance.destroy();
  const sweep = cachedMetrics.validation_threshold_sweep || [];
  const labels = sweep.map(s => s.threshold);
  const profits = sweep.map(s => s.net_profit_saved_inr);

  thCostChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Net Margin Saved on Validation (₹)',
        data: profits,
        borderColor: '#dc2626',
        backgroundColor: 'rgba(220, 38, 38, 0.08)',
        borderWidth: 2.5,
        fill: true,
        pointRadius: 1,
        pointHoverRadius: 5,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Net Profit Saved: ₹${ctx.parsed.y.toLocaleString('en-IN')}`
          }
        }
      },
      scales: {
        x: { title: { display: true, text: 'Cutoff Threshold (T)', font: { size: 11 } } },
        y: { title: { display: true, text: 'Net Saved (₹)', font: { size: 11 } } }
      }
    }
  });
}

// Audit Logs Management
async function refreshAuditLogs() {
  try {
    const res = await fetch("/api/v1/audit-logs?limit=100");
    if (!res.ok) return;
    const data = await res.json();
    cachedAuditLogs = data.logs || [];
    filterAuditLogs();
  } catch (err) {
    console.error("Failed to load audit logs:", err);
  }
}

function filterAuditLogs() {
  const filter = document.getElementById("audit-filter-tier")?.value || "ALL";
  const tbody = document.getElementById("audit-table-tbody");
  if (!tbody) return;

  let filtered = cachedAuditLogs;
  if (filter !== "ALL") {
    filtered = cachedAuditLogs.filter(l => l.risk_tier === filter);
  }

  if (filtered.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="9" class="py-8 text-center text-slate-400">
          No audit records found matching '${filter}'. Run an analysis in Live Scoring to log new transactions!
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = filtered.map(l => {
    let badgeClass = "bg-emerald-50 text-emerald-700 border-emerald-200";
    if (l.risk_tier === "HIGH") badgeClass = "bg-red-50 text-red-700 border-red-200";
    else if (l.risk_tier === "MEDIUM") badgeClass = "bg-amber-50 text-amber-700 border-amber-200";

    const formattedTime = l.created_at ? new Date(l.created_at).toLocaleTimeString() : "--";

    return `
      <tr onclick="inspectAuditRecord(${l.id})" class="hover:bg-slate-50 cursor-pointer transition-colors">
        <td class="py-2.5 px-3 font-mono text-slate-500 text-[11px]">${formattedTime}</td>
        <td class="py-2.5 px-3 font-mono font-semibold text-slate-900">${l.order_id}</td>
        <td class="py-2.5 px-3 font-mono text-slate-600">${l.customer_id}</td>
        <td class="py-2.5 px-3 text-slate-700">${l.product_category}</td>
        <td class="py-2.5 px-3 font-mono text-slate-900">₹${l.order_amount.toLocaleString()}</td>
        <td class="py-2.5 px-3 text-slate-600">${l.payment_method}</td>
        <td class="py-2.5 px-3 font-mono font-bold text-slate-900">${(l.risk_score * 100).toFixed(1)}%</td>
        <td class="py-2.5 px-3">
          <span class="px-2 py-0.5 rounded text-[10px] font-bold border ${badgeClass}">${l.risk_tier}</span>
        </td>
        <td class="py-2.5 px-3 text-right font-mono text-[11px] text-slate-600">${l.action}</td>
      </tr>
    `;
  }).join("");
}

// Modal Inspection of Audit Record
function inspectAuditRecord(logId) {
  const record = cachedAuditLogs.find(l => l.id === logId);
  if (!record) return;

  const content = document.getElementById("modal-content");
  content.innerHTML = `
    <div class="grid grid-cols-2 gap-2 p-3 bg-slate-50 rounded-xl border border-slate-200">
      <div><span class="text-slate-500 block">Order ID:</span> <strong class="font-mono text-slate-900">${record.order_id}</strong></div>
      <div><span class="text-slate-500 block">Customer ID:</span> <strong class="font-mono text-slate-900">${record.customer_id}</strong></div>
      <div><span class="text-slate-500 block">Amount:</span> <strong class="font-mono text-slate-900">₹${record.order_amount.toLocaleString()}</strong></div>
      <div><span class="text-slate-500 block">Discount:</span> <strong class="font-mono text-slate-900">${(record.discount_percentage * 100).toFixed(0)}%</strong></div>
      <div><span class="text-slate-500 block">Category:</span> <strong class="text-slate-900">${record.product_category}</strong></div>
      <div><span class="text-slate-500 block">Pincode Zone:</span> <strong class="text-slate-900">${record.pincode_tier}</strong></div>
    </div>

    <div class="p-3 bg-slate-50 rounded-xl border border-slate-200 space-y-1">
      <div class="flex justify-between">
        <span class="text-slate-500">Calculated Return Probability:</span>
        <strong class="font-mono text-slate-900">${(record.risk_score * 100).toFixed(1)}%</strong>
      </div>
      <div class="flex justify-between">
        <span class="text-slate-500">Assigned Risk Tier:</span>
        <strong class="font-bold text-red-600">${record.risk_tier}</strong>
      </div>
      <div class="flex justify-between">
        <span class="text-slate-500">Gating Decision Action:</span>
        <strong class="font-mono text-slate-900">${record.action}</strong>
      </div>
      <div class="flex justify-between">
        <span class="text-slate-500">Estimated Margin Loss Avoided:</span>
        <strong class="font-mono text-emerald-600">+₹${record.potential_savings_inr.toFixed(2)}</strong>
      </div>
    </div>

    <div class="space-y-1.5">
      <span class="font-bold text-slate-800 text-xs block">Top SHAP Drivers:</span>
      ${(record.top_drivers || []).map(d => `
        <div class="p-2 bg-slate-50 rounded-lg border border-slate-100 flex justify-between items-center text-xs">
          <span>${d.display_name}</span>
          <span class="font-semibold text-red-600">${d.direction === 'INCREASES_RISK' ? '▲ Risk Driver' : '▼ Trust Signal'}</span>
        </div>
      `).join("") || "<span class='text-slate-400 text-xs'>No drivers recorded</span>"}
    </div>
  `;

  document.getElementById("audit-modal").classList.remove("hidden");
}

function closeAuditModal() {
  document.getElementById("audit-modal").classList.add("hidden");
}

// AI Assistant Action Handlers
function askAIAssistant(prompt) {
  document.getElementById("ai-query-input").value = prompt;
  handleAISubmit();
}

function handleAISubmit() {
  const query = document.getElementById("ai-query-input").value.trim();
  const box = document.getElementById("ai-response-box");
  if (!query) return;

  box.classList.remove("hidden");
  
  if (query.toLowerCase().includes("today") || query.toLowerCase().includes("exposure")) {
    const total = cachedSummary?.orders_scored || 0;
    const high = cachedSummary?.high_risk_orders || 0;
    const savings = cachedSummary?.estimated_savings_inr || 0;
    box.innerHTML = `
      <strong>Riskora Exposure Summary:</strong> Out of ${total} total scored checkout sessions, ${high} orders exceeded the optimal threshold cutoff T* (0.32). Dynamic COD gating has safeguarded an estimated ₹${savings.toFixed(2)} in reverse transit depreciation while preserving low-risk conversions.
    `;
  } else {
    box.innerHTML = `
      <strong>Riskora Analysis:</strong> Operating at threshold T* = 0.32 maximizes net INR profit on validation data. Categorical propensities in Apparel and high COD usage remain the primary drivers of RTO variance.
    `;
  }
}
