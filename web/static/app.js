"use strict";

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

// ---- minimal vanilla-JS canvas line chart, no dependency ----
function drawLineChart(canvas, series, opts) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  const padL = 50, padR = 20, padT = 20, padB = 40;
  ctx.clearRect(0, 0, W, H);

  const allX = series.flatMap((s) => s.points.map((p) => p.x));
  const allY = series.flatMap((s) => s.points.map((p) => p.y));
  const xMin = Math.min(...allX), xMax = Math.max(...allX);
  const yMin = opts.yMin ?? Math.min(0, ...allY);
  const yMax = opts.yMax ?? Math.max(1, ...allY);

  const xScale = (x) => padL + ((x - xMin) / ((xMax - xMin) || 1)) * (W - padL - padR);
  const yScale = (y) => H - padB - ((y - yMin) / ((yMax - yMin) || 1)) * (H - padT - padB);

  const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const axisColor = isDark ? "#30363d" : "#e0e0e0";
  const textColor = isDark ? "#9aa4af" : "#5a5a5a";

  ctx.strokeStyle = axisColor;
  ctx.fillStyle = textColor;
  ctx.font = "11px sans-serif";
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, H - padB);
  ctx.lineTo(W - padR, H - padB);
  ctx.stroke();

  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const v = yMin + ((yMax - yMin) * i) / yTicks;
    const y = yScale(v);
    ctx.strokeStyle = axisColor;
    ctx.globalAlpha = 0.4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillText(v.toFixed(2), 6, y + 3);
  }

  const xTickValues = [...new Set(allX)].sort((a, b) => a - b);
  xTickValues.forEach((v) => {
    const x = xScale(v);
    ctx.fillText(String(v), x - 4, H - padB + 16);
  });

  ctx.fillText(opts.xLabel || "", W / 2 - 20, H - 6);

  series.forEach((s) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.points.forEach((p, i) => {
      const x = xScale(p.x), y = yScale(p.y);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = s.color;
    s.points.forEach((p) => {
      ctx.beginPath();
      ctx.arc(xScale(p.x), yScale(p.y), 2.5, 0, 2 * Math.PI);
      ctx.fill();
    });
  });
}

function renderLegend(el, series) {
  el.innerHTML = series
    .map(
      (s) =>
        `<span><span style="display:inline-block;width:0.75rem;height:0.75rem;border-radius:2px;background:${s.color};margin-right:0.35rem;vertical-align:middle;"></span>${s.label}</span>`
    )
    .join("");
}

const COLORS = {
  successInitial: "#f0883e",
  successFinal: "#d1242f",
  cleanPrAuc: "#1a7f37",
};

async function loadEvasionCurve() {
  const data = await fetchJSON("/api/evasion-curve");
  const records = data.records;
  const canvas = document.getElementById("evasion-chart");
  const series = [
    { label: "attack success (before evolution)", color: COLORS.successInitial, points: records.map((r) => ({ x: r.loop_generation, y: r.attack_success_rate_initial })) },
    { label: "attack success (after evolution)", color: COLORS.successFinal, points: records.map((r) => ({ x: r.loop_generation, y: r.attack_success_rate_final })) },
    { label: "clean-set PR-AUC", color: COLORS.cleanPrAuc, points: records.map((r) => ({ x: r.loop_generation, y: r.clean_pr_auc })) },
  ];
  drawLineChart(canvas, series, { xLabel: "loop (retraining) generation", yMin: 0, yMax: 1 });
  renderLegend(document.getElementById("evasion-legend"), series);
  return records;
}

async function loadAttackCurve(records) {
  const data = await fetchJSON("/api/attack-curve");
  const rows = data.rows.map((r) => ({
    loop_generation: Number(r.loop_generation),
    generation: Number(r.generation),
    mean_evasion_score: Number(r.mean_evasion_score),
    attack_success_rate: Number(r.attack_success_rate),
  }));

  const select = document.getElementById("loop-gen-select");
  const loopGens = [...new Set(rows.map((r) => r.loop_generation))].sort((a, b) => a - b);
  select.innerHTML = loopGens.map((g) => `<option value="${g}">loop generation ${g}</option>`).join("");

  function render() {
    const chosen = Number(select.value);
    const subset = rows.filter((r) => r.loop_generation === chosen);
    const canvas = document.getElementById("attack-chart");
    const series = [
      { label: "mean evasion score", color: COLORS.cleanPrAuc, points: subset.map((r) => ({ x: r.generation, y: r.mean_evasion_score })) },
      { label: "attack success rate", color: COLORS.successFinal, points: subset.map((r) => ({ x: r.generation, y: r.attack_success_rate })) },
    ];
    drawLineChart(canvas, series, { xLabel: "EA generation (within this loop generation)", yMin: 0, yMax: 1 });
  }
  select.addEventListener("change", render);
  render();
}

function renderCard(container, label, value) {
  const div = document.createElement("div");
  div.className = "card";
  div.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>`;
  container.appendChild(div);
}

async function loadMetrics() {
  const data = await fetchJSON("/api/metrics");
  const m = data.metrics;
  const el = document.getElementById("metrics-cards");
  el.innerHTML = "";
  renderCard(el, "PR-AUC", m.pr_auc.toFixed(4));
  renderCard(el, "ROC-AUC", m.roc_auc.toFixed(4));
  renderCard(el, "F1", m.f1.toFixed(4));
  renderCard(el, "Brier score", m.brier_score.toFixed(4));
  Object.entries(m.recall_at_fpr || {}).forEach(([fpr, r]) => renderCard(el, `recall @ FPR ${fpr}`, r.toFixed(3)));
  Object.entries(m.precision_at_k || {}).forEach(([k, p]) => renderCard(el, `precision @ k=${k}`, p.toFixed(3)));
  if (m.latency_ms) {
    renderCard(el, "p50 latency", `${m.latency_ms.p50_ms.toFixed(2)} ms`);
    renderCard(el, "p99 latency", `${m.latency_ms.p99_ms.toFixed(2)} ms`);
  }
}

async function loadFidelity() {
  try {
    const data = await fetchJSON("/api/fidelity");
    const f = data.fidelity;
    const el = document.getElementById("fidelity-cards");
    el.innerHTML = "";
    renderCard(el, "marginal similarity (mean)", f.marginal_similarity_mean.toFixed(4));
    renderCard(el, "correlation similarity", f.correlation_similarity.toFixed(4));
    renderCard(el, "DCR median ratio", f.dcr_median_ratio.toFixed(4));
    document.getElementById("fidelity-note").textContent = (f.notes || []).join(" ");
  } catch (e) {
    document.getElementById("fidelity-note").textContent = `Not available: ${e.message}`;
  }
}

async function loadTaxonomy() {
  const data = await fetchJSON("/api/taxonomy");
  const tbody = document.querySelector("#taxonomy-table tbody");
  tbody.innerHTML = data.rows
    .map(
      (r) =>
        `<tr><td>${r.surface}</td><td>${r.vector}</td><td>${r.rail}</td><td>${r.severity}</td><td>${r.source.name}</td></tr>`
    )
    .join("");
}

let currentTxn = null;

async function loadSampleTransaction(label) {
  currentTxn = await fetchJSON(`/api/sample-transaction?label=${label}`);
  document.getElementById("txn-preview").textContent = JSON.stringify(currentTxn, null, 2);
  document.getElementById("score-btn").disabled = false;
  document.getElementById("score-result").innerHTML = "";
}

async function scoreCurrentTransaction() {
  if (!currentTxn) return;
  const result = await fetchJSON("/api/mock/transactions/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(currentTxn),
  });
  const el = document.getElementById("score-result");
  const cls = result.flagged ? "blocked" : "authorized";
  const verdict = result.flagged ? "FLAGGED" : "not flagged";
  el.innerHTML = `<span class="${cls}">${verdict}</span> — score ${result.score.toFixed(4)} (threshold ${result.threshold})`;
}

async function runMandateDemo() {
  const data = await fetchJSON("/api/mandate-demo", { method: "POST" });
  const el = document.getElementById("mandate-result");
  el.innerHTML = data.scenarios
    .map((s) => {
      const cls = s.authorized ? "authorized" : "blocked";
      const verdict = s.authorized ? "AUTHORIZED" : "BLOCKED";
      const reasons = s.errors.length ? ` (${s.errors.join("; ")})` : "";
      return `<div class="scenario"><strong>${s.name}</strong>: <span class="${cls}">${verdict}</span>${reasons}</div>`;
    })
    .join("");
}

async function init() {
  const records = await loadEvasionCurve();
  await loadAttackCurve(records);
  await loadMetrics();
  await loadFidelity();
  await loadTaxonomy();

  document.getElementById("load-legit-btn").addEventListener("click", () => loadSampleTransaction(0));
  document.getElementById("load-fraud-btn").addEventListener("click", () => loadSampleTransaction(1));
  document.getElementById("score-btn").addEventListener("click", scoreCurrentTransaction);
  document.getElementById("mandate-demo-btn").addEventListener("click", runMandateDemo);
}

init().catch((e) => {
  console.error(e);
  document.body.insertAdjacentHTML("afterbegin", `<div style="background:#d1242f;color:white;padding:1rem;">Dashboard failed to load: ${e.message}. Have you run make data/train/loop/eval/fidelity?</div>`);
});
