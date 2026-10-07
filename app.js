/* ═══════════════════════════════════════════════════════════════
   CineNeural AI — app.js
   Interactive Decision Network demo (browser-side simulation)
═══════════════════════════════════════════════════════════════ */

"use strict";

// ──────────────────────────────────────────────────────────────
// Simulated CPT Data (mirrors the trained pgmpy model output)
// These probabilities approximate what the fitted BN produces.
// ──────────────────────────────────────────────────────────────
const CPT_DATA = {
  // P(rating_positive=1 | user_activity, user_rating_bucket, item_quality, item_popularity, item_genre)
  // Simplified marginals for demo inference
  base: 0.52,

  user_activity: { low: -0.10, medium: 0.0, high: +0.09 },
  user_rating_bucket: { harsh: -0.18, neutral: 0.0, generous: +0.16 },
  item_quality: { low: -0.22, medium: 0.0, high: +0.21 },
  item_popularity: { niche: -0.06, moderate: 0.0, popular: +0.05 },

  // Genre effect on P(like) — from empirical CPT
  item_genre: {
    "Action":      +0.04,
    "Adventure":   +0.06,
    "Animation":   +0.09,
    "Comedy":      +0.07,
    "Crime":       +0.05,
    "Drama":       +0.03,
    "Fantasy":     +0.08,
    "Horror":      -0.08,
    "Mystery":     +0.04,
    "Romance":     +0.02,
    "Sci-Fi":      +0.05,
    "Thriller":    +0.03,
  },

  // Gender-genre cross (small effect)
  gender_genre: {
    "M": { "Romance": -0.05, "Action": +0.03, "Animation": +0.01 },
    "F": { "Romance": +0.07, "Action": -0.02, "Drama":     +0.04 },
  },
};

const UTILITY_POS = +1.0;
const UTILITY_NEG = -0.5;
const EU_THRESHOLD = 0.30;

// ──────────────────────────────────────────────────────────────
// Sample movie database (simulated MovieLens 100K top movies)
// ──────────────────────────────────────────────────────────────
const MOVIES = [
  { id: 50,   title: "Star Wars (1977)",          genre: "Sci-Fi",    popularity: "popular",  quality: "high"   },
  { id: 1,    title: "Toy Story (1995)",           genre: "Animation", popularity: "popular",  quality: "high"   },
  { id: 258,  title: "Contact (1997)",             genre: "Sci-Fi",    popularity: "moderate", quality: "high"   },
  { id: 127,  title: "Godfather, The (1972)",      genre: "Crime",     popularity: "moderate", quality: "high"   },
  { id: 174,  title: "Raiders of the Lost Ark",   genre: "Adventure", popularity: "popular",  quality: "high"   },
  { id: 69,   title: "Forrest Gump (1994)",        genre: "Drama",     popularity: "popular",  quality: "high"   },
  { id: 100,  title: "Fargo (1996)",               genre: "Crime",     popularity: "moderate", quality: "high"   },
  { id: 294,  title: "Liar Liar (1997)",           genre: "Comedy",    popularity: "popular",  quality: "medium" },
  { id: 286,  title: "English Patient, The",       genre: "Romance",   popularity: "moderate", quality: "high"   },
  { id: 313,  title: "Titanic (1997)",             genre: "Romance",   popularity: "popular",  quality: "high"   },
  { id: 64,   title: "Shawshank Redemption (1994)",genre: "Drama",     popularity: "popular",  quality: "high"   },
  { id: 181,  title: "Return of Jedi (1983)",      genre: "Action",    popularity: "popular",  quality: "high"   },
  { id: 25,   title: "Braveheart (1995)",          genre: "Action",    popularity: "popular",  quality: "high"   },
  { id: 132,  title: "Wizard of Oz, The (1939)",   genre: "Fantasy",   popularity: "moderate", quality: "high"   },
  { id: 11,   title: "Seven (1995)",               genre: "Thriller",  popularity: "popular",  quality: "high"   },
  { id: 7,    title: "Twelve Monkeys (1995)",      genre: "Sci-Fi",    popularity: "moderate", quality: "high"   },
  { id: 322,  title: "Sleepers (1996)",            genre: "Crime",     popularity: "moderate", quality: "medium" },
  { id: 234,  title: "Jurassic Park (1993)",       genre: "Adventure", popularity: "popular",  quality: "medium" },
  { id: 169,  title: "Wrong Trousers, The (1993)", genre: "Animation", popularity: "niche",    quality: "high"   },
  { id: 210,  title: "Indiana Jones & Temple",     genre: "Adventure", popularity: "popular",  quality: "medium" },
];

// ──────────────────────────────────────────────────────────────
// Inference Engine (browser-side BN approximation)
// ──────────────────────────────────────────────────────────────

function computeProbLike(evidence) {
  let logit = logit_from_prob(CPT_DATA.base);

  logit += logit_delta(CPT_DATA.user_activity[evidence.user_activity] || 0);
  logit += logit_delta(CPT_DATA.user_rating_bucket[evidence.user_rating_bucket] || 0);
  logit += logit_delta(CPT_DATA.item_quality[evidence.item_quality] || 0);
  logit += logit_delta(CPT_DATA.item_popularity[evidence.item_popularity] || 0);
  logit += logit_delta(CPT_DATA.item_genre[evidence.item_genre] || 0);

  // Gender-genre cross
  const genderEffect = CPT_DATA.gender_genre[evidence.user_gender] || {};
  logit += logit_delta(genderEffect[evidence.item_genre] || 0);

  return sigmoid(logit);
}

function logit_from_prob(p) { return Math.log(p / (1 - p)); }
function logit_delta(d) { return d * 4; }   // scale delta to logit space
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function computeEU(probLike) {
  return probLike * UTILITY_POS + (1 - probLike) * UTILITY_NEG;
}

function decide(eu) {
  return eu >= EU_THRESHOLD ? "RECOMMEND" : "SKIP";
}

// ──────────────────────────────────────────────────────────────
// Navigation
// ──────────────────────────────────────────────────────────────

function showSection(name) {
  document.querySelectorAll(".section").forEach(s => s.classList.add("hidden"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));

  const section = document.getElementById(`section-${name}`);
  const btn     = document.getElementById(`nav-${name}`);
  if (section) {
    section.classList.remove("hidden");
    section.style.animation = "none";
    section.offsetHeight; // reflow
    section.style.animation = "";
  }
  if (btn) btn.classList.add("active");

  // Draw charts when section becomes visible
  if (name === "network") drawFullDAG();
  if (name === "metrics") drawPrecisionCurve();
}

// ──────────────────────────────────────────────────────────────
// Decision Network Canvas — mini visualiser
// ──────────────────────────────────────────────────────────────

const DN_NODES = [
  { id: "user_activity",      label: "User\nActivity",      x: 0.12, y: 0.15, type: "user",   state: "medium" },
  { id: "user_rating_bucket", label: "Rating\nStyle",       x: 0.12, y: 0.55, type: "user",   state: "neutral" },
  { id: "user_gender",        label: "Gender",              x: 0.12, y: 0.85, type: "user",   state: "M" },
  { id: "item_genre",         label: "Genre",               x: 0.50, y: 0.75, type: "item",   state: "Action" },
  { id: "item_popularity",    label: "Popularity",          x: 0.50, y: 0.20, type: "item",   state: "moderate" },
  { id: "item_quality",       label: "Quality",             x: 0.50, y: 0.47, type: "item",   state: "medium" },
  { id: "rating_positive",    label: "Rating\nPositive",    x: 0.78, y: 0.42, type: "chance", state: "?" },
  { id: "utility",            label: "Utility\nNode",       x: 0.93, y: 0.22, type: "utility",state: "U" },
  { id: "decision",           label: "DECIDE",              x: 0.93, y: 0.68, type: "decision",state: "?" },
];

const DN_EDGES = [
  ["user_activity",      "rating_positive"],
  ["user_rating_bucket", "rating_positive"],
  ["user_gender",        "item_genre"],
  ["item_genre",         "rating_positive"],
  ["item_popularity",    "rating_positive"],
  ["item_quality",       "rating_positive"],
  ["rating_positive",    "utility"],
  ["rating_positive",    "decision"],
];

const NODE_COLORS = {
  user:     { fill: "rgba(139,92,246,0.25)", stroke: "#8b5cf6", text: "#c4b5fd" },
  item:     { fill: "rgba(6,182,212,0.2)",   stroke: "#06b6d4", text: "#67e8f9" },
  chance:   { fill: "rgba(245,158,11,0.2)",  stroke: "#f59e0b", text: "#fcd34d" },
  utility:  { fill: "rgba(16,185,129,0.2)",  stroke: "#10b981", text: "#6ee7b7" },
  decision: { fill: "rgba(244,63,94,0.2)",   stroke: "#f43f5e", text: "#fca5a5" },
};

function drawDNCanvas(probLike = null, eu = null) {
  const canvas = document.getElementById("dn-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;

  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = "rgba(8,11,20,0.6)";
  ctx.fillRect(0, 0, W, H);

  const getXY = (n) => ({ x: n.x * W, y: n.y * H });

  // Draw edges with glow
  DN_EDGES.forEach(([fromId, toId]) => {
    const from = DN_NODES.find(n => n.id === fromId);
    const to   = DN_NODES.find(n => n.id === toId);
    if (!from || !to) return;
    const fp = getXY(from), tp = getXY(to);

    ctx.save();
    ctx.shadowColor = "#8b5cf6";
    ctx.shadowBlur  = 8;
    ctx.strokeStyle = "rgba(139,92,246,0.45)";
    ctx.lineWidth   = 1.5;
    ctx.beginPath();

    // Curved arrow
    const mx = (fp.x + tp.x) / 2;
    const my = (fp.y + tp.y) / 2 - 15;
    ctx.moveTo(fp.x, fp.y);
    ctx.quadraticCurveTo(mx, my, tp.x, tp.y);
    ctx.stroke();

    // Arrowhead
    const angle = Math.atan2(tp.y - my, tp.x - mx);
    ctx.fillStyle = "rgba(139,92,246,0.7)";
    ctx.beginPath();
    ctx.moveTo(tp.x, tp.y);
    ctx.lineTo(tp.x - 9 * Math.cos(angle - 0.4), tp.y - 9 * Math.sin(angle - 0.4));
    ctx.lineTo(tp.x - 9 * Math.cos(angle + 0.4), tp.y - 9 * Math.sin(angle + 0.4));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  });

  // Draw nodes
  DN_NODES.forEach(node => {
    const { x, y } = getXY(node);
    const colors = NODE_COLORS[node.type];
    const r = node.type === "decision" ? 24 : 22;

    ctx.save();
    ctx.shadowColor = colors.stroke;
    ctx.shadowBlur  = 18;

    // Node shape: decision node = diamond
    if (node.type === "decision") {
      ctx.beginPath();
      ctx.moveTo(x, y - r);
      ctx.lineTo(x + r, y);
      ctx.lineTo(x, y + r);
      ctx.lineTo(x - r, y);
      ctx.closePath();
    } else if (node.type === "utility") {
      // Diamond with rounded look (hexagon)
      drawHexagon(ctx, x, y, r);
    } else {
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
    }

    ctx.fillStyle   = colors.fill;
    ctx.strokeStyle = colors.stroke;
    ctx.lineWidth   = 2;
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // Label
    ctx.fillStyle  = colors.text;
    ctx.font       = "bold 9px Inter, sans-serif";
    ctx.textAlign  = "center";
    ctx.textBaseline = "middle";
    const lines = node.label.split("\n");
    lines.forEach((line, i) => {
      ctx.fillText(line, x, y + (i - (lines.length - 1) / 2) * 10);
    });
  });

  // Highlight rating_positive node with probability if available
  if (probLike !== null) {
    const rp = DN_NODES.find(n => n.id === "rating_positive");
    const { x, y } = getXY(rp);
    ctx.fillStyle = "rgba(245,158,11,0.9)";
    ctx.font = "bold 10px JetBrains Mono, monospace";
    ctx.textAlign = "center";
    ctx.fillText(`${(probLike * 100).toFixed(0)}%`, x, y + 32);
  }
}

function drawHexagon(ctx, cx, cy, r) {
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i - Math.PI / 6;
    const px = cx + r * Math.cos(angle);
    const py = cy + r * Math.sin(angle);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.closePath();
}

// ──────────────────────────────────────────────────────────────
// Full DAG Canvas (Network section)
// ──────────────────────────────────────────────────────────────

function drawFullDAG() {
  const canvas = document.getElementById("dag-full-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Background gradient
  const grad = ctx.createLinearGradient(0, 0, W, H);
  grad.addColorStop(0, "rgba(15,21,37,0.8)");
  grad.addColorStop(1, "rgba(8,11,20,0.8)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  const FULL_NODES = [
    { id: "user_activity",      label: "user_activity",      x: 0.08, y: 0.20, type: "user" },
    { id: "user_rating_bucket", label: "user_rating_bucket", x: 0.08, y: 0.50, type: "user" },
    { id: "user_gender",        label: "user_gender",        x: 0.08, y: 0.80, type: "user" },
    { id: "item_genre",         label: "item_genre",         x: 0.40, y: 0.72, type: "item" },
    { id: "item_popularity",    label: "item_popularity",    x: 0.40, y: 0.22, type: "item" },
    { id: "item_quality",       label: "item_quality",       x: 0.40, y: 0.50, type: "item" },
    { id: "rating_positive",    label: "rating_positive",    x: 0.68, y: 0.43, type: "chance" },
    { id: "utility",            label: "Utility(U)",         x: 0.88, y: 0.20, type: "utility" },
    { id: "decision",           label: "DECIDE",             x: 0.88, y: 0.70, type: "decision" },
  ];

  const getXY = n => ({ x: n.x * W, y: n.y * H });

  DN_EDGES.forEach(([fromId, toId]) => {
    const from = FULL_NODES.find(n => n.id === fromId);
    const to   = FULL_NODES.find(n => n.id === toId);
    if (!from || !to) return;
    const fp = getXY(from), tp = getXY(to);

    ctx.save();
    ctx.shadowColor = "#8b5cf6";
    ctx.shadowBlur  = 12;
    ctx.strokeStyle = "rgba(139,92,246,0.5)";
    ctx.lineWidth   = 2;
    ctx.setLineDash([]);
    ctx.beginPath();
    const cp = { x: (fp.x + tp.x) / 2, y: (fp.y + tp.y) / 2 - 20 };
    ctx.moveTo(fp.x, fp.y);
    ctx.quadraticCurveTo(cp.x, cp.y, tp.x, tp.y);
    ctx.stroke();

    // Arrow
    const angle = Math.atan2(tp.y - cp.y, tp.x - cp.x);
    ctx.fillStyle = "#8b5cf6";
    ctx.beginPath();
    ctx.moveTo(tp.x, tp.y);
    ctx.lineTo(tp.x - 11 * Math.cos(angle - 0.35), tp.y - 11 * Math.sin(angle - 0.35));
    ctx.lineTo(tp.x - 11 * Math.cos(angle + 0.35), tp.y - 11 * Math.sin(angle + 0.35));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  });

  FULL_NODES.forEach(node => {
    const { x, y } = getXY(node);
    const colors = NODE_COLORS[node.type];
    const rx = 68, ry = 20;

    ctx.save();
    ctx.shadowColor = colors.stroke;
    ctx.shadowBlur  = 20;

    if (node.type === "decision" || node.type === "utility") {
      // Rectangle with rounded corners
      roundRect(ctx, x - rx, y - ry, rx * 2, ry * 2, 8);
    } else {
      ctx.beginPath();
      ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
    }
    ctx.fillStyle   = colors.fill;
    ctx.strokeStyle = colors.stroke;
    ctx.lineWidth   = 2;
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle   = colors.text;
    ctx.font        = "600 11px JetBrains Mono, monospace";
    ctx.textAlign   = "center";
    ctx.textBaseline= "middle";
    ctx.fillText(node.label, x, y);

    // Type badge
    const typeLabel = { user:"Chance", item:"Chance", chance:"Chance", utility:"Utility", decision:"Decision" }[node.type];
    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "500 9px Inter, sans-serif";
    ctx.fillText(typeLabel, x, y + 28);
  });
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

// ──────────────────────────────────────────────────────────────
// Gauge Chart
// ──────────────────────────────────────────────────────────────

function drawGauge(prob) {
  const canvas = document.getElementById("gauge-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H - 10, r = 80;
  const startAngle = Math.PI, sweepAngle = Math.PI;

  // Background arc
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, startAngle + sweepAngle);
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth   = 14;
  ctx.lineCap     = "round";
  ctx.stroke();

  // Color gradient based on prob
  const color = prob > 0.6 ? "#10b981" : prob > 0.4 ? "#f59e0b" : "#f43f5e";

  // Value arc
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, startAngle + sweepAngle * prob);
  ctx.strokeStyle = color;
  ctx.lineWidth   = 14;
  ctx.lineCap     = "round";
  ctx.shadowColor = color;
  ctx.shadowBlur  = 15;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Needle
  const angle = Math.PI + Math.PI * prob;
  const nx = cx + (r - 5) * Math.cos(angle);
  const ny = cy + (r - 5) * Math.sin(angle);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(nx, ny);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth   = 2;
  ctx.stroke();

  // Center dot
  ctx.beginPath();
  ctx.arc(cx, cy, 5, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();

  // Min/max labels
  ctx.fillStyle  = "rgba(139,96,176,0.7)";
  ctx.font       = "500 10px Inter, sans-serif";
  ctx.textAlign  = "left";
  ctx.fillText("0%", cx - r - 10, cy + 16);
  ctx.textAlign  = "right";
  ctx.fillText("100%", cx + r + 10, cy + 16);

  // Value display
  document.getElementById("gauge-value").textContent = `${(prob * 100).toFixed(1)}%`;
  document.getElementById("gauge-value").style.color = color;
}

// ──────────────────────────────────────────────────────────────
// Precision@K Curve (metrics section)
// ──────────────────────────────────────────────────────────────

function drawPrecisionCurve() {
  const canvas = document.getElementById("prec-curve-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Simulated P@K values for K = 1..20
  const kValues = [1,2,3,4,5,6,7,8,9,10,12,15,20];
  const dnPrec  = [0.61,0.55,0.50,0.46,0.43,0.40,0.38,0.36,0.34,0.312,0.29,0.26,0.22];
  const svdPrec = [0.68,0.62,0.57,0.53,0.49,0.46,0.43,0.40,0.38,0.348,0.32,0.29,0.24];
  const popPrec = [0.38,0.35,0.31,0.28,0.25,0.23,0.21,0.20,0.19,0.187,0.17,0.15,0.13];

  const pad = { t:20, r:20, b:40, l:50 };
  const W2 = W - pad.l - pad.r;
  const H2 = H - pad.t - pad.b;

  const toXY = (k, p) => ({
    x: pad.l + (k - 1) / (kValues[kValues.length-1] - 1) * W2,
    y: pad.t + (1 - p / 0.75) * H2,
  });

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth   = 1;
  [0,0.25,0.50,0.75].forEach(p => {
    const y = pad.t + (1 - p / 0.75) * H2;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillStyle = "rgba(139,96,176,0.5)";
    ctx.font      = "9px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText((p * 100).toFixed(0) + "%", pad.l - 5, y + 3);
  });

  // Axes labels
  ctx.fillStyle = "rgba(139,150,176,0.7)";
  ctx.font      = "10px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("K (cut-off)", pad.l + W2 / 2, H - 5);

  // Draw line helper
  function drawLine(data, color, dash=[]) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.setLineDash(dash);
    ctx.shadowColor = color;
    ctx.shadowBlur  = 8;
    ctx.beginPath();
    kValues.forEach((k, i) => {
      const pt = toXY(k, data[i]);
      i === 0 ? ctx.moveTo(pt.x, pt.y) : ctx.lineTo(pt.x, pt.y);
    });
    ctx.stroke();
    // Dots
    kValues.forEach((k, i) => {
      const pt = toXY(k, data[i]);
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    });
    ctx.restore();
  }

  drawLine(dnPrec,  "#8b5cf6");
  drawLine(svdPrec, "#06b6d4", [4, 3]);
  drawLine(popPrec, "#f59e0b", [2, 4]);

  // Legend
  const legend = [
    { label: "Decision Network", color: "#8b5cf6", dash: [] },
    { label: "SVD",             color: "#06b6d4", dash: [4,3] },
    { label: "Popularity",      color: "#f59e0b", dash: [2,4] },
  ];
  legend.forEach(({ label, color, dash }, i) => {
    const lx = pad.l + 5, ly = pad.t + 10 + i * 18;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 20, ly); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = "rgba(240,244,255,0.8)";
    ctx.font      = "10px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(label, lx + 26, ly + 3);
  });
}

// ──────────────────────────────────────────────────────────────
// Generate Recommendations Grid
// ──────────────────────────────────────────────────────────────

function generateRecsGrid(evidence) {
  const grid = document.getElementById("recs-grid");
  grid.innerHTML = "";

  const scored = MOVIES.map(movie => {
    const ev = Object.assign({}, evidence, {
      item_genre:      movie.genre,
      item_popularity: movie.popularity,
      item_quality:    movie.quality,
    });
    const prob = computeProbLike(ev);
    const eu   = computeEU(prob);
    const dec  = decide(eu);
    return { ...movie, prob, eu, dec };
  }).sort((a, b) => b.eu - a.eu);

  scored.forEach((movie, idx) => {
    const card = document.createElement("div");
    card.className = "rec-card";
    card.style.animationDelay = `${idx * 0.04}s`;
    card.innerHTML = `
      <div class="rec-rank">#${idx + 1}</div>
      <span class="rec-decision-badge ${movie.dec === 'RECOMMEND' ? 'badge-recommend' : 'badge-skip'}">
        ${movie.dec === 'RECOMMEND' ? '✓ REC' : '✗ SKIP'}
      </span>
      <div class="rec-title">${movie.title}</div>
      <div class="rec-genre">${movie.genre}</div>
      <div class="rec-scores">
        <span class="rec-score-pill pill-prob">P=${(movie.prob*100).toFixed(0)}%</span>
        <span class="rec-score-pill pill-eu">EU=${movie.eu.toFixed(3)}</span>
      </div>
    `;
    card.onclick = () => explainCard(movie, evidence);
    grid.appendChild(card);
  });
}

function explainCard(movie, evidence) {
  const ev = Object.assign({}, evidence, {
    item_genre:      movie.genre,
    item_popularity: movie.popularity,
    item_quality:    movie.quality,
  });
  const prob = computeProbLike(ev);
  const eu   = computeEU(prob);
  const dec  = decide(eu);
  updateResultDisplay(prob, eu, dec, ev, movie.title);
  window.scrollTo({ top: document.getElementById("result-row").offsetTop - 80, behavior: "smooth" });
}

// ──────────────────────────────────────────────────────────────
// Run Inference
// ──────────────────────────────────────────────────────────────

function runInference() {
  const evidence = {
    user_activity:      document.getElementById("select-activity").value,
    user_rating_bucket: document.getElementById("select-rating-style").value,
    user_gender:        document.getElementById("select-gender").value,
    item_genre:         document.getElementById("select-genre").value,
    item_popularity:    document.getElementById("select-popularity").value,
    item_quality:       document.getElementById("select-quality").value,
  };

  const prob = computeProbLike(evidence);
  const eu   = computeEU(prob);
  const dec  = decide(eu);

  updateResultDisplay(prob, eu, dec, evidence);
  generateRecsGrid(evidence);
  drawDNCanvas(prob, eu);

  // Animate EU display
  document.getElementById("eu-value").textContent = eu.toFixed(3);
  document.getElementById("eu-decision").textContent =
    `${dec === "RECOMMEND" ? "✅ Recommend" : "❌ Skip"} (θ = ${EU_THRESHOLD})`;

  // Update user stat chips
  document.getElementById("stat-activity").textContent =
    evidence.user_activity.charAt(0).toUpperCase() + evidence.user_activity.slice(1);
  document.getElementById("stat-style").textContent =
    evidence.user_rating_bucket.charAt(0).toUpperCase() + evidence.user_rating_bucket.slice(1);
}

function updateResultDisplay(prob, eu, dec, evidence, title = null) {
  // Gauge
  drawGauge(prob);

  // Decision box
  const icon  = document.getElementById("decision-icon");
  const label = document.getElementById("decision-label");
  const euEl  = document.getElementById("decision-eu");
  icon.textContent  = dec === "RECOMMEND" ? "✅" : "❌";
  label.textContent = dec;
  label.className   = `decision-label decision-${dec.toLowerCase()}`;
  euEl.textContent  = `EU = ${eu.toFixed(3)}`;

  // Explanation
  const expBox = document.getElementById("explanation-text");
  const parts  = [];

  if (title) parts.push(`<strong>"${title}"</strong><br/>`);
  if (evidence.user_activity === "high")       parts.push("You are an <strong>active rater</strong> — your preferences are well-characterised.");
  if (evidence.user_rating_bucket === "generous") parts.push("You tend to <strong>rate generously</strong>, increasing the prior likelihood of enjoying any film.");
  if (evidence.user_rating_bucket === "harsh")    parts.push("You are a <strong>selective critic</strong> — only high-quality films pass your threshold.");

  parts.push(`The item is a <strong>${evidence.item_quality}-quality</strong>, <strong>${evidence.item_popularity}</strong> <strong>${evidence.item_genre}</strong> film.`);
  parts.push(`<br/><strong>P(like | evidence) = ${(prob*100).toFixed(1)}%</strong>`);
  parts.push(`<strong>EU(recommend) = P·(+1.0) + (1−P)·(−0.5) = ${eu.toFixed(3)}</strong>`);
  parts.push(eu >= EU_THRESHOLD
    ? `EU ≥ θ (${EU_THRESHOLD}) → <span style="color:#10b981">RECOMMEND ✓</span>`
    : `EU < θ (${EU_THRESHOLD}) → <span style="color:#f43f5e">SKIP ✗</span>`);

  expBox.innerHTML = parts.join(" ");

  // CPT factor chips
  const factorBox = document.getElementById("cpt-factors");
  factorBox.innerHTML = "";
  const factors = [
    `Activity: ${evidence.user_activity}`,
    `Style: ${evidence.user_rating_bucket}`,
    `Genre: ${evidence.item_genre}`,
    `Popularity: ${evidence.item_popularity}`,
    `Quality: ${evidence.item_quality}`,
  ];
  factors.forEach(f => {
    const span = document.createElement("span");
    span.className = "cpt-factor";
    span.textContent = f;
    factorBox.appendChild(span);
  });
}

// ──────────────────────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────────────────────

window.addEventListener("DOMContentLoaded", () => {
  // Draw initial network
  drawDNCanvas();

  // Fire default inference to populate grid
  setTimeout(runInference, 300);

  // Animate metric bars
  setTimeout(() => {
    document.querySelectorAll(".mhc-fill").forEach(el => {
      const w = el.style.width;
      el.style.width = "0%";
      setTimeout(() => { el.style.width = w; }, 100);
    });
  }, 200);
});
