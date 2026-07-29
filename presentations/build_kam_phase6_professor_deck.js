#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const ASSET_DIR = path.join(ROOT, "kam_phase6_professor_assets");
const OUTPUT = path.join(ROOT, "KAM_Phase6_Professor_Presentation.pptx");
const W = 1600;
const H = 900;

const C = {
  navy: "#132238",
  blue: "#2B6CB0",
  blueDark: "#1E4E82",
  blueLight: "#DCEAF7",
  amber: "#D69E2E",
  amberLight: "#F7E8BF",
  paper: "#F7F5EF",
  white: "#FFFFFF",
  gray: "#6B7280",
  gray2: "#98A1AE",
  line: "#D9DEE6",
  pale: "#ECEFF3",
  ink2: "#334155",
};

function esc(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function rect(x, y, w, h, fill, rx = 18, stroke = "none", sw = 0) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
}

function line(x1, y1, x2, y2, stroke = C.line, sw = 3, dash = "") {
  const d = dash ? ` stroke-dasharray="${dash}"` : "";
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${sw}" stroke-linecap="round"${d}/>`;
}

function circle(cx, cy, r, fill, stroke = "none", sw = 0) {
  return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
}

function arrow(x1, y1, x2, y2, color = C.navy, sw = 4) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const a = 13;
  const p1x = x2 - a * Math.cos(angle - Math.PI / 6);
  const p1y = y2 - a * Math.sin(angle - Math.PI / 6);
  const p2x = x2 - a * Math.cos(angle + Math.PI / 6);
  const p2y = y2 - a * Math.sin(angle + Math.PI / 6);
  return `${line(x1, y1, x2, y2, color, sw)}<polygon points="${x2},${y2} ${p1x},${p1y} ${p2x},${p2y}" fill="${color}"/>`;
}

function text(x, y, lines, size = 30, color = C.navy, weight = 400, opts = {}) {
  const {
    anchor = "start",
    lineHeight = Math.round(size * 1.22),
    family = "DejaVu Sans",
    italic = false,
    letterSpacing = 0,
  } = opts;
  const arr = Array.isArray(lines) ? lines : [lines];
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" fill="${color}" font-family="${family}" font-size="${size}" font-weight="${weight}"${italic ? ' font-style="italic"' : ""}${letterSpacing ? ` letter-spacing="${letterSpacing}"` : ""}>${arr
    .map((v, i) => `<tspan x="${x}" dy="${i === 0 ? 0 : lineHeight}">${esc(v)}</tspan>`)
    .join("")}</text>`;
}

function pill(x, y, w, label, fill = C.blueLight, color = C.blueDark) {
  return `${rect(x, y, w, 38, fill, 19)}${text(x + w / 2, y + 26, label, 18, color, 700, { anchor: "middle", letterSpacing: 0.5 })}`;
}

function header(slideNo, title, kicker = "KAM · PHASE VI") {
  return [
    text(70, 54, kicker, 17, C.blueDark, 700, { letterSpacing: 1.6 }),
    text(70, 112, title, 42, C.navy, 700),
    line(70, 142, 1530, 142, C.line, 2),
    text(1530, 54, String(slideNo).padStart(2, "0"), 18, C.gray, 700, { anchor: "end" }),
  ].join("");
}

function footer(note = "") {
  return `${note ? text(70, 853, note, 16, C.gray, 400) : ""}${text(1530, 853, "Sparse Separable Memory", 15, C.gray2, 500, { anchor: "end" })}`;
}

function svg(body) {
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  ${rect(0, 0, W, H, C.paper, 0)}
  ${body}
</svg>`;
}

function slide1() {
  let b = "";
  b += text(76, 82, "KERNEL-ADAPTIVE MEMORY · PHASE VI", 19, C.blueDark, 700, { letterSpacing: 1.7 });
  b += text(76, 235, ["Sparse Separable", "Memory for", "Sequence Models"], 72, C.navy, 760, { lineHeight: 82 });
  b += text(80, 535, ["Can a sequence model gain large, adaptable memory", "without activating all of it for every token?"], 29, C.ink2, 400, { lineHeight: 42 });
  b += pill(80, 650, 290, "PROFESSOR DISCUSSION DECK", C.amberLight, "#7A5713");
  b += text(80, 730, "July 2026", 22, C.gray, 500);
  b += rect(910, 105, 570, 650, C.white, 36, C.line, 2);
  const selected = new Set(["1-3", "3-2", "4-4"]);
  const nodes = [];
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 5; c++) {
      const key = `${r}-${c}`;
      const cx = 985 + c * 100;
      const cy = 180 + r * 95;
      const isSel = selected.has(key);
      nodes.push(circle(cx, cy, isSel ? 20 : 12, isSel ? C.blue : C.pale, isSel ? C.blueDark : C.line, isSel ? 4 : 2));
    }
  }
  b += line(1045, 320, 1285, 465, C.blue, 5);
  b += line(1285, 465, 1385, 655, C.blue, 5);
  b += line(1045, 320, 1185, 560, C.blue, 5);
  b += nodes.join("");
  b += text(1195, 790, "Large memory · small active path", 20, C.blueDark, 700, { anchor: "middle" });
  return svg(b);
}

function denseGrid(x, y, sparse = false) {
  let b = "";
  const cols = sparse ? 8 : 6;
  const rows = sparse ? 6 : 6;
  const gap = sparse ? 29 : 35;
  const selected = new Set(["1-5", "3-2", "4-6"]);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const key = `${r}-${c}`;
      const active = !sparse || selected.has(key);
      b += rect(x + c * gap, y + r * gap, sparse ? 18 : 22, sparse ? 18 : 22, active ? C.blue : C.pale, 4, active ? C.blueDark : C.line, 1.5);
    }
  }
  return b;
}

function slide2() {
  let b = header(2, "The capacity–computation tension");
  b += rect(70, 190, 690, 480, C.white, 28, C.line, 2);
  b += pill(105, 220, 155, "DENSE PATH", C.pale, C.navy);
  b += text(105, 305, ["Every token touches", "the same large block."], 34, C.navy, 700, { lineHeight: 43 });
  b += denseGrid(108, 425, false);
  b += arrow(360, 522, 500, 522, C.gray, 4);
  b += text(540, 475, ["Stored", "capacity", "≈", "active", "work"], 22, C.gray, 600, { lineHeight: 29 });
  b += rect(800, 190, 730, 480, C.white, 28, C.blueLight, 3);
  b += pill(835, 220, 175, "SPARSE MEMORY", C.blueLight, C.blueDark);
  b += text(835, 305, ["Every token consults", "only a few entries."], 34, C.navy, 700, { lineHeight: 43 });
  b += denseGrid(840, 420, true);
  b += arrow(1110, 522, 1230, 522, C.blue, 4);
  b += text(1265, 475, ["Large", "potential", "memory", "+", "small", "active path"], 21, C.blueDark, 650, { lineHeight: 27 });
  b += rect(170, 720, 1260, 80, C.navy, 20);
  b += text(800, 772, "Decouple how much the model can store from how much it must activate.", 29, C.white, 650, { anchor: "middle" });
  b += footer("Attention handles context; sparse memory supplies selectively accessed capacity.");
  return svg(b);
}

function slide3() {
  let b = header(3, "The core intuition: route, retrieve, blend");
  const cards = [
    { x: 80, title: "ROUTE", subtitle: ["Find the relevant", "neighborhood"], fill: C.blueLight, icon: "route" },
    { x: 475, title: "RETRIEVE", subtitle: ["Combine a few", "local experts"], fill: C.white, icon: "retrieve" },
    { x: 870, title: "BLEND", subtitle: ["Add memory through", "a conservative gate"], fill: C.white, icon: "blend" },
    { x: 1265, title: "OUTPUT", subtitle: ["Context + selected", "reusable capacity"], fill: C.amberLight, icon: "output" },
  ];
  cards.forEach((c, i) => {
    b += rect(c.x, 235, 275, 390, c.fill, 28, i === 0 ? C.blue : C.line, i === 0 ? 3 : 2);
    b += text(c.x + 28, 282, c.title, 19, i === 3 ? "#7A5713" : C.blueDark, 750, { letterSpacing: 1.2 });
    if (c.icon === "route") {
      for (let r = 0; r < 4; r++) for (let col = 0; col < 5; col++) b += circle(c.x + 52 + col * 38, 355 + r * 38, 8, (r === 1 && col === 3) || (r === 2 && col === 1) ? C.blue : C.pale);
    } else if (c.icon === "retrieve") {
      b += rect(c.x + 45, 340, 52, 90, C.pale, 8);
      b += rect(c.x + 112, 315, 52, 115, C.blue, 8);
      b += rect(c.x + 179, 355, 52, 75, C.blueLight, 8);
    } else if (c.icon === "blend") {
      b += line(c.x + 58, 365, c.x + 205, 365, C.gray, 6);
      b += line(c.x + 58, 425, c.x + 205, 425, C.blue, 6);
      b += circle(c.x + 132, 395, 26, C.white, C.navy, 3);
      b += text(c.x + 132, 403, "+", 26, C.navy, 700, { anchor: "middle" });
    } else {
      b += circle(c.x + 138, 390, 72, C.amber, C.white, 5);
      b += text(c.x + 138, 402, "h⁺", 36, C.white, 700, { anchor: "middle" });
    }
    b += text(c.x + 28, 515, c.subtitle, 24, C.navy, 650, { lineHeight: 33 });
    if (i < cards.length - 1) b += arrow(c.x + 300, 430, cards[i + 1].x - 25, 430, C.navy, 4);
  });
  b += rect(310, 700, 980, 78, C.navy, 20);
  b += text(800, 751, "Large potential memory  ·  small active path  ·  safe residual entry", 27, C.white, 650, { anchor: "middle" });
  b += footer("The gate starts near zero, so the memory path must earn influence.");
  return svg(b);
}

function slide4() {
  let b = header(4, "One technical slide: sparse separable KAM", "MINIMAL FORMAL VIEW");
  b += rect(70, 190, 690, 520, C.white, 28, C.line, 2);
  b += text(105, 240, "THREE LINES OF NOTATION", 18, C.blueDark, 750, { letterSpacing: 1.2 });
  const equations = [
    ["1", "I(z) = TopK(similarity(z, keys))"],
    ["2", "memory(u) = Σ selected weight × local expert(u)"],
    ["3", "output = Transformer path + small FFN + gate × memory"],
  ];
  equations.forEach((e, i) => {
    const y = 290 + i * 120;
    b += circle(125, y + 16, 22, i === 2 ? C.amber : C.blue, C.white, 3);
    b += text(125, y + 24, e[0], 18, C.white, 700, { anchor: "middle" });
    b += text(170, y + 24, e[1], i === 1 ? 21 : 23, C.navy, 600, { family: "DejaVu Sans Mono" });
    if (i < 2) b += line(110, y + 72, 710, y + 72, C.pale, 2);
  });
  b += rect(800, 190, 730, 520, C.blueLight, 28, C.blue, 3);
  b += text(835, 240, "WHY “SEPARABLE”?", 18, C.blueDark, 750, { letterSpacing: 1.2 });
  b += rect(850, 300, 245, 145, C.white, 20);
  b += text(972, 345, "GEOMETRY", 18, C.blueDark, 750, { anchor: "middle" });
  b += text(972, 390, ["Where should", "the model look?"], 23, C.navy, 650, { anchor: "middle", lineHeight: 30 });
  b += rect(1235, 300, 245, 145, C.white, 20);
  b += text(1357, 345, "ALGEBRA", 18, "#7A5713", 750, { anchor: "middle" });
  b += text(1357, 390, ["What should it", "return there?"], 23, C.navy, 650, { anchor: "middle", lineHeight: 30 });
  b += arrow(1110, 372, 1218, 372, C.navy, 4);
  b += text(1164, 520, ["Different objects", "can learn on", "different timescales."], 31, C.navy, 720, { anchor: "middle", lineHeight: 40 });
  b += rect(70, 745, 1460, 72, C.navy, 18);
  b += text(105, 790, "FAIRNESS", 18, C.amber, 750, { letterSpacing: 1 });
  b += text(255, 790, "Same data + token budget  ·  matched total parameters  ·  active parameters/FLOPs reported  ·  strong controls", 20, C.white, 550);
  b += footer();
  return svg(b);
}

function slide5() {
  let b = header(5, "Two timescales: learn the map, then stabilize it");
  const x0 = 110;
  const x1 = 1450;
  const freezeX = x0 + 0.8 * (x1 - x0);
  b += text(110, 205, "TRAINING PROGRESS", 18, C.gray, 700, { letterSpacing: 1.2 });
  b += line(x0, 315, x1, 315, C.line, 8);
  const stages = [
    { x: 150, label: "ADAPT", sub: "keys + experts learn" },
    { x: 475, label: "STABILIZE", sub: "geometry slows" },
    { x: freezeX, label: "FREEZE", sub: "≈ 80% of tokens" },
    { x: 1310, label: "FINAL TUNE", sub: "stable routing" },
  ];
  stages.forEach((s, i) => {
    b += circle(s.x, 315, 18, i === 2 ? C.amber : C.blue, C.paper, 5);
    b += text(s.x, 365, s.label, 19, i === 2 ? "#7A5713" : C.blueDark, 750, { anchor: "middle" });
    b += text(s.x, 397, s.sub, 16, C.gray, 500, { anchor: "middle" });
  });
  b += line(freezeX, 245, freezeX, 680, C.amber, 3, "9 8");
  b += pill(freezeX - 70, 220, 140, "LOCKED MAP", C.amberLight, "#7A5713");
  b += text(115, 480, "GEOMETRY", 18, C.blueDark, 750, { letterSpacing: 1 });
  b += `<path d="M 265 515 C 430 430, 650 475, 820 500 S 1040 512, ${freezeX} 512 L 1460 512" fill="none" stroke="${C.blue}" stroke-width="7" stroke-linecap="round"/>`;
  b += text(1470, 520, "frozen", 16, C.blueDark, 650);
  b += text(115, 610, "ALGEBRA", 18, "#7A5713", 750, { letterSpacing: 1 });
  b += `<path d="M 265 645 C 430 555, 635 590, 800 615 S 1080 600, 1230 625 S 1400 590, 1460 605" fill="none" stroke="${C.amber}" stroke-width="7" stroke-linecap="round"/>`;
  b += rect(1010, 705, 455, 90, C.white, 18, C.line, 2);
  b += text(1035, 740, "Library analogy", 17, C.gray, 700);
  b += text(1035, 773, "shelving stabilizes; contents keep improving", 20, C.navy, 600);
  b += rect(105, 705, 820, 90, C.navy, 18);
  b += text(135, 742, "Lifecycle test", 17, C.amber, 750);
  b += text(135, 775, "pre-freeze learning + post-freeze drift ≤ 1e-10", 22, C.white, 600);
  b += footer("Fixed-key success does not prove that learned geometry helps.");
  return svg(b);
}

function branchCard(x, y, w, title, lines, accent = C.blue) {
  return `${rect(x, y, w, 150, C.white, 22, C.line, 2)}${rect(x, y, 8, 150, accent, 4)}${text(x + 28, y + 42, title, 18, accent === C.amber ? "#7A5713" : C.blueDark, 750, { letterSpacing: 0.7 })}${text(x + 28, y + 78, lines, 20, C.navy, 550, { lineHeight: 28 })}`;
}

function slide6() {
  let b = header(6, "Current method and the research branches");
  b += rect(610, 315, 380, 205, C.navy, 28);
  b += text(800, 365, "CURRENT CANDIDATE", 17, C.amber, 750, { anchor: "middle", letterSpacing: 1 });
  b += text(800, 425, "T-KAM-F", 42, C.white, 760, { anchor: "middle" });
  b += text(800, 472, ["fixed / data-centered keys", "sparse top-K · local experts · gate"], 19, C.pale, 500, { anchor: "middle", lineHeight: 28 });
  b += branchCard(70, 200, 410, "LEARNED GEOMETRY", ["Can the support map improve", "without becoming unstable?"]);
  b += branchCard(70, 580, 410, "ONLINE ADAPTATION", ["Can values or experts recover", "quickly after regime change?"], C.amber);
  b += branchCard(1120, 200, 410, "ALTERNATING OPTIMIZATION", ["Fast algebra; conservative", "geometry updates."]);
  b += branchCard(1120, 580, 410, "DUAL MEMORY", ["Slow persistent bank +", "small recent episodic bank."], C.amber);
  b += branchCard(595, 615, 410, "EFFICIENT ROUTING", ["Exact, chunked, product-key", "lookup at larger bank sizes."]);
  b += line(480, 275, 610, 365, C.line, 3);
  b += line(480, 655, 610, 480, C.line, 3);
  b += line(990, 365, 1120, 275, C.line, 3);
  b += line(990, 480, 1120, 655, C.line, 3);
  b += line(800, 520, 800, 615, C.line, 3);
  b += rect(285, 805, 1030, 46, C.amberLight, 16);
  b += text(800, 836, "Evidence is branch-specific: a fixed-key win cannot validate the other hypotheses.", 20, "#704F12", 650, { anchor: "middle" });
  return svg(b);
}

function metricCard(x, value, label, sub, accent = C.blue) {
  return `${rect(x, 205, 335, 205, C.white, 24, C.line, 2)}${text(x + 28, 285, value, 54, accent, 780)}${text(x + 28, 330, label, 20, C.navy, 700)}${text(x + 28, 372, sub, 17, C.gray, 500)}`;
}

function slide7() {
  let b = header(7, "What would count as convincing evidence?");
  b += metricCard(70, "156", "fixed experiment rows", "four NVIDIA L4 GPUs");
  b += metricCard(435, "30", "primary paired seeds", "TinyStories · KAM-F vs T-WIDE");
  b += metricCard(800, "24", "replication pairs", "Tiny Shakespeare");
  b += metricCard(1165, "50M", "training tokens / row", "held-out endpoint", C.amber);
  b += text(70, 485, "A prespecified evidence gate", 25, C.navy, 700);
  const gates = [
    { x: 90, label: "FAIR", sub: "same data + budget" },
    { x: 395, label: "HELD-OUT", sub: "test endpoint" },
    { x: 700, label: "PAIRED", sub: "matched seeds" },
    { x: 1005, label: "REPLICATED", sub: "second corpus" },
  ];
  gates.forEach((g, i) => {
    b += rect(g.x, 550, 250, 130, i === 3 ? C.blue : C.white, 22, i === 3 ? C.blueDark : C.line, 2);
    b += text(g.x + 125, 600, g.label, 21, i === 3 ? C.white : C.blueDark, 760, { anchor: "middle" });
    b += text(g.x + 125, 640, g.sub, 17, i === 3 ? C.blueLight : C.gray, 500, { anchor: "middle" });
    if (i < gates.length - 1) b += arrow(g.x + 260, 615, gates[i + 1].x - 10, 615, C.navy, 4);
  });
  b += rect(1270, 550, 260, 130, C.amberLight, 22);
  b += text(1400, 597, "2% MARGIN", 21, "#7A5713", 760, { anchor: "middle" });
  b += text(1400, 635, ["confidence interval must", "clear the threshold"], 16, "#704F12", 550, { anchor: "middle", lineHeight: 23 });
  b += rect(175, 735, 1250, 70, C.navy, 18);
  b += text(800, 780, "Primary + replication must both pass  ·  no optional stopping  ·  no seed replacement", 23, C.white, 600, { anchor: "middle" });
  b += footer();
  return svg(b);
}

function slide8() {
  let b = header(8, "Intermediate evidence: promising direction, not a conclusion");
  b += rect(70, 185, 730, 560, C.white, 28, C.line, 2);
  b += text(105, 232, "Pilot validation loss at registered checkpoint", 23, C.navy, 700);
  b += text(105, 265, "Lower is better · n = 3 paired seeds", 17, C.gray, 500);
  const chartX = 145;
  const chartY = 630;
  const chartH = 285;
  const maxV = 3.2;
  b += line(chartX, chartY - chartH, chartX, chartY, C.gray2, 2);
  b += line(chartX, chartY, 700, chartY, C.gray2, 2);
  [0, 1, 2, 3].forEach((v) => {
    const yy = chartY - (v / maxV) * chartH;
    b += line(chartX, yy, 700, yy, C.pale, 2);
    b += text(chartX - 16, yy + 6, String(v), 16, C.gray, 500, { anchor: "end" });
  });
  const bars = [
    { x: 250, name: "T-KAM-F", value: 2.0081, color: C.blue },
    { x: 500, name: "T-WIDE", value: 2.8833, color: C.gray2 },
  ];
  bars.forEach((bar) => {
    const h = (bar.value / maxV) * chartH;
    b += rect(bar.x, chartY - h, 145, h, bar.color, 10);
    b += text(bar.x + 72, chartY - h - 18, bar.value.toFixed(4), 21, C.navy, 750, { anchor: "middle" });
    b += text(bar.x + 72, chartY + 34, bar.name, 18, C.navy, 700, { anchor: "middle" });
  });
  b += pill(535, 285, 205, "30.4% LOWER", C.blueLight, C.blueDark);
  b += rect(105, 690, 660, 92, C.amberLight, 18);
  b += text(130, 724, "Directional only", 18, "#7A5713", 750);
  b += text(130, 758, "exact p = 0.25  ·  Holm-adjusted p = 0.75", 19, "#704F12", 550);
  b += rect(840, 185, 690, 560, C.white, 28, C.line, 2);
  b += text(875, 232, "Confirmation campaign status", 23, C.navy, 700);
  b += text(875, 265, "July 28, 2026 · 6:40 PM EDT", 17, C.gray, 500);
  b += text(875, 360, "43 / 156", 62, C.blue, 780);
  b += text(875, 405, "complete", 22, C.navy, 700);
  const px = 875;
  const py = 455;
  const pw = 585;
  const doneW = pw * (43 / 156);
  const runW = pw * (4 / 156);
  b += rect(px, py, pw, 48, C.pale, 20);
  b += `<path d="M ${px + 20} ${py} H ${px + doneW} V ${py + 48} H ${px + 20} A 20 20 0 0 1 ${px} ${py + 28} V ${py + 20} A 20 20 0 0 1 ${px + 20} ${py}" fill="${C.blue}"/>`;
  b += rect(px + doneW, py, Math.max(runW, 16), 48, C.amber, 0);
  b += text(875, 545, "43 complete", 18, C.blueDark, 700);
  b += text(1035, 545, "4 running", 18, "#7A5713", 700);
  b += text(1175, 545, "109 pending", 18, C.gray, 700);
  b += text(875, 615, "0 failures", 34, C.navy, 760);
  b += text(875, 670, ["Partial confirmation effects remain uninspected", "until the complete fixed sample finishes."], 20, C.gray, 550, { lineHeight: 29 });
  b += rect(240, 785, 1120, 58, C.navy, 16);
  b += text(800, 823, "Promising enough to confirm; not strong enough to conclude.", 25, C.white, 650, { anchor: "middle" });
  return svg(b);
}

function slide9() {
  let b = header(9, "The decision after confirmation");
  const paths = [
    {
      x: 70,
      n: "01",
      title: ["ROBUST FIXED-KEY", "ADVANTAGE"],
      body: ["Accelerate T-KAM-F.", "Scale memory banks and", "test larger settings."],
      color: C.blue,
    },
    {
      x: 565,
      n: "02",
      title: ["FIXED HELPS;", "LEARNED FAILS"],
      body: ["Keep stable geometry.", "Stop learned-memory", "benefit claims."],
      color: C.amber,
    },
    {
      x: 1060,
      n: "03",
      title: ["NO REPLICATED", "QUALITY EDGE"],
      body: ["Reject the general claim.", "Retain only supported", "niches or diagnostics."],
      color: C.gray,
    },
  ];
  paths.forEach((p) => {
    b += rect(p.x, 200, 440, 350, C.white, 28, C.line, 2);
    b += circle(p.x + 55, 255, 28, p.color);
    b += text(p.x + 55, 263, p.n, 17, C.white, 750, { anchor: "middle" });
    b += text(p.x + 35, 335, p.title, 27, C.navy, 760, { lineHeight: 34 });
    b += line(p.x + 35, 405, p.x + 405, 405, C.pale, 2);
    b += text(p.x + 35, 455, p.body, 21, C.ink2, 550, { lineHeight: 30 });
  });
  b += text(70, 620, "Questions for discussion", 25, C.navy, 700);
  const qs = [
    "Is a 2% loss improvement the right scientific threshold?",
    "Should scale-up optimize quality, active compute, or adaptation speed?",
    "Which failure mode would be most informative for the broader direction?",
  ];
  qs.forEach((q, i) => {
    const y = 670 + i * 52;
    b += circle(91, y - 6, 8, i === 0 ? C.blue : i === 1 ? C.amber : C.gray);
    b += text(115, y, q, 21, C.navy, 550);
  });
  b += rect(775, 630, 755, 170, C.navy, 24);
  b += text(1152, 690, ["The goal is a narrow,", "testable claim—"], 31, C.white, 650, { anchor: "middle", lineHeight: 40 });
  b += text(1152, 773, "not a universal memory architecture claim.", 21, C.amber, 650, { anchor: "middle" });
  b += footer();
  return svg(b);
}

const slides = [
  {
    svg: slide1,
    notes: "Open with the narrow claim. KAM is being tested as a complement to a strong sequence model, not as a replacement for attention.",
  },
  {
    svg: slide2,
    notes: "The motivating tension is storage versus active work. Attention handles relationships in the current context; sparse memory is meant to provide selectively accessed reusable transformations.",
  },
  {
    svg: slide3,
    notes: "Explain only the three actions: route to a neighborhood, retrieve a small mixture of local experts, then blend through a gate. The zero-initialized gate lets the model begin as its baseline.",
  },
  {
    svg: slide4,
    notes: "This is the only technical slide. Separable means that routing geometry and returned algebra can be treated as different objects with different update schedules. Emphasize matched data, token budget, parameters, and strong controls.",
  },
  {
    svg: slide5,
    notes: "The intended lifecycle is explicit: keys learn early, stabilize, freeze near 80%, and remain essentially unchanged during final tuning. The confirmation campaign audits key gradients, freeze timing, and post-freeze drift.",
  },
  {
    svg: slide6,
    notes: "These branches answer different questions. A fixed-key win cannot be used to claim learned geometry, online adaptation, dual memory, or a particular router works.",
  },
  {
    svg: slide7,
    notes: "The fixed-sample campaign is decision-grade. The primary and independent replication must both pass. The confidence interval must clear the 2% scientific margin rather than merely touch zero.",
  },
  {
    svg: slide8,
    notes: "The corrected pilot effect is large but based on only three pairs. p-values are not significant. It justifies confirmation, not acceleration. Do not discuss partial effects from the ongoing fixed-sample run.",
  },
  {
    svg: slide9,
    notes: "Close by making every outcome useful. Ask the professor whether the 2% threshold and the prioritization among quality, active compute, and adaptation are appropriate.",
  },
];

function generateSvgs() {
  fs.mkdirSync(ASSET_DIR, { recursive: true });
  slides.forEach((slide, i) => {
    const filename = path.join(ASSET_DIR, `slide-${String(i + 1).padStart(2, "0")}.svg`);
    fs.writeFileSync(filename, slide.svg(), "utf8");
  });
  fs.writeFileSync(
    path.join(ASSET_DIR, "manifest.json"),
    JSON.stringify(
      {
        title: "Sparse Separable Memory for Sequence Models",
        slideCount: slides.length,
        generatedAt: new Date().toISOString(),
        source: "KAM_Phase6_Professor_Deck_Outline.md",
      },
      null,
      2,
    ),
  );
}

async function buildPptx() {
  const PptxGenJS = require("/tmp/kam_slides_tools/node_modules/pptxgenjs");
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "KAM Research";
  pptx.company = "Kernel-Adaptive Memory";
  pptx.subject = "Phase VI research update";
  pptx.title = "Sparse Separable Memory for Sequence Models";
  pptx.lang = "en-US";
  pptx.theme = {
    headFontFace: "Aptos Display",
    bodyFontFace: "Aptos",
    lang: "en-US",
  };
  slides.forEach((slideDef, i) => {
    const png = path.join(ASSET_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`);
    if (!fs.existsSync(png)) throw new Error(`Missing rendered slide: ${png}`);
    const slide = pptx.addSlide();
    slide.background = { color: "F7F5EF" };
    slide.addImage({ path: png, x: 0, y: 0, w: 13.333333, h: 7.5 });
    slide.addNotes(slideDef.notes);
  });
  await pptx.writeFile({ fileName: OUTPUT });
}

async function main() {
  const mode = process.argv[2] || "--svg";
  if (mode === "--svg") {
    generateSvgs();
    console.log(`Generated ${slides.length} SVG slides in ${ASSET_DIR}`);
  } else if (mode === "--pptx") {
    await buildPptx();
    console.log(`Wrote ${OUTPUT}`);
  } else {
    throw new Error(`Unknown mode: ${mode}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
