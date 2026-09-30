// Builds presentation/BING_Objectness_Group10.pptx from results/ and results/figures/.
// Usage: node presentation/build_deck.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ROOT = path.join(__dirname, "..");
const FIG = (f) => path.join(ROOT, "results", "figures", f);
const R = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "results_synthetic.json")));

const C = {
  dark: "141A24", ink: "1B1F2A", muted: "5B6270", accent: "E8772E", teal: "1C7293",
  tint: "F1F4F7", white: "FFFFFF", line: "D9DEE5", soft: "AEB8C6",
};
const HEAD = "Cambria", BODY = "Calibri";
const pct = (v) => (100 * v).toFixed(1) + "%";
const T = R.test, RL = R.real;
const B = T["BING (binary)"], BR = RL["BING (binary)"];
const SS = T["Selective Search (fast)"], SSR = RL["Selective Search (fast)"];
const BSS = T["BING (binary) @SS subset"];
const sp = R.speed;
const bestFps = Math.max(...Object.entries(sp).filter(([k]) => k.includes("numba")).map(([, v]) => v));

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "Group 10";
pres.company = "23CSE373 Computer Vision";
pres.subject = "BING objectness estimation";
pres.title = "Fast Object Proposal Generation Using Handcrafted Gradient Features";

let n = 0;
const slides = [];
const _add = pres.addSlide.bind(pres);
pres.addSlide = (...a) => { const s = _add(...a); slides.push(s); return s; };
function base(title, kicker) {
  const s = pres.addSlide();
  n += 1;
  s.background = { color: C.white };
  if (kicker) s.addText(kicker.toUpperCase(), { x: 0.6, y: 0.35, w: 9, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: C.accent, charSpacing: 2, margin: 0, isTextBox: true });
  s.addText(title, { x: 0.6, y: 0.62, w: 12.1, h: 0.8, fontFace: HEAD, fontSize: 32, bold: true, color: C.ink, margin: 0, isTextBox: true });
  s.addText(String(n), { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 10, color: C.muted, align: "right", margin: 0, isTextBox: true });
  return s;
}
function bullets(s, items, x, y, w, h, size = 16) {
  const runs = [];
  items.forEach((t, i) => {
    const last = i === items.length - 1;
    if (Array.isArray(t)) {
      runs.push({ text: t[0], options: { bullet: true, bold: true, color: C.ink, paraSpaceAfter: 9 } });
      runs.push({ text: t[1], options: { color: C.muted, breakLine: !last } });
    } else {
      runs.push({ text: t, options: { bullet: true, paraSpaceAfter: 9, breakLine: !last } });
    }
  });
  s.addText(runs, { x, y, w, h, fontFace: BODY, fontSize: size, color: C.ink, valign: "top", margin: 0, isTextBox: true });
}
function img(s, file, x, y, w, h) {
  s.addImage({ path: FIG(file), x, y, w, h, sizing: { type: "contain", w, h } });
}
function stat(s, x, y, w, big, label, color = C.accent) {
  s.addText(big, { x, y, w, h: 0.9, fontFace: HEAD, fontSize: 44, bold: true, color, margin: 0, isTextBox: true });
  s.addText(label, { x, y: y + 0.9, w, h: 0.6, fontFace: BODY, fontSize: 13, color: C.muted, margin: 0, valign: "top", isTextBox: true });
}
function card(s, x, y, w, h, fill = C.tint) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.12 });
}
function numCircle(s, x, y, k, fill = C.accent) {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.46, h: 0.46, fill: { color: fill }, line: { color: fill } });
  s.addText(String(k), { x, y, w: 0.46, h: 0.46, fontFace: BODY, fontSize: 14, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 1 title
{
  const s = pres.addSlide(); n += 1;
  s.background = { color: C.dark };
  s.addImage({ path: FIG("ng_title.png"), x: 7.2, y: 0, w: 6.13, h: 7.5, sizing: { type: "cover", w: 6.13, h: 7.5 } });
  s.addText("23CSE373 · COMPUTER VISION · GROUP 10", { x: 0.6, y: 0.7, w: 6.4, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: C.accent, charSpacing: 2, margin: 0, isTextBox: true });
  s.addText("Fast Object Proposal Generation Using Handcrafted Gradient Features", { x: 0.6, y: 1.2, w: 6.3, h: 2.4, fontFace: HEAD, fontSize: 36, bold: true, color: C.white, margin: 0, valign: "top", isTextBox: true });
  s.addText("A from-scratch re-implementation and study of BING — Binarized Normed Gradients for Objectness Estimation (Cheng, Zhang, Lin, Torr · CVPR 2014)", { x: 0.6, y: 3.65, w: 6.2, h: 0.9, fontFace: BODY, fontSize: 15, italic: true, color: C.soft, margin: 0, valign: "top", isTextBox: true });
  const team = [["Abhinav Dileep", "AM.SC.U4CSE23202"], ["Ayyappadas M T", "AM.SC.U4CSE23209"], ["Hari Sankar A", "AM.SC.U4CSE23221"], ["Parthiv M", "AM.SC.U4CSE23240"], ["Sai Kishen K M", "AM.SC.U4CSE23248"]];
  team.forEach(([nm, roll], i) => {
    s.addText([{ text: nm, options: { bold: true, color: C.white } }, { text: "   " + roll, options: { color: C.soft } }], { x: 0.6, y: 4.85 + i * 0.4, w: 6.3, h: 0.36, fontFace: BODY, fontSize: 14, margin: 0, isTextBox: true });
  });
}

// ---------------------------------------------------------------- 2 problem
{
  const s = base("Why object proposals?", "Problem & motivation");
  bullets(s, [
    ["Detection is expensive. ", "A sliding-window detector must classify every position × scale × aspect ratio — easily 10⁵–10⁶ windows per image."],
    ["Most windows are background. ", "Only a handful of windows actually contain an object."],
    ["Objectness first. ", "A cheap, class-agnostic score ranks windows so that a detector only examines the top ~1000."],
    ["Question we answer: ", "“Which regions of this image are likely to contain objects?” — not “what is the object?”"],
  ], 0.6, 1.7, 6.6, 4.5, 17);
  card(s, 7.7, 1.7, 5.0, 5.0);
  stat(s, 8.1, 2.0, 4.3, "~10⁵+", "windows a naïve sliding-window scan evaluates per image", C.muted);
  stat(s, 8.1, 3.55, 4.3, pct(B.dr_at["1000"]), "of objects recovered by our BING in the top 1000 proposals (synthetic test set)", C.accent);
  stat(s, 8.1, 5.05, 4.3, `${Math.round(bestFps)} fps`, "fastest configuration of our Python/Numba implementation", C.teal);
}

// ---------------------------------------------------------------- 3 key idea
{
  const s = base("BING in one idea", "Base paper");
  bullets(s, [
    ["Objects are stand-alone things with closed boundaries. ", "Resize any window to a tiny 8×8 and look at its gradient magnitudes: object windows show a ring of strong edges around a quieter centre."],
    ["Normed gradients (NG). ", "g = min(|gx| + |gy|, 255) — a 64-D feature that is invariant to translation, scale and aspect ratio of the window."],
    ["Linear model, two stages. ", "An SVM filter w scores every window; a second per-size SVM calibrates scores across the 36 window sizes."],
    ["Binarize everything. ", "w and g are approximated by binary vectors so a window score costs only a few BITWISE AND + POPCOUNT ops → 300 fps in the paper's C++."],
  ], 0.6, 1.7, 6.9, 5.2, 16);
  img(s, "filter_w.png", 7.6, 1.8, 5.2, 1.6);
  s.addText("The learned 8×8 filter w (red = positive weight) and its binary approximations. It is a “closed boundary” template: positive on the border, negative inside.", { x: 7.6, y: 3.5, w: 5.2, h: 0.9, fontFace: BODY, fontSize: 12, italic: true, color: C.muted, margin: 0, isTextBox: true });
  img(s, "calibration.png", 8.6, 4.35, 3.4, 2.9);
}

// ---------------------------------------------------------------- 4 pipeline
{
  const s = base("Proposed system — 8 stages", "Methodology");
  const steps = [
    ["Input image", "any size; resized so the longer side ≤ 500 px"],
    ["Pre-processing", "colour space (RGB / HSV / Gray / Lab)"],
    ["Candidates", "36 quantised sizes {10…320}², every 8×8 position"],
    ["Gradient features", "image resized per size, [-1 0 1] gradient"],
    ["Binarized NG", "top Ng=4 bit-planes packed into 64-bit words"],
    ["Objectness score", "s = ⟨w, g⟩ via AND + POPCOUNT (Nw=2 bases)"],
    ["Ranking", "NMS on score map, per-size calibration v·s + t"],
    ["Top proposals", "sorted boxes + objectness heat-map"],
  ];
  steps.forEach(([t, d], i) => {
    const col = i % 4, row = Math.floor(i / 4);
    const x = 0.6 + col * 3.1, y = 1.8 + row * 2.6;
    card(s, x, y, 2.8, 2.2, row === 0 ? C.tint : "FDF1E8");
    numCircle(s, x + 0.2, y + 0.22, i + 1, row === 0 ? C.teal : C.accent);
    s.addText(t, { x: x + 0.2, y: y + 0.82, w: 2.4, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.2, y: y + 1.28, w: 2.45, h: 0.85, fontFace: BODY, fontSize: 12.5, color: C.muted, margin: 0, valign: "top", isTextBox: true });
    if (col < 3) s.addText("→", { x: x + 2.8, y: y + 0.85, w: 0.3, h: 0.4, fontFace: BODY, fontSize: 20, color: C.soft, align: "center", margin: 0, isTextBox: true });
  });
}

// ---------------------------------------------------------------- 5 NG feature
{
  const s = base("Normed-gradient features at every scale", "Stages 3–5");
  img(s, "ng_pipeline.png", 0.5, 1.55, 8.4, 3.9);
  bullets(s, [
    "The whole image is resized once per window size (w, h) to (8W/w, 8H/h) — then every 8×8 patch of the NG map is one candidate window.",
    "Top row: NG maps for three window sizes. Bottom row: the four most significant bit-planes b₁…b₄ that form the binarized feature.",
    "g ≈ Σₖ 2^(8−k) bₖ — keeping Ng = 4 bits loses almost nothing (see ablation).",
  ], 9.2, 1.7, 3.6, 5.0, 14);
  s.addText("Sliding a 64-bit window: each new pixel updates a row byte with one shift + OR, and 8 row bytes form the 64-bit word (paper Alg. 2).", { x: 0.6, y: 5.8, w: 8.3, h: 0.8, fontFace: BODY, fontSize: 13, italic: true, color: C.muted, margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 6 binarization math
{
  const s = base("From dot product to bit operations", "Stages 5–6");
  card(s, 0.6, 1.7, 6.0, 4.9);
  s.addText([
    { text: "1 · Approximate the filter (Alg. 1)", options: { bold: true, breakLine: true, fontSize: 16, color: C.ink } },
    { text: "w ≈ Σⱼ βⱼ aⱼ,  aⱼ ∈ {−1, +1}⁶⁴  (greedy: aⱼ = sign(residual))", options: { breakLine: true, color: C.muted } },
    { text: " ", options: { breakLine: true } },
    { text: "2 · Approximate the feature", options: { bold: true, breakLine: true, fontSize: 16, color: C.ink } },
    { text: "g ≈ Σₖ 2^(8−k) bₖ,  bₖ ∈ {0, 1}⁶⁴  (top Ng bit-planes)", options: { breakLine: true, color: C.muted } },
    { text: " ", options: { breakLine: true } },
    { text: "3 · Score with bits only", options: { bold: true, breakLine: true, fontSize: 16, color: C.ink } },
    { text: "⟨aⱼ, bₖ⟩ = 2·popcount(aⱼ⁺ AND bₖ) − popcount(bₖ)", options: { breakLine: true, color: C.muted } },
    { text: "s ≈ Σⱼ βⱼ Σₖ 2^(8−k) ⟨aⱼ, bₖ⟩", options: { color: C.muted } },
  ], { x: 0.9, y: 1.95, w: 5.5, h: 4.5, fontFace: BODY, fontSize: 15, valign: "top", margin: 0, isTextBox: true });
  img(s, "filter_bases.png", 6.9, 1.7, 5.9, 1.9);
  bullets(s, [
    `With Nw = 2 and Ng = 4 a window needs 4 × (1 + 2) = 12 POPCOUNTs instead of 64 multiply-adds.`,
    `Accuracy is unchanged: DR@1000 ${pct(B.dr_at["1000"])} (binary) vs ${pct(T["BING (float w)"].dr_at["1000"])} (float w) on the synthetic test set.`,
    "Unit-tested: the bitwise score equals ⟨w̃, g̃⟩ computed with floats to 1e-3.",
  ], 6.9, 3.9, 5.9, 2.8, 14);
}

// ---------------------------------------------------------------- 7 implementation
{
  const s = base("What we built", "Implementation");
  const mods = [
    ["bing/features.py", "NG maps, colour spaces, [-1 0 1] vs Sobel, per-box features"],
    ["bing/binary.py", "Alg. 1 filter binarization, 64-bit packing, AND+POPCOUNT scorer (numpy)"],
    ["bing/fast.py", "Numba JIT kernel of Alg. 2 — shift/OR feature packing, parallel popcount"],
    ["bing/model.py", "Two-stage training (LinearSVC + per-size calibration), NMS, ranking, heat-map"],
    ["bing/metrics.py", "DR-#WIN, MABO, AUC, recall-vs-IoU"],
    ["bing/datasets.py", "Synthetic scene generator, real photos, PASCAL VOC 2007 loader"],
  ];
  mods.forEach(([f, d], i) => {
    const y = 1.7 + i * 0.78;
    s.addText(f, { x: 0.6, y, w: 2.6, h: 0.6, fontFace: "Courier New", fontSize: 14, bold: true, color: C.teal, margin: 0, valign: "middle", isTextBox: true });
    s.addText(d, { x: 3.2, y, w: 4.6, h: 0.6, fontFace: BODY, fontSize: 13.5, color: C.ink, margin: 0, valign: "middle", isTextBox: true });
  });
  card(s, 8.3, 1.7, 4.4, 4.7);
  s.addText([
    { text: "Scripts", options: { bold: true, fontSize: 16, breakLine: true } },
    { text: "make_synthetic.py · run_experiments.py · make_figures.py · demo.py (image / webcam)", options: { color: C.muted, breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Stack", options: { bold: true, fontSize: 16, breakLine: true } },
    { text: "Python · OpenCV · NumPy · scikit-learn · Numba · Matplotlib", options: { color: C.muted, breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Reference", options: { bold: true, fontSize: 16, breakLine: true } },
    { text: "Design follows torrvision/Objectness (C++); re-written from scratch in Python, no deep learning.", options: { color: C.muted } },
  ], { x: 8.6, y: 1.95, w: 3.9, h: 4.3, fontFace: BODY, fontSize: 14, color: C.ink, valign: "top", margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 8 data
{
  const s = base("Evaluation data", "Experimental setup");
  img(s, "dataset_synthetic.png", 0.6, 1.6, 6.0, 2.35);
  img(s, "dataset_real.png", 6.8, 1.6, 6.0, 2.35);
  s.addText("Synthetic benchmark", { x: 0.6, y: 4.1, w: 6, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: C.ink, margin: 0, isTextBox: true });
  bullets(s, [
    `${R.n_train} train / ${R.n_test} test images, 1–5 objects each (${T["BING (binary)"].num_objects} test objects)`,
    "Textured / smooth backgrounds + random clutter lines (edges that are not objects), shading, noise, blur",
  ], 0.6, 4.55, 6.0, 1.8, 14);
  s.addText("Real photographs", { x: 6.8, y: 4.1, w: 6, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: C.ink, margin: 0, isTextBox: true });
  bullets(s, [
    `${R.n_real} unseen natural images, ${BR.num_objects} hand-labelled boxes (people, cup, coins, motorcycle, pagoda …)`,
    "Model trained only on synthetic data → tests generalisation of the ‘closed boundary’ cue",
  ], 6.8, 4.55, 6.0, 1.8, 14);
  s.addText("Metrics: detection rate (DR, IoU ≥ 0.5) vs number of proposals, MABO, AUC, runtime. A VOC 2007 loader is included for the paper's full protocol.", { x: 0.6, y: 6.45, w: 12.2, h: 0.5, fontFace: BODY, fontSize: 12.5, italic: true, color: C.muted, margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 9 results synthetic
{
  const s = base("Recall vs number of proposals", "Results · synthetic test set");
  img(s, "dr_synthetic.png", 0.5, 1.5, 7.6, 5.0);
  stat(s, 8.6, 1.7, 4.2, pct(B.dr_at["1000"]), "DR@1000 — BING (binary)");
  stat(s, 8.6, 3.25, 4.2, pct(T["BING-Diversified"].dr_at["1000"]), "DR@1000 — BING-Diversified (RGB + HSV + Gray)", C.teal);
  stat(s, 8.6, 4.8, 4.2, pct(T["Random boxes"].dr_at["1000"]), "DR@1000 — random boxes (no objectness)", C.muted);
}

// ---------------------------------------------------------------- 10 comparison table
{
  const s = base("How does BING compare?", "Results · baselines");
  const rows = [["Method", "DR@100", "DR@1k", "MABO@1k", "AUC", "img/s †"]];
  const order = ["BING (binary)", "BING (float w)", "BING stage I only", "BING-Diversified", "Selective Search (fast)", "Random boxes", "Sliding windows"];
  order.forEach((m) => {
    const r = T[m];
    rows.push([m + (r.n_images ? ` *` : ""), pct(r.dr_at["100"]), pct(r.dr_at["1000"]), r.mabo_at["1000"].toFixed(3), r.auc.toFixed(3), r.fps > 500 ? "—" : r.fps.toFixed(1)]);
  });
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || (i === 1 && j === 0), color: i === 0 ? C.white : C.ink, fill: { color: i === 0 ? C.dark : (i % 2 ? C.white : C.tint) }, align: j ? "center" : "left" } }))),
    { x: 0.6, y: 1.7, w: 8.2, colW: [2.5, 1.05, 1.15, 1.35, 0.95, 1.2], fontFace: BODY, fontSize: 13, rowH: 0.48, border: { type: "solid", color: C.line, pt: 0.5 } });
  s.addText(`* Selective Search on ${SS.n_images} test images (it is ~${Math.round(BSS.fps / SS.fps)}× slower); BING on the same subset: DR@1000 ${pct(BSS.dr_at["1000"])}. † End-to-end timing during evaluation (default INTER_AREA resizing, Python overhead) — see the Speed slide for the controlled benchmark.`, { x: 0.6, y: 5.75, w: 8.0, h: 0.8, fontFace: BODY, fontSize: 11.5, italic: true, color: C.muted, margin: 0, isTextBox: true });
  card(s, 9.0, 1.7, 3.8, 4.8);
  s.addText([
    { text: "Take-aways", options: { bold: true, fontSize: 17, breakLine: true } },
    { text: "Stage II calibration matters: +" + (100 * (B.dr_at["100"] - T["BING stage I only"].dr_at["100"])).toFixed(0) + " pts DR@100.", options: { bullet: true, breakLine: true } },
    { text: "Binary ≈ float accuracy.", options: { bullet: true, breakLine: true } },
    { text: "Selective Search localises tighter (higher MABO) but is " + Math.round(BSS.fps / SS.fps) + "× slower.", options: { bullet: true, breakLine: true } },
    { text: "Exactly the trade-off the paper reports: BING is excellent at IoU 0.5, weaker at high IoU.", options: { bullet: true } },
  ], { x: 9.25, y: 1.95, w: 3.35, h: 4.4, fontFace: BODY, fontSize: 13.5, color: C.ink, valign: "top", margin: 0, paraSpaceAfter: 6, isTextBox: true });
}

// ---------------------------------------------------------------- 11 real photos
{
  const s = base("Generalisation to real photographs", "Results · real images");
  img(s, "dr_real.png", 0.5, 1.5, 6.2, 4.1);
  img(s, "found_real.png", 6.9, 1.5, 6.0, 5.2);
  s.addText(`Trained only on synthetic scenes, BING recovers ${pct(BR.dr_at["1000"])} of ${BR.num_objects} real objects in its top 1000 proposals (Diversified: ${pct(RL["BING-Diversified"].dr_at["1000"])}; random: ${pct(RL["Random boxes"].dr_at["1000"])}). Right: best proposal per object among the top 1000 — green = found (IoU ≥ 0.5), red = missed.`, { x: 0.6, y: 5.75, w: 6.1, h: 1.2, fontFace: BODY, fontSize: 12.5, color: C.muted, margin: 0, valign: "top", isTextBox: true });
}

// ---------------------------------------------------------------- 12 qualitative
{
  const s = base("What BING looks at", "Visualising objectness");
  img(s, "top8_real.png", 0.5, 1.5, 7.0, 5.4);
  img(s, "heatmaps_real.png", 7.7, 1.6, 5.2, 3.9);
  s.addText("Left: the 8 highest-ranked proposals. Right: objectness heat-map (sum of the top 2000 proposal scores per pixel) — mass concentrates on the cup, the pagoda, the flower and the person rather than on texture.", { x: 7.7, y: 5.6, w: 5.1, h: 1.3, fontFace: BODY, fontSize: 12.5, color: C.muted, margin: 0, valign: "top", isTextBox: true });
}

// ---------------------------------------------------------------- 13 speed
{
  const s = base("Speed", "Computational efficiency");
  img(s, "speed.png", 0.5, 1.5, 7.2, 4.1);
  img(s, "stage_time.png", 7.9, 1.55, 4.9, 2.0);
  bullets(s, [
    `Numba JIT of the bitwise kernel: ${sp["numba|binary|linear"].toFixed(0)} img/s vs ${sp["numpy|binary|linear"].toFixed(0)} img/s in pure numpy.`,
    `OpenCV's SIMD float correlation is still competitive in Python (${sp["numba|float|linear"].toFixed(0)} img/s): the paper's 300 fps comes from hand-written C++ with SSE popcount.`,
    `Per image, resizing dominates (${R.stage_ms.resize.toFixed(1)} ms with INTER_AREA) — scoring all windows with bit operations takes only ${R.stage_ms.scoring.toFixed(1)} ms.`,
    "Selective Search: ~1 img/s — BING is 1–2 orders of magnitude faster.",
  ], 7.9, 3.8, 4.9, 3.0, 13.5);
  s.addText("Blue = binarized scoring, orange = float filter. INTER_AREA gives anti-aliased resizing; bilinear matches the original C++ and is ~2× faster.", { x: 0.6, y: 5.75, w: 7.0, h: 0.8, fontFace: BODY, fontSize: 12, italic: true, color: C.muted, margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 14 ablations
{
  const s = base("Ablation studies", "Contributions · analysis");
  img(s, "ablations.png", 0.5, 1.45, 8.6, 4.6);
  const A = R.ablations || {};
  const g = (k, n = "1000") => (A[k] ? pct(A[k].dr_at[n]) : "—");
  bullets(s, [
    ["Gradient representation: ", `[-1 0 1] ${g("colour=RGB")} vs Sobel ${g("gradient=Sobel 3x3")} — the simplest mask is enough.`],
    ["Colour space: ", `RGB ${g("colour=RGB")}, HSV ${g("colour=HSV")}, Lab ${g("colour=LAB")}, Gray ${g("colour=GRAY")}; they are complementary (Diversified).`],
    ["Binarization: ", `Nw = 1 → ${g("Nw=1")}, Nw = 2 → ${g("Nw=2")}; Ng = 1 → ${g("Ng=1")}, Ng = 4 → ${g("Ng=4")}.`],
    ["Data: ", `only 10 training images already give ${g("train imgs=10")} — 64 weights are easy to learn.`],
  ], 9.3, 1.6, 3.6, 5.3, 12.5);
  s.addText("All numbers: DR@1000 on the synthetic test set; bars also show DR@100.", { x: 0.6, y: 6.2, w: 8.4, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: C.muted, margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 15 proposal quality vs IoU
{
  const s = base("Proposal quality vs localisation precision", "Contributions · analysis");
  img(s, "recall_iou_synthetic.png", 0.5, 1.5, 6.2, 4.2);
  img(s, "mabo_synthetic.png", 6.8, 1.5, 6.2, 4.2);
  s.addText("BING's quantised window sizes (powers of two) and 8-pixel stride give coarse boxes: recall is high at IoU 0.5 but falls quickly as the threshold rises, and MABO saturates around " + B.mabo_at["1000"].toFixed(2) + ". Segmentation-based Selective Search is slower but tighter. This matches known analyses of BING (Hosang et al., TPAMI 2016).", { x: 0.6, y: 5.85, w: 12.2, h: 1.0, fontFace: BODY, fontSize: 13, color: C.muted, margin: 0, valign: "top", isTextBox: true });
}

// ---------------------------------------------------------------- 16 contributions
{
  const s = base("Planned contributions → delivered", "Proposal check-list");
  const items = [
    ["Analysis of gradient representations", "[-1 0 1] vs Sobel, 4 colour spaces, bit depth Ng"],
    ["Proposal quality vs computational cost", "binary vs float, numpy vs Numba, Nw/Ng trade-offs, BING vs Selective Search"],
    ["Different numbers of proposals", "DR / MABO curves from 1 to 5000 proposals, per-size budget ablation"],
    ["Visualisation of objectness scores", "learned filter, bit-planes, calibration map, heat-maps, top-k boxes"],
    ["Different image categories / datasets", "synthetic benchmark, 10 real photos, VOC 2007 loader"],
    ["Optimisation / modernisation", "pure-Python package + Numba Alg. 2 kernel, unit tests, webcam demo"],
  ];
  items.forEach(([t, d], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 6.2, y = 1.7 + row * 1.65;
    card(s, x, y, 5.9, 1.4);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.45, w: 0.5, h: 0.5, fill: { color: C.teal }, line: { color: C.teal } });
    s.addText("✓", { x: x + 0.25, y: y + 0.45, w: 0.5, h: 0.5, fontFace: BODY, fontSize: 16, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.95, y: y + 0.2, w: 4.8, h: 0.45, fontFace: HEAD, fontSize: 16, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.95, y: y + 0.66, w: 4.8, h: 0.6, fontFace: BODY, fontSize: 13, color: C.muted, margin: 0, valign: "top", isTextBox: true });
  });
}

// ---------------------------------------------------------------- 17 limitations
{
  const s = base("Limitations & future work", "Discussion");
  bullets(s, [
    ["Localisation is coarse. ", "Power-of-two sizes and an 8-px stride cap MABO; refine top boxes (e.g. edge-based box regression) or add intermediate sizes."],
    ["Evaluation scale. ", "PASCAL VOC could not be downloaded in our sandbox; we provide the loader and scripts to reproduce the paper's protocol (train on trainval, test on test)."],
    ["Python ceiling. ", "Numba closes much of the gap, but a SIMD C++/Cython popcount kernel would be needed for the paper's 300 fps."],
    ["Synthetic-to-real gap. ", "Training on real annotated images (VOC / COCO) would raise real-photo recall further."],
    ["Next step. ", "Feed BING proposals into a classifier (HOG + SVM or a small CNN) for a complete detector."],
  ], 0.6, 1.7, 7.6, 5.2, 15);
  img(s, "found_synthetic.png", 8.5, 1.7, 4.3, 3.3);
  s.addText("Misses (red) are mostly low-contrast objects or objects cut by clutter lines.", { x: 8.5, y: 5.1, w: 4.3, h: 0.7, fontFace: BODY, fontSize: 12, italic: true, color: C.muted, margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 18 conclusion
{
  const s = pres.addSlide(); n += 1;
  s.background = { color: C.dark };
  s.addText("CONCLUSION", { x: 0.6, y: 0.6, w: 6, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: C.accent, charSpacing: 2, margin: 0, isTextBox: true });
  s.addText("Simple gradients, a 64-weight linear model and bit operations are enough to find most objects — fast.", { x: 0.6, y: 1.0, w: 12, h: 1.4, fontFace: HEAD, fontSize: 30, bold: true, color: C.white, margin: 0, valign: "top", isTextBox: true });
  const st = [[pct(B.dr_at["1000"]), "objects found in top 1000\n(synthetic test)"], [pct(BR.dr_at["1000"]), "objects found in top 1000\n(real photos, zero real training)"], [`${Math.round(bestFps)}`, "images / second\n(Python + Numba, 4 CPU cores)"], ["12", "POPCOUNTs per window\ninstead of 64 multiply-adds"]];
  st.forEach(([v, l], i) => {
    s.addText(v, { x: 0.6 + i * 3.1, y: 3.0, w: 2.9, h: 1.0, fontFace: HEAD, fontSize: 44, bold: true, color: i % 2 ? C.white : C.accent, margin: 0, isTextBox: true });
    s.addText(l, { x: 0.6 + i * 3.1, y: 4.0, w: 2.9, h: 0.9, fontFace: BODY, fontSize: 14, color: C.soft, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText([
    { text: "References  ", options: { bold: true, color: C.white } },
    { text: "[1] M.-M. Cheng, Z. Zhang, W.-Y. Lin, P. H. S. Torr, “BING: Binarized Normed Gradients for Objectness Estimation at 300fps,” CVPR 2014.  [2] J. Uijlings et al., “Selective Search for Object Recognition,” IJCV 2013.  [3] B. Alexe, T. Deselaers, V. Ferrari, “Measuring the Objectness of Image Windows,” TPAMI 2012.  [4] J. Hosang et al., “What makes for effective detection proposals?,” TPAMI 2016.  [5] P. Arbeláez et al., “Multiscale Combinatorial Grouping,” CVPR 2014.", options: { color: C.soft } },
  ], { x: 0.6, y: 5.5, w: 12.1, h: 1.3, fontFace: BODY, fontSize: 11, margin: 0, valign: "top", isTextBox: true });
  s.addText("Thank you — questions?", { x: 0.6, y: 6.85, w: 6, h: 0.4, fontFace: HEAD, fontSize: 16, italic: true, color: C.white, margin: 0, isTextBox: true });
}


const NOTES = [
  "Introduce the group and the topic. Our project re-implements and studies BING, a CVPR 2014 method that finds likely object regions in an image at 300 frames per second, using only handcrafted gradient features — no deep learning.",
  "Object detection is costly because a sliding-window detector must classify a huge number of windows. Object proposals first pick a small number of promising windows. We ask only 'where are objects likely to be', not 'what are they'. The numbers on the right are our own results, explained later.",
  "BING's key observation: objects are stand-alone things with well-defined closed boundaries. If you shrink any window to 8x8 and compute gradient magnitudes, windows around objects show a ring of strong edges. A single 64-weight linear filter learns this. The figure shows the filter we learned: positive (red) on the border, negative inside. The second stage calibrates each window size — the small matrix shows the learned calibration weight per size; sizes that never contain objects are dropped.",
  "These are the eight stages from our proposal. The input is resized, 36 window sizes are considered, gradient features are computed, binarized, scored with bit operations, non-maximum suppression and calibration rank them, and the top boxes are output.",
  "Instead of cropping every window, we resize the whole image once per window size so that each 8x8 patch in the resized normed-gradient map is exactly one window. The bottom row shows the four most significant bit planes used by the binarized feature.",
  "The filter is approximated by a weighted sum of plus/minus-one vectors (Algorithm 1 of the paper), and the feature by its top four bit planes. Then the dot product becomes AND plus POPCOUNT operations on 64-bit integers. Our unit tests confirm the bitwise score equals the approximated float score, and accuracy is the same as the float filter.",
  "We wrote everything from scratch in Python, following the structure of the authors' C++ repository. The Numba module is a JIT-compiled version of the paper's Algorithm 2. There are scripts for data generation, experiments, figures and a live demo, and unit tests.",
  "We could not download PASCAL VOC in our environment, so we built a synthetic benchmark with ground-truth boxes, including distracting clutter lines, textures, shading and noise. To test generalisation we hand-labelled 58 objects in 10 real photos. The model never sees real images during training. A VOC loader is included to reproduce the paper's protocol.",
  "Detection rate: the fraction of objects covered by at least one proposal with IoU at least 0.5, as a function of the number of proposals. BING reaches about 95 percent at 1000 proposals, far above random boxes; combining three colour spaces (Diversified) pushes it higher.",
  "Comparison with baselines. Calibration (stage II) clearly helps at small proposal budgets. Binary and float are equally accurate. Selective Search, a segmentation-based method, gives tighter boxes (higher MABO) but is dozens of times slower. This is the same trade-off reported in the literature.",
  "On real photographs the model trained only on synthetic shapes still finds most objects in the top 1000, showing that the 'closed boundary' cue transfers. The right panel shows, per object, the best proposal found — green if found, red if missed.",
  "Visualising objectness: top-ranked boxes and a heat-map built by accumulating proposal scores. Objectness concentrates on real objects rather than on textured background.",
  "Speed: the JIT-compiled bitwise kernel is several times faster than pure numpy. In Python, OpenCV's SIMD float correlation is still competitive, because the paper's 300 fps relies on hand-optimised C++. Most of our per-image time is resizing, not scoring.",
  "Ablations. The simple [-1 0 1] gradient is as good as Sobel. RGB is the best single colour space; the others are complementary. One binary basis loses some accuracy; two are enough. Four bits of the gradient suffice. Very few training images are needed because the model has only 64 weights.",
  "BING's weakness is localisation precision: recall is high at IoU 0.5 but drops at stricter thresholds, and MABO is lower than Selective Search. This is because of the coarse power-of-two window sizes and 8-pixel stride.",
  "Every contribution we listed in the proposal is covered.",
  "Limitations and what we would do next: better localisation, full VOC evaluation, a C++ popcount kernel, training on real data, and adding a classifier to make a full detector.",
  "Summary of the key numbers. Thank you — we are happy to take questions and can run the live demo (scripts/demo.py).",
];
NOTES.forEach((t, i) => slides[i] && slides[i].addNotes(t));

const out = path.join(__dirname, "BING_Objectness_Group10.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote", out));
