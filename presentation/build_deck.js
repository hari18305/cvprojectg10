// Builds presentation/BING_Objectness_Group10.pptx from results/ and results/figures/.
// Usage: node presentation/build_deck.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ROOT = path.join(__dirname, "..");
const FIG = (f) => path.join(ROOT, "results", "figures", f);
const R = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "results_synthetic.json")));
const R1 = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "tier1_synthetic.json")));
const V = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "voc.json")));
const VF = V["BING (ours, released-code sizes, greedy NMS)"], VO = V["BING (authors' C++)"];
const VFS = V["BING (ours, released-code sizes, greedy NMS) @SS subset"];
const T1s = (k) => R1.test["BING (ours)"].by_size[k]["dr@100"];
const T1o = (k) => R1.test["BING (authors' C++)"].by_size[k]["dr@100"];

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


const SEC = ["1 · Introduction", "2 · Problem definition", "3 · Proposed solution & novelty", "4 · Design & solution", "5 · Result analysis"];
const A = R.ablations || {};
const g = (k, m = "1000") => (A[k] ? pct(A[k].dr_at[m]) : "—");

// ---------------------------------------------------------------- 2 introduction
{
  const s = base("Why object proposals?", SEC[0]);
  bullets(s, [
    ["Object detection is expensive. ", "A sliding-window detector must classify every position × scale × aspect ratio, which is 10⁵–10⁶ windows per image."],
    ["Most windows are background. ", "Only a handful of windows actually contain an object."],
    ["Object proposals. ", "A cheap, class-agnostic objectness score ranks the windows, so the detector only examines the top ~1000."],
    ["Base paper: BING (CVPR 2014). ", "Handcrafted gradient features + bit operations produce proposals at 300 fps, with no deep learning."],
  ], 0.6, 1.7, 6.6, 4.8, 17);
  card(s, 7.7, 1.7, 5.0, 5.0);
  stat(s, 8.1, 2.0, 4.3, "~10⁵+", "windows a naïve sliding-window scan evaluates per image", C.muted);
  stat(s, 8.1, 3.55, 4.3, "~1000", "proposals that a detector needs to check instead", C.teal);
  stat(s, 8.1, 5.05, 4.3, "300 fps", "BING's reported speed in the paper's C++ implementation", C.accent);
}

// ---------------------------------------------------------------- 3 problem definition
{
  const s = base("Problem definition", SEC[1]);
  card(s, 0.6, 1.7, 12.1, 1.35, "FDF1E8");
  s.addText([
    { text: "Given an image I, output a ranked list of N bounding boxes such that as many objects as possible are covered (IoU ≥ 0.5) by a small N, in milliseconds, using only handcrafted features.", options: { bold: true } },
  ], { x: 0.9, y: 1.85, w: 11.5, h: 1.05, fontFace: HEAD, fontSize: 19, color: C.ink, valign: "middle", margin: 0, isTextBox: true });
  const cols = [
    ["Input", "An RGB image of any size, any content"],
    ["Output", "Boxes (x₁, y₁, x₂, y₂) ranked by objectness score, plus an objectness heat-map"],
    ["Constraints", "Class-agnostic, real-time, CPU only, no deep networks"],
    ["Success measures", "Detection rate vs #proposals (DR@N), MABO (box tightness), runtime (fps)"],
  ];
  cols.forEach(([t, d], i) => {
    const x = 0.6 + i * 3.1;
    card(s, x, 3.35, 2.85, 2.2);
    numCircle(s, x + 0.2, 3.55, i + 1, C.teal);
    s.addText(t, { x: x + 0.2, y: 4.1, w: 2.5, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.2, y: 4.52, w: 2.5, h: 0.95, fontFace: BODY, fontSize: 13, color: C.muted, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("Scope: we answer “Which regions are likely to contain objects?”, not “What is the object?” (classification is a later stage).", { x: 0.6, y: 5.85, w: 12.1, h: 0.6, fontFace: BODY, fontSize: 14, italic: true, color: C.muted, margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 4 proposed solution & novelty
{
  const s = base("Proposed solution: BING, rebuilt and extended", SEC[2]);
  bullets(s, [
    ["Key idea. ", "Objects have closed boundaries. Shrink any window to 8×8 and look at gradient magnitudes: object windows show a ring of edges."],
    ["64-weight linear model. ", "An SVM filter learns this “closed boundary” template (right); a second stage calibrates each window size."],
    ["Bits instead of floats. ", "Filter and features are binarized, so each window costs a few AND + POPCOUNT operations."],
  ], 0.6, 1.65, 6.4, 3.0, 15);
  img(s, "filter_w.png", 7.3, 1.6, 5.5, 1.75);
  s.addText("Our novelty beyond the paper", { x: 0.6, y: 4.3, w: 8, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: C.accent, margin: 0, isTextBox: true });
  const nov = [
    ["From-scratch Python", "No reuse of the C++ code; Numba JIT kernel of the paper's Alg. 2"],
    ["Head-to-head on VOC", "Compiled and ran the authors' C++ code; same images, same evaluator"],
    ["Two missing details", "Window sizes and suppression from the released code, not the paper"],
    ["Benchmarks + ablations", "Synthetic scenes, 58 real objects, gradient / colour / bits / data"],
  ];
  nov.forEach(([t, d], i) => {
    const x = 0.6 + i * 3.1;
    card(s, x, 4.8, 2.85, 1.9);
    s.addText(t, { x: x + 0.2, y: 4.95, w: 2.5, h: 0.45, fontFace: HEAD, fontSize: 15, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.2, y: 5.42, w: 2.5, h: 1.2, fontFace: BODY, fontSize: 12.5, color: C.muted, margin: 0, valign: "top", isTextBox: true });
  });
}

// ---------------------------------------------------------------- 5 design: pipeline
{
  const s = base("System design: 8-stage pipeline", SEC[3]);
  const steps = [
    ["Input image", "longer side ≤ 500 px"],
    ["Pre-processing", "colour space: RGB / HSV / Gray / Lab"],
    ["Candidates", "36 power-of-two sizes (16…512, clipped), every 8×8 position"],
    ["Gradient features", "image resized per size, [-1 0 1] gradient"],
    ["Binarized NG", "top Ng=4 bit-planes packed into 64-bit words"],
    ["Objectness score", "s = ⟨w, g⟩ via AND + POPCOUNT (Nw=2)"],
    ["Ranking", "greedy NMS per score map, per-size calibration v·s + t"],
    ["Top proposals", "sorted boxes + objectness heat-map"],
  ];
  steps.forEach(([t, d], i) => {
    const col = i % 4, row = Math.floor(i / 4);
    const x = 0.6 + col * 3.1, y = 1.75 + row * 2.45;
    card(s, x, y, 2.8, 2.1, row === 0 ? C.tint : "FDF1E8");
    numCircle(s, x + 0.2, y + 0.2, i + 1, row === 0 ? C.teal : C.accent);
    s.addText(t, { x: x + 0.2, y: y + 0.78, w: 2.4, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.2, y: y + 1.24, w: 2.45, h: 0.8, fontFace: BODY, fontSize: 12.5, color: C.muted, margin: 0, valign: "top", isTextBox: true });
    if (col < 3) s.addText("→", { x: x + 2.8, y: y + 0.8, w: 0.3, h: 0.4, fontFace: BODY, fontSize: 20, color: C.soft, align: "center", margin: 0, isTextBox: true });
  });
  s.addText("Training: stage I is a linear SVM on 8×8 NG features of object boxes vs random background windows; stage II is a 1-D calibration per window size.", { x: 0.6, y: 6.5, w: 12.1, h: 0.4, fontFace: BODY, fontSize: 13, italic: true, color: C.muted, margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 6 design: features + binarization
{
  const s = base("Normed gradients → bit operations", SEC[3]);
  img(s, "ng_pipeline.png", 0.5, 1.55, 7.2, 3.35);
  s.addText("Top: NG maps for three window sizes (each 8×8 patch = one window). Bottom: the 4 bit-planes forming the binarized feature.", { x: 0.6, y: 4.95, w: 7.0, h: 0.6, fontFace: BODY, fontSize: 12, italic: true, color: C.muted, margin: 0, isTextBox: true });
  card(s, 7.95, 1.6, 4.8, 5.1);
  s.addText([
    { text: "Approximate the filter", options: { bold: true, breakLine: true, color: C.ink } },
    { text: "w ≈ Σⱼ βⱼ aⱼ,   aⱼ ∈ {−1, +1}⁶⁴", options: { breakLine: true, color: C.muted } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Approximate the feature", options: { bold: true, breakLine: true, color: C.ink } },
    { text: "g ≈ Σₖ 2^(8−k) bₖ,   bₖ ∈ {0, 1}⁶⁴", options: { breakLine: true, color: C.muted } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Score with bits only", options: { bold: true, breakLine: true, color: C.ink } },
    { text: "⟨aⱼ, bₖ⟩ = 2·popcnt(aⱼ⁺ AND bₖ) − popcnt(bₖ)", options: { breakLine: true, color: C.muted } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "12 POPCOUNTs per window instead of 64 multiply-adds.", options: { bold: true, color: C.accent } },
  ], { x: 8.25, y: 1.85, w: 4.3, h: 4.7, fontFace: BODY, fontSize: 15, valign: "top", margin: 0, isTextBox: true });
  bullets(s, [
    "Implemented in Python: bing/ package (features, binary, Numba fast path, model, metrics, datasets); unit tests prove bitwise score = float score.",
  ], 0.6, 5.75, 7.0, 1.0, 13);
}

// ---------------------------------------------------------------- 7 design: data & setup
{
  const s = base("Experimental setup", SEC[3]);
  img(s, "dataset_synthetic.png", 0.6, 1.6, 6.0, 2.35);
  img(s, "dataset_real.png", 6.8, 1.6, 6.0, 2.35);
  const cols = [
    ["PASCAL VOC 2007", `The paper's benchmark: train on ${V.n_train} trainval images, test on ${V.n_test} test images (${V.n_objects.toLocaleString("en")} objects, “difficult” ones ignored).`],
    ["Synthetic scenes", `${R.n_train} train / ${R.n_test} test images: shapes with texture and shading on cluttered backgrounds (edges that are not objects).`],
    ["Real photographs", `${R.n_real} photos, ${BR.num_objects} hand-labelled objects, never used for training: tests generalisation.`],
  ];
  cols.forEach(([t, d], i) => {
    const x = 0.6 + i * 4.1;
    s.addText(t, { x, y: 4.15, w: 3.9, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: i === 0 ? C.accent : C.ink, margin: 0, isTextBox: true });
    s.addText(d, { x, y: 4.6, w: 3.8, h: 1.3, fontFace: BODY, fontSize: 13.5, color: C.muted, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("Metrics: detection rate DR@N (share of objects covered with IoU ≥ 0.5 by the top N proposals), MABO (mean best overlap), images/s. Baselines: the authors' own C++ BING, Selective Search, sliding windows, random boxes. All scored with the same evaluator.", { x: 0.6, y: 6.1, w: 12.2, h: 0.8, fontFace: BODY, fontSize: 12.5, italic: true, color: C.muted, margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 8 results: VOC headline
{
  const s = base("PASCAL VOC 2007: reproducing the paper", SEC[4]);
  img(s, "voc_curves.png", 0.4, 1.5, 8.4, 3.1);
  stat(s, 9.1, 1.6, 3.8, pct(VF.dr_at["1000"]), "objects found in the top 1000, our final BING");
  stat(s, 9.1, 3.1, 3.8, pct(VO.dr_at["1000"]), "the authors' C++ code, same images and evaluator", C.teal);
  stat(s, 9.1, 4.6, 3.8, "96.2%", "reported in the paper (authors' evaluator)", C.muted);
  bullets(s, [
    ["Our rebuild matches the authors' code ", `on their own benchmark: ${pct(VF.dr_at["1000"])} vs ${pct(VO.dr_at["1000"])} at 1000, ${pct(VF.dr_at["5000"])} vs ${pct(VO.dr_at["5000"])} at 5000; box tightness ${VF.mabo_at["1000"].toFixed(3)} vs ${VO.mabo_at["1000"].toFixed(3)}.`],
    ["By the authors' own evaluator ", `their code scores 96.0% here, so the paper's number reproduces.`],
  ], 0.6, 4.85, 8.2, 2.0, 13.5);
}

// ---------------------------------------------------------------- 9 results: comparison with existing models
{
  const s = base("Comparison with existing methods", SEC[4]);
  const T1 = R1.test, ssVoc = V["Selective Search (fast)"], ssSyn = T1["Selective Search (fast)"];
  const rows = [["Method", "VOC DR@1k", "VOC MABO", "Synth. DR@1k", "Real DR@1k", "img/s (1 thread)"]];
  rows.push(["BING — ours (Python + Numba)", pct(VF.dr_at["1000"]), VF.mabo_at["1000"].toFixed(3), pct(B.dr_at["1000"]), pct(BR.dr_at["1000"]), `${Math.round(V.speed_ours["1 thread"])}`]);
  rows.push(["BING — authors' C++ code", pct(VO.dr_at["1000"]), VO.mabo_at["1000"].toFixed(3), pct(T1["BING (authors' C++)"].dr_at["1000"]), pct(R1.real["BING (authors' C++), shrunk to 500 px"].dr_at["1000"]), `${Math.round(1 / VO.sec_per_img_reported)}`]);
  rows.push([`Selective Search (fast) *`, pct(ssVoc.dr_at["1000"]), ssVoc.mabo_at["1000"].toFixed(3), pct(ssSyn.dr_at["1000"]), pct(R1.real["Selective Search (fast)"].dr_at["1000"]), ssVoc.fps.toFixed(1)]);
  rows.push(["Random boxes", pct(V["Random boxes"].dr_at["1000"]), V["Random boxes"].mabo_at["1000"].toFixed(3), pct(T["Random boxes"].dr_at["1000"]), pct(RL["Random boxes"].dr_at["1000"]), "—"]);
  rows.push(["Sliding windows (random order)", "—", "—", pct(T["Sliding windows"].dr_at["1000"]), pct(RL["Sliding windows"].dr_at["1000"]), "—"]);
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || i === 1, color: i === 0 ? C.white : C.ink, fill: { color: i === 0 ? C.dark : (i === 1 ? "FDF1E8" : (i % 2 ? C.white : C.tint)) }, align: j ? "center" : "left" } }))),
    { x: 0.6, y: 1.65, w: 12.1, colW: [3.6, 1.6, 1.5, 1.8, 1.6, 2.0], fontFace: BODY, fontSize: 13.5, rowH: 0.5, border: { type: "solid", color: C.line, pt: 0.5 } });
  s.addText(`* Selective Search on ${V.ss_subset_ids.length} random VOC test images (our BING on the same images: ${pct(VFS.dr_at["1000"])}); it runs multi-threaded. The authors' code reaches ${Math.round(1 / VO.sec_per_img_4threads)} img/s on 4 threads.`, { x: 0.6, y: 4.85, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 11.5, italic: true, color: C.muted, margin: 0, isTextBox: true });
  bullets(s, [
    ["BING vs Selective Search: ", `BING finds more objects on VOC (${pct(VF.dr_at["1000"])} vs ${pct(ssVoc.dr_at["1000"])}) and is ~${Math.round(V.speed_ours["1 thread"] / ssVoc.fps)}× faster, but its boxes are looser (MABO ${VF.mabo_at["1000"].toFixed(2)} vs ${ssVoc.mabo_at["1000"].toFixed(2)}).`],
    ["Ours vs the authors' code: ", `same accuracy on VOC; ours generalises better to real photos when trained on synthetic data, and is slightly faster single-threaded (${Math.round(V.speed_ours["1 thread"])} vs ${Math.round(1 / VO.sec_per_img_reported)} img/s).`],
  ], 0.6, 5.45, 12.1, 1.5, 13.5);
}

// ---------------------------------------------------------------- 10 results: what made the difference
{
  const s = base("Two details the paper leaves out", SEC[4]);
  const rows = [["Our BING on VOC 2007", "DR@1k", "DR@5k", "MABO"]];
  const cfg = [["Paper's text: sizes 10–320, local-max NMS (first version)", "BING (ours)"],
               ["+ released code's window sizes (16–512, clipped)", "BING (ours, released-code sizes)"],
               ["+ released code's greedy NMS only", "BING (ours, greedy NMS)"],
               ["Both (final)", "BING (ours, released-code sizes, greedy NMS)"]];
  cfg.forEach(([lab, k]) => rows.push([lab, pct(V[k].dr_at["1000"]), pct(V[k].dr_at["5000"]), V[k].mabo_at["1000"].toFixed(3)]));
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || i === 4, color: i === 0 ? C.white : C.ink, fill: { color: i === 0 ? C.dark : (i === 4 ? "FDF1E8" : (i % 2 ? C.white : C.tint)) }, align: j ? "center" : "left" } }))),
    { x: 0.6, y: 1.6, w: 7.6, colW: [4.6, 1.0, 1.0, 1.0], fontFace: BODY, fontSize: 12.5, rowH: 0.5, border: { type: "solid", color: C.line, pt: 0.5 } });
  bullets(s, [
    ["Window sizes: ", "the paper lists 10…320; the released code uses 16…512 and clips windows to the image, so near full-image boxes exist. VOC is full of large objects."],
    ["Suppression: ", "keeping only local maxima of each score map throws away good windows; the code's greedy suppression keeps the best 130 spread-out windows per size."],
    ["Lesson: ", "reproducing a paper needs its code, not just its text."],
  ], 8.5, 1.6, 4.4, 3.4, 13);
  img(s, "voc_per_class.png", 0.5, 4.3, 12.3, 2.75);
}

// ---------------------------------------------------------------- 11 results: ablations & speed
{
  const s = base("What matters? Ablations and speed", SEC[4]);
  img(s, "ablations.png", 0.5, 1.45, 7.6, 4.1);
  img(s, "speed.png", 8.3, 1.45, 4.6, 2.6);
  bullets(s, [
    ["Gradient: ", `[-1 0 1] ${g("colour=RGB")} ≈ Sobel ${g("gradient=Sobel 3x3")}`],
    ["Colour: ", `RGB ${g("colour=RGB")} > Lab ${g("colour=LAB")} > HSV ${g("colour=HSV")} > Gray ${g("colour=GRAY")}`],
    ["Bits: ", `Ng = 1 → ${g("Ng=1")}; Ng ≥ 2 → ${g("Ng=2")}`],
    ["Data: ", `10 images → ${g("train imgs=10")}; 600 → ${g("colour=RGB")}`],
    ["Real photos: ", `${pct(BR.dr_at["1000"])} found with no real training`],
  ], 8.3, 4.2, 4.6, 2.8, 12.5);
  s.addText("DR@1000 on the synthetic test set (bars: blue DR@100, orange DR@1000).", { x: 0.6, y: 5.65, w: 7.5, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: C.muted, margin: 0, isTextBox: true });
  s.addText(`Limitation: power-of-two sizes and an 8-px stride give coarse boxes (VOC MABO ${VF.mabo_at["1000"].toFixed(2)}); small objects such as bottles are hardest.`, { x: 0.6, y: 6.1, w: 7.5, h: 0.7, fontFace: BODY, fontSize: 13, bold: true, color: C.ink, margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 12 conclusion
{
  const s = pres.addSlide(); n += 1;
  s.background = { color: C.dark };
  s.addText("CONCLUSION", { x: 0.6, y: 0.6, w: 6, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: C.accent, charSpacing: 2, margin: 0, isTextBox: true });
  s.addText("Simple gradients, a 64-weight linear model and bit operations find 94% of objects on PASCAL VOC, matching the authors' own code.", { x: 0.6, y: 1.0, w: 12, h: 1.4, fontFace: HEAD, fontSize: 28, bold: true, color: C.white, margin: 0, valign: "top", isTextBox: true });
  const st = [[pct(VF.dr_at["1000"]), "VOC 2007 objects found in\ntop 1000 (authors' code " + pct(VO.dr_at["1000"]) + ")"], [pct(BR.dr_at["1000"]), "real photos, no real\ntraining images"], [`${Math.round(V.speed_ours["1 thread"])}`, "images / second, 1 thread\n(authors' C++: " + Math.round(1 / VO.sec_per_img_reported) + ")"], ["12", "POPCOUNTs per window\ninstead of 64 multiply-adds"]];
  st.forEach(([v, l], i) => {
    s.addText(v, { x: 0.6 + i * 3.1, y: 2.7, w: 2.9, h: 1.0, fontFace: HEAD, fontSize: 44, bold: true, color: i % 2 ? C.white : C.accent, margin: 0, isTextBox: true });
    s.addText(l, { x: 0.6 + i * 3.1, y: 3.7, w: 2.9, h: 0.9, fontFace: BODY, fontSize: 14, color: C.soft, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText([
    { text: "Future work  ", options: { bold: true, color: C.white } },
    { text: "tighter boxes (refinement / more sizes) · C++ popcount kernel and multi-threading · train-on-some-classes generalisation · add a classifier for a complete detector", options: { color: C.soft } },
  ], { x: 0.6, y: 4.85, w: 12.1, h: 0.6, fontFace: BODY, fontSize: 14, margin: 0, isTextBox: true });
  s.addText([
    { text: "References  ", options: { bold: true, color: C.white } },
    { text: "[1] M.-M. Cheng, Z. Zhang, W.-Y. Lin, P. H. S. Torr, “BING: Binarized Normed Gradients for Objectness Estimation at 300fps,” CVPR 2014.  [2] J. Uijlings et al., “Selective Search for Object Recognition,” IJCV 2013.  [3] B. Alexe, T. Deselaers, V. Ferrari, “Measuring the Objectness of Image Windows,” TPAMI 2012.  [4] J. Hosang et al., “What makes for effective detection proposals?,” TPAMI 2016.", options: { color: C.soft } },
  ], { x: 0.6, y: 5.65, w: 12.1, h: 1.0, fontFace: BODY, fontSize: 11, margin: 0, valign: "top", isTextBox: true });
  s.addText("Thank you, questions?", { x: 0.6, y: 6.85, w: 6, h: 0.4, fontFace: HEAD, fontSize: 16, italic: true, color: C.white, margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- speaker plan + notes
// Who presents which slides. Change the names here and rebuild to reassign slides.
const SPEAKERS = ["Abhinav Dileep", "Ayyappadas M T", "Parthiv M", "Hari Sankar A", "Sai Kishen K M"];
const P = (i) => SPEAKERS[i];
// [speaker index, target seconds, script]. Each script is written to be read aloud.
const SCRIPT = [
  [0, 35, `Good morning. We are Group 10, and our project is "Fast Object Proposal Generation Using Handcrafted Gradient Features". It is based on the CVPR 2014 paper BING, by Cheng, Zhang, Lin and Torr. BING finds the regions of an image that are likely to contain objects, at around 300 frames per second, using only image gradients and bit operations, with no deep learning. I will introduce the problem, and then my teammates will cover our solution, design and results.`],
  [0, 45, `Why do we need object proposals? A classic detector slides a classifier over every position, every size and every shape of window. That is a hundred thousand windows or more per image, and almost all of them are background. Object proposals solve this. A cheap, class-agnostic score ranks all the windows first, so the expensive detector only has to look at the top thousand or so. BING is one of the fastest ways to do this: the paper reports 300 images per second.`],
  [0, 40, `Here is our problem definition, in the box: given an image, output a ranked list of boxes that covers as many objects as possible with few boxes, in milliseconds, using only handcrafted features. The constraints: no class labels, real time, CPU only, no deep networks. We measure three things: the detection rate, meaning the share of objects found in the top N boxes; MABO, meaning how tight the boxes are; and speed. We answer where objects are, not what they are.`],
  [1, 55, `Thank you, Abhinav. BING's key idea is simple: objects have closed outlines. Shrink any window to 8 by 8 pixels and look at its gradient strength, and a window around an object shows a ring of strong edges. A linear model with just 64 weights learns this template; on the right is the filter we learned, positive on the border and negative inside. To make it fast, filter and features are turned into bits. Beyond the paper, we made four contributions, shown in the cards: a from-scratch Python implementation; a head-to-head comparison with the authors' own C++ code; two details the paper's text leaves out, which we found that way; and our own benchmarks and ablations.`],
  [1, 45, `This is our system design, the same eight stages as in our proposal. The input image is shrunk to at most 500 pixels and converted to a colour space. We consider 36 window sizes. For each size, the whole image is resized so that every 8 by 8 patch corresponds to exactly one window. We compute the gradients, binarize them, and score every window with bit operations. Greedy suppression keeps the best spread-out windows, a per-size calibration makes scores comparable across sizes, and the top boxes come out.`],
  [2, 50, `Thank you, Ayyappadas. On the left, the top row shows the gradient maps of a coffee-cup image for three window sizes. The bottom row shows the four most significant bit planes of one map. The trick is on the right. The 64-weight filter is approximated by a weighted sum of plus-and-minus-one vectors. Each gradient value is approximated by its top four bits. Each set of 64 bits fits in one 64-bit integer, so a dot product becomes an AND followed by a POPCOUNT. That is 12 popcount operations per window instead of 64 multiply-adds. Our unit tests confirm that the bitwise score exactly equals the score computed with floats.`],
  [2, 45, `For evaluation we used three datasets. The main one is PASCAL VOC 2007, the paper's own benchmark: we train on 5,011 images and test on 4,952 images with over 12,000 objects. We also generated a synthetic benchmark of 900 images with cluttered backgrounds, and we hand-labelled 58 objects in 10 real photographs that are never used for training. Every method is scored with the same evaluator. As baselines we used the authors' own C++ code, Selective Search, sliding windows and random boxes.`],
  [3, 55, `Thank you, Parthiv. This is our main result, on PASCAL VOC 2007. Our final BING finds ${pct(VF.dr_at["1000"])} of all objects in its top 1000 proposals. The authors' own C++ code, on the same images with the same evaluator, finds ${pct(VO.dr_at["1000"])}. So our rebuild matches the original. When we use the authors' own evaluator, their code scores 96.0 percent, against 96.2 percent reported in the paper, so the paper's result reproduces. The dashed blue line is our first version. It stopped at 75 percent, and Sai Kishen will explain why in a moment.`],
  [3, 55, `This table compares BING with other methods. On VOC, BING finds more objects in the top 1000 than Selective Search, ${pct(VF.dr_at["1000"])} against ${pct(V["Selective Search (fast)"].dr_at["1000"])}, and it is about 70 times faster. But Selective Search draws tighter boxes, which shows in the MABO column. Compared with the authors' code, our version has the same accuracy on VOC, generalises better to real photos when trained only on synthetic data, and is slightly faster on one thread, ${Math.round(V.speed_ours["1 thread"])} against ${Math.round(1 / VO.sec_per_img_reported)} images per second. Their C++ code scales better to four threads.`],
  [4, 55, `Thank you, Hari. Our first version followed the paper's text exactly and reached only 75 percent on VOC. By running the authors' released code, we found two details that are not in the paper. First, window sizes: the paper lists 10 to 320 pixels, but the code uses 16 to 512 and clips large windows to the image, and VOC has many large objects. Second, suppression: we kept only local peaks, but the code uses greedy suppression, which keeps more good windows. As the table shows, each helps, and together they close the gap. Below is recall per VOC class: thin objects like bottles are hardest; cats and dogs are easiest.`],
  [4, 45, `Our ablation studies show what matters. The simple minus-one, zero, one gradient mask works as well as Sobel. RGB is the best single colour space. Just two bits of gradient are enough. And because the model has only 64 weights, even 10 training images give good results. The main limitation is that BING's boxes are coarse, because the window sizes are powers of two and the stride is 8 pixels, so recall drops when tighter boxes are required.`],
  [4, 35, `To conclude: simple gradients, a 64-weight linear model and bit operations find 94 percent of objects on PASCAL VOC, matching the authors' own code. Our version works on real photos without real training data, runs at ${Math.round(V.speed_ours["1 thread"])} images per second on one thread, and needs only 12 popcounts per window. In future we would like to make the boxes tighter, speed up multi-threading, and add a classifier to build a complete detector. Thank you. We are happy to take questions, and we can show a live webcam demo.`],
];
const HANDOVER = [, , "Ayyappadas will now explain our proposed solution.", , "Parthiv will now explain the bit-level trick and our experimental setup.", ,
  "Hari will now present our results.", , "Sai Kishen will now explain what made the difference, and conclude."];
// Target time per slide from the script length: 140 words per minute, rounded to 5 s.
SCRIPT.forEach((r) => { r[1] = Math.max(20, Math.round(r[2].split(/\s+/).length / 140 * 60 / 5) * 5); });
const fmt = (sec) => `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
const md = ["# Speaker script — Group 10", "",
  "Generated by `presentation/build_deck.js`; edit `SPEAKERS` / `SCRIPT` there and rebuild to change it.", "",
  "| Speaker | Slides | Time |", "|---|---|---|"];
SPEAKERS.forEach((name, k) => {
  const mine = SCRIPT.map((r, i) => [r, i]).filter(([r]) => r[0] === k);
  md.push(`| ${name} | ${mine.map(([, i]) => i + 1).join(", ")} | ~${fmt(mine.reduce((a, [r]) => a + r[1], 0))} |`);
});
md.push(`| **Total** | 1–${SCRIPT.length} | **~${fmt(SCRIPT.reduce((a, r) => a + r[1], 0))}** |`, "");
let clock = 0;
SCRIPT.forEach(([who, sec, text], i) => {
  const start = clock;
  clock += sec;
  const full = HANDOVER[i] ? `${text} ${HANDOVER[i]}` : text;
  const next = SCRIPT[i + 1];
  const cue = next && next[0] !== who ? `→ Hand over to ${P(next[0])}.` : "";
  if (slides[i]) {
    slides[i].addNotes(`SLIDE ${i + 1} · ${P(who)} · ~${sec} s (clock ${fmt(start)}–${fmt(clock)})\n\n${full}${cue ? "\n\n" + cue : ""}`);
    if (i > 0) {  // small presenter credit on every slide after the title
      const last = i === SCRIPT.length - 1;
      slides[i].addText(`Presented by ${P(who)}`, { x: last ? 7.3 : 0.6, y: last ? 6.9 : 7.08, w: last ? 5.4 : 6, h: 0.28, fontFace: BODY, fontSize: 9.5, color: last ? C.soft : C.muted, align: last ? "right" : "left", margin: 0, isTextBox: true });
    }
  }
  md.push(`## Slide ${i + 1} — ${P(who)} (~${sec} s, ${fmt(start)}–${fmt(clock)})`, "", full, "");
  if (cue) md.push(`**${cue}**`, "");
});
fs.writeFileSync(path.join(__dirname, "SPEAKER_NOTES.md"), md.join("\n"));

const out = path.join(__dirname, "BING_Objectness_Group10.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote", out));
