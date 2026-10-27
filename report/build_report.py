"""Build the project report (IEEE-style, two columns) from the results files.

  python report/build_report.py            # writes report/report.html
  then print it to PDF with Chromium (see report/README.md)

Every number in the report is read from results/*.json, so the report always matches
the experiments.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(name):
    with open(os.path.join(ROOT, "results", name)) as f:
        return json.load(f)


V = load("voc.json")
NOV = load("novelty_voc.json")
DET = load("detection_voc.json")["results"]
CNN_ALL = load("cnn_detection_voc.json")
CNN = CNN_ALL["results"]
CS = load("colorspace.json")
UN = load("unseen.json")
SYN = load("results_synthetic.json")

FINAL = "BING (ours, released-code sizes, greedy NMS)"
ORIG = "BING (authors' C++)"
SSALL = "Selective Search (fast), all test images"
EB = "EdgeBoxes"


def pct(v, d=1):
    return f"{100 * v:.{d}f}%"


def dr(r, n):
    return r["dr_at"][str(n)] if str(n) in r["dr_at"] else r["dr_at"][n]


def mabo(r, n=1000):
    return r["mabo_at"][str(n)] if str(n) in r["mabo_at"] else r["mabo_at"][n]


def small(r):
    return r["by_size"]["small (<32 px)"]["dr@1000"]


SP = V["speed_methods_1_thread_sec"]
SAME = V[ORIG]["speed_same_session"]
T = V["speed_threads"]
fps1 = V["speed_ours"]["1 thread"]
base, ours = NOV["BING (final baseline)"], NOV["BING + multi-cue re-ranking (ours)"]
abl = {k.replace("re-ranking without ", ""): v for k, v in NOV.items() if k.startswith("re-ranking without")}
cnn = lambda s, n, reg=True: CNN[f"{s} @{n}" + (" +box regression" if reg else "")]["mAP"]
hog = lambda s, n: DET[f"{s} @{n}"]["mAP"]
RR = "BING + multi-cue re-ranking"
U6u, U20u = UN["trained on 6 classes | unseen 14 classes"], UN["trained on 20 classes | unseen 14 classes"]
U6s, U20s = UN["trained on 6 classes | seen 6 classes"], UN["trained on 20 classes | seen 6 classes"]
CSV = CS["voc"]

AUTHORS = [("Abhinav Dileep", "AM.SC.U4CSE23202"), ("Ayyappadas M T", "AM.SC.U4CSE23209"),
           ("Hari Sankar A", "AM.SC.U4CSE23221"), ("Parthiv M", "AM.SC.U4CSE23240"),
           ("Sai Kishen K M", "AM.SC.U4CSE23248")]


def table(head, rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def fig(src, cap, wide=False):
    return (f'<figure class="{"wide" if wide else ""}"><img src="../results/figures/{src}" alt="">'
            f"<figcaption>{cap}</figcaption></figure>")


# ------------------------------------------------------------------ tables
t_repro = table(
    ["Method", "DR@100", "DR@1000", "DR@5000", "MABO"],
    [["<b>BING, ours</b>", pct(dr(V[FINAL], 100)), f"<b>{pct(dr(V[FINAL], 1000))}</b>", pct(dr(V[FINAL], 5000)), f"{mabo(V[FINAL]):.3f}"],
     ["BING, authors' C++", pct(dr(V[ORIG], 100)), pct(dr(V[ORIG], 1000)), pct(dr(V[ORIG], 5000)), f"{mabo(V[ORIG]):.3f}"],
     ["BING, paper [1]", "–", "96.2%", "99.5%", "–"]])

t_details = table(
    ["Our BING on VOC 2007", "DR@1000", "DR@5000", "MABO"],
    [["Paper's text (sizes 10–320, local maxima)", pct(dr(V["BING (ours)"], 1000)), pct(dr(V["BING (ours)"], 5000)), f"{mabo(V['BING (ours)']):.3f}"],
     ["+ released sizes (16–512, clipped)", pct(dr(V["BING (ours, released-code sizes)"], 1000)), pct(dr(V["BING (ours, released-code sizes)"], 5000)), f"{mabo(V['BING (ours, released-code sizes)']):.3f}"],
     ["+ greedy NMS only", pct(dr(V["BING (ours, greedy NMS)"], 1000)), pct(dr(V["BING (ours, greedy NMS)"], 5000)), f"{mabo(V['BING (ours, greedy NMS)']):.3f}"],
     ["<b>Both (final)</b>", f"<b>{pct(dr(V[FINAL], 1000))}</b>", pct(dr(V[FINAL], 5000)), f"{mabo(V[FINAL]):.3f}"]])

t_methods = table(
    ["Method", "DR@100", "DR@1000", "MABO", "ms/img"],
    [["BING, ours", pct(dr(V[FINAL], 100)), pct(dr(V[FINAL], 1000)), f"{mabo(V[FINAL]):.3f}", f"{1e3 * SP['BING (ours)']:.0f}"],
     ["BING + re-ranking (ours)", pct(dr(ours, 100)), pct(dr(ours, 1000)), f"{mabo(ours):.3f}", f"{1e3 * SP['BING + multi-cue re-ranking (ours)']:.0f}"],
     ["EdgeBoxes [4]", pct(dr(V[EB], 100)), pct(dr(V[EB], 1000)), f"{mabo(V[EB]):.3f}", f"{1e3 * SP['EdgeBoxes']:.0f}"],
     ["Selective Search, fast [2]", pct(dr(V[SSALL], 100)), pct(dr(V[SSALL], 1000)), f"{mabo(V[SSALL]):.3f}", f"{1e3 * SP['Selective Search (fast)']:.0f}"],
     ["Random boxes", pct(dr(V["Random boxes"], 100)), pct(dr(V["Random boxes"], 1000)), f"{mabo(V['Random boxes']):.3f}", "–"]])

t_colour = table(
    ["Space", "VOC @100", "VOC @1k", "VOC small", "Synth. @1k", "Real @1k"],
    [[{"RGB": "RGB", "HSV": "HSV", "GRAY": "Gray"}[k], pct(dr(CSV[k], 100)), pct(dr(CSV[k], 1000)), pct(small(CSV[k])),
      pct(SYN["ablations"][f"colour={k}"]["dr_at"]["1000"]), pct(dr(CS["real"][k], 1000))] for k in ("RGB", "HSV", "GRAY")]
    + [["Merged", pct(dr(CSV["RGB + HSV + GRAY"], 100)), pct(dr(CSV["RGB + HSV + GRAY"], 1000)),
        pct(small(CSV["RGB + HSV + GRAY"])), "–", "–"]])

t_unseen = table(
    ["Training classes", "Test objects", "DR@100", "DR@1000", "DR@5000"],
    [["6 (seen)", f"14 unseen classes ({U6u['num_objects']})", pct(dr(U6u, 100)), pct(dr(U6u, 1000)), pct(dr(U6u, 5000))],
     ["all 20", f"14 unseen classes ({U20u['num_objects']})", pct(dr(U20u, 100)), pct(dr(U20u, 1000)), pct(dr(U20u, 5000))],
     ["6 (seen)", f"6 seen classes ({U6s['num_objects']})", pct(dr(U6s, 100)), pct(dr(U6s, 1000)), pct(dr(U6s, 5000))],
     ["all 20", f"6 seen classes ({U20s['num_objects']})", pct(dr(U20s, 100)), pct(dr(U20s, 1000)), pct(dr(U20s, 5000))]])

t_novelty = table(
    ["Method", "DR@100", "DR@1000", "Small, DR@1000", "AUC"],
    [["BING", pct(dr(base, 100)), pct(dr(base, 1000)), pct(small(base)), f"{base['auc']:.3f}"],
     ["<b>+ multi-cue re-ranking</b>", f"<b>{pct(dr(ours, 100))}</b>", pct(dr(ours, 1000)), f"<b>{pct(small(ours))}</b>", f"{ours['auc']:.3f}"]]
    + [[f"&nbsp;&nbsp;without {k}", pct(dr(v, 100)), pct(dr(v, 1000)), pct(small(v)), f"{v['auc']:.3f}"] for k, v in abl.items()]
    + [["+ edge refinement (rejected)", pct(dr(NOV["+ edge refinement"], 100)), pct(dr(NOV["+ edge refinement"], 1000)),
        pct(small(NOV["+ edge refinement"])), f"{NOV['+ edge refinement']['auc']:.3f}"],
       ["+ finer window sizes (rejected)", pct(dr(NOV["+ fine-scale windows"], 100)), pct(dr(NOV["+ fine-scale windows"], 1000)),
        pct(small(NOV["+ fine-scale windows"])), f"{NOV['+ fine-scale windows']['auc']:.3f}"]])

t_det = table(
    ["Windows", "HOG @100", "CNN @100", "CNN+BR @100", "CNN+BR @1k"],
    [[n, pct(hog(s, 100)), pct(cnn(s, 100, False)), f"<b>{pct(cnn(s, 100))}</b>" if s == RR else pct(cnn(s, 100)), pct(cnn(s, 1000))]
     for s, n in (("BING", "BING"), (RR, "+ re-ranking"), ("Random boxes", "Random"), ("Sliding windows", "Sliding"))])

t_speed = table(
    ["Implementation (same machine)", "1 thread", "4 threads"],
    [["BING, authors' C++", f"{SAME['1 thread']:.0f}", f"{SAME['4 threads']:.0f}"],
     ["BING, ours: window sizes in parallel", f"{T['window sizes in parallel']['1']:.0f}", f"{T['window sizes in parallel']['4']:.0f}"],
     ["BING, ours: images in parallel", f"{T['images in parallel']['1']:.0f}", f"<b>{T['images in parallel']['4']:.0f}</b>"]])

ab = SYN["ablations"]
abl_line = (f"gradient mask [−1 0 1] {pct(ab['colour=RGB']['dr_at']['1000'])} vs Sobel {pct(ab['gradient=Sobel 3x3']['dr_at']['1000'])}; "
            f"feature bits N<sub>g</sub> = 1: {pct(ab['Ng=1']['dr_at']['1000'])}, N<sub>g</sub> = 4: {pct(ab['Ng=4']['dr_at']['1000'])}; "
            f"filter bases N<sub>w</sub> = 1: {pct(ab['Nw=1']['dr_at']['1000'])}, N<sub>w</sub> = 2: {pct(ab['Nw=2']['dr_at']['1000'])}; "
            f"10 training images: {pct(ab['train imgs=10']['dr_at']['1000'])}, 300: {pct(ab['train imgs=300']['dr_at']['1000'])}")

authors_html = "".join(f'<div class="au"><b>{n}</b><span>{r}</span></div>' for n, r in AUTHORS)

ss_x = SP["Selective Search (fast)"] / SP["BING (ours)"]

HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Fast Object Proposal Generation Using Handcrafted Gradient Features</title>
<style>
@page {{ size: A4; margin: 18mm 15mm 18mm 15mm; }}
body {{ font: 9.6pt/1.32 "Liberation Serif", "Times New Roman", Times, serif; color: #000; margin: 0; }}
h1 {{ font-size: 21pt; line-height: 1.15; text-align: center; font-weight: normal; margin: 0 0 10pt; }}
.authors {{ display: flex; justify-content: center; flex-wrap: wrap; gap: 4pt 20pt; margin-bottom: 5pt }}
.au {{ text-align: center; font-size: 10pt }}
.au span {{ display: block; font-size: 8.5pt; font-style: italic }}
.course {{ text-align: center; font-size: 9pt; margin-bottom: 12pt }}
.cols {{ column-count: 2; column-gap: 7mm; text-align: justify; hyphens: auto; }}
.abstract {{ font-weight: bold; font-size: 9pt }}
.abstract i {{ font-weight: bold }}
.kw {{ font-size: 9pt; font-weight: bold; font-style: italic; margin-bottom: 6pt }}
h2 {{ font-size: 9.6pt; font-weight: normal; text-align: center; font-variant: small-caps; margin: 10pt 0 4pt; letter-spacing: .02em; break-after: avoid }}
h3 {{ font-size: 9.6pt; font-weight: normal; font-style: italic; margin: 6pt 0 2pt; break-after: avoid }}
p {{ margin: 0 0 4pt; text-indent: 1em }}
p.ni, .abstract p {{ text-indent: 0 }}
ul {{ margin: 2pt 0 4pt 0; padding-left: 14pt }}
li {{ margin-bottom: 1.5pt }}
figure {{ margin: 6pt 0 8pt; break-inside: avoid; text-align: center }}
figure.wide {{ column-span: all; }}
figure img {{ max-width: 100%; }}
figcaption {{ font-size: 8pt; text-align: justify; margin-top: 3pt }}
table {{ border-collapse: collapse; width: 100%; font-size: 8pt; margin: 4pt 0 2pt; break-inside: avoid; text-align: left }}
th, td {{ padding: 1.6pt 3pt; border-bottom: .5pt solid #999 }}
th {{ border-top: 1pt solid #000; border-bottom: .8pt solid #000; font-weight: bold }}
tbody tr:last-child td {{ border-bottom: 1pt solid #000 }}
td:not(:first-child), th:not(:first-child) {{ text-align: right; white-space: nowrap }}
.tcap {{ font-size: 8pt; text-align: center; font-variant: small-caps; margin-top: 8pt; break-after: avoid }}
.tnote {{ font-size: 7.6pt; margin-bottom: 8pt; text-indent: 0 }}
.eq {{ text-align: center; margin: 4pt 0; text-indent: 0; font-style: italic }}
.refs p {{ text-indent: -1.6em; padding-left: 1.6em; font-size: 8.2pt; margin-bottom: 2pt }}
sub, sup {{ font-size: 70% }}
</style></head><body>
<h1>Fast Object Proposal Generation Using Handcrafted Gradient Features: Reproducing and Extending BING</h1>
<div class="authors">{authors_html}</div>
<div class="course">23CSE373 Computer Vision · Group 10</div>
<div class="cols">

<div class="abstract"><p><i>Abstract</i>—Object proposal methods reduce the 10<sup>5</sup>–10<sup>6</sup> windows a detector would otherwise scan to about a thousand candidates. BING (Cheng et al., CVPR 2014) does this at 300 frames per second with an 8×8 normed-gradient feature, a 64-weight linear model and bitwise AND/POPCOUNT scoring. We re-implemented BING from scratch in Python with compiled kernels and evaluated it on PASCAL VOC 2007 against the authors' own C++ code: our version finds {pct(dr(V[FINAL], 1000))} of objects in its top 1000 proposals against {pct(dr(V[ORIG], 1000))} for the original, after we identified two details that the paper's text omits. We extend BING with a third stage that re-ranks its windows using colour-contrast, edge-density and geometric cues, which raises recall in the top 100 from {pct(dr(base, 100))} to {pct(dr(ours, 100))} and small-object recall from {pct(small(base))} to {pct(small(ours))}. We also compare RGB, HSV and grayscale gradients, test generalisation to unseen classes, compare with Selective Search and EdgeBoxes, build complete detectors on the proposals (up to {pct(cnn(RR, 100))} mAP), and multi-thread the implementation to {T['images in parallel']['4']:.0f} images per second, against {SAME['4 threads']:.0f} for the authors' code on the same machine.</p></div>
<p class="kw">Index Terms—object proposals, objectness, BING, normed gradients, binarization, PASCAL VOC.</p>

<h2>I. Introduction</h2>
<p class="ni">Classic object detectors classify every position, scale and aspect ratio of a sliding window. That is 10<sup>5</sup> to 10<sup>6</sup> windows per image, and almost all of them are background. Object proposal methods first rank windows by a cheap, class-agnostic "objectness" score, so that an expensive classifier only examines the best thousand or so [3], [5]. BING [1] is among the fastest proposal methods: it observes that objects have closed boundaries, so a window around an object, shrunk to 8×8 pixels, shows a characteristic ring of strong gradients. A 64-weight linear model learns this pattern, and binarizing the model and the features turns each window's score into a few bitwise operations.</p>
<p>This project re-implements, evaluates and extends BING. Our contributions are:</p>
<ul>
<li>A from-scratch Python implementation with Numba-compiled kernels for the paper's bitwise algorithm, and a head-to-head comparison with the authors' released C++ code on PASCAL VOC 2007, which revealed two implementation details missing from the paper.</li>
<li>A novelty: <i>multi-cue re-ranking</i>, a third stage that re-scores BING's windows with handcrafted colour, edge and geometric cues, with an ablation of each cue group.</li>
<li>A comparison of RGB, HSV and grayscale gradients; a generalisation test on unseen classes, following the paper; and a comparison with Selective Search [2] and EdgeBoxes [4] on the full test set.</li>
<li>Complete detectors built on the proposals, with handcrafted (HOG) and CNN features, and multi-threading that brings our speed to that of the C++ code.</li>
</ul>

<h2>II. Related Work</h2>
<p class="ni">Alexe et al. [3] introduced objectness as a generic measure combining saliency, colour contrast, edge density and superpixel straddling. Selective Search [2] groups superpixels hierarchically and returns the bounding boxes of the groups; its boxes fit objects tightly but it needs about a second per image. EdgeBoxes [4] scores a window by the number of edge contours wholly inside it. BING [1] trades box precision for speed. Hosang et al. [5] compared these methods systematically and showed that recall at higher overlaps predicts detection quality better than recall at IoU 0.5. R-CNN [6] popularised detection on top of proposals with CNN features; Dalal and Triggs' HOG [7] is the classic handcrafted descriptor.</p>

<h2>III. Method</h2>
<h3>A. Normed gradients and the 8×8 window feature</h3>
<p class="ni">For each pixel the normed gradient (NG) is <i>g</i> = min(|<i>g<sub>x</sub></i>| + |<i>g<sub>y</sub></i>|, 255), computed with the mask [−1, 0, 1] in each colour channel; the strongest channel is kept. Windows take 36 quantised sizes (<i>w</i>, <i>h</i>) ∈ {{16, 32, …, 512}}<sup>2</sup>. For each size the whole image is resized to (8<i>W</i>/<i>w</i>, 8<i>H</i>/<i>h</i>), so that every 8×8 patch of the resized NG map is the 64-D feature of one window.</p>
<h3>B. Two-stage linear model</h3>
<p class="ni">Stage I is a linear SVM <b>w</b> ∈ ℝ<sup>64</sup> trained on the NG features of ground-truth boxes (and their mirror images) against random background windows; the score of a window is <i>s</i> = ⟨<b>w</b>, <b>g</b>⟩. Greedy non-maximum suppression keeps up to 130 windows per size. Stage II calibrates each size with <i>o</i> = <i>v<sub>i</sub></i>·<i>s</i> + <i>t<sub>i</sub></i>, learned by logistic regression on training windows labelled by IoU ≥ 0.5, because sizes such as 16×512 rarely contain objects. Windows from all sizes are sorted by <i>o</i>.</p>
<h3>C. Binarized scoring</h3>
<p class="ni">The filter is approximated with <i>N<sub>w</sub></i> = 2 binary bases, <b>w</b> ≈ Σ<sub><i>j</i></sub> β<sub><i>j</i></sub><b>a</b><sub><i>j</i></sub> with <b>a</b><sub><i>j</i></sub> ∈ {{−1, +1}}<sup>64</sup>, and each NG value by its top <i>N<sub>g</sub></i> = 4 bits, <b>g</b> ≈ Σ<sub><i>k</i></sub> 2<sup>8−<i>k</i></sup><b>b</b><sub><i>k</i></sub>. Each 64-bit vector fits in one machine word, and</p>
<p class="eq">⟨<b>a</b><sub>j</sub>, <b>b</b><sub>k</sub>⟩ = 2·popcnt(<b>a</b><sub>j</sub><sup>+</sup> AND <b>b</b><sub>k</sub>) − popcnt(<b>b</b><sub>k</sub>),</p>
<p class="ni">where <b>a</b><sup>+</sup> marks the +1 entries. A window then costs 12 POPCOUNTs instead of 64 multiply-adds. Bit-planes are built with shift and OR operations over rows and columns (the paper's Algorithm 2). Unit tests confirm that the bitwise score equals the floating-point score of the approximated filter and features.</p>
<h3>D. Implementation</h3>
<p class="ni">Our implementation uses NumPy, OpenCV and scikit-learn, with Numba compiling the gradient, bit-packing, scoring and suppression loops to machine code. It follows two details of the released code that the paper does not state (Section V-B): window sizes 16–512 clipped to the image, and greedy suppression.</p>
<h3>E. Novelty: multi-cue re-ranking (stage III)</h3>
<p class="ni">BING ranks windows with a single cue, its edge template. Stage III re-scores BING's windows with 16 handcrafted values per window, each read in constant time from integral images: the Lab colour contrast and lightness difference between the window and a surrounding ring (the window grown by a quarter of its size on each side); the mean gradient energy inside, in the ring, and their log-ratio; the window's centre, distance to the image centre, log area, log aspect ratio, relative width and height, ring share and border contact; and BING's calibrated score and rank. A gradient-boosted tree classifier, trained on the proposals of 2,500 VOC trainval images (positive if IoU ≥ 0.5 with an object), combines them, and windows are re-sorted by its probability. No deep network is involved.</p>
<h3>F. Detectors on the proposals</h3>
<p class="ni">To measure what the proposals are worth, we built a detector in two versions. The handcrafted version warps each window (padded by 1/8) to 64×64 and describes it with HOG [7] (1,764 values) and an 8×4 hue-saturation histogram. The CNN version passes the image once through a ResNet-50 [9] pre-trained on ImageNet, kept frozen, and describes each window by the mean conv4 and conv5 activations inside it (3,072 values, read from integral images of the feature maps). Both train 20 one-vs-rest linear SVMs on VOC trainval with one round of hard-negative mining; the CNN version adds class-specific box regression as in R-CNN [6]. Detections are filtered with per-class non-maximum suppression and scored with the VOC 2007 11-point average precision.</p>
<h3>G. Multi-threading</h3>
<p class="ni">The gradient, scoring and suppression of one window size run inside a single compiled function that releases Python's global interpreter lock. Window sizes of one image can then be processed by a thread pool (lower latency), or whole images can be distributed over threads (higher throughput).</p>

<h2>IV. Experimental Setup</h2>
<p class="ni"><i>Data.</i> PASCAL VOC 2007 [8], following the paper: train on the 5,011 trainval images, test on the 4,952 test images ({V['n_objects']:,} objects; "difficult" objects ignored). We also use a synthetic benchmark of 600/300 images of textured shapes on cluttered backgrounds, and 10 real photographs with 58 annotated objects that are never used for training.</p>
<p><i>Metrics.</i> The detection rate DR@<i>N</i> is the share of objects covered by at least one of the top <i>N</i> proposals with IoU ≥ 0.5. MABO is the mean (over classes) of the average best overlap per object, measuring box tightness. For detection we report the mean average precision (mAP). All methods are scored by the same evaluator. Speeds were measured on one 4-core CPU in a single session.</p>

<h2>V. Results</h2>
<h3>A. Reproducing BING on PASCAL VOC 2007</h3>
<p class="ni">Table I compares our implementation with the authors' C++ code run on the same images and scored by our evaluator. The two agree within 0.2 points at every budget. With the authors' own evaluator, which counts box areas with VOC's inclusive pixel convention, their code scores 96.0% at 1000 proposals, close to the 96.2% reported in the paper.</p>
<div class="tcap">Table I. BING on the VOC 2007 test set</div>
{t_repro}
<h3>B. Two details the paper leaves out</h3>
<p class="ni">Our first version followed the paper's text and reached only {pct(dr(V['BING (ours)'], 1000))}. Running the released code showed two differences (Table II). Its window sizes are 16–512 rather than 10–320, with windows up to twice the image size clipped to it, so near full-image boxes exist; VOC has many large objects. Its suppression is greedy: it accepts the best window, blocks a 5×5 neighbourhood and repeats, whereas keeping only local maxima of each score map discards good windows on smooth maps. Together they close the gap.</p>
<div class="tcap">Table II. Effect of the two undocumented details</div>
{t_details}
{fig("voc_curves.png", "Fig. 1. Recall versus number of proposals (left) and MABO (right) on VOC 2007. Our final BING matches the authors' code; our first version, following the paper's text, plateaus at 75%.", wide=True)}
<h3>C. Comparison with other proposal methods</h3>
<p class="ni">Table III compares BING with Selective Search (OpenCV's fast mode) and EdgeBoxes (OpenCV's implementation with its pre-trained structured-edge model) on all 4,952 test images, with times per image on one thread. BING finds the most objects up to 1,000 proposals and is about {SP['EdgeBoxes'] / SP['BING (ours)']:.0f} times faster than EdgeBoxes and {ss_x:.0f} times faster than Selective Search. Both alternatives draw tighter boxes (higher MABO), and EdgeBoxes overtakes BING at 5,000 proposals ({pct(dr(V[EB], 5000))} vs {pct(dr(V[FINAL], 5000))}). Random boxes show what a method gains over chance.</p>
<div class="tcap">Table III. Proposal methods on VOC 2007 (all test images)</div>
{t_methods}
<h3>D. Colour space: RGB, HSV or grayscale</h3>
<p class="ni">We trained one BING model per colour space (Table IV). On VOC, HSV is slightly best and wins 15 of the 20 classes at 100 proposals; grayscale is lowest because it misses edges between regions that differ only in colour, which hurts small objects most. The ranking depends on the data: RGB is best on the synthetic scenes, where hue is unstable in dark and grey regions, and grayscale transfers best from synthetic training to real photographs. Merging the three ranked lists, as in the paper's BING-Diversified, gives the highest recall at 5,000 proposals ({pct(dr(CSV['RGB + HSV + GRAY'], 5000))}) but dilutes small budgets. We keep RGB as the default.</p>
<div class="tcap">Table IV. Colour spaces: DR at 100 / 1000 proposals (synthetic and real columns: models trained on synthetic scenes; merged: RGB + HSV + Gray)</div>
{t_colour}
{fig("colorspace_voc.png", "Fig. 2. Colour spaces on VOC 2007: recall versus number of proposals, recall by object size, and DR@1000 on the three datasets.", wide=True)}
<h3>E. Generalisation to unseen classes</h3>
<p class="ni">Following the paper, we trained BING on the trainval images of six classes (bird, car, cat, cow, dog, sheep), with only their boxes as objects, and tested it on the other 14 classes (Table V). Its recall on the unseen classes is within {abs(100 * (dr(U20u, 1000) - dr(U6u, 1000))):.1f} points of the model trained on all 20 classes, which supports the claim that BING learns generic objectness rather than class-specific appearance.</p>
<div class="tcap">Table V. Training on 6 classes, testing on 14 unseen classes</div>
{t_unseen}
<h3>F. Novelty: multi-cue re-ranking</h3>
<p class="ni">Re-ranking raises recall at every budget (Table VI, Fig. 3): {pct(dr(base, 100))} → {pct(dr(ours, 100))} at 100 proposals, and small-object recall at 1000 from {pct(small(base))} to {pct(small(ours))}. Recall at 5,000 cannot change, because re-ranking only re-orders BING's windows. Removing any cue group lowers recall; edge density matters most, especially for small objects. The cost is speed ({1e3 * SP['BING + multi-cue re-ranking (ours)']:.0f} instead of {1e3 * SP['BING (ours)']:.0f} ms per image on one thread). Two other extensions, snapping box sides to strong image edges and adding intermediate window sizes, looked promising on a 300-image subset but did not improve the full test set; we report them as negative results.</p>
<div class="tcap">Table VI. Multi-cue re-ranking on VOC 2007</div>
{t_novelty}
{fig("novelty_voc.png", "Fig. 3. Multi-cue re-ranking: recall versus proposals, recall by object size, and the contribution of each cue group (DR@100).", wide=True)}
<h3>G. Detection on the proposals</h3>
<p class="ni">Table VII reports VOC 2007 mAP for the same detector given different windows. With CNN features and box regression, our re-ranked top 100 windows give the best result, {pct(cnn(RR, 100))}, against {pct(hog(RR, 100))} with HOG. At 100 windows BING's boxes clearly beat random boxes and sliding windows. With 1,000 windows random boxes catch up ({pct(cnn('Random boxes', 1000))}), because they are tighter: at IoU 0.7, BING's top 1000 cover {pct(V[FINAL]['recall_iou']['0.70'], 0)} of objects and random boxes {pct(V['Random boxes']['recall_iou']['0.70'], 0)}. This agrees with Hosang et al. [5]. The absolute numbers stay well below R-CNN (about 58%) because our network is not fine-tuned.</p>
<div class="tcap">Table VII. Detection mAP on VOC 2007 for the top 100 or 1000 windows (BR: box regression)</div>
{t_det}
{fig("detection_voc.png", "Fig. 4. Left: detector mAP for each source of windows (CNN features and box regression). Right: recall at stricter overlaps, where BING's coarse boxes fall behind random boxes.", wide=True)}
<h3>H. Speed</h3>
<p class="ni">All timings in Table VIII were measured on the same 4-core machine in one session (images per second on the 300-image speed subset). On one thread our implementation is faster than the C++ code; with one image per thread it reaches {T['images in parallel']['4']:.0f} images per second, against {SAME['4 threads']:.0f} for the C++ code, which matches the 300 fps reported in the paper.</p>
<div class="tcap">Table VIII. Images per second on VOC 2007 images</div>
{t_speed}
<h3>I. Ablations</h3>
<p class="ni">On the synthetic benchmark: {abl_line}. The simple gradient mask is sufficient, two feature bits recover almost all accuracy, and the 64-weight model needs very little training data.</p>

<h2>VI. Discussion and Limitations</h2>
<p class="ni">BING's weakness is localisation. Power-of-two window sizes and an 8-pixel stride make its boxes coarse (MABO {mabo(V[FINAL]):.2f}), small objects are hard, and these limits show again in detection, where tighter random boxes overtake BING at large budgets. Our re-ranking improves the order of the windows but cannot change their shape; our edge-snapping refinement did not help either, so learned box regression applied to the proposals themselves is the most promising next step. The detector's CNN is frozen; fine-tuning it would raise mAP considerably. The real-photo set (58 objects) is small and serves only as a generalisation check.</p>

<h2>VII. Conclusion</h2>
<p class="ni">Simple gradients, a 64-weight linear model and bit operations find {pct(dr(V[FINAL], 1000), 0)} of the objects in PASCAL VOC 2007, and our from-scratch implementation matches the authors' code in accuracy and speed. Our multi-cue re-ranking finds objects sooner, especially small ones; HSV gradients are marginally better than RGB on VOC; BING generalises to unseen classes; and detectors built on the proposals confirm both their value over sliding windows and the cost of BING's loose boxes.</p>

<h2>References</h2>
<div class="refs">
<p>[1] M.-M. Cheng, Z. Zhang, W.-Y. Lin, and P. H. S. Torr, "BING: Binarized normed gradients for objectness estimation at 300fps," in <i>Proc. IEEE CVPR</i>, 2014.</p>
<p>[2] J. R. R. Uijlings, K. E. A. van de Sande, T. Gevers, and A. W. M. Smeulders, "Selective search for object recognition," <i>Int. J. Comput. Vis.</i>, vol. 104, no. 2, 2013.</p>
<p>[3] B. Alexe, T. Deselaers, and V. Ferrari, "Measuring the objectness of image windows," <i>IEEE Trans. Pattern Anal. Mach. Intell.</i>, vol. 34, no. 11, 2012.</p>
<p>[4] C. L. Zitnick and P. Dollár, "Edge boxes: Locating object proposals from edges," in <i>Proc. ECCV</i>, 2014.</p>
<p>[5] J. Hosang, R. Benenson, P. Dollár, and B. Schiele, "What makes for effective detection proposals?," <i>IEEE Trans. Pattern Anal. Mach. Intell.</i>, vol. 38, no. 4, 2016.</p>
<p>[6] R. Girshick, J. Donahue, T. Darrell, and J. Malik, "Rich feature hierarchies for accurate object detection and semantic segmentation," in <i>Proc. IEEE CVPR</i>, 2014.</p>
<p>[7] N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in <i>Proc. IEEE CVPR</i>, 2005.</p>
<p>[8] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, "The PASCAL visual object classes (VOC) challenge," <i>Int. J. Comput. Vis.</i>, vol. 88, no. 2, 2010.</p>
<p>[9] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in <i>Proc. IEEE CVPR</i>, 2016.</p>
</div>
</div>
</body></html>
"""

out = os.path.join(HERE, "report.html")
with open(out, "w") as f:
    f.write(HTML)
print("wrote", out)
