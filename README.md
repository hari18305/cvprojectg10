# Fast Object Proposal Generation Using Handcrafted Gradient Features

**23CSE373 Computer Vision — Group 10**

| Name | Roll number |
|---|---|
| Abhinav Dileep | AM.SC.U4CSE23202 |
| Ayyappadas M T | AM.SC.U4CSE23209 |
| Hari Sankar A | AM.SC.U4CSE23221 |
| Parthiv M | AM.SC.U4CSE23240 |
| Sai Kishen K M | AM.SC.U4CSE23248 |

**Base paper:** M.-M. Cheng, Z. Zhang, W.-Y. Lin, P. H. S. Torr, *"BING: Binarized Normed Gradients for Objectness Estimation at 300fps,"* CVPR 2014.
**Base repository:** [torrvision/Objectness](https://github.com/torrvision/Objectness) (C++). This project re-implements the method from scratch in Python (OpenCV, NumPy, scikit-learn, Numba). It does not use any deep learning. The authors' C++ code is also built and run on the same data as a reference (section 5.7).

**Presentation:** [`presentation/BING_Objectness_Group10.pptx`](presentation/BING_Objectness_Group10.pptx), 12 slides for a 10-minute talk, following the five required sections (introduction, problem definition, proposed solution and novelty, design and solution, result analysis), with timed speaker notes. A PDF export is also included.

---

## 1. What the system does

It answers *"Which regions of this image are likely to contain objects?"*: given an image, it returns a ranked list of bounding boxes (object proposals) and an objectness heat-map.

```
Input image → Pre-processing → Candidate windows (36 sizes) → Gradient features
→ Binarized normed gradients → Objectness score → Ranking (NMS + calibration) → Top proposals
```

| Stage | Implementation |
|---|---|
| Pre-processing | The longer side is limited to 500 px (VOC-like). Colour space can be RGB, HSV, Lab or Gray. |
| Candidate windows | 36 quantised sizes (w, h) ∈ {10, 20, 40, 80, 160, 320}². The image is resized to (8W/w, 8H/h), so every 8×8 patch is one window. |
| Gradient features | Normed gradient `g = min(|gx|+|gy|, 255)` with a `[-1 0 1]` mask, taking the max over colour channels. |
| Binarized NG | The top `Ng = 4` bit-planes are packed into 64-bit words with shift/OR operations (paper Alg. 2). |
| Objectness score | Stage I: linear SVM filter `w` (64 weights). It is approximated by `Nw = 2` binary bases (Alg. 1), and scoring uses only `AND` + `POPCOUNT`. |
| Ranking | Non-maximum suppression on each score map, keeping the top 130 per size. Stage II computes a per-size calibration `o = v·s + t`. |
| Output | Boxes sorted by calibrated objectness, plus a heat-map. |

## 2. Repository layout

```
bing/
  features.py    NG maps, colour spaces, [-1 0 1] vs Sobel, per-box training features
  binary.py      Alg. 1 filter binarization, 64-bit packing, AND+POPCOUNT scorer (numpy)
  fast.py        Numba JIT kernels: NG map + Alg. 2 bitwise scoring (parallel)
  model.py       BING class: two-stage training, NMS, ranking, heat-map, save/load
  metrics.py     DR-#WIN curves, MABO, AUC, recall-vs-IoU
  datasets.py    synthetic scene generator, real-photo set, PASCAL VOC 2007 loader
scripts/
  make_synthetic.py   generate the synthetic benchmark
  run_experiments.py  train + every experiment → results/results_<dataset>.json
  make_figures.py     all plots → results/figures/
  demo.py             run on any image or a webcam
  export_voc_format.py  write our datasets in PASCAL VOC layout (for the authors' code)
  run_tier1.py        authors' code vs ours, full Selective Search, per-size/shape recall
baselines/original_bing/  build + run the authors' C++ BING (CLI entry point, OpenCV 4 patch)
presentation/         slide deck (.pptx + .pdf) and its generator (build_deck.js)
models/               trained models (RGB / HSV / Gray)
data/real/            10 real photos with 58 hand-labelled boxes
tests/                unit tests (bitwise score == float score, numba == numpy, IoU …)
```

## 3. Quick start

```bash
pip install -r requirements.txt

# Try it on an image. Writes results/demo/<name>_top20.jpg and <name>_heat.jpg
python scripts/demo.py --image data/real/coffee.jpg --top 20
python scripts/demo.py --webcam                     # live demo

# Reproduce everything (about 25 min on 4 CPU cores)
python scripts/make_synthetic.py                    # 600 train / 300 test scenes
python scripts/run_experiments.py                   # train + evaluate + ablations
python scripts/make_figures.py                      # figures
node presentation/build_deck.js                     # rebuild the slides (needs `npm i pptxgenjs`)
python -m pytest tests -q
```

To follow the paper's protocol on **PASCAL VOC 2007** (train on `trainval`, test on `test`):

```bash
python scripts/run_experiments.py --dataset voc --voc-root /path/to/VOCdevkit/VOC2007
```

To run the **authors' C++ code** on the same images and compare (needs `cmake`, a C++ compiler and `libopencv-dev`):

```bash
python scripts/export_voc_format.py --out /tmp/voc_synth                        # synthetic train/test
python scripts/export_voc_format.py --out /tmp/voc_real500 --test real --max-side 500
bash baselines/original_bing/run_original.sh /tmp/voc_synth 1                    # 1 thread
bash baselines/original_bing/run_original.sh /tmp/voc_real500 1
python scripts/run_tier1.py --orig-synth /tmp/voc_synth --orig-real-500 /tmp/voc_real500
```

The same `run_original.sh` works on a real VOC 2007 folder.

Using the model from Python:

```python
import cv2
from bing import BING
model = BING.load("models/bing_synthetic_rgb.pkl")
boxes, scores = model.propose(cv2.imread("image.jpg"), top=1000)   # boxes: x1, y1, x2, y2
```

## 4. Evaluation data

* **Synthetic benchmark** (`data/synthetic`): 600 training and 300 test images of 320–500 px, containing 1–5 objects each (915 test objects). Objects are ellipses, rounded rectangles, polygons and blobs, with random colour, shading, texture and inner parts, and sometimes an outline. Backgrounds are textured or smooth colour fields crossed by random clutter lines, which are strong edges that do not belong to any object. Noise and blur are added.
* **Real photographs** (`data/real`): 10 natural images that ship with scikit-image, scikit-learn and matplotlib, with 58 hand-labelled boxes (people, cup, saucer, spoon, coins, motorcycle, pagoda, flowers, …). The model never sees a real image during training.
* **PASCAL VOC 2007**: a loader is included and verified on our data exported in VOC format, but the dataset itself could not be downloaded in our build environment.

Metrics: **DR**, the fraction of objects covered by a proposal with IoU ≥ 0.5, reported for N proposals. **MABO** is the mean best IoU per object. **AUC** is the mean DR over a log-spaced #WIN axis from 1 to 5000.

## 5. Results

### 5.1 Comparison with baselines (synthetic test set, 300 images)

| Method | DR@100 | DR@1000 | MABO@1000 | AUC |
|---|---|---|---|---|
| **BING (binary, Nw=2, Ng=4)** | **84.0 %** | **95.1 %** | 0.658 | **0.825** |
| BING (float filter w) | 82.7 % | 94.6 % | 0.658 | 0.820 |
| BING stage I only (no calibration) | 73.2 % | 93.8 % | 0.653 | 0.758 |
| BING-Diversified (RGB + HSV + Gray) | 79.0 % | **97.0 %** | 0.675 | 0.808 |
| BING, authors' C++ code (same data, same evaluator) | 80.2 % | 94.9 % | 0.639 | 0.794 |
| Selective Search, fast mode (all 300 images) | **88.4 %** | **97.5 %** | **0.895** | 0.796 |
| Random boxes | 26.3 % | 53.6 % | 0.518 | 0.365 |
| Sliding windows (random order) | 0.8 % | 7.0 % | 0.314 | 0.052 |

Selective Search runs at about 1.4 img/s; our BING at 160 img/s (4 threads) and the authors' C++ BING at 104 img/s (1 thread) or 374 img/s (4 threads).

### 5.2 Real photographs (58 objects, model trained on synthetic data only)

| Method | DR@100 | DR@1000 | MABO@1000 |
|---|---|---|---|
| **BING (binary)** | 60.3 % | 86.2 % | 0.655 |
| BING-Diversified | 44.8 % | 91.4 % | 0.658 |
| BING stage I only | 31.0 % | 75.9 % | 0.606 |
| BING, authors' C++ code (images shrunk to 500 px) | 27.6 % | 75.9 % | 0.571 |
| BING, authors' C++ code (original size) | 29.3 % | 70.7 % | 0.542 |
| Selective Search (fast) | 77.6 % | 100 % | 0.885 |
| Random boxes | 24.1 % | 48.3 % | 0.517 |

### 5.3 Speed (images/s, 4-core CPU, about 450×300 px images, 100 test images)

| Backend | Scoring | Resize | img/s |
|---|---|---|---|
| numpy | binary | INTER_AREA | 25 |
| numpy | float (OpenCV correlation) | INTER_AREA | 43 |
| numpy | binary | bilinear | 37 |
| numpy | float | bilinear | 82 |
| **Numba** | **binary (AND + POPCOUNT)** | **bilinear** | **160** |
| Numba | float | bilinear | 188 |

Per image, time breaks down as follows: resize 12.9 ms (INTER_AREA), gradient 1.5 ms, bitwise scoring 3.7 ms, NMS 1.6 ms. Training takes about 30 s per model on 600 images.

### 5.4 Ablations (DR@1000 on the synthetic test set)

| Factor | Result |
|---|---|
| Gradient mask | `[-1 0 1]` 95.1 %, Sobel 95.1 %. The simplest mask is enough. |
| Colour space | RGB 95.1 %, HSV 94.1 %, Lab 94.1 %, Gray 89.3 %. The spaces are complementary: Diversified reaches 97.0 %. |
| Resize interpolation | INTER_AREA 95.1 %, bilinear 95.6 %. Bilinear is also about 2–5× faster. |
| Filter bases Nw | 1 → 95.1 %, 2 → 95.1 %, 3 → 94.9 %, 4 → 95.3 %. DR@100 is 82.8 / 84.0 / 82.8 / 83.7 %. |
| Feature bits Ng | 1 → 92.8 %, 2 → 95.8 %, 3 → 95.1 %, 4 → 95.1 %, 5 → 95.2 %, 6 → 94.9 %. |
| Proposals per size | 10 → 85.7 %, 30 → 93.0 %, 60 → 94.4 %, 130 → 95.1 %, 250 → 95.1 %. |
| Training images | 10 → 79.2 %, 50 → 92.6 %, 150 → 94.2 %, 300 → 95.4 %, 600 → 95.1 %. |

### 5.5 Observations

1. **Binarization is almost free.** With 2 binary bases and 4 bit-planes, the score needs 12 POPCOUNTs per window instead of 64 multiply-adds, and recall does not drop. The unit tests confirm that the bitwise score equals the float dot product of the approximations.
2. **Stage II calibration matters** at small budgets: it adds 11 points of DR@100 on synthetic images and 29 points on real ones. Some sizes (for example 10×320) rarely contain objects.
3. **The main weakness of BING is localisation.** Recall is high at IoU 0.5 but falls quickly at stricter thresholds, and MABO (≈0.66) is well below that of Selective Search (≈0.88). This follows from the power-of-two sizes and the 8-pixel stride, and it matches published analyses (Hosang et al., TPAMI 2016).
4. **The "closed boundary" cue transfers.** A model trained only on synthetic shapes finds 86 % of real objects in its top 1000 proposals.
5. **In Python, speed depends mostly on engineering.** Resizing dominates the runtime. OpenCV's SIMD float correlation is competitive with our Numba popcount kernel. The paper's 300 fps comes from hand-optimised C++.

### 5.7 Our implementation vs the authors' C++ code

The authors' code ([torrvision/Objectness](https://github.com/torrvision/Objectness)) was built with three small compatibility fixes for OpenCV 4 (`baselines/original_bing/patch_opencv4.sh`; the algorithm is untouched). It was trained on the same 600 synthetic images and tested on the same images, and its proposals were scored with our evaluator.

| | Ours (Python + Numba) | Authors' C++ |
|---|---|---|
| DR@100 / DR@1000, synthetic | **84.0 %** / **95.1 %** | 80.2 % / 94.9 % |
| MABO@1000, synthetic | **0.658** | 0.639 |
| DR@1000, real photos | **86.2 %** | 75.9 % (shrunk to 500 px) |
| Speed, 1 thread | – | 104 img/s (9.6 ms) |
| Speed, 4 threads | 160 img/s | **374 img/s** (2.7 ms) |
| Window sizes | {10 … 320}² (as in the paper) | {16 … 512}² (powers of two between 10 and 500) |

Our rebuild matches the authors' code in recall and is slightly better at small budgets and on real photos. On the real photos the authors' model misses the largest objects (people, cat, pagoda). The C++ code is about 2.3× faster, which is the gap between Numba and hand-written C++ with hardware popcount.

**Recall by object size and shape** (DR@100, synthetic):

| | small (<32 px) | medium | large (>96 px) | ellipse | rounded rect | blob | polygon | star |
|---|---|---|---|---|---|---|---|---|
| Ours | 65 % | 84 % | 94 % | 92 % | 88 % | 82 % | 81 % | 77 % |
| Authors' C++ | 49 % | 81 % | 95 % | 89 % | 82 % | 79 % | 78 % | 74 % |
| Selective Search | 84 % | 86 % | 95 % | 91 % | 90 % | 89 % | 90 % | 83 % |

Small objects and thin, star-shaped outlines are the hardest for BING. With 1000 proposals every group reaches at least 91 %.

### 5.6 Figures

All figures are in `results/figures/`:

* `dr_*.png`, `mabo_*.png`, `recall_iou_*.png`: accuracy curves
* `filter_w.png`, `filter_bases.png`: the learned filter and its binary approximations
* `ng_pipeline.png`: NG maps and bit-planes
* `calibration.png`: stage II weight per window size
* `top8_*.png`, `found_*.png`, `heatmaps_real.png`: qualitative results
* `speed.png`, `stage_time.png`, `ablations.png`
* `tier1_vs_original.png`, `tier1_breakdown.png`: authors' code vs ours, recall by size and shape

## 6. Limitations and future work

* Localisation is coarse. Refining the top boxes or adding intermediate window sizes would raise MABO.
* The full PASCAL VOC 2007 evaluation still has to be run. The official host only serves plain HTTP, which our build environment cannot reach; the loader and scripts are ready (`run_experiments.py --dataset voc`, `run_original.sh`).
* A SIMD C++/Cython popcount kernel would be needed to reach the paper's 300 fps.
* Training on real annotated images would close the gap between synthetic and real data.
* Feeding the proposals into a classifier (HOG + SVM, or a small CNN) would give a complete detector.

## References

1. M.-M. Cheng, Z. Zhang, W.-Y. Lin, P. H. S. Torr, "BING: Binarized Normed Gradients for Objectness Estimation at 300fps," CVPR 2014.
2. J. Uijlings, K. van de Sande, T. Gevers, A. Smeulders, "Selective Search for Object Recognition," IJCV 2013.
3. B. Alexe, T. Deselaers, V. Ferrari, "Measuring the Objectness of Image Windows," TPAMI 2012.
4. J. Hosang, R. Benenson, P. Dollár, B. Schiele, "What makes for effective detection proposals?," TPAMI 2016.
5. P. Arbeláez, J. Pont-Tuset, J. T. Barron, F. Marques, J. Malik, "Multiscale Combinatorial Grouping," CVPR 2014.
