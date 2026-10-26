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

---

## 1. What the system does

It answers *"Which regions of this image are likely to contain objects?"*: given an image, it returns a ranked list of bounding boxes (object proposals) and an objectness heat-map.

```
Input image → Pre-processing → Candidate windows (36 sizes) → Gradient features
→ Binarized normed gradients → Objectness score → Ranking (NMS + calibration) → Top proposals
→ (our extension) multi-cue re-ranking → (our extension) HOG + SVM detector
```

| Stage | Implementation |
|---|---|
| Pre-processing | The longer side is limited to 500 px (VOC-like). Colour space can be RGB, HSV, Lab or Gray. |
| Candidate windows | 36 quantised sizes: (w, h) ∈ {10, 20, …, 320}² as in the paper, or {16, 32, …, 512}² clipped to the image as in the authors' code (used for VOC). The image is resized to (8W/w, 8H/h), so every 8×8 patch is one window. |
| Gradient features | Normed gradient `g = min(|gx|+|gy|, 255)` with a `[-1 0 1]` mask, taking the max over colour channels. |
| Binarized NG | The top `Ng = 4` bit-planes are packed into 64-bit words with shift/OR operations (paper Alg. 2). |
| Objectness score | Stage I: linear SVM filter `w` (64 weights). It is approximated by `Nw = 2` binary bases (Alg. 1), and scoring uses only `AND` + `POPCOUNT`. |
| Ranking | Greedy non-maximum suppression on each score map (as in the authors' code), keeping 130 windows per size. Stage II computes a per-size calibration `o = v·s + t`. |
| Output | Boxes sorted by calibrated objectness, plus a heat-map. |

## 2. Repository layout

```
bing/
  features.py    NG maps, colour spaces, [-1 0 1] vs Sobel, per-box training features
  binary.py      Alg. 1 filter binarization, 64-bit packing, AND+POPCOUNT scorer (numpy)
  fast.py        Numba JIT kernels: NG map + Alg. 2 bitwise scoring (parallel)
  model.py       BING class: two-stage training, NMS, ranking, heat-map, save/load,
                 window sizes scored in parallel threads (n_jobs)
  rerank.py      our stage III: multi-cue re-ranking of BING's proposals (section 6.1)
  refine.py      gradient-guided box refinement (tried; did not help, section 6.1)
  detector.py    complete detector on proposals: HOG + colour features, linear SVMs, VOC AP
  cnn.py         CNN window features (frozen ResNet-50), joint linear SVMs, box regression
  metrics.py     DR-#WIN curves, MABO, AUC, recall-vs-IoU
  datasets.py    synthetic scene generator, real-photo set, PASCAL VOC 2007 loader
scripts/
  make_synthetic.py   generate the synthetic benchmark
  run_experiments.py  train + every experiment → results/results_<dataset>.json
  make_figures.py     all plots → results/figures/
  demo.py             run on any image or a webcam (--model, --camera, --list-cameras)
  export_voc_format.py  write our datasets in PASCAL VOC layout (for the authors' code)
  run_tier1.py        authors' code vs ours, full Selective Search, per-size/shape recall
  prepare_voc.py      download PASCAL VOC 2007 and prepare it for both implementations
  run_voc.py          VOC 2007 evaluation: ours, authors' code, Selective Search, per-class recall
  measure_speed.py    controlled speed benchmark (1 and 4 threads)
  train_rerank.py     train the stage III re-ranker on VOC 2007 trainval
  run_novelty.py      our extensions vs plain BING on the VOC 2007 test set
  run_rerank_ablation.py  which cues the re-ranker relies on
  run_detection.py    detector built on proposals, VOC 2007 mAP per window source
  run_cnn_detection.py  the same with CNN features and box regression
baselines/original_bing/  build + run the authors' C++ BING (CLI entry point, OpenCV 4 patch)
presentation/         slide deck (.pptx + .pdf) and the speaker script
models/               trained models: synthetic (RGB / HSV / Gray) and PASCAL VOC 2007
data/real/            10 real photos with 58 hand-labelled boxes
tests/                unit tests (bitwise score == float score, numba == numpy, IoU …)
```

## 3. Quick start

Run these from the project folder (the one containing `scripts/`), with Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

**Demo on an image.** Saves two pictures in `results/demo/` (no window opens): `<name>_top20.jpg` with the 20 best boxes, and `<name>_heat.jpg` with the objectness heat-map.

```bash
python scripts/demo.py --image data/real/coffee.jpg
python scripts/demo.py --image photo.jpg --top 50 --model voc   # 50 boxes, model trained on VOC 2007
```

**Our extensions in the demo** (these use the VOC model automatically):

```bash
python scripts/demo.py --image photo.jpg --compare     # BING vs + re-ranking, side by side
python scripts/demo.py --image photo.jpg --rerank      # top boxes after our re-ranking
python scripts/demo.py --image photo.jpg --detect      # our detector: CNN features if torch is installed, else HOG
python scripts/demo.py --webcam --camera 1 --rerank    # live, with re-ranking
```

Add `--show` to open the saved pictures in a window.

`--model synthetic` (default) ranks distinct objects first, so its top boxes make the clearest picture. `--model voc` finds more objects within 1000 proposals but ranks large, near full-image boxes first.

**Live webcam demo.** Opens a window with the top boxes and the frame rate; press `q` to quit.

```bash
python scripts/demo.py --list-cameras              # which camera numbers deliver frames
python scripts/demo.py --webcam --camera 1          # e.g. a USB webcam (built-in is usually 0)
```

The first run takes a few seconds longer while the fast (Numba) code compiles.

**Troubleshooting**

| Problem | Fix |
|---|---|
| `pip.exe was blocked by ... policy` | Use `python -m pip ...` instead of `pip ...`. |
| `The function is not implemented ... imshow` | The display-less OpenCV build is installed. Run the two lines below. |
| `module 'cv2' has no attribute 'VideoCapture'` | Two OpenCV packages overwrote each other. Run the two lines below. |
| `can't grab frame` / wrong camera | Close other apps using the camera, allow desktop apps in Windows camera privacy settings, and pick the camera with `--list-cameras` / `--camera N`. |

```bash
python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
python -m pip install --force-reinstall --no-cache-dir "opencv-contrib-python>=4.8,<5"
```

**Reproduce everything** (synthetic and real-photo experiments, about 25 min on 4 CPU cores):

```bash
python scripts/make_synthetic.py                    # 600 train / 300 test scenes
python scripts/run_experiments.py                   # train + evaluate + ablations
python scripts/make_figures.py                      # figures
python -m pytest tests -q
```

To follow the paper's protocol on **PASCAL VOC 2007** (train on `trainval`, test on `test`):

```bash
python scripts/prepare_voc.py --dir /data/voc                       # download + prepare (~900 MB)
bash baselines/original_bing/run_original.sh /data/voc/VOC2007_original_bing/ 1   # authors' code
python scripts/run_voc.py --voc-root /data/voc/VOCdevkit/VOC2007 --orig /data/voc/VOC2007_original_bing
python scripts/run_voc.py --voc-root /data/voc/VOCdevkit/VOC2007 --sizes released --nms greedy --only-ours
python scripts/measure_speed.py --voc-root /data/voc/VOCdevkit/VOC2007
```

Our extensions on VOC 2007 (section 6):

```bash
python scripts/train_rerank.py --voc-root /data/voc/VOCdevkit/VOC2007          # stage III re-ranker
python scripts/run_novelty.py --voc-root /data/voc/VOCdevkit/VOC2007           # re-ranking vs plain BING
python scripts/run_rerank_ablation.py --voc-root /data/voc/VOCdevkit/VOC2007   # which cues matter
python scripts/run_detection.py --voc-root /data/voc/VOCdevkit/VOC2007         # HOG detector (~2 h on 4 cores)
python scripts/run_cnn_detection.py --voc-root /data/voc/VOCdevkit/VOC2007     # CNN-feature detector (~2 h; needs torch)
python scripts/measure_speed.py --voc-root /data/voc/VOCdevkit/VOC2007 --threads-only
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
* **PASCAL VOC 2007**: the paper's benchmark, downloaded with `scripts/prepare_voc.py` (5,011 trainval / 4,952 test images).

Metrics: **DR**, the fraction of objects covered by a proposal with IoU ≥ 0.5, reported for N proposals. **MABO** is the mean best IoU per object. **AUC** is the mean DR over a log-spaced #WIN axis from 1 to 5000.

## 5. Results

All methods are scored with the same evaluator (`bing/metrics.py`). **DR@N** is the share of objects covered by at least one of the top N proposals with IoU ≥ 0.5; **MABO** is the mean best overlap per object.

### 5.1 PASCAL VOC 2007 (the paper's benchmark)

Train on the 5,011 trainval images, test on the 4,952 test images (12,032 objects; "difficult" objects ignored, as in the paper).

| Method | DR@100 | DR@1000 | DR@5000 | MABO@1000 | img/s (1 thread) |
|---|---|---|---|---|---|
| **BING, ours (final)** | 67.2 % | **94.0 %** | **96.0 %** | **0.651** | **58** |
| BING, authors' C++ code | 68.8 % | 93.8 % | 95.7 % | 0.646 | 50 (196 on 4 threads) |
| BING, paper (reported) | – | 96.2 % | 99.5 % | – | 300 (paper's hardware) |
| Selective Search, fast (300 random test images)\* | 36.2 % | 88.2 % | 93.5 % | 0.707 | 0.8 (multi-threaded) |
| Random boxes | 46.5 % | 72.9 % | 86.8 % | 0.621 | – |

\*Our BING on the same 300 images: DR@1000 94.3 %.

The authors' code scores 96.0 % DR@1000 with its own evaluator (which uses VOC's inclusive-pixel box areas), so the paper's result reproduces; our evaluator gives it 93.8 %. Our implementation matches it. Our speed is 58 img/s on 1 thread. Section 6.3 adds multi-threading: 186 img/s on 4 threads, against 196 for the C++ code.

**Two details the paper leaves out.** Our first version followed the paper's text and plateaued at 76 %:

| Our BING on VOC 2007 | DR@1000 | DR@5000 | MABO@1000 |
|---|---|---|---|
| Paper's text: sizes {10…320}², local-maximum NMS (first version) | 75.1 % | 75.8 % | 0.576 |
| + released code's window sizes ({16…512}², clipped to the image) | 79.2 % | 79.7 % | 0.607 |
| + released code's greedy NMS only | 92.2 % | 94.4 % | 0.621 |
| **Both (final)** | 94.0 % | 96.0 % | 0.651 |

1. **Window sizes.** The paper lists sizes {10, 20, …, 320}. The released code uses {16, 32, …, 512} and keeps windows up to twice the image size, clipped to the image, so near full-image boxes exist. VOC has many large objects.
2. **Non-maximum suppression.** Keeping only strict local maxima of each score map throws away good windows on smooth score maps. The released code suppresses greedily: accept the best window, block its 5×5 neighbourhood, repeat until 130 windows per size.

Greedy NMS is now the default everywhere; the released window sizes are an option (`BING(sizes=RELEASED_SIZES, clip_large=True)`, used for VOC). Per-class recall is in `results/figures/voc_per_class.png`: bottles (small, thin) are hardest (≈83 %), cats and dogs easiest (≈99 %).

### 5.2 Synthetic test set (300 images)

| Method | DR@100 | DR@1000 | MABO@1000 | AUC |
|---|---|---|---|---|
| **BING (binary, Nw=2, Ng=4)** | 82.3 % | 98.5 % | 0.667 | 0.835 |
| BING (float filter w) | 82.5 % | 98.4 % | 0.666 | 0.832 |
| BING stage I only (no calibration) | 71.3 % | 94.1 % | 0.651 | 0.759 |
| BING-Diversified (RGB + HSV + Gray) | 77.6 % | 96.1 % | 0.672 | 0.802 |
| BING, authors' C++ code (same data) | 80.2 % | 94.9 % | 0.639 | 0.794 |
| Selective Search, fast mode (all 300 images) | 88.4 % | 97.5 % | 0.895 | 0.796 |
| Random boxes | 26.3 % | 53.6 % | 0.518 | 0.365 |
| Sliding windows (random order) | 0.8 % | 7.0 % | 0.314 | 0.052 |

### 5.3 Real photographs (58 objects, trained on synthetic data only)

| Method | DR@100 | DR@1000 | MABO@1000 |
|---|---|---|---|
| **BING (ours)** | 60.3 % | 87.9 % | 0.656 |
| BING-Diversified | 44.8 % | 87.9 % | 0.652 |
| BING stage I only | 31.0 % | 72.4 % | 0.591 |
| BING, authors' C++ code (images shrunk to 500 px) | 27.6 % | 75.9 % | 0.571 |
| Selective Search (fast) | 77.6 % | 100.0 % | 0.885 |
| Random boxes | 24.1 % | 48.3 % | 0.517 |

### 5.4 Speed (images/s, 4 threads, synthetic images ≈450×300 px)

| Backend | Scoring | Resize | img/s |
|---|---|---|---|
| numpy | binary | INTER_AREA | 25 |
| numpy | float (OpenCV correlation) | INTER_AREA | 38 |
| numpy | binary | bilinear | 36 |
| numpy | float | bilinear | 64 |
| **Numba | binary (AND + POPCOUNT) | bilinear** | 80 |
| Numba | float | bilinear | 147 |

One thread: 88 img/s (Numba, binary, bilinear). Per image (INTER_AREA): resize 12.1 ms, gradient 1.3 ms, bitwise scoring 5.1 ms, greedy NMS 3.8 ms.

### 5.5 Ablations (DR@1000 on the synthetic test set)

| Factor | Result |
|---|---|
| Gradient mask | `[-1 0 1]` 98.5 %, Sobel 99.3 %. |
| Colour space | RGB 98.5 %, Lab 97.4 %, HSV 96.7 %, Gray 94.5 %. |
| Resize interpolation | INTER_AREA 98.5 %, bilinear 98.4 %. |
| NMS | greedy 98.5 %, local maxima 95.1 %. |
| Filter bases Nw | 1 → 98.1 %, 2 → 98.5 %, 3 → 98.4 %, 4 → 98.5 %. |
| Feature bits Ng | 1 → 95.6 %, 2 → 98.4 %, 4 → 98.5 %, 6 → 98.5 %. |
| Proposals per size | 10 → 85.7 %, 30 → 95.0 %, 60 → 97.0 %, 130 → 98.5 %, 250 → 98.6 %. |
| Training images | 10 → 91.6 %, 50 → 96.9 %, 150 → 98.5 %, 300 → 98.5 %, 600 → 98.5 %. |

### 5.6 Recall by object size and shape (DR@100, synthetic)

| | small (<32 px) | medium | large (>96 px) | ellipse | rounded rect | blob | polygon | star |
|---|---|---|---|---|---|---|---|---|
| Ours | 56 % | 82 % | 96 % | 91 % | 90 % | 81 % | 79 % | 72 % |
| Authors' C++ | 49 % | 81 % | 95 % | 89 % | 82 % | 79 % | 78 % | 74 % |
| Selective Search | 84 % | 86 % | 95 % | 91 % | 90 % | 89 % | 90 % | 83 % |

### 5.7 Observations

1. **Our implementation reproduces BING.** On VOC 2007 it matches the authors' code (94.0 % vs 93.8 % DR@1000) once the two details above are included.
2. **Binarization is almost free.** 12 POPCOUNTs per window replace 64 multiply-adds with no loss of recall; the unit tests confirm the bitwise score equals the float dot product of the approximations.
3. **Stage II calibration matters** at small budgets (DR@100).
4. **BING's weakness is localisation.** Recall is high at IoU 0.5 but MABO (≈0.65) is well below Selective Search (≈0.7–0.9), from the power-of-two sizes and 8-pixel stride (Hosang et al., TPAMI 2016). Small objects are hardest.
5. **The "closed boundary" cue transfers.** Trained only on synthetic shapes, our model finds 88 % of objects in real photos.

### 5.8 Figures

All figures are in `results/figures/`:

* `voc_curves.png`, `voc_per_class.png`: PASCAL VOC 2007
* `dr_*.png`, `mabo_*.png`, `recall_iou_*.png`: accuracy curves (synthetic, real)
* `tier1_vs_original.png`, `tier1_breakdown.png`: authors' code vs ours, recall by size and shape
* `filter_w.png`, `filter_bases.png`, `ng_pipeline.png`, `calibration.png`: model internals
* `top8_*.png`, `found_*.png`, `heatmaps_real.png`: qualitative results
* `speed.png`, `stage_time.png`, `ablations.png`, `speed_threads.png`
* `novelty_voc.png`, `novelty_example.png`: our multi-cue re-ranking (section 6.1)
* `detection_voc.png`: the detector built on the proposals (CNN features), and why localisation matters (section 6.2)

## 6. Our extensions beyond the paper

### 6.1 Novelty: multi-cue re-ranking (stage III)

BING ranks windows with a single cue, its 8×8 edge template. We add a third stage (`bing/rerank.py`) that re-scores BING's proposals with cheap, handcrafted cues. Each cue is read from integral images, so it costs O(1) per window:

| Cue group | Values per window |
|---|---|
| Colour contrast | distance between the mean Lab colour inside the window and in a ring around it (the window grown by ¼ of its size per side); lightness difference |
| Edge density | mean gradient magnitude inside, in the ring, and their log-ratio |
| Geometry | centre position, distance to the image centre, log area, log aspect ratio, relative width and height, ring share, touches the border |
| BING | calibrated score and log rank |

A gradient-boosted tree model (scikit-learn `HistGradientBoostingClassifier`) combines these 16 values. It is trained on the proposals of 2,500 VOC 2007 trainval images, labelled positive when IoU ≥ 0.5 with an object (`scripts/train_rerank.py`), and the proposals are re-sorted by its probability. No deep network is involved.

Full VOC 2007 test set (4,952 images, 12,032 objects), `scripts/run_novelty.py` and `scripts/run_rerank_ablation.py`:

| Method | DR@100 | DR@1000 | DR@5000 | MABO@1000 | small objects DR@1000 | AUC | img/s |
|---|---|---|---|---|---|---|---|
| BING (final) | 67.2 % | 94.0 % | 96.0 % | 0.651 | 39.3 % | 0.719 | 67 |
| **BING + multi-cue re-ranking (ours)** | **71.0 %** | **94.7 %** | 96.0 % | **0.653** | **53.3 %** | **0.741** | 34 |
| re-ranking without colour contrast | 69.5 % | 94.5 % | 96.0 % | 0.652 | 51.6 % | 0.732 | |
| re-ranking without edge density | 69.4 % | 94.3 % | 96.0 % | 0.651 | 45.0 % | 0.732 | |
| re-ranking without geometry | 70.2 % | 94.6 % | 96.0 % | 0.652 | 54.8 % | 0.736 | |

Small objects are those under 32 px (516 of the 12,032). DR@5000 cannot change, because re-ranking only re-orders BING's windows.

* Re-ranking raises recall at every budget (`results/figures/novelty_voc.png`). The gain is largest at small budgets, which is where a detector works.
* Every cue group helps. Edge density matters most for small objects: without it, small-object recall falls from 53.3 % to 45.0 %.
* The cost is speed: 34 instead of 67 img/s, still about 40 times faster than Selective Search.
* **Tried and rejected** (kept in the repository as negative results): snapping each box side to the strongest nearby edge line (`bing/refine.py`), and adding intermediate window sizes. On a 300-image subset both looked promising, but on the full test set neither raised recall (DR@1000 93.6 % and 93.9 %).

`results/figures/novelty_example.png` shows one test image: BING's top 20 windows find one of three objects, the re-ranked top 20 find all three.

### 6.2 A complete detector built on the proposals

Proposals exist to feed a detector, so we built one (`bing/detector.py`, `scripts/run_detection.py`), in the style of R-CNN but with handcrafted features only:

* each window is padded by 1/8 per side, warped to 64×64 and described with HOG (9 bins, 8×8 cells, 1,764 values) plus an 8×4 hue/saturation histogram (1,796 values in all);
* one linear SVM per VOC class, trained on VOC 2007 trainval: positives are the ground-truth boxes and their mirror images (25,216 patches), negatives are BING proposals overlapping no object (IoU < 0.3), then one round of hard-negative mining on 1,500 trainval images (about 14,500 per class);
* at test time the top N windows are scored, per-class non-maximum suppression (IoU 0.3) is applied, and the VOC 2007 11-point average precision is computed ("difficult" objects ignored).

VOC 2007 test, mean AP over the 20 classes, same classifier for every source of windows:

| Windows | top 100 | top 500 | top 1000 |
|---|---|---|---|
| BING | 6.8 % | 6.0 % | 5.6 % |
| **BING + multi-cue re-ranking (ours)** | 7.4 % | 5.9 % | 5.4 % |
| Random boxes | 6.8 % | 8.2 % | 8.4 % |
| Sliding windows | 0.5 % | 1.0 % | 1.1 % |

* Proposals matter: at 100 windows, BING's windows give about 15 times the mAP of sliding windows, and our re-ranking gives the best result (7.4 %).
* HOG with a linear SVM is a weak classifier, so all numbers are low (for comparison, DPM reaches about 34 % and R-CNN about 58 % on this test set).
* With more windows, random boxes overtake BING (8.4 % at 1000). The reason is localisation: BING's top 1000 cover 30 % of objects at IoU ≥ 0.7, random boxes 43 % (`results/figures/detection_voc.png`). A detector needs tight boxes, which agrees with Hosang et al. (TPAMI 2016): recall at stricter overlaps predicts detection quality. BING's coarse boxes, not its recall at IoU 0.5, limit it here.

#### With CNN features

HOG limits the detector above, so we also trained it on features from a ResNet-50 pre-trained on ImageNet (`bing/cnn.py`, `scripts/run_cnn_detection.py`). The network is frozen (no fine-tuning) and runs once per image, with the longer side scaled to 600 px. Each window is described by the mean conv4 and conv5 activations inside it (3,072 values, square-rooted and L2-normalised), read from integral images of the feature maps, so 1,000 windows cost about 75 ms. The 20 linear SVMs are trained jointly with L-BFGS on ground-truth boxes plus windows with IoU ≥ 0.7, against background windows, with one round of hard-negative mining. Class-specific box regression (as in R-CNN, ridge regression on windows with IoU ≥ 0.6) then moves each box. Only the detector uses a deep network; the proposals still come from BING.

| Windows | top 100 | top 100 + box regression | top 300 + box regression | top 1000 + box regression |
|---|---|---|---|---|
| BING | 16.9 % | 18.8 % | 17.6 % | 13.9 % |
| **BING + multi-cue re-ranking (ours)** | 16.7 % | 19.1 % | 17.6 % | 13.1 % |
| Random boxes | 11.4 % | 13.8 % | 15.0 % | 16.0 % |
| Sliding windows | 0.5 % | 0.6 % | 2.1 % | 2.7 % |

* CNN features raise the best result from 7.4 % (HOG) to **19.1 %** mAP, with our re-ranked top 100 windows.
* At 100 windows, BING's boxes clearly beat random boxes (18.8 % vs 13.8 %) and sliding windows (0.6 %).
* Box regression adds about 2 points for every source.
* With more windows, random boxes still overtake BING (16.0 % at 1000): BING's loose boxes remain the limit, as with HOG.
* Without fine-tuning, the numbers stay well below R-CNN (about 58 %). Per test image: ResNet-50 pass 353 ms, BING 19 ms, re-ranking 16 ms (4 CPU threads).

The demo uses this detector when torch is installed: `python scripts/demo.py --image photo.jpg --detect`.

### 6.3 Multi-threading

The released C++ code is fast because it uses all cores. Our first version did not scale (58 img/s on 1 thread, 57 on 4). We added two modes (`bing/model.py`, `bing/fast.py`):

* **Window sizes in parallel** (`BING(n_jobs=4)`): the 36 window sizes of one image are processed by a thread pool. The gradient, bitwise scoring and greedy NMS of one size run in one GIL-free Numba call, and the largest score maps are scheduled first. This lowers the latency of a single image (useful for the webcam demo).
* **Images in parallel** (`model.propose_batch(images, n_jobs=4)`): one image per thread, for throughput.

VOC 2007 test images, 4-core CPU (`python scripts/measure_speed.py --voc-root ... --threads-only`):

| Threads | window sizes in parallel | images in parallel | authors' C++ |
|---|---|---|---|
| 1 | 59 img/s | 57 img/s | 50 img/s |
| 2 | 80 img/s | 104 img/s | |
| 4 | 106 img/s | **186 img/s** | 196 img/s |

The fused kernel orders tied scores differently from the single-threaded path, so about 1 % of the boxes in the top 1000 differ.

## 7. Limitations and future work

* Localisation is coarse (power-of-two sizes, 8-pixel stride). Our edge-snapping refinement and finer window sizes did not fix it (section 6.1); learned box regression is the next thing to try, and section 6.2 shows it is what detection needs.
* The detector's CNN is frozen; fine-tuning it on VOC (as R-CNN does) would raise mAP a lot.
* The re-ranker halves the speed (34 vs 67 img/s); a smaller tree model or fewer cues would recover some of it.
* Training on real annotated images would close the gap between synthetic and real data.

## References

1. M.-M. Cheng, Z. Zhang, W.-Y. Lin, P. H. S. Torr, "BING: Binarized Normed Gradients for Objectness Estimation at 300fps," CVPR 2014.
2. J. Uijlings, K. van de Sande, T. Gevers, A. Smeulders, "Selective Search for Object Recognition," IJCV 2013.
3. B. Alexe, T. Deselaers, V. Ferrari, "Measuring the Objectness of Image Windows," TPAMI 2012.
4. J. Hosang, R. Benenson, P. Dollár, B. Schiele, "What makes for effective detection proposals?," TPAMI 2016.
5. P. Arbeláez, J. Pont-Tuset, J. T. Barron, F. Marques, J. Malik, "Multiscale Combinatorial Grouping," CVPR 2014.
