"""Render every figure used in the report / presentation from results/*.json."""
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing import BING  # noqa: E402
from bing.binary import approximate_filter  # noqa: E402
from bing.datasets import ROOT, get_dataset  # noqa: E402
from bing.features import normed_gradient, resize  # noqa: E402

FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold",
    "axes.titlesize": 13,
})

ORDER = ["BING (binary)", "BING-Diversified", "BING (float w)", "BING stage I only",
         "Selective Search (fast)", "Sliding windows", "Random boxes"]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), facecolor="white")
    plt.close(fig)


def curves(R, split, key, ylabel, fname, title):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, m in enumerate(ORDER):
        r = R[split].get(m)
        if not r:
            continue
        ls = "--" if m in ("BING (float w)", "BING stage I only") else "-"
        lab = m + (f" ({r['n_images']} imgs)" if "n_images" in r else "")
        ax.plot(r["curve_n"], r[key], ls, color=SERIES[i], lw=2, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("Number of proposals (#WIN)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, fname)


def recall_iou(R, split, fname, title):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, m in enumerate(ORDER):
        r = R[split].get(m)
        if not r:
            continue
        t = [float(k) for k in r["recall_iou"]]
        ax.plot(t, list(r["recall_iou"].values()), "-o", ms=4, color=SERIES[i], lw=2, label=m)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Detection rate @1000 proposals")
    ax.set_ylim(0, 1.0)
    ax.set_title(title, loc="left")
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, fname)


def filter_figs(R):
    w = np.array(R["w"])
    lim = np.abs(w).max()
    fig, axs = plt.subplots(1, 5, figsize=(12, 2.9))
    axs[0].imshow(w.reshape(8, 8), cmap="RdBu_r", vmin=-lim, vmax=lim)
    axs[0].set_title("learned w", fontsize=11)
    for n in range(1, 5):
        b, a = approximate_filter(w, n)
        wa = (b[:, None] * a).sum(0)
        err = np.linalg.norm(w - wa) / np.linalg.norm(w)
        axs[n].imshow(wa.reshape(8, 8), cmap="RdBu_r", vmin=-lim, vmax=lim)
        axs[n].set_title(f"Nw={n}\nerr {err:.2f}", fontsize=11)
    for a in axs:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    save(fig, "filter_w.png")
    # basis vectors
    b, a = approximate_filter(w, 4)
    fig, axs = plt.subplots(1, 4, figsize=(9.5, 2.8))
    for j in range(4):
        axs[j].imshow(a[j].reshape(8, 8), cmap="gray", vmin=-1, vmax=1)
        axs[j].set_title(f"a{j + 1}  beta={b[j]:.4f}", fontsize=10)
        axs[j].set_xticks([]); axs[j].set_yticks([]); axs[j].grid(False)
    save(fig, "filter_bases.png")


def ng_pipeline(img_path):
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    fig, axs = plt.subplots(2, 4, figsize=(13, 6))
    axs[0, 0].imshow(img[..., ::-1]); axs[0, 0].set_title("input")
    for i, (w, h) in enumerate([(160, 160), (80, 160), (40, 40)]):
        rw, rh = int(round(8 * W / w)), int(round(8 * H / h))
        g = normed_gradient(resize(img, rw, rh))
        axs[0, i + 1].imshow(g, cmap="gray"); axs[0, i + 1].set_title(f"NG map for {w}x{h} windows\n({rw}x{rh})", fontsize=10)
    g = normed_gradient(resize(img, int(8 * W / 40), int(8 * H / 40)))
    for k in range(4):
        axs[1, k].imshow((g >> (7 - k)) & 1, cmap="gray"); axs[1, k].set_title(f"bit plane b{k + 1} (weight 2^{7 - k})", fontsize=10)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    save(fig, "ng_pipeline.png")


def draw_boxes(img, boxes, gt=None, top=10):
    out = img.copy()
    if gt is not None:
        for x1, y1, x2, y2 in np.asarray(gt, int):
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 4)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), 2)
    cols = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (164, 123, 232)]
    for i in range(min(top, len(boxes)) - 1, -1, -1):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), cols[i % 5], 2)
    return out


def best_match(gt, props, top):
    """For each GT, the best proposal in the top-N (to show 'what was found')."""
    from bing.metrics import iou_matrix
    iou = iou_matrix(gt, props[:top])
    return props[iou.argmax(1)], iou.max(1)


def qualitative(model, data, fname, n=6, cols=3, title=None):
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.4 * rows))
    for a, (img, gt) in zip(axs.ravel(), data[:n]):
        b, s = model.propose(img)
        vis = draw_boxes(img, b, None, top=8)
        a.imshow(vis[..., ::-1]); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for a in axs.ravel()[len(data[:n]):]:
        a.axis("off")
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    save(fig, fname)


def recall_vis(model, data, fname, n=6, cols=3, top=1000):
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.4 * rows))
    for a, (img, gt) in zip(axs.ravel(), data[:n]):
        b, s = model.propose(img)
        bm, iou = best_match(gt, b, top)
        vis = img.copy()
        for g in gt.astype(int):
            cv2.rectangle(vis, tuple(g[:2]), tuple(g[2:]), (255, 255, 255), 3)
        for p, v in zip(bm.astype(int), iou):
            c = (27, 175, 122) if v >= 0.5 else (72, 73, 227)
            cv2.rectangle(vis, tuple(p[:2]), tuple(p[2:]), c, 2)
        a.imshow(vis[..., ::-1]); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.set_title(f"{(iou >= 0.5).sum()}/{len(gt)} objects found", fontsize=10)
    for a in axs.ravel()[len(data[:n]):]:
        a.axis("off")
    fig.tight_layout()
    save(fig, fname)


def heatmaps(model, data, fname, n=4):
    fig, axs = plt.subplots(2, n, figsize=(4.2 * n, 6.2))
    for i, (img, gt) in enumerate(data[:n]):
        heat = model.objectness_heatmap(img)
        axs[0, i].imshow(img[..., ::-1]); axs[1, i].imshow(heat, cmap="inferno")
        for a in axs[:, i]:
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
    axs[0, 0].set_ylabel("input"); axs[1, 0].set_ylabel("objectness map")
    fig.tight_layout()
    save(fig, fname)


def speed_figs(R):
    sp = R["speed"]
    labels, vals = [], []
    for k in ["numpy|float|area", "numpy|binary|area", "numba|float|area", "numba|binary|area",
              "numpy|float|linear", "numpy|binary|linear", "numba|float|linear", "numba|binary|linear"]:
        if k in sp:
            b, t, i = k.split("|")
            labels.append(f"{b} · {t} · {'INTER_AREA' if i == 'area' else 'bilinear'}")
            vals.append(sp[k])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(vals))[::-1]
    cols = [SERIES[0] if "binary" in l else SERIES[1] for l in labels]
    ax.barh(y, vals, color=cols, height=0.62)
    for yy, v in zip(y, vals):
        ax.text(v + 3, yy, f"{v:.0f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("images / second (4-core CPU, ~450x300 px)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Proposal speed by implementation", loc="left")
    save(fig, "speed.png")
    st = R["stage_ms"]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ks = list(st)
    ax.barh(range(len(ks))[::-1], [st[k] for k in ks], color=SERIES[0], height=0.6)
    for yy, k in zip(range(len(ks))[::-1], ks):
        ax.text(st[k] + 0.05, yy, f"{st[k]:.2f} ms", va="center", fontsize=10)
    ax.set_yticks(range(len(ks))[::-1]); ax.set_yticklabels(ks)
    ax.set_xlabel("ms per image (sum over all window sizes)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Where the time goes (numba · binary)", loc="left")
    save(fig, "stage_time.png")


def ablation_figs(R):
    A = R.get("ablations")
    if not A:
        return
    groups = {
        "Colour space": ["colour=RGB", "colour=HSV", "colour=LAB", "colour=GRAY"],
        "Gradient / resize": ["colour=RGB", "gradient=Sobel 3x3", "resize=bilinear"],
        "Filter bases Nw": ["Nw=1", "Nw=2", "Nw=3", "Nw=4"],
        "Feature bits Ng": ["Ng=1", "Ng=2", "Ng=3", "Ng=4", "Ng=5", "Ng=6"],
        "Proposals per size": ["per-size=10", "per-size=30", "per-size=60", "per-size=130", "per-size=250"],
        "Training images": ["train imgs=10", "train imgs=50", "train imgs=150", "train imgs=300", "colour=RGB"],
    }
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
    for a, (g, keys) in zip(axs.ravel(), groups.items()):
        keys = [k for k in keys if k in A]
        d100 = [A[k]["dr_at"]["100"] if "100" in A[k]["dr_at"] else A[k]["dr_at"][100] for k in keys]
        d1k = [A[k]["dr_at"]["1000"] if "1000" in A[k]["dr_at"] else A[k]["dr_at"][1000] for k in keys]
        x = np.arange(len(keys))
        a.bar(x - 0.2, d100, 0.38, color=SERIES[0], label="DR@100")
        a.bar(x + 0.2, d1k, 0.38, color=SERIES[1], label="DR@1000")
        for xx, v in zip(x, d1k):
            a.text(xx + 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5)
        lab = [k.split("=")[1] if k != "colour=RGB" or g == "Colour space" else "default" for k in keys]
        if g == "Training images":
            lab[-1] = "600"
        a.set_xticks(x); a.set_xticklabels(lab, fontsize=9)
        a.set_ylim(0, 1.08); a.set_title(g, loc="left"); a.grid(axis="x", visible=False)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "ablations.png")


def calib_fig(R):
    sizes = [10, 20, 40, 80, 160, 320]
    M = np.full((6, 6), np.nan)
    for k, (v, t) in R["calib"].items():
        w, h = map(int, k.split("x"))
        M[sizes.index(h), sizes.index(w)] = v
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(M, cmap="Blues")
    for i in range(6):
        for j in range(6):
            txt = "–" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if (not np.isnan(M[i, j]) and M[i, j] > np.nanmax(M) * 0.6) else INK)
    ax.set_xticks(range(6)); ax.set_xticklabels(sizes); ax.set_yticks(range(6)); ax.set_yticklabels(sizes)
    ax.set_xlabel("window width"); ax.set_ylabel("window height"); ax.grid(False)
    ax.set_title("Stage II weight v per window size", loc="left", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "calibration.png")


def tier1_figs():
    path = os.path.join(ROOT, "results", "tier1_synthetic.json")
    if not os.path.exists(path):
        return
    T = json.load(open(path))
    methods = [("BING (ours)", SERIES[0], "-"), ("BING (authors' C++)", SERIES[1], "-"),
               ("BING (authors' C++), shrunk to 500 px", SERIES[1], "-"),
               ("Selective Search (fast)", SERIES[4], "-")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (split, title) in zip(axs, (("test", "Synthetic test set (300 images)"), ("real", "Real photos (10 images)"))):
        for name, col, ls in methods:
            r = T[split].get(name)
            if r:
                lab = "BING (authors' C++)" if name.startswith("BING (authors") else name
                ax.plot(r["curve_n"], r["dr_curve"], ls, color=col, lw=2, label=lab)
        ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_title(title, loc="left")
        ax.set_xlabel("Number of proposals (#WIN)")
    axs[0].set_ylabel("Detection rate (IoU ≥ 0.5)")
    axs[0].legend(loc="upper left", fontsize=9.5)
    fig.tight_layout(); save(fig, "tier1_vs_original.png")

    groups = [("by_size", ["small (<32 px)", "medium (32-96 px)", "large (>96 px)"], "Object size"),
              ("by_category", ["ellipse", "rounded_rect", "blob", "polygon", "star"], "Object shape")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [3, 5]})
    names = [("BING (ours)", SERIES[0]), ("BING (authors' C++)", SERIES[1]), ("Selective Search (fast)", SERIES[4])]
    for ax, (key, cats, title) in zip(axs, groups):
        x = np.arange(len(cats))
        for i, (name, col) in enumerate(names):
            v = [T["test"][name][key][c]["dr@100"] for c in cats]
            ax.bar(x + (i - 1) * 0.27, v, 0.25, color=col, label=name)
        ax.set_xticks(x); ax.set_xticklabels([c.split(" (")[0].replace("_", " ") + (f"\n({c.split(' (')[1]}" if " (" in c else "") for c in cats], fontsize=9.5)
        ax.set_ylim(0, 1.05); ax.set_title(f"DR@100 by {title.lower()} (synthetic)", loc="left"); ax.grid(axis="x", visible=False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "tier1_breakdown.png")


def voc_figs():
    path = os.path.join(ROOT, "results", "voc.json")
    if not os.path.exists(path):
        return
    V = json.load(open(path))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    methods = [("BING (ours)", SERIES[0]), ("BING (authors' C++)", SERIES[1]),
               ("Selective Search (fast)", SERIES[4]), ("Random boxes", SERIES[6])]
    for name, col in methods:
        r = V.get(name)
        if not r:
            continue
        lab = name + (f" ({len(V['ss_subset_ids'])} imgs)" if name.startswith("Selective") else "")
        axs[0].plot(r["curve_n"], r["dr_curve"], color=col, lw=2, label=lab)
        axs[1].plot(r["curve_n"], r["mabo_curve"], color=col, lw=2, label=lab)
    axs[0].axhline(0.962, color=INK2, lw=1, ls=":")
    axs[0].text(1.2, 0.975, "paper: 96.2% @1000", fontsize=9, color=INK2)
    for ax, yl, t in ((axs[0], "Detection rate (IoU ≥ 0.5)", "Recall vs #proposals — VOC 2007 test"),
                      (axs[1], "MABO", "Box tightness — VOC 2007 test")):
        ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.set_xlabel("Number of proposals (#WIN)")
        ax.set_ylabel(yl); ax.set_title(t, loc="left")
    axs[0].legend(loc="lower right", fontsize=9)
    fig.tight_layout(); save(fig, "voc_curves.png")

    ours, orig = V["BING (ours)"]["by_class"], V.get("BING (authors' C++)", {}).get("by_class", {})
    classes = sorted(ours, key=lambda c: ours[c]["dr@1000"])
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(classes))
    ax.bar(x - 0.2, [ours[c]["dr@1000"] for c in classes], 0.38, color=SERIES[0], label="BING (ours)")
    if orig:
        ax.bar(x + 0.2, [orig[c]["dr@1000"] for c in classes], 0.38, color=SERIES[1], label="BING (authors' C++)")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=9.5)
    ax.set_ylim(0.6, 1.0); ax.set_ylabel("DR@1000"); ax.grid(axis="x", visible=False)
    ax.set_title("Recall at 1000 proposals per VOC class", loc="left")
    ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout(); save(fig, "voc_per_class.png")


def main(tag="synthetic"):
    R = json.load(open(os.path.join(ROOT, "results", f"results_{tag}.json")))
    curves(R, "test", "dr_curve", "Detection rate (IoU ≥ 0.5)", "dr_synthetic.png", "Recall vs #proposals — synthetic test set")
    curves(R, "real", "dr_curve", "Detection rate (IoU ≥ 0.5)", "dr_real.png", "Recall vs #proposals — real photos")
    curves(R, "test", "mabo_curve", "MABO", "mabo_synthetic.png", "Mean average best overlap — synthetic")
    curves(R, "real", "mabo_curve", "MABO", "mabo_real.png", "Mean average best overlap — real photos")
    recall_iou(R, "test", "recall_iou_synthetic.png", "Recall vs IoU @1000 — synthetic")
    recall_iou(R, "real", "recall_iou_real.png", "Recall vs IoU @1000 — real photos")
    filter_figs(R)
    speed_figs(R)
    ablation_figs(R)
    calib_fig(R)
    tier1_figs()
    voc_figs()
    model = BING.load(os.path.join(ROOT, "models", f"bing_{tag}_rgb.pkl"))
    real = list(get_dataset("real", "test"))
    syn = list(get_dataset("synthetic", "test", 12))
    ng_pipeline(os.path.join(ROOT, "data", "real", "coffee.jpg"))
    qualitative(model, real, "top8_real.png", n=6)
    qualitative(model, syn, "top8_synthetic.png", n=6)
    recall_vis(model, real, "found_real.png", n=9)
    recall_vis(model, syn, "found_synthetic.png", n=6)
    heatmaps(model, [real[i] for i in (0, 2, 6, 8)], "heatmaps_real.png")
    # title-slide art: NG map of a real photo
    im = cv2.imread(os.path.join(ROOT, "data", "real", "motorcycle.jpg"))
    g = normed_gradient(im).astype(np.float32)
    g = cv2.GaussianBlur(np.sqrt(g / 255.0), (0, 0), 0.6)
    H, W = g.shape
    crop = g[:, int(W * 0.12):int(W * 0.12) + int(H * 0.82)]
    rgb = (plt.get_cmap("inferno")(np.clip(crop * 1.1, 0, 1))[..., :3] * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(FIG, "ng_title.png"), rgb[..., ::-1])
    # dataset samples
    fig, axs = plt.subplots(2, 4, figsize=(14, 5.4))
    for a, (img, gt) in zip(axs.ravel(), syn[:8]):
        v = img.copy()
        for g in gt.astype(int):
            cv2.rectangle(v, tuple(g[:2]), tuple(g[2:]), (0, 230, 0), 2)
        a.imshow(v[..., ::-1]); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.tight_layout(); save(fig, "dataset_synthetic.png")
    fig, axs = plt.subplots(2, 5, figsize=(16, 5.6))
    for a, (img, gt) in zip(axs.ravel(), real):
        v = img.copy()
        for g in gt.astype(int):
            cv2.rectangle(v, tuple(g[:2]), tuple(g[2:]), (0, 230, 0), 3)
        a.imshow(v[..., ::-1]); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.tight_layout(); save(fig, "dataset_real.png")
    print("figures in", FIG)


if __name__ == "__main__":
    main(*sys.argv[1:])
