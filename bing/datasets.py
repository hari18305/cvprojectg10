"""Datasets: synthetic object scenes, real annotated photos and PASCAL VOC.

Every dataset is an iterable of (image_bgr_uint8, boxes[N,4] x1,y1,x2,y2).
"""
import glob
import json
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- synthetic
def _textures():
    from skimage import data
    tex = [data.brick(), data.grass(), data.gravel()]
    return [t if t.ndim == 2 else cv2.cvtColor(t, cv2.COLOR_RGB2GRAY) for t in tex]


def _smooth_field(rng, H, W, cells, ch=3):
    f = rng.uniform(0, 1, (cells, cells, ch)).astype(np.float32)
    return cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC).reshape(H, W, ch)


def _texture_layer(rng, tex, H, W):
    t = tex[rng.integers(len(tex))]
    s = rng.uniform(0.3, 1.0)
    th, tw = int(t.shape[0] * s), int(t.shape[1] * s)
    t = cv2.resize(t, (max(tw, 8), max(th, 8)))
    reps = (int(np.ceil(H / t.shape[0])) + 1, int(np.ceil(W / t.shape[1])) + 1)
    t = np.tile(t, reps)[:H, :W].astype(np.float32) / 255.0
    return (t - t.mean())[..., None]


def _background(rng, tex, H, W):
    base = _smooth_field(rng, H, W, rng.integers(2, 6))
    base = 0.25 + 0.5 * base
    kind = rng.integers(3)
    if kind == 0:   # textured surface
        base += rng.uniform(0.2, 0.6) * _texture_layer(rng, tex, H, W)
    elif kind == 1:  # mild texture + fine noise
        base += 0.15 * _texture_layer(rng, tex, H, W)
    img = np.clip(base, 0, 1)
    # clutter: thin lines / curves that are edges but not objects
    img8 = (img * 255).astype(np.uint8)
    for _ in range(rng.integers(3, 15)):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p = rng.integers(0, [W, H], (rng.integers(2, 5), 2)).astype(np.int32)
        cv2.polylines(img8, [p], False, c, int(rng.integers(1, 3)), cv2.LINE_AA)
    return img8.astype(np.float32) / 255.0


SHAPE_NAMES = ("ellipse", "rounded_rect", "polygon", "star", "blob")


def _shape_mask(rng, w, h):
    """Return (soft mask in [0,1], index into SHAPE_NAMES)."""
    m = np.zeros((h, w), np.uint8)
    kind = rng.integers(5)
    if kind == 0:  # ellipse
        cv2.ellipse(m, (w // 2, h // 2), (w // 2 - 1, h // 2 - 1), 0, 0, 360, 255, -1, cv2.LINE_AA)
    elif kind == 1:  # rectangle with rounded corners
        r = int(min(w, h) * rng.uniform(0, 0.3))
        cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
        cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
        for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
            cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    else:  # random star-shaped blob / polygon
        n = rng.integers(5, 14)
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = rng.uniform(0.55 if kind == 2 else 0.3, 1.0, n)
        if kind == 4:  # smooth blob
            ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
            k = rng.normal(0, 0.12, (3, 2))
            rad = 0.75 + sum(k[i, 0] * np.cos((i + 2) * ang) + k[i, 1] * np.sin((i + 2) * ang)
                             for i in range(3))
        pts = np.stack([np.cos(ang) * rad, np.sin(ang) * rad], 1)
        pts = (pts - pts.min(0)) / (pts.max(0) - pts.min(0) + 1e-9)
        pts = (pts * [w - 1, h - 1]).astype(np.int32)
        cv2.fillPoly(m, [pts], 255, cv2.LINE_AA)
    return m.astype(np.float32) / 255.0, int(kind)


def _object_appearance(rng, tex, w, h, bg_mean):
    col = rng.uniform(0, 1, 3)
    # enforce some (random, sometimes low) contrast with the background
    contrast = rng.uniform(0.12, 0.6)
    d = col - bg_mean
    col = np.clip(bg_mean + d / (np.linalg.norm(d) + 1e-6) * contrast * np.sqrt(3), 0, 1)
    app = np.ones((h, w, 3), np.float32) * col
    # shading (illumination gradient)
    gx, gy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    a = rng.uniform(0, 2 * np.pi)
    app += (rng.uniform(0, 0.25) * (np.cos(a) * gx + np.sin(a) * gy))[..., None]
    if rng.random() < 0.5:  # surface texture
        app += rng.uniform(0.05, 0.3) * _texture_layer(rng, tex, h, w)
    if rng.random() < 0.3:  # inner parts (e.g. stripes / spots)
        for _ in range(rng.integers(1, 4)):
            c = tuple(float(v) for v in rng.uniform(0, 1, 3))
            cv2.circle(app, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                       int(min(w, h) * rng.uniform(0.05, 0.2)), c, -1, cv2.LINE_AA)
    return np.clip(app, 0, 1)


def synth_scene(rng, tex):
    H, W = int(rng.integers(240, 400)), int(rng.integers(320, 500))
    img = _background(rng, tex, H, W)
    boxes, kinds = [], []
    for _ in range(int(rng.integers(1, 6))):
        for _try in range(20):
            s = np.exp(rng.uniform(np.log(24), np.log(0.9 * min(H, W))))
            ar = np.exp(rng.uniform(np.log(0.35), np.log(2.8)))
            w, h = int(s * np.sqrt(ar)), int(s / np.sqrt(ar))
            if w < 12 or h < 12 or w >= W - 2 or h >= H - 2:
                continue
            x, y = int(rng.integers(0, W - w)), int(rng.integers(0, H - h))
            b = [x, y, x + w, y + h]
            if boxes:
                from .metrics import iou_matrix
                if iou_matrix(np.array(boxes), np.array([b])).max() > 0.2:
                    continue
                # do not hide an existing object's box under a new one
                if any(b[0] <= o[0] and b[1] <= o[1] and b[2] >= o[2] and b[3] >= o[3] for o in boxes):
                    continue
            break
        else:
            continue
        m, kind = _shape_mask(rng, w, h)
        m = m[..., None]
        bgm = img[y:y + h, x:x + w].reshape(-1, 3).mean(0)
        app = _object_appearance(rng, tex, w, h, bgm)
        if rng.random() < 0.5:  # dark outline
            m8 = (m[..., 0] * 255).astype(np.uint8)
            edge = cv2.morphologyEx(m8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            app[edge > 0] *= rng.uniform(0.3, 0.8)
        img[y:y + h, x:x + w] = img[y:y + h, x:x + w] * (1 - m) + app * m
        ys, xs = np.nonzero(m[..., 0] > 0.5)
        boxes.append([x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1])
        kinds.append(SHAPE_NAMES[kind])
    img = np.clip(img + rng.normal(0, rng.uniform(0.005, 0.03), img.shape), 0, 1)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
    img8 = (img * 255).astype(np.uint8)
    return img8, np.array(boxes, float).reshape(-1, 4), kinds


def make_synthetic(out_dir, n, seed):
    rng = np.random.default_rng(seed)
    tex = _textures()
    os.makedirs(out_dir, exist_ok=True)
    ann, cats = {}, {}
    for i in range(n):
        img, boxes, kinds = synth_scene(rng, tex)
        name = f"{i:05d}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ann[name] = boxes.tolist()
        cats[name] = kinds
    with open(os.path.join(out_dir, "annotations.json"), "w") as f:
        json.dump(ann, f)
    with open(os.path.join(out_dir, "categories.json"), "w") as f:
        json.dump(cats, f)


class JsonDataset:
    """Folder of images + annotations.json {filename: [[x1,y1,x2,y2], ...]}."""

    def __init__(self, folder, limit=None):
        self.folder = folder
        with open(os.path.join(folder, "annotations.json")) as f:
            self.ann = json.load(f)
        self.names = sorted(self.ann)[:limit]
        cat_path = os.path.join(folder, "categories.json")
        self.categories = None  # {filename: [category of each box]} when available
        if os.path.exists(cat_path):
            with open(cat_path) as f:
                self.categories = json.load(f)

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        for n in self.names:
            yield cv2.imread(os.path.join(self.folder, n)), np.array(self.ann[n], float).reshape(-1, 4)


class VOCDataset:
    """PASCAL VOC 2007 (as used in the BING paper: train on trainval, test on test).

    root: path to VOCdevkit/VOC2007.  'difficult' objects are ignored, as in the paper.
    """

    def __init__(self, root, split="trainval", limit=None):
        self.root = root
        with open(os.path.join(root, "ImageSets", "Main", f"{split}.txt")) as f:
            self.ids = [l.strip() for l in f if l.strip()][:limit]

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for i in self.ids:
            img = cv2.imread(os.path.join(self.root, "JPEGImages", i + ".jpg"))
            tree = ET.parse(os.path.join(self.root, "Annotations", i + ".xml"))
            boxes = []
            for obj in tree.findall("object"):
                if int(obj.findtext("difficult", "0")):
                    continue
                bb = obj.find("bndbox")
                x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                boxes.append([x1 - 1, y1 - 1, x2, y2])  # VOC is 1-based, inclusive
            yield img, np.array(boxes, float).reshape(-1, 4)


def get_dataset(name, split, limit=None, voc_root=None):
    if name == "synthetic":
        return JsonDataset(os.path.join(ROOT, "data", "synthetic", split), limit)
    if name == "real":
        return JsonDataset(os.path.join(ROOT, "data", "real"), limit)
    if name == "voc":
        return VOCDataset(voc_root, "trainval" if split == "train" else "test", limit)
    raise ValueError(name)
