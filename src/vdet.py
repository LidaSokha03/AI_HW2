"""Shared utilities for the VisDrone multi-object video detection homework (UCU AI 2026).

Every notebook uses this module for the same steps, so all models share one protocol:
  * finding / downloading the competition data (Kaggle or Colab),
  * building the frame subset (every N-th frame, resized once),
  * COCO evaluation that respects ignored regions (iscrowd=1),
  * Ultralytics train / predict helpers,
  * error analysis, plots, result logging and the submission file.
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

SLUG = "multiple-object-video-detection-hw-2-ucu-ai-2026"
CLASS_NAMES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
SEED = 42
METRIC_KEYS = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR500"]


# =====================================================================================
# Platform, paths and data download
# =====================================================================================
def platform() -> str:
    """'kaggle', 'colab' or 'local'."""
    if os.path.exists("/kaggle/input"):
        return "kaggle"
    if "google.colab" in sys.modules or os.path.exists("/content"):
        return "colab"
    return "local"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_paths() -> dict:
    """Working folder (kept between sessions) and fast local data folder for each platform."""
    plat = platform()
    if plat == "kaggle":
        return dict(platform=plat, work="/kaggle/working", data="/tmp/vdet_data")
    if plat == "colab":
        drive = "/content/drive/MyDrive"
        work = f"{drive}/vdet" if os.path.isdir(drive) else "/content/work"
        return dict(platform=plat, work=work, data="/content/vdet_data")
    # laptop: keep data and results outside the repository (they are large)
    return dict(platform=plat, work=os.environ.get("VDET_WORK", os.path.expanduser("~/vdet_work")),
                data=os.environ.get("VDET_DATA", os.path.expanduser("~/vdet_data")))


def _kaggle_credentials() -> None:
    """Find Kaggle API credentials. Checked in this order:
    ~/.kaggle/kaggle.json, env vars, token files (~/.kaggle/access_token, <repo>/.kaggle_token), Colab secrets,
    kaggle*.json in Google Drive (Colab) / Downloads / current folder (laptop). Never commit them."""
    home = os.path.expanduser("~/.kaggle")
    target = f"{home}/kaggle.json"
    if os.path.exists(target) or os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN"):
        return
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [f"{home}/access_token", f"{repo_root}/.kaggle_token", ".kaggle_token", "../.kaggle_token",
              "/content/drive/MyDrive/.kaggle_token"]:
        if os.path.exists(p):  # new-style token (KGAT_...) saved as plain text; .kaggle_token is git-ignored
            os.environ["KAGGLE_API_TOKEN"] = open(p).read().strip()
            return
    try:
        from google.colab import userdata
        for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
            try:
                v = userdata.get(k)
            except Exception:
                v = None
            if v:
                os.environ[k] = v
    except ImportError:
        pass
    if os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN"):
        return
    patterns = ["/content/kaggle*.json", "/content/drive/MyDrive/kaggle*.json",
                os.path.expanduser("~/Downloads/kaggle*.json"), "kaggle*.json", "../kaggle*.json"]
    for pattern in patterns:
        for p in sorted(glob.glob(pattern)):
            os.makedirs(home, exist_ok=True)
            shutil.copy(p, target)
            os.chmod(target, 0o600)
            print(f"Kaggle credentials copied from {p} to {target}")
            return
    raise RuntimeError(
        "Kaggle credentials not found. Kaggle -> Settings -> API -> Create New Token, then:\n"
        "  laptop: leave kaggle.json in Downloads (or put it into ~/.kaggle/);\n"
        "  Colab:  put kaggle.json into the root of Google Drive (My Drive).\n"
        "If Kaggle shows a token text instead of a file, save that text into ~/.kaggle/access_token.")


def _download_comp(root: str) -> str:
    """Download and unzip the competition data into root/<slug> (only the first time)."""
    import zipfile
    comp = f"{root}/{SLUG}"
    found = glob.glob(f"{comp}/test_images.csv") + glob.glob(f"{comp}/*/test_images.csv")
    if found:
        return os.path.dirname(found[0])
    _kaggle_credentials()
    import kaggle  # authenticates on import, so credentials must be set first
    os.makedirs(root, exist_ok=True)
    print(f"Downloading competition data to {root} (one time, can take 10-30 min)...")
    kaggle.api.competition_download_files(SLUG, path=root, quiet=False)
    zip_path = f"{root}/{SLUG}.zip"
    print("Unzipping...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(comp)
    os.remove(zip_path)
    found = glob.glob(f"{comp}/test_images.csv") + glob.glob(f"{comp}/*/test_images.csv")
    return os.path.dirname(found[0])


def get_comp(paths: dict) -> str:
    """Path to the original competition folder.
    Kaggle: the attached input. Colab / laptop: downloaded once with the Kaggle API."""
    if paths["platform"] == "kaggle":
        for p in [f"/kaggle/input/competitions/{SLUG}", f"/kaggle/input/{SLUG}"]:
            if os.path.exists(f"{p}/test_images.csv"):
                return p
        raise FileNotFoundError("Competition data not found: right panel -> Add Input -> Competitions.")
    if os.environ.get("VDET_COMP"):
        return os.environ["VDET_COMP"]
    root = "/content/raw" if paths["platform"] == "colab" else f"{paths['data']}/raw"
    return _download_comp(root)


# =====================================================================================
# Frame subset: every N-th frame, resized once so that the longer side <= max_side
# =====================================================================================
def _tag(train_step, val_step, max_side, mask_ignore, lowres_frac) -> str:
    tag = f"s{train_step}-{val_step}_m{max_side}"
    if mask_ignore:
        tag += "_mask"
    if lowres_frac:
        tag += f"_lowres{int(round(lowres_frac * 100))}"
    return tag


def prepare_frames(comp: str, out: str, train_step: int, val_step: int, max_side: int,
                   mask_ignore: bool = False, lowres_frac: float = 0.0, workers: int | None = None) -> None:
    """Copy every N-th frame (resized), its YOLO label and a COCO ground-truth json for that subset.

    * Ground truth json keeps ORIGINAL pixel coordinates (incl. iscrowd ignore boxes) and stores the
      resize factor per image in images[i]["scale"], so predictions are always evaluated at the
      original resolution (same AP_small definition for all models).
    * mask_ignore: paint ignored regions grey in TRAIN frames (they contain unlabelled objects).
    * lowres_frac: share of TRAIN frames that are down-scaled to 680/960 px and up-scaled back,
      to imitate the low-resolution test frames.
    """
    t0 = time.time()
    workers = workers or min(16, 2 * (os.cpu_count() or 2))
    rng = random.Random(SEED)
    for split, step in (("train", train_step), ("val", val_step)):
        with open(f"{comp}/{split}/annotations_coco.json") as f:
            coco = json.load(f)
        images = [im for im in coco["images"] if (im["frame_id"] - 1) % step == 0]
        keep = {im["id"] for im in images}
        anns = [a for a in coco["annotations"] if a["image_id"] in keep]
        categories = coco["categories"]
        del coco

        ignore = defaultdict(list)
        for a in anns:
            if a["iscrowd"] == 1 and a["category_id"] == 1:
                ignore[a["image_id"]].append(a["bbox"])

        jobs = []
        for im in images:
            im["scale"] = min(1.0, max_side / max(im["width"], im["height"]))
            lowres = split == "train" and lowres_frac > 0 and rng.random() < lowres_frac
            jobs.append((im, lowres))

        def work(job):
            im, lowres = job
            s = im["scale"]
            dst = f"{out}/{split}/{im['file_name']}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            img = Image.open(f"{comp}/{split}/{im['file_name']}").convert("RGB")
            size = (round(im["width"] * s), round(im["height"] * s))
            if s < 1:
                img = img.resize(size, Image.Resampling.BILINEAR)
            if mask_ignore and split == "train" and ignore.get(im["id"]):
                draw = ImageDraw.Draw(img)
                for x, y, w, h in ignore[im["id"]]:
                    draw.rectangle([x * s, y * s, (x + w) * s, (y + h) * s], fill=(114, 114, 114))
            if lowres:
                long_side = random.Random(im["id"]).choice([680, 960])
                f = long_side / max(size)
                small = img.resize((round(size[0] * f), round(size[1] * f)), Image.Resampling.BILINEAR)
                img = small.resize(size, Image.Resampling.BILINEAR)
            img.save(dst, quality=95)

            label_src = f"{comp}/{split}/{im['file_name']}".replace("/images/", "/labels/")[:-4] + ".txt"
            label_dst = dst.replace("/images/", "/labels/")[:-4] + ".txt"
            os.makedirs(os.path.dirname(label_dst), exist_ok=True)
            if os.path.exists(label_src):
                shutil.copyfile(label_src, label_dst)
            else:
                open(label_dst, "w").close()

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, jobs))

        with open(f"{out}/{split}_gt.json", "w") as f:
            json.dump(dict(info={"subset_step": step, "max_side": max_side},
                           categories=categories, images=images, annotations=anns), f)
        print(f"{split}: {len(images)} frames, {sum(a['iscrowd'] == 0 for a in anns)} objects")

    with open(f"{out}/data.yaml", "w") as f:
        f.write(f"path: {out}\ntrain: train/images\nval: val/images\nnames:\n")
        f.write("".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    with open(f"{out}/done.json", "w") as f:
        json.dump(dict(train_step=train_step, val_step=val_step, max_side=max_side,
                       mask_ignore=mask_ignore, lowres_frac=lowres_frac), f)
    print(f"prepared in {time.time() - t0:.0f}s -> {out}")


def setup(train_step: int = 10, val_step: int = 5, max_side: int = 1600,
          mask_ignore: bool = False, lowres_frac: float = 0.0, smoke: bool = False) -> dict:
    """Prepare (or restore from cache) the frame subset and return all paths as a dict `S`.

    S["work"]      - outputs kept between sessions (runs, results, submissions);
                     smoke=True writes to <work>/smoke, so test runs never mix with real ones
    S["prep"]      - prepared train/val frames, labels, *_gt.json and data.yaml
    S["data_yaml"] - Ultralytics config
    S["test_root"] - folder with test_images.csv and test/images/...
    """
    seed_everything()
    S = get_paths()
    os.makedirs(S["work"], exist_ok=True)
    tag = _tag(train_step, val_step, max_side, mask_ignore, lowres_frac)
    prep = f"{S['data']}/{tag}"
    cache_tar = f"{S['work']}/cache/{tag}.tar"
    raw_test = f"{S['data']}/raw_test"

    if not os.path.exists(f"{prep}/done.json"):
        if S["platform"] == "colab" and os.path.exists(cache_tar):
            print("Restoring prepared data from Drive cache...")
            with tarfile.open(cache_tar) as t:
                try:
                    t.extractall(S["data"], filter="data")
                except TypeError:
                    t.extractall(S["data"])
        else:
            comp = get_comp(S)
            prepare_frames(comp, prep, train_step, val_step, max_side, mask_ignore, lowres_frac)
            if S["platform"] == "colab":
                if not os.path.exists(f"{raw_test}/test_images.csv"):
                    shutil.copytree(f"{comp}/test", f"{raw_test}/test")
                    shutil.copy(f"{comp}/test_images.csv", raw_test)
                print("Saving prepared data to Drive cache (next sessions skip the download)...")
                os.makedirs(os.path.dirname(cache_tar), exist_ok=True)
                with tarfile.open(cache_tar, "w") as t:
                    t.add(prep, arcname=tag)
                    t.add(raw_test, arcname="raw_test")

    if S["platform"] == "colab":
        S["test_root"] = raw_test
    else:
        S["test_root"] = get_comp(S)
    S.update(prep=prep, data_yaml=f"{prep}/data.yaml", tag=tag)
    if smoke:
        S["work"] = f"{S['work']}/smoke"
    for d in ["runs", "results", "figs", "submissions"]:
        os.makedirs(f"{S['work']}/{d}", exist_ok=True)
    return S


# =====================================================================================
# Image lists for prediction: dicts {key, path, scale}; boxes are divided by `scale`
# =====================================================================================
def val_items(S: dict) -> list[dict]:
    with open(f"{S['prep']}/val_gt.json") as f:
        gt = json.load(f)
    return [dict(key=im["id"], path=f"{S['prep']}/val/{im['file_name']}", scale=im["scale"])
            for im in gt["images"]]


def lowres_val_items(S: dict, long_side: int = 680) -> list[dict]:
    """Val frames down-scaled to `long_side` px, like the low-resolution test frames."""
    with open(f"{S['prep']}/val_gt.json") as f:
        gt = json.load(f)
    items = []
    for im in gt["images"]:
        f_ = long_side / max(im["width"], im["height"])
        dst = f"{S['prep']}/val_lowres{long_side}/{im['file_name']}"
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            img = Image.open(f"{S['prep']}/val/{im['file_name']}")
            img.resize((round(im["width"] * f_), round(im["height"] * f_)),
                       Image.Resampling.BILINEAR).save(dst, quality=95)
        items.append(dict(key=im["id"], path=dst, scale=f_))
    return items


def test_items(S: dict) -> list[dict]:
    df = pd.read_csv(f"{S['test_root']}/test_images.csv")
    return [dict(key=r.image_id, path=f"{S['test_root']}/{r.file_name}", scale=1.0)
            for r in df.itertuples()]


# =====================================================================================
# Evaluation
# =====================================================================================
def coco_eval(gt_json: str, preds: list[dict], per_class: bool = False) -> dict:
    """COCO mAP with up to 500 detections per image (as on the leaderboard).

    Ignored regions are iscrowd=1 boxes in the ground truth, so detections inside them are not
    counted as false positives. `preds` use original pixel coordinates and category ids 1..8.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not preds:
        return {k: 0.0 for k in METRIC_KEYS}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_json)
        dt = gt.loadRes(preds)
        E = COCOeval(gt, dt, "bbox")
        E.params.maxDets = [1, 100, 500]
        E.evaluate()
        E.accumulate()
        E.summarize()
    s = E.stats
    res = dict(mAP=s[0], AP50=s[1], AP75=s[2], AP_small=s[3], AP_medium=s[4], AP_large=s[5], AR500=s[8])
    if per_class:
        prec = E.eval["precision"]  # [IoU, recall, class, area, maxDets]
        for k, cat_id in enumerate(E.params.catIds):
            p = prec[:, :, k, 0, -1]
            res[f"AP_{CLASS_NAMES[cat_id - 1]}"] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return {k: round(float(v), 4) for k, v in res.items()}


def measure_speed(predict_one, paths: list[str], n: int = 100, warmup: int = 5) -> dict:
    """Average time per image (incl. reading, pre- and post-processing), batch size 1."""
    import torch
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for p in paths[:warmup]:
        predict_one(p)
    sync()
    sel = paths[warmup:warmup + n] or paths[:n]
    t = time.perf_counter()
    for p in sel:
        predict_one(p)
    sync()
    ms = (time.perf_counter() - t) / len(sel) * 1000
    return dict(ms_per_img=round(ms, 1), FPS=round(1000 / ms, 1))


def log_result(S: dict, notebook: str, run: str, metrics: dict, **extra) -> pd.DataFrame:
    """Append one row to work/results/<notebook>.csv (one row per run, re-running overwrites it)."""
    path = f"{S['work']}/results/{notebook}.csv"
    row = pd.DataFrame([{"notebook": notebook, "run": run, **metrics, **extra}])
    if os.path.exists(path):
        old = pd.read_csv(path)
        row = pd.concat([old[old.run != run], row], ignore_index=True)
    row.to_csv(path, index=False)
    return row


def _search_inputs(rel: str) -> list[str]:
    """Find a file in /kaggle/input outputs of other notebooks (skips the competition data)."""
    hits = []
    if os.path.exists("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input"):
            dirs[:] = [d for d in dirs if d not in ("competitions", SLUG, "images", "labels", "ignore")]
            p = os.path.join(root, rel)
            if os.path.exists(p):
                hits.append(p)
    return hits


def find_file(S: dict, rel: str) -> str | None:
    """Look for `rel` (e.g. 'runs/yolo_base/weights/best.pt') in work dir, then in Kaggle inputs."""
    p = f"{S['work']}/{rel}"
    if os.path.exists(p):
        return p
    hits = _search_inputs(rel)
    return hits[0] if hits else None


def load_results(S: dict) -> pd.DataFrame:
    """All logged results: this work dir, Kaggle inputs and the repository's results/ folder
    (commit results/<notebook>.csv to GitHub to compare runs made on different platforms)."""
    repo_results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    files = glob.glob(f"{repo_results}/*.csv") + glob.glob(f"{S['work']}/results/*.csv")
    for nb in ["02_faster_rcnn", "03_retinanet", "04_yolo", "05_rtdetr", "06_final_pipeline"]:
        if not os.path.exists(f"{S['work']}/results/{nb}.csv"):
            files += _search_inputs(f"results/{nb}.csv")[:1]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.drop_duplicates(["notebook", "run"], keep="last").reset_index(drop=True)


# =====================================================================================
# Ultralytics (YOLO and RT-DETR)
# =====================================================================================
def fp16_args() -> dict:
    """FP16 inference on GPU. New Ultralytics versions use `quantize=16`, older ones `half=True`."""
    import torch
    if not torch.cuda.is_available():
        return {}
    from ultralytics.cfg import get_cfg
    return {"quantize": 16} if "quantize" in vars(get_cfg()) else {"half": True}


def ultra_train(S: dict, name: str, model_src: str, epochs: int, imgsz: int, batch: int,
                family: str = "yolo", load_weights: str | None = None, **kwargs) -> str:
    """Train an Ultralytics model into work/runs/<name>. Skips finished runs, resumes interrupted ones.

    The full Ultralytics log goes to runs/<name>/train_log.txt; the notebook only gets one line per epoch.
    """
    import logging
    from ultralytics import RTDETR, YOLO
    from ultralytics.utils import LOGGER
    cls = RTDETR if family == "rtdetr" else YOLO
    project = f"{S['work']}/runs"
    run_dir = f"{project}/{name}"
    csv, best, last = f"{run_dir}/results.csv", f"{run_dir}/weights/best.pt", f"{run_dir}/weights/last.pt"

    if os.path.exists(best) and os.path.exists(csv) and len(pd.read_csv(csv)) >= epochs:
        print(f"[{name}] already trained - skipping")
        return run_dir

    out, t0 = sys.stdout, time.time()

    printed = set()

    def on_epoch_end(trainer):
        if trainer.epoch >= trainer.epochs or trainer.epoch in printed:  # final re-validation of best.pt
            return
        printed.add(trainer.epoch)
        m = trainer.metrics or {}
        out.write(f"[{name}] epoch {trainer.epoch + 1}/{trainer.epochs}  "
                  f"val mAP50-95 (ultralytics) = {m.get('metrics/mAP50-95(B)', float('nan')):.4f}  "
                  f"({(time.time() - t0) / 60:.0f} min)\n")
        out.flush()

    def quiet_train(model, **train_kwargs):
        os.makedirs(run_dir, exist_ok=True)
        model.add_callback("on_fit_epoch_end", on_epoch_end)
        level = LOGGER.level
        LOGGER.setLevel(logging.WARNING)
        try:
            with open(f"{run_dir}/train_log.txt", "a") as log, \
                    contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                model.train(**train_kwargs)
        finally:
            LOGGER.setLevel(level)
        print(f"[{name}] done in {(time.time() - t0) / 60:.0f} min -> {run_dir}")

    if os.path.exists(last) and os.path.exists(csv):
        try:
            print(f"[{name}] resuming interrupted run")
            quiet_train(cls(last), resume=True)
            return run_dir
        except Exception as e:  # e.g. run finished but best.pt missing
            print(f"[{name}] resume failed ({e}), training from scratch")
    model = cls(model_src)
    if load_weights:
        model.load(load_weights)
    kwargs.setdefault("workers", min(8, os.cpu_count() or 2))
    quiet_train(model, data=S["data_yaml"], project=project, name=name, exist_ok=True, epochs=epochs,
                imgsz=imgsz, batch=batch, seed=SEED, **kwargs)
    return run_dir


def ultra_predict(model, items: list[dict], imgsz: int, conf: float = 0.001, iou: float = 0.6,
                  max_det: int = 500, batch: int = 8, augment: bool = False) -> list[dict]:
    """Predict with an Ultralytics model; returns COCO-style detections in original coordinates."""
    fp16 = fp16_args()
    preds = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        results = model.predict([it["path"] for it in chunk], imgsz=imgsz, conf=conf, iou=iou,
                                max_det=max_det, augment=augment, verbose=False, **fp16)
        for it, r in zip(chunk, results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy() / it["scale"]
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            preds += [dict(image_id=it["key"], category_id=int(c) + 1, score=round(float(s), 5),
                           bbox=[round(float(x1), 2), round(float(y1), 2),
                                 round(float(x2 - x1), 2), round(float(y2 - y1), 2)])
                      for (x1, y1, x2, y2), s, c in zip(xyxy, scores, classes)]
    return preds


def ultra_tiled_predict(model, items: list[dict], imgsz: int, grid=(2, 2), overlap: float = 0.2,
                        conf: float = 0.001, iou: float = 0.6, max_det: int = 500,
                        add_full: bool = True) -> list[dict]:
    """SAHI-style inference: predict on overlapping tiles (+ the full frame) and merge with NMS.

    Every tile is resized to `imgsz`, so small objects become larger for the model.
    """
    import torch
    from torchvision.ops import batched_nms
    fp16 = fp16_args()
    preds = []
    for it in items:
        img = np.array(Image.open(it["path"]).convert("RGB"))[:, :, ::-1]  # BGR for Ultralytics
        H, W = img.shape[:2]
        nx, ny = grid
        tw, th = int(W / (nx - (nx - 1) * overlap)), int(H / (ny - (ny - 1) * overlap))
        xs = np.linspace(0, W - tw, nx).astype(int)
        ys = np.linspace(0, H - th, ny).astype(int)
        crops, offsets = [], []
        for y0 in ys:
            for x0 in xs:
                crops.append(np.ascontiguousarray(img[y0:y0 + th, x0:x0 + tw]))
                offsets.append((x0, y0))
        if add_full:
            crops.append(np.ascontiguousarray(img))
            offsets.append((0, 0))
        boxes, scores, classes = [], [], []
        for r, (x0, y0) in zip(model.predict(crops, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                                             verbose=False, **fp16), offsets):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes.append(r.boxes.xyxy.float().cpu() + torch.tensor([x0, y0, x0, y0], dtype=torch.float32))
            scores.append(r.boxes.conf.float().cpu())
            classes.append(r.boxes.cls.cpu().long())
        if not boxes:
            continue
        boxes, scores, classes = torch.cat(boxes), torch.cat(scores), torch.cat(classes)
        keep = batched_nms(boxes, scores, classes, iou)[:max_det]
        for (x1, y1, x2, y2), s, c in zip((boxes[keep] / it["scale"]).tolist(), scores[keep].tolist(),
                                          classes[keep].tolist()):
            preds.append(dict(image_id=it["key"], category_id=int(c) + 1, score=round(s, 5),
                              bbox=[round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]))
    return preds


def read_ultra_results(run_dir: str) -> pd.DataFrame:
    df = pd.read_csv(f"{run_dir}/results.csv")
    df.columns = [c.strip() for c in df.columns]
    return df


# =====================================================================================
# Submission
# =====================================================================================
def write_submission(S: dict, preds: list[dict], name: str, max_det: int = 500) -> str:
    """Write submissions/<name>.csv: `class score x y w h` groups, class 0..7, original pixels."""
    by_img = defaultdict(list)
    for p in preds:
        by_img[p["image_id"]].append(p)
    df = pd.read_csv(f"{S['test_root']}/test_images.csv")
    rows = []
    for image_id in df.image_id:
        ps = sorted(by_img.get(image_id, []), key=lambda p: -p["score"])[:max_det]
        rows.append(" ".join(f"{p['category_id'] - 1} {p['score']:.4f} {p['bbox'][0]:.1f} {p['bbox'][1]:.1f} "
                             f"{p['bbox'][2]:.1f} {p['bbox'][3]:.1f}" for p in ps))
    path = f"{S['work']}/submissions/{name}.csv"
    pd.DataFrame({"image_id": df.image_id, "PredictionString": rows}).to_csv(path, index=False)
    n = [len(r.split()) // 6 for r in rows]
    print(f"saved {path}: {len(rows)} rows, {np.mean(n):.0f} detections per frame on average")
    return path


# =====================================================================================
# Error analysis and plots
# =====================================================================================
def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between xywh boxes a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def _ioa(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Share of each box in a covered by each box in b."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iou = _iou(a, b)
    area_a, area_b = (a[:, 2] * a[:, 3])[:, None], (b[:, 2] * b[:, 3])[None]
    inter = iou * (area_a + area_b) / (1 + iou)
    return inter / np.maximum(area_a, 1e-9)


def classify_frame(gts, crowd, dets, score_thr=0.3, iou_thr=0.5):
    """Label every detection (TP / duplicate / classification / localization / background / ignored)
    and every GT object (found / missed) for one frame. gts, dets: arrays [x, y, w, h, cls(, score)]."""
    dets = dets[dets[:, 5] >= score_thr] if len(dets) else dets
    dets = dets[np.argsort(-dets[:, 5])] if len(dets) else dets
    iou = _iou(dets[:, :4], gts[:, :4]) if len(dets) else np.zeros((0, len(gts)))
    ign = _ioa(dets[:, :4], crowd) if len(dets) and len(crowd) else np.zeros((len(dets), 0))
    matched = np.zeros(len(gts), bool)
    det_type = []
    for i in range(len(dets)):
        same = gts[:, 4] == dets[i, 4] if len(gts) else np.zeros(0, bool)
        cand = np.where(same & ~matched & (iou[i] >= iou_thr))[0]
        if len(cand):
            matched[cand[np.argmax(iou[i, cand])]] = True
            det_type.append("TP")
        elif ign.shape[1] and ign[i].max() > 0.5:
            det_type.append("ignored")
        elif len(gts) and (iou[i][same] >= iou_thr).any():
            det_type.append("duplicate")
        elif len(gts) and (iou[i] >= iou_thr).any():
            det_type.append("classification")
        elif len(gts) and (iou[i][same] >= 0.1).any():
            det_type.append("localization")
        else:
            det_type.append("background")
    return dets, det_type, matched


def _frame_arrays(gt: dict):
    """Group ground truth by image: real objects and ignore boxes as numpy arrays."""
    objs, crowd = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        if a["iscrowd"] == 0:
            objs[a["image_id"]].append(a["bbox"] + [a["category_id"]])
        elif a["category_id"] == 1:
            crowd[a["image_id"]].append(a["bbox"])
    return objs, crowd


def error_breakdown(S: dict, preds: list[dict], score_thr: float = 0.3, iou_thr: float = 0.5) -> pd.DataFrame:
    """Simple TIDE-like count of error types on the val subset at a fixed score threshold."""
    with open(f"{S['prep']}/val_gt.json") as f:
        gt = json.load(f)
    objs, crowd = _frame_arrays(gt)
    dets_by = defaultdict(list)
    for p in preds:
        dets_by[p["image_id"]].append(p["bbox"] + [p["category_id"], p["score"]])
    det_counts, miss_by_size, found_by_size = defaultdict(int), defaultdict(int), defaultdict(int)
    for im in gt["images"]:
        g = np.array(objs.get(im["id"], []), float).reshape(-1, 5)
        c = np.array(crowd.get(im["id"], []), float).reshape(-1, 4)
        d = np.array(dets_by.get(im["id"], []), float).reshape(-1, 6)
        _, types, matched = classify_frame(g, c, d, score_thr, iou_thr)
        for t in types:
            det_counts[t] += 1
        area = g[:, 2] * g[:, 3]
        size = np.where(area < 32 ** 2, "small", np.where(area < 96 ** 2, "medium", "large"))
        for s_, m in zip(size, matched):
            (found_by_size if m else miss_by_size)[s_] += 1
    det_df = pd.Series(det_counts, name="detections").reindex(
        ["TP", "duplicate", "classification", "localization", "background", "ignored"]).fillna(0).astype(int)
    miss = pd.DataFrame({"found": found_by_size, "missed": miss_by_size}).reindex(["small", "medium", "large"]).fillna(0)
    miss["miss_rate"] = (miss.missed / (miss.found + miss.missed).clip(lower=1)).round(3)
    print(f"Detections with score >= {score_thr}:")
    print(det_df.to_string())
    print("\nGround-truth objects by size:")
    print(miss.astype({"found": int, "missed": int}).to_string())
    return miss


def plot_errors(S: dict, preds: list[dict], image_ids: list[int], score_thr: float = 0.3,
                iou_thr: float = 0.5, titles: list[str] | None = None, save: str | None = None):
    """Show TP (green), false positives (red), missed GT (yellow) and ignored regions (grey)."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    with open(f"{S['prep']}/val_gt.json") as f:
        gt = json.load(f)
    ims = {im["id"]: im for im in gt["images"]}
    objs, crowd = _frame_arrays(gt)
    dets_by = defaultdict(list)
    for p in preds:
        dets_by[p["image_id"]].append(p["bbox"] + [p["category_id"], p["score"]])
    n = len(image_ids)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(22, 6.5 * ((n + 1) // 2)), squeeze=False)
    for ax, iid, k in zip(axes.ravel(), image_ids, range(n)):
        im = ims[iid]
        s = im["scale"]
        ax.imshow(Image.open(f"{S['prep']}/val/{im['file_name']}"))
        g = np.array(objs.get(iid, []), float).reshape(-1, 5)
        c = np.array(crowd.get(iid, []), float).reshape(-1, 4)
        d = np.array(dets_by.get(iid, []), float).reshape(-1, 6)
        d_sel, types, matched = classify_frame(g, c, d, score_thr, iou_thr)
        for x, y, w, h in c:
            ax.add_patch(patches.Rectangle((x * s, y * s), w * s, h * s, color="grey", alpha=0.4))
        for (x, y, w, h, _), m in zip(g, matched):
            if not m:
                ax.add_patch(patches.Rectangle((x * s, y * s), w * s, h * s, fill=False, ec="yellow", lw=1.2))
        for (x, y, w, h, _, _), t in zip(d_sel, types):
            if t != "ignored":
                ax.add_patch(patches.Rectangle((x * s, y * s), w * s, h * s, fill=False, lw=0.9,
                                               ec="lime" if t == "TP" else "red"))
        fp = sum(t not in ("TP", "ignored") for t in types)
        ax.set_title((titles[k] + "\n" if titles else "") +
                     f"{im['video']} frame {im['frame_id']}: TP={types.count('TP')}, FP={fp}, "
                     f"missed={int((~matched).sum())} (score>={score_thr})", fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=100)
    plt.show()


def worst_frames(S: dict, preds: list[dict], k: int = 4, score_thr: float = 0.3) -> list[int]:
    """Val frames with the most missed objects (useful for picking failure cases)."""
    with open(f"{S['prep']}/val_gt.json") as f:
        gt = json.load(f)
    objs, crowd = _frame_arrays(gt)
    dets_by = defaultdict(list)
    for p in preds:
        dets_by[p["image_id"]].append(p["bbox"] + [p["category_id"], p["score"]])
    scores = []
    for im in gt["images"]:
        g = np.array(objs.get(im["id"], []), float).reshape(-1, 5)
        c = np.array(crowd.get(im["id"], []), float).reshape(-1, 4)
        d = np.array(dets_by.get(im["id"], []), float).reshape(-1, 6)
        _, types, matched = classify_frame(g, c, d, score_thr)
        fp = sum(t not in ("TP", "ignored") for t in types)
        scores.append((int((~matched).sum()) + fp, im["video"], im["id"]))
    # one frame per video, so the examples show different scenes
    seen, out = set(), []
    for _, vid, iid in sorted(scores, reverse=True):
        if vid not in seen:
            seen.add(vid)
            out.append(iid)
        if len(out) == k:
            break
    return out


def results_table(rows: dict) -> pd.DataFrame:
    """{run_name: metrics_dict} -> DataFrame (one row per run)."""
    return pd.DataFrame(rows).T


def plot_size_bars(df: pd.DataFrame, title: str, save: str | None = None):
    import matplotlib.pyplot as plt
    cols = ["AP_small", "AP_medium", "AP_large", "mAP"]
    ax = df[cols].astype(float).plot.bar(figsize=(9, 4), rot=0)
    ax.set_title(title)
    ax.set_ylabel("AP")
    ax.legend(ncol=4, fontsize=9)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=110)
    plt.show()
