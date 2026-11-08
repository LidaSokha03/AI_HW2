"""Torchvision detectors (Faster R-CNN, RetinaNet) for the VisDrone homework.

Contents: dataset, model builders, GIoU loss for Faster R-CNN,
a training loop with AMP + checkpoint/resume, prediction and data-driven anchors.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.models.detection import (FasterRCNN, RetinaNet, fasterrcnn_resnet50_fpn,
                                          retinanet_resnet50_fpn)
from torchvision.models.detection import roi_heads as tv_roi_heads
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetHead
from torchvision.ops import generalized_box_iou_loss

import vdet

NUM_CLASSES = 9  # 8 classes + background (index 0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================================
# Data
# =====================================================================================
class DetDataset(Dataset):
    """Prepared frames + boxes. Ignore boxes (iscrowd=1) are NOT training targets."""

    def __init__(self, gt_json: str, img_root: str, train: bool = True, limit: int | None = None):
        with open(gt_json) as f:
            gt = json.load(f)
        self.images = gt["images"][:limit] if limit else gt["images"]
        self.root, self.train = img_root, train
        self.anns = defaultdict(list)
        for a in gt["annotations"]:
            x, y, w, h = a["bbox"]
            if a["iscrowd"] == 0 and w > 1 and h > 1:
                self.anns[a["image_id"]].append((x, y, x + w, y + h, a["category_id"]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        img = read_image(f"{self.root}/{im['file_name']}", ImageReadMode.RGB)  # uint8 [3, H, W]
        a = torch.tensor(self.anns.get(im["id"], []), dtype=torch.float32).reshape(-1, 5)
        boxes, labels = a[:, :4] * im["scale"], a[:, 4].long()
        if self.train and torch.rand(1).item() < 0.5:  # horizontal flip
            img = img.flip(-1)
            W = img.shape[-1]
            boxes = torch.stack([W - boxes[:, 2], boxes[:, 1], W - boxes[:, 0], boxes[:, 3]], dim=1)
        return img, {"boxes": boxes, "labels": labels}, im["id"]


class ImageList(Dataset):
    """Images for prediction (val or test items from vdet)."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return read_image(self.items[i]["path"], ImageReadMode.RGB), i


def collate(batch):
    return tuple(zip(*batch))


# =====================================================================================
# Models
# =====================================================================================
def build_faster_rcnn(max_side: int = 1280, pretrained: bool = True, **kwargs) -> FasterRCNN:
    """Faster R-CNN R50-FPN from COCO weights with a new 9-class box predictor.

    min_size=max_size=max_side: frames are resized so that the longer side = max_side.
    kwargs go to FasterRCNN (e.g. rpn_post_nms_top_n_train, box_positive_fraction).
    """
    kwargs.setdefault("box_detections_per_img", 500)
    kwargs.setdefault("box_score_thresh", 0.01)
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None,
                                    weights_backbone="DEFAULT" if pretrained else None,
                                    min_size=max_side, max_size=max_side, **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def build_retinanet(max_side: int = 1280, pretrained: bool = True, anchor_sizes=None, aspect_ratios=None,
                    reg_loss: str = "l1", **kwargs) -> RetinaNet:
    """RetinaNet R50-FPN from COCO weights.

    Default anchors: sizes 32..512 (one per FPN level P3-P7, 3 scales each), ratios 0.5/1/2.
    With custom anchors both heads are re-initialised (number of anchors per location may change).
    reg_loss: 'l1' (default), 'smooth_l1', 'giou', 'diou', 'ciou'.
    """
    kwargs.setdefault("detections_per_img", 500)
    kwargs.setdefault("score_thresh", 0.01)
    model = retinanet_resnet50_fpn(weights="DEFAULT" if pretrained else None,
                                   weights_backbone="DEFAULT" if pretrained else None,
                                   min_size=max_side, max_size=max_side, **kwargs)
    if anchor_sizes is not None:
        model.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        n_anchors = model.anchor_generator.num_anchors_per_location()[0]
        model.head = RetinaNetHead(model.backbone.out_channels, n_anchors, NUM_CLASSES)
    else:
        n_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(model.backbone.out_channels,
                                                                     n_anchors, NUM_CLASSES)
    model.head.regression_head._loss_type = reg_loss
    return model


def default_retinanet_anchors():
    sizes = tuple((x, int(x * 2 ** (1 / 3)), int(x * 2 ** (2 / 3))) for x in [32, 64, 128, 256, 512])
    return sizes, ((0.5, 1.0, 2.0),) * len(sizes)


def suggest_anchors(gt_json: str, model_side: int = 1280, n_ratios: int = 3, seed: int = 42):
    """Anchors grounded in the data (YOLOv2-style idea, simplified):

    * object size = sqrt(w*h) after resizing the frame to `model_side`;
    * smallest base size = power of two closest to the 5th percentile (between 8 and 32),
      then doubled for each of the 5 FPN levels, with 3 scales per level (2^0, 2^1/3, 2^2/3);
    * aspect ratios (h/w) = k-means centres of log(h/w).
    """
    with open(gt_json) as f:
        gt = json.load(f)
    ims = {im["id"]: im for im in gt["images"]}
    w, h = [], []
    for a in gt["annotations"]:
        if a["iscrowd"] == 0 and a["bbox"][2] > 1 and a["bbox"][3] > 1:
            im = ims[a["image_id"]]
            f = model_side / max(im["width"], im["height"])
            w.append(a["bbox"][2] * f)
            h.append(a["bbox"][3] * f)
    w, h = np.array(w), np.array(h)
    size = np.sqrt(w * h)
    q = np.percentile(size, [5, 25, 50, 75, 95])
    base = int(np.clip(2 ** round(math.log2(q[0])), 8, 32))
    sizes = tuple(tuple(int(round(base * 2 ** lvl * 2 ** (k / 3))) for k in range(3)) for lvl in range(5))

    log_r = np.log(h / w)
    rng = np.random.default_rng(seed)
    centers = np.quantile(log_r, np.linspace(0.2, 0.8, n_ratios))
    sample = rng.choice(log_r, size=min(len(log_r), 200_000), replace=False)
    for _ in range(50):  # 1-D k-means
        assign = np.argmin(np.abs(sample[:, None] - centers[None]), axis=1)
        centers = np.array([sample[assign == k].mean() if (assign == k).any() else centers[k]
                            for k in range(n_ratios)])
    ratios = tuple(sorted(round(float(np.exp(c)), 2) for c in centers))
    stats = pd.Series(q, index=["p5", "p25", "p50", "p75", "p95"]).round(1)
    return sizes, (ratios,) * 5, stats


# =====================================================================================
# Loss modifications (monkey-patching torchvision internals; always set before training)
# =====================================================================================
_ORIG_FASTRCNN_LOSS = tv_roi_heads.fastrcnn_loss


def _decode_unit(deltas: torch.Tensor, weights=(10.0, 10.0, 5.0, 5.0)) -> torch.Tensor:
    """Decode box deltas relative to a unit proposal. IoU/GIoU do not change under the per-proposal
    shift and scaling, so GIoU computed here equals GIoU of the real boxes."""
    dx, dy = deltas[:, 0] / weights[0], deltas[:, 1] / weights[1]
    clip = math.log(1000.0 / 16)
    dw, dh = (deltas[:, 2] / weights[2]).clamp(max=clip), (deltas[:, 3] / weights[3]).clamp(max=clip)
    pw, ph = torch.exp(dw), torch.exp(dh)
    return torch.stack([dx - pw / 2, dy - ph / 2, dx + pw / 2, dy + ph / 2], dim=1)


def _giou_fastrcnn_loss(class_logits, box_regression, labels, regression_targets, weight=10.0):
    """Same as torchvision's fastrcnn_loss, but the box term is GIoU (weight 10, as in MMDetection)."""
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)
    pos = torch.where(labels > 0)[0]
    N = class_logits.shape[0]
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    pred = _decode_unit(box_regression[pos, labels[pos]].float())
    target = _decode_unit(regression_targets[pos].float())
    box_loss = weight * generalized_box_iou_loss(pred, target, reduction="sum") / labels.numel()
    return classification_loss, box_loss


def set_frcnn_box_loss(kind: str = "smooth_l1") -> None:
    tv_roi_heads.fastrcnn_loss = _giou_fastrcnn_loss if kind == "giou" else _ORIG_FASTRCNN_LOSS



# =====================================================================================
# Training and prediction
# =====================================================================================
@torch.no_grad()
def predict(model, items: list[dict], batch_size: int = 4, workers: int = 2) -> list[dict]:
    """COCO-style detections in original pixel coordinates (boxes divided by item scale)."""
    model.eval().to(DEVICE)
    loader = DataLoader(ImageList(items), batch_size=batch_size, num_workers=workers, collate_fn=collate)
    preds = []
    for imgs, idx in loader:
        imgs = [im.to(DEVICE, non_blocking=True).float().div_(255) for im in imgs]
        with torch.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
            outs = model(imgs)
        for out, i in zip(outs, idx):
            it = items[i]
            b = (out["boxes"].float().cpu().numpy() / it["scale"]).round(2)
            for (x1, y1, x2, y2), s, c in zip(b, out["scores"].float().cpu().numpy(), out["labels"].cpu().numpy()):
                preds.append(dict(image_id=it["key"], category_id=int(c), score=round(float(s), 5),
                                  bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]))
    return preds


def speed(model, items: list[dict], n: int = 100) -> dict:
    model.eval().to(DEVICE)

    @torch.no_grad()
    def one(path):
        img = read_image(path, ImageReadMode.RGB).to(DEVICE).float().div_(255)
        with torch.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
            model([img])

    return vdet.measure_speed(one, [it["path"] for it in items], n=n)


def train(model, S: dict, run: str, epochs: int = 6, batch_size: int = 4, lr: float = 0.005,
          lr_drop_epoch: int | None = None, warmup_iters: int = 300, workers: int | None = None,
          max_iters_per_epoch: int | None = None, eval_limit: int | None = None) -> pd.DataFrame:
    """SGD + linear warm-up + x0.1 LR drop, AMP, grad clipping. Evaluates COCO mAP on val after
    every epoch. Checkpoints to work/runs/<run>/ and resumes automatically after a disconnect.
    Returns the per-epoch history (also saved as history.csv)."""
    vdet.seed_everything()
    out = f"{S['work']}/runs/{run}"
    os.makedirs(out, exist_ok=True)
    workers = workers if workers is not None else min(8, os.cpu_count() or 2)
    lr_drop_epoch = lr_drop_epoch if lr_drop_epoch is not None else max(1, epochs - 1)

    ds = DetDataset(f"{S['prep']}/train_gt.json", f"{S['prep']}/train", train=True)
    g = torch.Generator().manual_seed(vdet.SEED)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate,
                        generator=g, drop_last=True, pin_memory=DEVICE.type == "cuda",
                        persistent_workers=workers > 0)
    iters = min(len(loader), max_iters_per_epoch or len(loader))
    val = vdet.val_items(S)[:eval_limit] if eval_limit else vdet.val_items(S)

    model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    drop_at = lr_drop_epoch * iters
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / warmup_iters) * (0.1 if it >= drop_at else 1.0))
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    history, start, best = [], 0, -1.0
    if os.path.exists(f"{out}/last.pt"):
        ck = torch.load(f"{out}/last.pt", map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        history, start = ck["history"], ck["epoch"] + 1
        best = max([h["mAP"] for h in history], default=-1.0)
        print(f"[{run}] resumed after epoch {start}")
    if start >= epochs:
        print(f"[{run}] already trained - skipping")
        return pd.DataFrame(history)

    for epoch in range(start, epochs):
        model.train()
        t0, sums, n = time.time(), defaultdict(float), 0
        for it, (imgs, targets, _) in enumerate(loader):
            if it >= iters:
                break
            imgs = [im.to(DEVICE, non_blocking=True).float().div_(255) for im in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
                losses = model(imgs, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                print("non-finite loss, batch skipped")
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in losses.items():
                sums[k] += v.item()
            n += 1
        train_time = time.time() - t0
        metrics = vdet.coco_eval(f"{S['prep']}/val_gt.json", predict(model, val, workers=workers))
        row = dict(epoch=epoch + 1, train_min=round(train_time / 60, 1), lr=opt.param_groups[0]["lr"],
                   **{k: round(v / max(n, 1), 4) for k, v in sums.items()}, **metrics)
        history.append(row)
        print(f"[{run}] epoch {epoch + 1}/{epochs}  loss={sum(sums.values()) / max(n, 1):.3f}  "
              f"mAP={metrics['mAP']:.4f}  AP50={metrics['AP50']:.4f}  ({train_time / 60:.1f} min)")
        if metrics["mAP"] > best:
            best = metrics["mAP"]
            torch.save(model.state_dict(), f"{out}/best.pt")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), history=history, epoch=epoch), f"{out}/last.pt")
        pd.DataFrame(history).to_csv(f"{out}/history.csv", index=False)
    return pd.DataFrame(history)


def load_best(model, S: dict, run: str):
    path = vdet.find_file(S, f"runs/{run}/best.pt")
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


@torch.no_grad()
def rpn_recall(model, S: dict, items: list[dict], topn=(100, 300, 1000), iou_thr: float = 0.5,
               batch_size: int = 2) -> pd.DataFrame:
    """Share of GT objects covered by at least one of the top-N RPN proposals (IoU >= iou_thr)."""
    with open(f"{S['prep']}/val_gt.json") as f:
        gt = json.load(f)
    objs = defaultdict(list)
    for a in gt["annotations"]:
        if a["iscrowd"] == 0:
            objs[a["image_id"]].append(a["bbox"])
    model.eval().to(DEVICE)
    old_pre, old_post = dict(model.rpn._pre_nms_top_n), dict(model.rpn._post_nms_top_n)
    model.rpn._pre_nms_top_n["testing"] = max(max(topn), old_pre["testing"])
    model.rpn._post_nms_top_n["testing"] = max(topn)
    keys = ["all", "small", "medium", "large"]
    hits, total = {n: dict.fromkeys(keys, 0) for n in topn}, dict.fromkeys(keys, 0)
    loader = DataLoader(ImageList(items), batch_size=batch_size, num_workers=2, collate_fn=collate)
    for imgs, idx in loader:
        imgs = [im.to(DEVICE).float().div_(255) for im in imgs]
        orig_sizes = [tuple(im.shape[-2:]) for im in imgs]
        with torch.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
            images, _ = model.transform(imgs)
            feats = model.backbone(images.tensors)
            proposals, _ = model.rpn(images, feats)
        for p, (h0, w0), (h, w), i in zip(proposals, orig_sizes, images.image_sizes, idx):
            it = items[i]
            g = np.array(objs.get(it["key"], []), float).reshape(-1, 4)
            if not len(g):
                continue
            p = p.float().cpu().numpy() * np.array([w0 / w, h0 / h, w0 / w, h0 / h]) / it["scale"]
            p = np.c_[p[:, :2], p[:, 2:] - p[:, :2]]
            area = g[:, 2] * g[:, 3]
            size = np.where(area < 32 ** 2, "small", np.where(area < 96 ** 2, "medium", "large"))
            total["all"] += len(g)
            for s_ in size:
                total[s_] += 1
            for n in topn:
                iou = vdet._iou(g, p[:n])
                ok = iou.max(1) >= iou_thr if iou.shape[1] else np.zeros(len(g), bool)
                hits[n]["all"] += int(ok.sum())
                for s_, o in zip(size, ok):
                    hits[n][s_] += int(o)
    model.rpn._pre_nms_top_n.update(old_pre)
    model.rpn._post_nms_top_n.update(old_post)
    return pd.DataFrame({f"top {n}": {k: hits[n][k] / max(total[k], 1) for k in keys} for n in topn}).round(3)
