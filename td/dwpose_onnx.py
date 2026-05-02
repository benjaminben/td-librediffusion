"""
Minimal DWPose detector backed by onnxruntime-gpu, replacing
controlnet_aux.dwpose.DWposeDetector's mmpose+mmdet+mmcv stack with
direct ONNX inference. Used by openpose_top.py when DWPose is selected.

The mmpose stack doesn't install cleanly on Windows + Py 3.11 + torch
2.6 (mmcv source build fails on pkg_resources, chumpy fails on missing
pip in its build env). Since controlnet_aux's renderer is pure
numpy/cv2 and only `Wholebody.__call__` depends on mmpose, we replace
just that one class.

Models pulled from yzd-v/DWPose on HuggingFace:
  yolox_l.onnx              640x640 person detector (~200 MB)
  dw-ll_ucoco_384.onnx      288x384 whole-body pose estimator (~120 MB)

Renderer is reused from controlnet_aux.dwpose.util so the rendered
stick figure matches what the openposev2 controlnet was trained on.
"""

import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from controlnet_aux.dwpose import util as dwpose_util
from controlnet_aux.util import HWC3, resize_image


# ---- YOLOX preprocessing / postprocessing ---------------------------------

YOLOX_INPUT_SIZE = (640, 640)  # (H, W)
YOLOX_STRIDES = [8, 16, 32]


def _yolox_letterbox(img_bgr, input_size=YOLOX_INPUT_SIZE):
    """Resize maintaining aspect into a 114-padded canvas. YOLOX standard."""
    H, W = img_bgr.shape[:2]
    th, tw = input_size
    scale = min(th / H, tw / W)
    nh, nw = int(H * scale), int(W * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, scale


def _yolox_postprocess(outputs, input_size=YOLOX_INPUT_SIZE, strides=YOLOX_STRIDES):
    """Decode YOLOX raw outputs [1,N,85] from FPN-grid form to absolute
    cxcywh in the model's 640x640 input space."""
    grids, expanded_strides = [], []
    th, tw = input_size
    for stride in strides:
        hsize, wsize = th // stride, tw // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((1, hsize * wsize, 1), stride))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def _nms(boxes, scores, iou_thresh=0.45):
    """Plain numpy NMS. boxes [N,4] xyxy."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thresh]
    return np.array(keep, dtype=np.int64)


# ---- DWPose preprocessing / SimCC decode ----------------------------------

# RTMPose / DWPose uses ImageNet RGB stats applied to BGR input
# (the model was trained with to_rgb=True so the order matches).
DWPOSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
DWPOSE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _crop_warp(img_bgr, bbox, target_wh):
    """Crop a person bbox with 1.25x padding + aspect-fix, warp to (W,H).
    Returns the cropped image and (center, scale) for inverse mapping."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * 1.25
    bh = (y2 - y1) * 1.25
    tw, th = target_wh
    target_aspect = tw / th
    if bw / bh > target_aspect:
        bh = bw / target_aspect
    else:
        bw = bh * target_aspect
    sx = bw / tw
    sy = bh / th
    M = np.array([
        [1.0 / sx, 0.0, tw / 2.0 - cx / sx],
        [0.0, 1.0 / sy, th / 2.0 - cy / sy],
    ], dtype=np.float32)
    cropped = cv2.warpAffine(img_bgr, M, (tw, th),
                             flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return cropped, (cx, cy), (sx, sy)


def _decode_simcc(simcc_x, simcc_y, center, scale, target_wh, split_ratio=2.0):
    """Argmax-based SimCC decode -> keypoints in original image coords."""
    x_locs = np.argmax(simcc_x[0], axis=1).astype(np.float32) / split_ratio
    y_locs = np.argmax(simcc_y[0], axis=1).astype(np.float32) / split_ratio
    sx_max = np.max(simcc_x[0], axis=1)
    sy_max = np.max(simcc_y[0], axis=1)
    scores = np.minimum(sx_max, sy_max)
    tw, th = target_wh
    cx, cy = center
    sx, sy = scale
    keypoints = np.stack([x_locs, y_locs], axis=-1)
    keypoints[:, 0] = (keypoints[:, 0] - tw / 2.0) * sx + cx
    keypoints[:, 1] = (keypoints[:, 1] - th / 2.0) * sy + cy
    return keypoints, scores


# ---- Wholebody replacement ------------------------------------------------

class WholebodyONNX:
    """Drop-in for controlnet_aux.dwpose.wholebody.Wholebody. Returns
    (keypoints, scores) in the same shape and OpenPose remap as the
    original mmpose-backed version."""

    def __init__(self, det_path, pose_path, device="cuda"):
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self.det = ort.InferenceSession(det_path, providers=providers)
        self.pose = ort.InferenceSession(pose_path, providers=providers)
        self.det_in = self.det.get_inputs()[0].name
        self.pose_in = self.pose.get_inputs()[0].name
        ps = self.pose.get_inputs()[0].shape  # [1,3,H,W]; some dims may be str (dynamic)
        h = ps[2] if isinstance(ps[2], int) else 288
        w = ps[3] if isinstance(ps[3], int) else 384
        self.pose_wh = (w, h)
        print(f"[dwpose-onnx] detector providers: {self.det.get_providers()}")
        print(f"[dwpose-onnx] pose model input WxH: {self.pose_wh}")

    def to(self, device):  # parity with controlnet_aux
        return self

    def __call__(self, oriImg):
        # Person detection
        canvas, scale = _yolox_letterbox(oriImg)
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        det_out = self.det.run(None, {self.det_in: x})[0]
        det_out = _yolox_postprocess(det_out)
        boxes_cxcywh = det_out[0, :, :4]
        obj = det_out[0, :, 4]
        cls = det_out[0, :, 5:]
        cls_idx = np.argmax(cls, axis=1)
        cls_top = cls[np.arange(len(cls_idx)), cls_idx]
        combined = obj * cls_top
        mask = (cls_idx == 0) & (combined > 0.5)
        if not mask.any():
            return np.zeros((0, 134, 2), dtype=np.float32), \
                   np.zeros((0, 134), dtype=np.float32)
        boxes_cxcywh = boxes_cxcywh[mask]
        combined = combined[mask]
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2.0
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2.0
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2.0
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2.0
        boxes = np.stack([x1, y1, x2, y2], axis=1) / scale
        keep = _nms(boxes, combined, iou_thresh=0.45)
        boxes = boxes[keep]

        # Pose estimation per-person
        all_kp, all_sc = [], []
        for bbox in boxes:
            cropped, center, crop_scale = _crop_warp(oriImg, bbox, self.pose_wh)
            xn = cropped.astype(np.float32)
            xn = (xn - DWPOSE_MEAN) / DWPOSE_STD
            xn = xn.transpose(2, 0, 1)[None]
            outs = self.pose.run(None, {self.pose_in: xn})
            simcc_x, simcc_y = outs[0], outs[1]
            kp, sc = _decode_simcc(simcc_x, simcc_y, center, crop_scale,
                                   self.pose_wh)
            all_kp.append(kp)
            all_sc.append(sc)
        kpts = np.stack(all_kp)  # [N,133,2]
        scs = np.stack(all_sc)   # [N,133]

        # Insert neck (between shoulders 5+6) and apply mmpose->openpose remap.
        # Same logic as controlnet_aux/dwpose/wholebody.py:Wholebody.__call__.
        neck = (kpts[:, 5] + kpts[:, 6]) / 2.0
        neck_sc = np.minimum(scs[:, 5], scs[:, 6])
        kpts = np.insert(kpts, 17, neck, axis=1)  # [N,134,2]
        scs = np.insert(scs, 17, neck_sc, axis=1)  # [N,134]
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        kp_copy = kpts.copy()
        sc_copy = scs.copy()
        kpts[:, openpose_idx] = kp_copy[:, mmpose_idx]
        scs[:, openpose_idx] = sc_copy[:, mmpose_idx]
        return kpts, scs


# ---- Top-level detector (public API) --------------------------------------

class DWposeDetectorONNX:
    """Drop-in for controlnet_aux.dwpose.DWposeDetector. Same __call__
    contract: takes a PIL Image (or array-like), returns a PIL Image of
    the rendered stick figure (or numpy if output_type != 'pil')."""

    def __init__(self, device="cuda", det_path=None, pose_path=None):
        if det_path is None:
            det_path = hf_hub_download("yzd-v/DWPose", "yolox_l.onnx")
        if pose_path is None:
            pose_path = hf_hub_download("yzd-v/DWPose", "dw-ll_ucoco_384.onnx")
        self.pose_estimation = WholebodyONNX(det_path, pose_path, device=device)

    def to(self, device):  # parity
        return self

    def __call__(self, input_image, detect_resolution=512, image_resolution=512,
                 output_type="pil", **kwargs):
        # Mirrors controlnet_aux/dwpose/__init__.py:DWposeDetector.__call__
        # but feeds our ONNX-backed Wholebody instead.
        input_image = cv2.cvtColor(np.array(input_image, dtype=np.uint8),
                                   cv2.COLOR_RGB2BGR)
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, _ = input_image.shape

        candidate, subset = self.pose_estimation(input_image)
        if candidate.shape[0] == 0:
            blank = np.zeros((H, W, 3), dtype=np.uint8)
            blank = cv2.resize(blank, (image_resolution, image_resolution),
                               interpolation=cv2.INTER_LINEAR)
            if output_type == "pil":
                blank = Image.fromarray(blank)
            return blank

        nums, _, locs = candidate.shape
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        body = candidate[:, :18].copy().reshape(nums * 18, locs)
        score = subset[:, :18]
        for i in range(len(score)):
            for j in range(len(score[i])):
                score[i][j] = int(18 * i + j) if score[i][j] > 0.3 else -1
        un_visible = subset < 0.3
        candidate[un_visible] = -1
        faces = candidate[:, 24:92]
        hands = candidate[:, 92:113]
        hands = np.vstack([hands, candidate[:, 113:]])

        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        canvas = dwpose_util.draw_bodypose(canvas, body, score)
        canvas = dwpose_util.draw_handpose(canvas, hands)
        canvas = dwpose_util.draw_facepose(canvas, faces)

        canvas = cv2.resize(canvas, (image_resolution, image_resolution),
                            interpolation=cv2.INTER_LINEAR)
        if output_type == "pil":
            canvas = Image.fromarray(canvas)
        return canvas
