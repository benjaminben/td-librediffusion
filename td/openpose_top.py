"""
TD Script TOP: webcam -> OpenPose stick figure for the librediffusion
TOP's input #2. Python Env must point at a conda env with controlnet_aux
+ torch + opencv installed. First cook downloads detector weights.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np


_detector = None
_detector_kind = None   # "dwpose" or "openpose"; reload when the menu changes
_env_path_added = None  # last successfully-applied env root; triggers reload on change
_dll_workaround_installed = False
_last_pose_rgba = None  # last successfully-rendered pose; reused on detector hiccups
_last_detect_time = 0.0  # monotonic seconds at last successful detection (FPS throttle)


def _install_dll_loading_workaround():
    """Patch os.add_dll_directory: AddDllDirectory raises WinError 206
    in TD's process even for short valid paths. Swallow it and fall
    back to extending PATH so torch's CUDA loader can still resolve."""
    global _dll_workaround_installed
    if _dll_workaround_installed or not hasattr(os, "add_dll_directory"):
        return
    _orig = os.add_dll_directory

    def _patched(path):
        try:
            return _orig(path)
        except (OSError, FileNotFoundError):
            p = str(path)
            current = os.environ.get("PATH", "")
            if p and p not in current:
                os.environ["PATH"] = p + os.pathsep + current
            return None  # torch ignores the cookie return value

    os.add_dll_directory = _patched
    _dll_workaround_installed = True


def onSetupParameters(scriptOp):
    page = scriptOp.appendCustomPage("OpenPose")
    page.appendFolder("Pythonenv", label="Python Env")
    page.appendInt("Outputsize", label="Output Size")[0].default = 512
    # DWPose: single ONNX pass, ~25-50 ms on a 4090, always renders body+hand+face.
    # OpenPose: classic controlnet_aux multi-scale pyramid, much slower but
    # included for A/B comparison against the controlnet's training distribution.
    detector_par = page.appendMenu("Detector", label="Detector")[0]
    detector_par.menuNames = ["DWPose", "OpenPose"]
    detector_par.menuLabels = ["DWPose (fast)", "OpenPose (classic)"]
    detector_par.default = "DWPose"
    page.appendToggle("Handsandface", label="Hands + Face (OpenPose only)")[0].default = True
    # Cap detection rate; throttled cooks re-emit the cached last pose. 0 = uncapped.
    fps_par = page.appendFloat("Maxdetectfps", label="Max Detect FPS")[0]
    fps_par.default = 15.0
    fps_par.normMin, fps_par.normMax = 0.0, 30.0


def _setup_python_env(scriptOp) -> bool:
    """Resolve the user-specified env's site-packages + DLL dirs and splice
    them into TD's Python on first cook (or when the param changes).

    Returns False if the param is unset or invalid -- caller should bail
    early so we don't try to import controlnet_aux from TD's bundled Python.
    """
    global _env_path_added, _detector

    env_root = scriptOp.par.Pythonenv.eval().strip()
    if not env_root:
        scriptOp.addWarning(
            "Python Env parameter is unset. Point it at the conda env that "
            "has controlnet_aux installed (e.g. C:\\Users\\Me\\miniconda3"
            "\\envs\\librediffusion)."
        )
        return False

    if env_root == _env_path_added:
        return True

    _detector = None  # env changed -> reload torch / model state from new env

    root = Path(env_root)
    if not root.exists():
        scriptOp.addError(f"Python env root does not exist: {env_root}")
        return False

    candidates = [root / "Lib" / "site-packages"]
    candidates.extend(root.glob("lib/python*/site-packages"))
    site_pkgs = next((c for c in candidates if c.exists()), None)
    if site_pkgs is None:
        scriptOp.addError(
            f"Could not find site-packages under {env_root}. "
            "Expected <env>\\Lib\\site-packages (conda Windows layout)."
        )
        return False

    site_str = str(site_pkgs)
    if site_str not in sys.path:
        sys.path.insert(0, site_str)

    # Pre-seed PATH so torch's CUDA loader has the DLL dirs even before
    # the add_dll_directory patch kicks in.
    extra_dirs = [
        root / "Lib" / "site-packages" / "torch" / "lib",
        root / "Library" / "bin",
    ]
    current_path = os.environ.get("PATH", "")
    for d in extra_dirs:
        if d.exists() and str(d) not in current_path:
            os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]

    _install_dll_loading_workaround()

    _env_path_added = env_root
    return True


def _detector_kind_param(scriptOp) -> str:
    """Read the Detector menu, falling back to 'dwpose' if the param hasn't
    been set up yet. TD's onSetupParameters only runs on TOP creation; an
    existing Script TOP from before this menu existed will lack the param
    until the user re-runs setup or recreates the TOP."""
    par = getattr(scriptOp.par, "Detector", None)
    if par is None:
        scriptOp.addWarning(
            "Detector menu missing -- right-click the Script TOP's OpenPose "
            "page tab -> Customize Page -> Setup Parameters (or delete the "
            "page so TD recreates it on next cook). Defaulting to DWPose."
        )
        return "dwpose"
    return par.eval().lower()


def _ensure_detector(scriptOp):
    """Load (or reload) the detector. Cached by kind; switching the Detector
    menu invalidates the cache so the other backend gets loaded fresh."""
    global _detector, _detector_kind
    kind = _detector_kind_param(scriptOp)
    if _detector is not None and _detector_kind == kind:
        return _detector
    _detector = None
    if kind == "dwpose":
        _detector = _load_dwpose(scriptOp)
    else:
        _detector = _load_openpose(scriptOp)
    if _detector is not None:
        _detector_kind = kind
    return _detector


def _load_dwpose(scriptOp):
    # Locate dwpose_onnx.py via TD's project.folder (the .toe directory).
    # Script TOP callbacks don't always populate __file__, so project.folder
    # is the reliable handle. Falls back to __file__ if project isn't in
    # globals (e.g. running this file outside TD for unit tests).
    script_dir = None
    try:
        script_dir = project.folder  # TD global
    except (NameError, AttributeError):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            pass
    if script_dir is None:
        scriptOp.addError(
            "Cannot locate dwpose_onnx.py: neither project.folder nor "
            "__file__ resolved. Place dwpose_onnx.py in the .toe folder."
        )
        return None
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    try:
        from dwpose_onnx import DWposeDetectorONNX
    except ImportError as e:
        scriptOp.addError(
            f"dwpose_onnx import failed: {e}. Verify dwpose_onnx.py exists "
            f"in {script_dir} alongside openpose_top.py."
        )
        return None
    det = DWposeDetectorONNX(device="cuda")
    print("[openpose] DWPose ONNX detector loaded (yolox_l + dw-ll_ucoco_384)")
    return det


def _load_openpose(scriptOp):
    try:
        import torch
        from controlnet_aux import OpenposeDetector
    except ImportError as e:
        scriptOp.addError(f"OpenposeDetector import failed: {e}")
        return None
    det = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    # By default controlnet_aux loads body/hand/face nets on CPU.
    if torch.cuda.is_available():
        if hasattr(det, "to"):
            det = det.to("cuda")
        else:
            for name in ("body_estimation", "hand_estimation", "face_estimation"):
                sub = getattr(det, name, None)
                if sub is not None and hasattr(sub, "model"):
                    sub.model.to("cuda")
        print("[openpose] classic OpenPose detector moved to CUDA")
    else:
        scriptOp.addWarning("torch.cuda.is_available() is False; OpenPose will run on CPU")
    return det


def _square_crop(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return arr[top:top + side, left:left + side]


def onCook(scriptOp):
    if not _setup_python_env(scriptOp):
        return

    detector = _ensure_detector(scriptOp)
    if detector is None:
        return

    # Detector throttle: skip detection if under min-interval since last run.
    global _last_detect_time, _last_pose_rgba
    fps_par = getattr(scriptOp.par, "Maxdetectfps", None)
    max_fps = float(fps_par.eval()) if fps_par is not None else 0.0
    if max_fps > 0.0 and _last_pose_rgba is not None \
            and (time.monotonic() - _last_detect_time) < (1.0 / max_fps):
        scriptOp.copyNumpyArray(_last_pose_rgba)
        return

    src = scriptOp.inputs[0] if len(scriptOp.inputs) else None
    out_size = int(scriptOp.par.Outputsize.eval())
    if src is None:
        # No input wired: emit black so the librediffusion TOP gets a stable
        # control image and can still skip inference cleanly upstream.
        scriptOp.copyNumpyArray(
            np.zeros((out_size, out_size, 4), dtype=np.uint8)
        )
        return

    # OpenposeDetector expects top-down PIL in RGB uint8. TD's input may be
    # uint8 [0,255] or float32 [0,1] depending on upstream TOP format.
    rgba = src.numpyArray(delayed=False)
    if rgba is None:
        return
    rgb_top_down = np.ascontiguousarray(rgba[::-1, :, :3])
    if rgb_top_down.dtype == np.uint8:
        rgb_u8 = rgb_top_down
    else:
        rgb_u8 = (np.clip(rgb_top_down, 0.0, 1.0) * 255.0).astype(np.uint8)

    rgb_u8 = _square_crop(rgb_u8)

    from PIL import Image
    pil = Image.fromarray(rgb_u8, mode="RGB").resize(
        (out_size, out_size), Image.LANCZOS
    )

    # controlnet_aux fails on degenerate hand / face crops (e.g. when wrist
    # keypoints collide), raising "height and width must be > 0" deep in
    # PIL. Catch transient failures and reuse the last good frame; only
    # emit black on the very first failure when we have no fallback yet.
    try:
        if _detector_kind == "dwpose":
            # DWPose always renders body + hand + face; no toggle.
            pose_pil = detector(pil, output_type="pil",
                                detect_resolution=out_size, image_resolution=out_size)
        else:
            pose_pil = detector(
                pil,
                hand_and_face=bool(scriptOp.par.Handsandface.eval()),
                detect_resolution=out_size,
                image_resolution=out_size,
            )
    except Exception as e:
        scriptOp.addWarning(f"detector failed on this frame: {e}")
        if _last_pose_rgba is not None and _last_pose_rgba.shape[0] == out_size:
            scriptOp.copyNumpyArray(_last_pose_rgba)
        else:
            scriptOp.copyNumpyArray(np.zeros((out_size, out_size, 4), dtype=np.uint8))
        return

    # Output uint8 RGBA, not float32. The librediffusion plugin's control
    # texture path mishandles 32-bit float input -- engine ignores
    # conditioning. 8-bit fixed (matching ab_pose.png) makes it work.
    pose = np.asarray(pose_pil)  # uint8 [0,255] from PIL
    pose = np.ascontiguousarray(pose[::-1])
    alpha = np.full((out_size, out_size, 1), 255, dtype=np.uint8)
    rgba_out = np.ascontiguousarray(np.concatenate([pose, alpha], axis=-1))
    _last_pose_rgba = rgba_out
    _last_detect_time = time.monotonic()
    scriptOp.copyNumpyArray(rgba_out)
