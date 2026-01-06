"""
photoUp.py — Batch Upscale GUI (Tkinter)
Modes:
  - Real-ESRGAN (realesrgan-ncnn-vulkan)
  - SeedVR2 (CLI dans ./seedvr2)

Fonctionnalités:
  - Batch dossier, sous-dossiers optionnels
  - Upscale x2/x3/x4/x5
      * Real-ESRGAN: -s 2/3/4 ; pour x5 => x4 puis resize Lanczos 1.25
      * SeedVR2: --resolution calculée à partir de la LARGEUR (width * factor)
  - Post-process micro-détails (OpenCV) + protection visage (MediaPipe) optionnelle
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
    
import os
import sys
import time
import threading
import queue
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np

# MediaPipe (optionnel / best-effort)
try:
    import mediapipe as mp  # type: ignore
    _MP_OK = hasattr(mp, "solutions")
except Exception:
    mp = None
    _MP_OK = False


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Settings:
    # Common
    mode: str  # "esrgan" or "seedvr2"
    input_dir: Path
    output_dir: Path
    factor: int  # 2..5
    preserve_subdirs: bool
    overwrite: bool

    # Post-process
    microdetail_strength: float  # 0..1
    microdetail_sigma_s: float
    microdetail_sigma_r: float
    unsharp_amount: float
    unsharp_radius: float
    unsharp_threshold: int

    face_protection: bool
    face_padding: float

    # ESRGAN
    esrgan_exe: Path
    esrgan_model: str  # ex: "realesrgan-x4plus"
    esrgan_models_dir: Optional[Path]

    # SeedVR2
    seedvr2_root: Path  # ./seedvr2
    seedvr2_python: Path
    seedvr2_cli: Path  # inference_cli.py
    seedvr2_seed: int
    seedvr2_max_resolution: int  # 0 = none
    seedvr2_extra_args: str  # raw


def list_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return files


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Real-ESRGAN
# ---------------------------

def run_realesrgan(
    exe_path: Path,
    in_path: Path,
    out_path: Path,
    scale: int,
    model_name: str,
    models_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    """
    Calls realesrgan-ncnn-vulkan.
    """
    cmd = [
        str(exe_path),
        "-i", str(in_path),
        "-o", str(out_path),
        "-s", str(scale),
        "-n", model_name,
    ]
    if models_dir is not None:
        cmd += ["-m", str(models_dir)]

    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        ok = (r.returncode == 0) and out_path.exists()
        msg = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
        return ok, msg.strip()
    except Exception as e:
        return False, f"Erreur lancement Real-ESRGAN: {e}"


# ---------------------------
# SeedVR2
# ---------------------------

def _find_seedvr2_python(seedvr2_root: Path) -> Path:
    """
    Heuristique:
      - ./seedvr2/venv/Scripts/python.exe (Windows)
      - ./seedvr2/.venv/Scripts/python.exe (Windows)
      - ./seedvr2/venv/bin/python (Linux)
      - ./seedvr2/.venv/bin/python (Linux)
      - fallback: sys.executable
    """
    candidates = [
        seedvr2_root / "venv" / "Scripts" / "python.exe",
        seedvr2_root / ".venv" / "Scripts" / "python.exe",
        seedvr2_root / "venv" / "bin" / "python",
        seedvr2_root / ".venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path(sys.executable)


def _split_args(s: str) -> List[str]:
    """
    Split args in a simple way (supports quoted strings).
    """
    import shlex
    s = (s or "").strip()
    return shlex.split(s) if s else []


def run_seedvr2_cli(
    python_exe: Path,
    cli_path: Path,
    in_path: Path,
    out_dir: Path,
    resolution: int,
    seed: int = 42,
    max_resolution: int = 0,
    extra_args: str = "",
) -> Tuple[bool, str, Optional[Path]]:
    """
    Appelle SeedVR2 inference_cli.py pour UNE image.

    SeedVR2 écrit dans un dossier de sortie. On détecte ensuite le fichier créé
    (diff entre listing avant/après) et on le renvoie pour pouvoir le renommer.

    Returns: ok, log, produced_file_path
    """
    ensure_dir(out_dir)

    before = {p for p in out_dir.glob("*") if p.is_file()}

    cmd = [
        str(python_exe),
        str(cli_path),
        str(in_path),
        "--output",
        str(out_dir),
        "--resolution",
        str(resolution),
        "--seed",
        str(seed),
    ]
    if max_resolution and max_resolution > 0:
        cmd += ["--max_resolution", str(max_resolution)]

    cmd += _split_args(extra_args)

    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        msg = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
        ok = (r.returncode == 0)

        after = {p for p in out_dir.glob("*") if p.is_file()}
        created = list(after - before)

        produced: Optional[Path] = None
        if created:
            # pick the most recent
            created.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            produced = created[0]

        return ok, msg.strip(), produced
    except Exception as e:
        return False, f"Erreur lancement SeedVR2: {e}", None


# ---------------------------
# Face detection + postprocess
# ---------------------------

def detect_faces_mediapipe(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Returns face bounding boxes in pixel coords: (x1,y1,x2,y2)
    """
    if not _MP_OK:
        return []

    mp_fd = mp.solutions.face_detection  # type: ignore[attr-defined]
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxes: List[Tuple[int, int, int, int]] = []

    with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        res = fd.process(rgb)
        if res.detections:
            for det in res.detections:
                bb = det.location_data.relative_bounding_box
                x1 = int(bb.xmin * w)
                y1 = int(bb.ymin * h)
                x2 = int((bb.xmin + bb.width) * w)
                y2 = int((bb.ymin + bb.height) * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
    return boxes


def build_face_mask(shape_hw: Tuple[int, int], boxes: List[Tuple[int, int, int, int]], pad_ratio: float) -> np.ndarray:
    """
    Binary mask for faces (255 inside padded face boxes, 0 elsewhere), blurred for smooth blend.
    """
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    for (x1, y1, x2, y2) in boxes:
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        xa = max(0, x1 - pad_x)
        ya = max(0, y1 - pad_y)
        xb = min(w - 1, x2 + pad_x)
        yb = min(h - 1, y2 + pad_y)
        cv2.rectangle(mask, (xa, ya), (xb, yb), 255, thickness=-1)

    if mask.max() > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=8, sigmaY=8)
    return mask


def unsharp_mask(bgr: np.ndarray, amount: float, radius: float, threshold: int) -> np.ndarray:
    if amount <= 0:
        return bgr

    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=radius, sigmaY=radius)
    sharpened = cv2.addWeighted(bgr, 1.0 + amount, blurred, -amount, 0)

    if threshold > 0:
        diff = cv2.absdiff(bgr, blurred)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        mask = (diff_gray > threshold).astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
        mask_f = (mask.astype(np.float32) / 255.0)[..., None]
        out = (bgr.astype(np.float32) * (1 - mask_f) + sharpened.astype(np.float32) * mask_f)
        return np.clip(out, 0, 255).astype(np.uint8)

    return sharpened


def microdetail_enhance(
    bgr: np.ndarray,
    sigma_s: float,
    sigma_r: float,
    strength: float,
    unsharp_amt: float,
    unsharp_radius: float,
    unsharp_threshold: int,
) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return bgr

    de = cv2.detailEnhance(bgr, sigma_s=float(sigma_s), sigma_r=float(sigma_r))
    us = unsharp_mask(de, amount=float(unsharp_amt), radius=float(unsharp_radius), threshold=int(unsharp_threshold))

    out = (bgr.astype(np.float32) * (1 - strength) + us.astype(np.float32) * strength)
    return np.clip(out, 0, 255).astype(np.uint8)


def postprocess_preserve_faces(bgr: np.ndarray, st: Settings) -> np.ndarray:
    enhanced = microdetail_enhance(
        bgr,
        sigma_s=st.microdetail_sigma_s,
        sigma_r=st.microdetail_sigma_r,
        strength=st.microdetail_strength,
        unsharp_amt=st.unsharp_amount,
        unsharp_radius=st.unsharp_radius,
        unsharp_threshold=st.unsharp_threshold,
    )

    if not st.face_protection or not _MP_OK:
        return enhanced

    boxes = detect_faces_mediapipe(bgr)
    if not boxes:
        return enhanced

    mask = build_face_mask(bgr.shape[:2], boxes, st.face_padding)
    mask_f = (mask.astype(np.float32) / 255.0)[..., None]
    out = enhanced.astype(np.float32) * (1 - mask_f) + bgr.astype(np.float32) * mask_f
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------
# GUI App
# ---------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Batch Upscale — Real-ESRGAN / SeedVR2 + Micro-détails (Face-safe)")
        self.geometry("1320x860")
        self.minsize(1150, 720)
        self.resizable(True, True)
        try:
            self.state("zoomed")  # Windows
        except Exception:
            pass

        self.work_q: "queue.Queue[tuple]" = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        # Variables
        self.mode = tk.StringVar(value="esrgan")  # "esrgan" | "seedvr2"
        self.input_dir = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value="")
        self.factor = tk.IntVar(value=4)

        self.preserve_subdirs = tk.BooleanVar(value=True)
        self.overwrite = tk.BooleanVar(value=False)

        # Post-process defaults (photo-safe)
        self.microdetail_strength = tk.DoubleVar(value=0.30)
        self.microdetail_sigma_s = tk.DoubleVar(value=25.0)
        self.microdetail_sigma_r = tk.DoubleVar(value=0.20)
        self.unsharp_amount = tk.DoubleVar(value=0.55)
        self.unsharp_radius = tk.DoubleVar(value=1.2)
        self.unsharp_threshold = tk.IntVar(value=6)

        self.face_protection = tk.BooleanVar(value=True if _MP_OK else False)
        self.face_padding = tk.DoubleVar(value=0.30)

        # ESRGAN
        self.esrgan_exe = tk.StringVar(value=str(self._guess_esrgan_exe()))
        self.esrgan_model = tk.StringVar(value="realesrgan-x4plus")
        self.esrgan_models_dir = tk.StringVar(value="")  # optional override

        # SeedVR2
        self.seedvr2_root = tk.StringVar(value=str(self._guess_seedvr2_root()))
        self.seedvr2_python = tk.StringVar(value=str(_find_seedvr2_python(Path(self.seedvr2_root.get()))))
        self.seedvr2_cli = tk.StringVar(value=str(Path(self.seedvr2_root.get()) / "inference_cli.py"))
        self.seedvr2_seed = tk.IntVar(value=42)
        self.seedvr2_max_resolution = tk.IntVar(value=0)
        self.seedvr2_extra_args = tk.StringVar(value="")

        self._build_ui()
        self._tick()

        if not _MP_OK:
            self._log("Note: MediaPipe non disponible => protection visage désactivée.\n")

    def _guess_esrgan_exe(self) -> Path:
        here = Path(__file__).resolve().parent
        candidates = [
            here / "realesrgan-ncnn-vulkan.exe",
            here / "realesrgan-ncnn-vulkan",
            here / "realesrgan.exe",
        ]
        for c in candidates:
            if c.exists():
                return c
        return Path("")

    def _guess_seedvr2_root(self) -> Path:
        here = Path(__file__).resolve().parent
        return here / "seedvr2"

    def _browse_dir(self, var: tk.StringVar):
        p = filedialog.askdirectory()
        if p:
            var.set(p)

    def _browse_file(self, var: tk.StringVar, title: str = "Choisir un fichier"):
        p = filedialog.askopenfilename(title=title)
        if p:
            var.set(p)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=12, pady=10)

        # Input / Output
        r = 0
        ttk.Label(root, text="Dossier d'entrée (images)").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(root, textvariable=self.input_dir, width=80).grid(row=r, column=1, sticky="we", **pad)
        ttk.Button(root, text="Parcourir", command=lambda: self._browse_dir(self.input_dir)).grid(row=r, column=2, **pad)

        r += 1
        ttk.Label(root, text="Dossier de sortie").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(root, textvariable=self.output_dir, width=80).grid(row=r, column=1, sticky="we", **pad)
        ttk.Button(root, text="Parcourir", command=lambda: self._browse_dir(self.output_dir)).grid(row=r, column=2, **pad)

        # Mode selection
        r += 1
        mode_box = ttk.LabelFrame(root, text="Mode d'upscaling")
        mode_box.grid(row=r, column=0, columnspan=3, sticky="we", padx=10, pady=8)

        ttk.Radiobutton(mode_box, text="Real-ESRGAN (rapide)", variable=self.mode, value="esrgan",
                        command=self._refresh_mode_visibility).grid(row=0, column=0, padx=10, pady=6, sticky="w")
        ttk.Radiobutton(mode_box, text="SeedVR2 (qualité / diffusion)", variable=self.mode, value="seedvr2",
                        command=self._refresh_mode_visibility).grid(row=0, column=1, padx=10, pady=6, sticky="w")

        ttk.Label(mode_box, text="Facteur").grid(row=0, column=2, padx=10, pady=6, sticky="e")
        factor_combo = ttk.Combobox(mode_box, textvariable=self.factor, values=[2, 3, 4, 5], width=5, state="readonly")
        factor_combo.grid(row=0, column=3, padx=10, pady=6, sticky="w")

        ttk.Checkbutton(mode_box, text="Conserver sous-dossiers", variable=self.preserve_subdirs)\
            .grid(row=0, column=4, padx=10, pady=6, sticky="w")
        ttk.Checkbutton(mode_box, text="Écraser si existe", variable=self.overwrite)\
            .grid(row=0, column=5, padx=10, pady=6, sticky="w")

        # ESRGAN settings
        r += 1
        self.esrgan_frame = ttk.LabelFrame(root, text="Paramètres Real-ESRGAN")
        self.esrgan_frame.grid(row=r, column=0, columnspan=3, sticky="we", padx=10, pady=8)

        ttk.Label(self.esrgan_frame, text="Binaire realesrgan-ncnn-vulkan").grid(row=0, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(self.esrgan_frame, textvariable=self.esrgan_exe, width=80).grid(row=0, column=1, padx=10, pady=6, sticky="we")
        ttk.Button(self.esrgan_frame, text="Choisir", command=lambda: self._browse_file(self.esrgan_exe, "Choisir realesrgan-ncnn-vulkan"))\
            .grid(row=0, column=2, padx=10, pady=6)

        ttk.Label(self.esrgan_frame, text="Modèle").grid(row=1, column=0, padx=10, pady=6, sticky="w")
        model_combo = ttk.Combobox(
            self.esrgan_frame,
            textvariable=self.esrgan_model,
            values=["realesrgan-x4plus", "realesrgan-x4plus-anime", "realesr-animevideov3"],
            width=28,
            state="readonly",
        )
        model_combo.grid(row=1, column=1, padx=10, pady=6, sticky="w")
        ttk.Label(self.esrgan_frame, text="(photo: realesrgan-x4plus)").grid(row=1, column=2, padx=10, pady=6, sticky="w")

        ttk.Label(self.esrgan_frame, text="Dossier models (optionnel)").grid(row=2, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(self.esrgan_frame, textvariable=self.esrgan_models_dir, width=80).grid(row=2, column=1, padx=10, pady=6, sticky="we")
        ttk.Button(self.esrgan_frame, text="Parcourir", command=lambda: self._browse_dir(self.esrgan_models_dir))\
            .grid(row=2, column=2, padx=10, pady=6)

        # SeedVR2 settings
        r += 1
        self.seedvr2_frame = ttk.LabelFrame(root, text="Paramètres SeedVR2 (dans ./seedvr2)")
        self.seedvr2_frame.grid(row=r, column=0, columnspan=3, sticky="we", padx=10, pady=8)

        ttk.Label(self.seedvr2_frame, text="Dossier seedvr2").grid(row=0, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(self.seedvr2_frame, textvariable=self.seedvr2_root, width=80).grid(row=0, column=1, padx=10, pady=6, sticky="we")
        ttk.Button(self.seedvr2_frame, text="Parcourir", command=lambda: self._browse_dir(self.seedvr2_root))\
            .grid(row=0, column=2, padx=10, pady=6)

        ttk.Label(self.seedvr2_frame, text="Python SeedVR2").grid(row=1, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(self.seedvr2_frame, textvariable=self.seedvr2_python, width=80).grid(row=1, column=1, padx=10, pady=6, sticky="we")
        ttk.Button(self.seedvr2_frame, text="Choisir", command=lambda: self._browse_file(self.seedvr2_python, "Choisir python.exe SeedVR2"))\
            .grid(row=1, column=2, padx=10, pady=6)

        ttk.Label(self.seedvr2_frame, text="CLI (inference_cli.py)").grid(row=2, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(self.seedvr2_frame, textvariable=self.seedvr2_cli, width=80).grid(row=2, column=1, padx=10, pady=6, sticky="we")
        ttk.Button(self.seedvr2_frame, text="Choisir", command=lambda: self._browse_file(self.seedvr2_cli, "Choisir inference_cli.py"))\
            .grid(row=2, column=2, padx=10, pady=6)

        ttk.Label(self.seedvr2_frame, text="Seed").grid(row=3, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(self.seedvr2_frame, textvariable=self.seedvr2_seed, width=10).grid(row=3, column=1, padx=10, pady=6, sticky="w")

        ttk.Label(self.seedvr2_frame, text="Max resolution (0=illimité)").grid(row=3, column=1, padx=120, pady=6, sticky="w")
        ttk.Entry(self.seedvr2_frame, textvariable=self.seedvr2_max_resolution, width=10).grid(row=3, column=1, padx=320, pady=6, sticky="w")

        ttk.Label(self.seedvr2_frame, text="Extra args (optionnel)").grid(row=4, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(self.seedvr2_frame, textvariable=self.seedvr2_extra_args, width=80).grid(row=4, column=1, padx=10, pady=6, sticky="we")
        ttk.Label(self.seedvr2_frame, text='ex: --blocks_to_swap 12 --dit_offload_device cpu').grid(row=4, column=2, padx=10, pady=6, sticky="w")

        # Post-process
        r += 1
        pp = ttk.LabelFrame(root, text="Post-traitement (OpenCV) + Protection visage (MediaPipe)")
        pp.grid(row=r, column=0, columnspan=3, sticky="we", padx=10, pady=8)

        ttk.Label(pp, text="Force micro-détails (0..1)").grid(row=0, column=0, padx=10, pady=4, sticky="w")
        ttk.Scale(pp, variable=self.microdetail_strength, from_=0.0, to=1.0, orient="horizontal", length=260)\
            .grid(row=0, column=1, padx=10, pady=4, sticky="w")
        ttk.Label(pp, textvariable=self.microdetail_strength, width=6).grid(row=0, column=2, padx=10, pady=4, sticky="w")

        ttk.Label(pp, text="detailEnhance sigma_s").grid(row=1, column=0, padx=10, pady=4, sticky="w")
        ttk.Entry(pp, textvariable=self.microdetail_sigma_s, width=10).grid(row=1, column=1, padx=10, pady=4, sticky="w")
        ttk.Label(pp, text="detailEnhance sigma_r").grid(row=2, column=0, padx=10, pady=4, sticky="w")
        ttk.Entry(pp, textvariable=self.microdetail_sigma_r, width=10).grid(row=2, column=1, padx=10, pady=4, sticky="w")

        ttk.Label(pp, text="Unsharp amount").grid(row=1, column=2, padx=10, pady=4, sticky="w")
        ttk.Entry(pp, textvariable=self.unsharp_amount, width=10).grid(row=1, column=3, padx=10, pady=4, sticky="w")
        ttk.Label(pp, text="Unsharp radius").grid(row=2, column=2, padx=10, pady=4, sticky="w")
        ttk.Entry(pp, textvariable=self.unsharp_radius, width=10).grid(row=2, column=3, padx=10, pady=4, sticky="w")
        ttk.Label(pp, text="Unsharp threshold").grid(row=3, column=2, padx=10, pady=4, sticky="w")
        ttk.Entry(pp, textvariable=self.unsharp_threshold, width=10).grid(row=3, column=3, padx=10, pady=4, sticky="w")

        ttk.Checkbutton(pp, text="Protéger les visages (recommandé)", variable=self.face_protection)\
            .grid(row=4, column=0, columnspan=2, padx=10, pady=6, sticky="w")
        ttk.Label(pp, text="Padding visage").grid(row=4, column=2, padx=10, pady=6, sticky="w")
        ttk.Scale(pp, variable=self.face_padding, from_=0.0, to=0.8, orient="horizontal", length=260)\
            .grid(row=4, column=3, padx=10, pady=6, sticky="w")

        if not _MP_OK:
            ttk.Label(pp, text="(MediaPipe indisponible => protection visage OFF)").grid(row=5, column=0, columnspan=4, padx=10, pady=4, sticky="w")

        # Progress + controls
        r += 1
        barfrm = ttk.Frame(root)
        barfrm.grid(row=r, column=0, columnspan=3, sticky="we", padx=10, pady=10)

        self.progress = ttk.Progressbar(barfrm, orient="horizontal", mode="determinate", length=900)
        self.progress.grid(row=0, column=0, padx=8, pady=6, sticky="we")
        barfrm.columnconfigure(0, weight=1)

        self.status = tk.StringVar(value="Prêt.")
        ttk.Label(barfrm, textvariable=self.status).grid(row=1, column=0, padx=8, pady=4, sticky="w")

        btnfrm = ttk.Frame(root)
        btnfrm.grid(row=r + 1, column=0, columnspan=3, sticky="we", padx=10, pady=8)

        self.btn_start = ttk.Button(btnfrm, text="Lancer", command=self.start)
        self.btn_stop = ttk.Button(btnfrm, text="Stop", command=self.stop, state="disabled")
        self.btn_start.grid(row=0, column=0, padx=8, pady=6, sticky="w")
        self.btn_stop.grid(row=0, column=1, padx=8, pady=6, sticky="w")

        # Log
        r += 2
        self.log = tk.Text(root, height=14)
        self.log.grid(row=r, column=0, columnspan=3, padx=10, pady=8, sticky="nsew")
        root.rowconfigure(r, weight=1)
        root.columnconfigure(1, weight=1)

        self.log.insert("end", "Log...\n")
        self.log.configure(state="disabled")

        self._refresh_mode_visibility()

    def _refresh_mode_visibility(self):
        m = self.mode.get()
        if m == "esrgan":
            self.esrgan_frame.grid()
            self.seedvr2_frame.grid_remove()
        else:
            self.seedvr2_frame.grid()
            self.esrgan_frame.grid_remove()

    def _log(self, s: str):
        self.log.configure(state="normal")
        self.log.insert("end", s.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _validate(self) -> Optional[Settings]:
        in_dir = Path(self.input_dir.get().strip())
        out_dir = Path(self.output_dir.get().strip())
        if not in_dir.exists():
            messagebox.showerror("Erreur", "Dossier d'entrée invalide.")
            return None
        ensure_dir(out_dir)

        factor = int(self.factor.get())
        if factor not in (2, 3, 4, 5):
            messagebox.showerror("Erreur", "Facteur invalide (2..5).")
            return None

        mode = self.mode.get().strip().lower()
        if mode not in ("esrgan", "seedvr2"):
            messagebox.showerror("Erreur", "Mode invalide.")
            return None

        # Face protection only if mediapipe is available
        face_prot = bool(self.face_protection.get()) and _MP_OK

        # ESRGAN
        esrgan_exe = Path(self.esrgan_exe.get().strip())
        esrgan_model = self.esrgan_model.get().strip()
        models_override = self.esrgan_models_dir.get().strip()
        esrgan_models_dir = Path(models_override) if models_override else (esrgan_exe.parent / "models" if esrgan_exe else None)

        if mode == "esrgan":
            if not esrgan_exe.exists():
                messagebox.showerror("Erreur", "Binaire Real-ESRGAN introuvable.")
                return None
            if esrgan_models_dir is not None and not esrgan_models_dir.exists():
                messagebox.showerror("Erreur", "Dossier models Real-ESRGAN introuvable.")
                return None

        # SeedVR2
        seed_root = Path(self.seedvr2_root.get().strip())
        seed_py = Path(self.seedvr2_python.get().strip())
        seed_cli = Path(self.seedvr2_cli.get().strip())

        if mode == "seedvr2":
            if not seed_root.exists():
                messagebox.showerror("Erreur", "Dossier seedvr2 introuvable.")
                return None
            if not seed_py.exists():
                messagebox.showerror("Erreur", "Python SeedVR2 introuvable.")
                return None
            if not seed_cli.exists():
                messagebox.showerror("Erreur", "inference_cli.py introuvable.")
                return None

        return Settings(
            mode=mode,
            input_dir=in_dir,
            output_dir=out_dir,
            factor=factor,
            preserve_subdirs=bool(self.preserve_subdirs.get()),
            overwrite=bool(self.overwrite.get()),
            microdetail_strength=float(self.microdetail_strength.get()),
            microdetail_sigma_s=float(self.microdetail_sigma_s.get()),
            microdetail_sigma_r=float(self.microdetail_sigma_r.get()),
            unsharp_amount=float(self.unsharp_amount.get()),
            unsharp_radius=float(self.unsharp_radius.get()),
            unsharp_threshold=int(self.unsharp_threshold.get()),
            face_protection=face_prot,
            face_padding=float(self.face_padding.get()),
            esrgan_exe=esrgan_exe,
            esrgan_model=esrgan_model,
            esrgan_models_dir=esrgan_models_dir,
            seedvr2_root=seed_root,
            seedvr2_python=seed_py,
            seedvr2_cli=seed_cli,
            seedvr2_seed=int(self.seedvr2_seed.get()),
            seedvr2_max_resolution=int(self.seedvr2_max_resolution.get()),
            seedvr2_extra_args=self.seedvr2_extra_args.get(),
        )

    def start(self):
        st = self._validate()
        if not st:
            return

        self.stop_flag.clear()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        self.worker_thread = threading.Thread(target=self._worker, args=(st,), daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.stop_flag.set()
        self.status.set("Arrêt demandé…")
        self.btn_stop.configure(state="disabled")

    def _resolve_out_path(self, st: Settings, in_path: Path, rel: Path) -> Path:
        out_base = st.output_dir / (rel.parent if st.preserve_subdirs else Path())
        ensure_dir(out_base)

        suffix = f"_{st.mode}_x{st.factor}"
        return out_base / (in_path.stem + suffix + in_path.suffix)

    def _worker(self, st: Settings):
        images = list_images(st.input_dir)
        total = len(images)

        self.work_q.put(("init", total))
        if total == 0:
            self.work_q.put(("done", "Aucune image trouvée."))
            return

        for idx, in_path in enumerate(images, start=1):
            if self.stop_flag.is_set():
                self.work_q.put(("done", "Arrêté."))
                return

            rel = in_path.relative_to(st.input_dir)
            out_path = self._resolve_out_path(st, in_path, rel)

            if out_path.exists() and not st.overwrite:
                self.work_q.put(("progress", idx, f"Skip (existe): {rel}"))
                continue

            try:
                if st.mode == "esrgan":
                    ok = self._process_esrgan_one(st, in_path, out_path)
                else:
                    ok = self._process_seedvr2_one(st, in_path, out_path)

                if ok:
                    self.work_q.put(("progress", idx, f"OK: {rel} -> {out_path.name}"))
                else:
                    self.work_q.put(("progress", idx, f"ERREUR: {rel} (voir log)"))

            except Exception as e:
                self.work_q.put(("progress", idx, f"ERREUR EXCEPTION: {rel}\n{e}"))

        self.work_q.put(("done", "Terminé."))

    def _process_esrgan_one(self, st: Settings, in_path: Path, out_path: Path) -> bool:
        # ESRGAN supports s=2/3/4; x5 => x4 + Lanczos 1.25
        factor = st.factor
        esrgan_scale = factor if factor in (2, 3, 4) else 4

        tmp_path = out_path.with_name(out_path.stem + "__tmp" + out_path.suffix)
        ok, msg = run_realesrgan(
            st.esrgan_exe,
            in_path,
            tmp_path,
            scale=esrgan_scale,
            model_name=st.esrgan_model,
            models_dir=st.esrgan_models_dir,
        )
        if not ok:
            self.work_q.put(("log", f"--- ESRGAN ERROR {in_path.name} ---\n{msg}\n"))
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return False

        bgr = cv2.imread(str(tmp_path), cv2.IMREAD_COLOR)
        if bgr is None:
            self.work_q.put(("log", f"ERREUR lecture image upscalée: {tmp_path.name}"))
            tmp_path.unlink(missing_ok=True)
            return False

        # If user asked x5: resize 5/4
        if factor == 5:
            h, w = bgr.shape[:2]
            new_w = int(round(w * 5 / 4))
            new_h = int(round(h * 5 / 4))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Post-process
        bgr_pp = postprocess_preserve_faces(bgr, st)

        ok2 = cv2.imwrite(str(out_path), bgr_pp)
        tmp_path.unlink(missing_ok=True)
        return bool(ok2)

    def _process_seedvr2_one(self, st: Settings, in_path: Path, out_path: Path) -> bool:
        # SeedVR2: resolution computed from ORIGINAL WIDTH (as requested)
        src = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if src is None:
            self.work_q.put(("log", f"ERREUR lecture: {in_path}"))
            return False

        h0, w0 = src.shape[:2]
        resolution = int(w0 * st.factor)

        # Temp output folder per-image (avoid collisions)
        tmp_dir = out_path.parent / "__seedvr2_tmp"
        ensure_dir(tmp_dir)

        ok, msg, produced = run_seedvr2_cli(
            python_exe=st.seedvr2_python,
            cli_path=st.seedvr2_cli,
            in_path=in_path,
            out_dir=tmp_dir,
            resolution=resolution,
            seed=st.seedvr2_seed,
            max_resolution=st.seedvr2_max_resolution,
            extra_args=st.seedvr2_extra_args,
        )
        if not ok:
            self.work_q.put(("log", f"--- SeedVR2 ERROR {in_path.name} ---\n{msg}\n"))
            return False
        if produced is None or not produced.exists():
            self.work_q.put(("log", f"SeedVR2: fichier de sortie introuvable pour {in_path.name}\nLog:\n{msg}\n"))
            return False

        bgr = cv2.imread(str(produced), cv2.IMREAD_COLOR)
        if bgr is None:
            self.work_q.put(("log", f"SeedVR2: sortie illisible: {produced.name}"))
            return False

        # Post-process
        bgr_pp = postprocess_preserve_faces(bgr, st)

        ok2 = cv2.imwrite(str(out_path), bgr_pp)
        # Cleanup temp produced file (best-effort)
        try:
            produced.unlink(missing_ok=True)
        except Exception:
            pass

        return bool(ok2)

    def _tick(self):
        try:
            while True:
                item = self.work_q.get_nowait()
                kind = item[0]

                if kind == "init":
                    total = item[1]
                    self.progress["value"] = 0
                    self.progress["maximum"] = max(1, total)
                    self.status.set(f"{total} image(s) à traiter.")
                    self._log(f"--- Démarrage: {total} image(s) ---")

                elif kind == "progress":
                    idx, msg = item[1], item[2]
                    self.progress["value"] = idx
                    self.status.set(f"{idx}/{int(self.progress['maximum'])}  {msg[:90]}")
                    self._log(msg)

                elif kind == "log":
                    self._log(item[1])

                elif kind == "done":
                    msg = item[1]
                    self.status.set(msg)
                    self._log(f"--- {msg} ---")
                    self.btn_start.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
        except queue.Empty:
            pass

        self.after(120, self._tick)


def main():
    # Better DPI scaling on Windows
    if sys.platform.startswith("win"):
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
