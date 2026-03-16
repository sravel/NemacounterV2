import sys
import os
import json
import csv
import logging
from pathlib import Path

log = logging.getLogger("LYRA.gui")

import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image as PILimage

# ── Force PyQt6 pour vispy / napari ───────────────────────────────────────────
os.environ["QT_API"] = "pyqt6"
import vispy
try:
    vispy.use_app("pyqt6")
except Exception:
    pass

# ── PyQt6 ─────────────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QCheckBox,
    QTabWidget, QProgressBar, QLineEdit, QComboBox,
    QFrame, QMessageBox, QScrollArea, QGroupBox,
    QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView,
    QTextEdit, QSplitter, QRadioButton,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QPixmap, QFont, QIcon, QColor

# ── Napari ────────────────────────────────────────────────────────────────────
import napari

# ── Modules LYRA ───────────────────────────────────────────────────────
import lyra.utils as utils
import lyra.common as common
# detection_workflow et edition_workflow sont importés localement dans RunWorker
# pour éviter de charger torch/ultralytics au démarrage de l'UI
from lyra.detection_engine import (
    LYRADetection,
    LYRASegmentation,
    add_masks_on_image,
    create_multicolored_masks_image,
)

# =============================================================================
#  CONSTANTES
# =============================================================================

IMG_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CSV_GLOB_SUFFIX = "_globinfo.csv"
CSV_SEG_SUFFIX  = "_segmentation_globinfo.csv"

# =============================================================================
#  WORKERS QThread
# =============================================================================

class ModelProbeWorker(QThread):
    """Charge un modèle YOLO en arrière-plan et retourne ses métadonnées."""
    ready = pyqtSignal(dict)

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            from ultralytics import YOLO
            m = YOLO(self.model_path)
            info = {
                "task":    m.task,
                "classes": dict(m.names),
                "nc":      len(m.names),
                "imgsz":   m.__dict__.get("overrides", {}).get("imgsz", "—"),
                "error":   None,
            }
        except Exception as e:
            info = {"task": "—", "classes": {}, "nc": 0, "imgsz": "—", "error": str(e)}
        self.ready.emit(info)


class ProjectScanWorker(QThread):
    """
    Scanne le répertoire de travail et retourne les stats complètes du projet.

    Résultat émis (dict) :
      images       : [chemin absolu, ...]  — images source (top-level)
      glob_csvs    : [...]  — *_globinfo.csv (détection YOLO)
      seg_csvs     : [...]  — *_segmentation_globinfo.csv (SAM)
      edit_csvs    : [...]  — *_edition_globinfo.csv (édition manuelle)
      json_exports : [...]  — *_roboflow_coco.json
      yolo_exports : [...]  — data.yaml (dataset YOLO train)
      img_info     : {basename: {yolo, sam, edition, exported}}
    """
    done = pyqtSignal(dict)

    def __init__(self, work_dir: str):
        super().__init__()
        self.work_dir = work_dir

    def run(self):
        result = {
            "images": [], "glob_csvs": [], "seg_csvs": [],
            "edit_csvs": [], "json_exports": [], "yolo_exports": [],
        }

        # ── Images top-level ─────────────────────────────────────────────
        for f in sorted(os.listdir(self.work_dir)):
            full = os.path.join(self.work_dir, f)
            if os.path.isfile(full) and os.path.splitext(f)[1].lower() in IMG_EXTENSIONS:
                result["images"].append(full)

        # ── Sous-dossiers résultats ───────────────────────────────────────
        for root, dirs, files in os.walk(self.work_dir):
            if root == self.work_dir:
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                continue
            dirs[:] = [d for d in dirs if d != "img"]
            for f in files:
                full = os.path.join(root, f)
                if f.endswith("_edition_globinfo.csv"):
                    result["edit_csvs"].append(full)
                elif f.endswith(CSV_SEG_SUFFIX):
                    result["seg_csvs"].append(full)
                elif f.endswith(CSV_GLOB_SUFFIX):
                    result["glob_csvs"].append(full)
                elif f.endswith("_roboflow_coco.json"):
                    result["json_exports"].append(full)
                elif f == "data.yaml":
                    result["yolo_exports"].append(full)

        # ── Agréger les infos par image ───────────────────────────────────
        img_info: dict[str, dict] = {}

        def _ensure(bn):
            if bn not in img_info:
                img_info[bn] = {
                    "yolo": None, "sam": None, "edition": None,
                    "exported": {"json": False, "yolo": False},
                    "full_path": "",
                }

        def _read_csv_counts(paths, key, extra_fn=None):
            for p in paths:
                try:
                    df = pd.read_csv(p, comment="#", usecols=["img_id", "object_type"])
                    real = df[df["object_type"].notna() & (df["object_type"] != "")]
                    for img_id, grp in real.groupby("img_id"):
                        bn = os.path.basename(str(img_id))
                        _ensure(bn)
                        entry = {"count": len(grp), "csv_path": p}
                        if extra_fn:
                            extra_fn(entry, grp)
                        img_info[bn][key] = entry
                except Exception:
                    pass

        def _yolo_extra(entry, grp):
            types = grp["object_type"].unique()
            entry["type"] = "mask" if "mask" in types else "box"

        _read_csv_counts(result["glob_csvs"],  "yolo",    _yolo_extra)
        _read_csv_counts(result["seg_csvs"],   "sam")
        _read_csv_counts(result["edit_csvs"],  "edition")

        # Diff édition vs référence
        for bn, info in img_info.items():
            if info["edition"] and (info["yolo"] or info["sam"]):
                ref = (info["sam"] or info["yolo"])["count"]
                info["edition"]["diff"] = info["edition"]["count"] - ref

        # ── Export per-CSV stem (précis, pas global) ──────────────────────
        # Construire un index stem → fichier d'export
        json_stems: set[str] = set()
        for jp in result["json_exports"]:
            stem = os.path.basename(jp).replace("_roboflow_coco.json", "")
            json_stems.add(stem)

        yolo_proj_dirs: set[str] = set()
        for yp in result["yolo_exports"]:
            yolo_proj_dirs.add(os.path.dirname(os.path.dirname(yp)))  # remonter au-dessus de labels/

        for bn, info in img_info.items():
            for key in ("yolo", "sam", "edition"):
                if info[key]:
                    csv_path = info[key].get("csv_path", "")
                    # Récupérer le stem du CSV qui contient cette image
                    csv_stem = os.path.basename(csv_path).replace("_globinfo.csv","")                                                           .replace("_segmentation_globinfo.csv","")                                                           .replace("_edition_globinfo.csv","")
                    if csv_stem in json_stems:
                        info["exported"]["json"] = True
                    csv_dir = os.path.dirname(csv_path)
                    if any(csv_dir.startswith(d) or d.startswith(csv_dir) for d in yolo_proj_dirs):
                        info["exported"]["yolo"] = True

        # Attacher le chemin complet de l'image
        for p in result["images"]:
            bn = os.path.basename(p)
            _ensure(bn)
            img_info[bn]["full_path"] = p

        result["img_info"] = img_info
        self.done.emit(result)


class RunWorker(QThread):
    """
    Lance detection_workflow, le workflow de segmentation ou d'édition.
    Utilise les callbacks natifs de detection_workflow pour la progression —
    plus de redirection stdout ni de duplication de logique.
    """
    progress = pyqtSignal(float)   # 0.0 → 1.0
    status   = pyqtSignal(str)     # texte affiché sous la barre
    log      = pyqtSignal(str)     # lignes dans la log box
    finished = pyqtSignal(bool, str)

    def __init__(self, mode: str, params: dict):
        super().__init__()
        self.mode   = mode
        self.params = params

    def run(self):
        try:
            if self.mode == "detection":
                self._run_detection()
            elif self.mode == "segmentation":
                self._run_segmentation()
            else:
                raise ValueError(f"Unknown run mode: {self.mode!r}")
            self.finished.emit(True, f"{self.mode.capitalize()} completed successfully.")
        except MemoryError:
            self.finished.emit(False, "Failed: insufficient memory.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e))

    # ── Détection / Segmentation YOLO ─────────────────────────────────────

    def _run_detection(self):
        """
        Appelle detection_workflow() avec les trois callbacks natifs :
          progress_callback → signal progress (float 0-1)
          status_callback   → signal status  (str)
          log_callback      → signal log     (str)
        Toute la logique de boucle, sauvegarde CSV, overlay est dans detection.py.
        """
        from lyra.detection_engine import detection_workflow
        detection_workflow(
            self.params,
            gui=True,
            progress_callback=self.progress.emit,
            status_callback=self.status.emit,
            log_callback=self.log.emit,
        )

    def _run_segmentation(self):
        from lyra.detection_engine import segmentation_workflow
        segmentation_workflow(
            self.params,
            progress_callback=self.progress.emit,
            status_callback=self.status.emit,
        )

    # _run_edition is no longer used — EditionWindow opens directly in main thread


class _MetricsWorker(QThread):
    """
    Background worker for the enhanced metrics report.
    Calls export_engine.export_metrics_report() and emits progress/status/finished.
    """
    progress = pyqtSignal(float)
    status   = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, globinfo_path: str, selected_groups: list,
                 image_paths: dict | None = None):
        super().__init__()
        self.path            = globinfo_path
        self.selected_groups = selected_groups
        self.image_paths     = image_paths

    def run(self):
        try:
            from lyra.export_engine import export_metrics_report
            result = export_metrics_report(
                self.path,
                selected_groups=self.selected_groups,
                image_paths=self.image_paths,
                progress_callback=self.progress.emit,
                status_callback=self.status.emit,
            )
            summary = os.path.basename(result["summary"])
            agg     = os.path.basename(result["aggregate"])
            self.finished.emit(True, f"Saved: {summary}  +  {agg}")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.finished.emit(False, str(e))


class ExportWorker(QThread):
    """Convertit un _globinfo.csv en JSON Roboflow COCO ou en dataset YOLO train."""
    progress = pyqtSignal(float)
    status   = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, globinfo_path: str, format: str = "roboflow",
                 output_dir: str = "",
                 train_pct: int = 70, val_pct: int = 15, test_pct: int = 15):
        super().__init__()
        self.path       = globinfo_path
        self.format     = format
        self.output_dir = output_dir
        self.train_pct  = train_pct
        self.val_pct    = val_pct
        self.test_pct   = test_pct

    def run(self):
        try:
            self._convert()
            label = "JSON" if self.format == "roboflow" else "YOLO dataset"
            self.finished.emit(True, f"{label} exported successfully.")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.finished.emit(False, str(e))

    def _convert(self):
        from lyra.export_engine import export_roboflow_coco, export_yolo_train
        if self.format == "yolo":
            export_yolo_train(
                self.path,
                self.output_dir,
                train_pct=self.train_pct,
                val_pct=self.val_pct,
                test_pct=self.test_pct,
                progress_callback=self.progress.emit,
                status_callback=self.status.emit,
            )
        else:
            export_roboflow_coco(
                self.path,
                progress_callback=self.progress.emit,
                status_callback=self.status.emit,
            )


# =============================================================================
#  WIDGETS UTILITAIRES
# =============================================================================

class PathRow(QWidget):
    """Bouton + label chemin sur une ligne."""
    def __init__(self, btn_text: str, placeholder: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.btn   = QPushButton(btn_text)
        self.btn.setFixedWidth(130)
        self.label = QLabel(placeholder)
        self.label.setObjectName("pathLabel")
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.btn)
        layout.addWidget(self.label)
        self.path = ""

    def set_path(self, path: str):
        self.path = path
        display = os.path.relpath(path) if path else ""
        self.label.setText(display or "(none)")
        self.label.setToolTip(path)


class LabeledSlider(QWidget):
    """Slider horizontal avec libellé de valeur."""
    valueChanged = pyqtSignal(float)

    def __init__(self, title: str, min_: int, max_: int, default: int,
                 scale: float = 0.01, fmt: str = "{:.2f}", parent=None):
        super().__init__(parent)
        self._scale = scale
        self._fmt   = fmt
        self._title = title
        v_layout = QVBoxLayout(self)
        v_layout.setContentsMargins(0, 0, 0, 4)
        self._lbl = QLabel(f"{title}: {fmt.format(default * scale)}")
        self._lbl.setObjectName("sliderLabel")
        self._sld = QSlider(Qt.Orientation.Horizontal)
        self._sld.setRange(min_, max_)
        self._sld.setValue(default)
        self._sld.valueChanged.connect(self._on_change)
        v_layout.addWidget(self._lbl)
        v_layout.addWidget(self._sld)

    def _on_change(self, v: int):
        self._lbl.setText(f"{self._title}: {self._fmt.format(v * self._scale)}")
        self.valueChanged.emit(v * self._scale)

    def value(self) -> float:
        return self._sld.value() * self._scale

    def setValue(self, v: float):
        self._sld.setValue(int(v / self._scale))


class RangeSlider(QWidget):
    """
    Slider à deux poignées sur une seule piste — style range selector.

    Valeurs :
      lo  = position de la poignée gauche  (0..100, défaut 70)  → % train
      hi  = position de la poignée droite  (0..100, défaut 85)  → % train+val

    Segments colorés :
      [0..lo]    bleu   → train
      [lo..hi]   vert   → val  (obligatoire, min 1 image si N > 0)
      [hi..100]  rouge  → test (optionnel, peut être 0)

    Signaux :
      changed(lo, hi)  émis à chaque déplacement
    """
    changed = pyqtSignal(int, int)   # lo, hi  (0-100)

    _TRACK_H = 6
    _HANDLE_R = 9   # rayon poignée
    _MIN_VAL  = 1   # val minimum = 1 % (arrondi à 1 image si N > 0)

    # Couleurs segments
    _COL_TRAIN = "#378ADD"
    _COL_VAL   = "#3B6D11"
    _COL_TEST  = "#D85A30"
    _COL_TRACK = "#45475a"
    _COL_HANDLE = "#ffffff"
    _COL_HANDLE_BORDER = "#888"

    def __init__(self, lo=70, hi=85, parent=None):
        super().__init__(parent)
        self._lo = max(0, min(99, lo))
        self._hi = max(self._lo + self._MIN_VAL, min(100, hi))
        self._drag: str | None = None   # "lo" | "hi" | None
        self.setMinimumHeight(36)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    # ── Accesseurs ────────────────────────────────────────────────────────────

    @property
    def lo(self) -> int: return self._lo

    @property
    def hi(self) -> int: return self._hi

    def set_values(self, lo: int, hi: int):
        lo = max(0, min(99, lo))
        hi = max(lo + self._MIN_VAL, min(100, hi))
        changed = (lo != self._lo or hi != self._hi)
        self._lo, self._hi = lo, hi
        self.update()
        if changed:
            self.changed.emit(self._lo, self._hi)

    # ── Géométrie ─────────────────────────────────────────────────────────────

    def _track_rect(self):
        """Retourne (x_start, x_end, y_center) de la piste."""
        m = self._HANDLE_R + 2
        w = self.width() - 2 * m
        y = self.height() // 2
        return m, m + w, y

    def _val_to_x(self, v: int) -> int:
        x0, x1, _ = self._track_rect()
        return round(x0 + (x1 - x0) * v / 100)

    def _x_to_val(self, x: int) -> int:
        x0, x1, _ = self._track_rect()
        v = round((x - x0) / max(1, x1 - x0) * 100)
        return max(0, min(100, v))

    # ── Rendu ─────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        x0, x1, yc = self._track_rect()
        h = self._TRACK_H
        r_track = h // 2

        x_lo = self._val_to_x(self._lo)
        x_hi = self._val_to_x(self._hi)

        # ── Piste de fond ─────────────────────────────────────────────────
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self._COL_TRACK))
        p.drawRoundedRect(x0, yc - h//2, x1 - x0, h, r_track, r_track)

        # ── Segment Train (bleu) ──────────────────────────────────────────
        if x_lo > x0:
            p.setBrush(QColor(self._COL_TRAIN))
            p.drawRoundedRect(x0, yc - h//2, x_lo - x0, h, r_track, r_track)

        # ── Segment Val (vert) ────────────────────────────────────────────
        if x_hi > x_lo:
            p.setBrush(QColor(self._COL_VAL))
            p.drawRect(x_lo, yc - h//2, x_hi - x_lo, h)

        # ── Segment Test (rouge) ──────────────────────────────────────────
        if x1 > x_hi:
            p.setBrush(QColor(self._COL_TEST))
            p.drawRoundedRect(x_hi, yc - h//2, x1 - x_hi, h, r_track, r_track)

        # ── Poignée Lo ────────────────────────────────────────────────────
        self._draw_handle(p, x_lo, yc)
        # ── Poignée Hi ────────────────────────────────────────────────────
        self._draw_handle(p, x_hi, yc)

        p.end()

    def _draw_handle(self, p, x, y):
        from PyQt6.QtGui import QColor, QPen
        r = self._HANDLE_R
        p.setPen(QPen(QColor(self._COL_HANDLE_BORDER), 1.5))
        p.setBrush(QColor(self._COL_HANDLE))
        p.drawEllipse(x - r, y - r, 2*r, 2*r)

    # ── Interactions ──────────────────────────────────────────────────────────

    def _hit(self, pos) -> str | None:
        x_lo = self._val_to_x(self._lo)
        x_hi = self._val_to_x(self._hi)
        yc   = self.height() // 2
        r    = self._HANDLE_R + 4
        dx_lo = abs(pos.x() - x_lo)
        dx_hi = abs(pos.x() - x_hi)
        dy    = abs(pos.y() - yc)
        if dy > r * 2:
            return None
        if dx_lo < dx_hi and dx_lo <= r:
            return "lo"
        if dx_hi <= dx_lo and dx_hi <= r:
            return "hi"
        # Zone de la piste → déplacer la poignée la plus proche
        if dx_lo <= r:
            return "lo"
        if dx_hi <= r:
            return "hi"
        return None

    def mousePressEvent(self, event):
        self._drag = self._hit(event.pos())
        if self._drag:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self._move_handle(event.pos().x())

    def mouseMoveEvent(self, event):
        if self._drag:
            self._move_handle(event.pos().x())
        else:
            hit = self._hit(event.pos())
            self.setCursor(
                Qt.CursorShape.SizeHorCursor if hit
                else Qt.CursorShape.ArrowCursor
            )

    def mouseReleaseEvent(self, event):
        self._drag = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def _move_handle(self, x: int):
        v = self._x_to_val(x)
        if self._drag == "lo":
            # Train peut aller de 0 à hi - MIN_VAL (val doit garder au moins MIN_VAL)
            new_lo = max(0, min(v, self._hi - self._MIN_VAL))
            if new_lo != self._lo:
                self._lo = new_lo
                self.update()
                self.changed.emit(self._lo, self._hi)
        elif self._drag == "hi":
            # Hi = train+val peut aller de lo+MIN_VAL à 100
            new_hi = max(self._lo + self._MIN_VAL, min(v, 100))
            if new_hi != self._hi:
                self._hi = new_hi
                self.update()
                self.changed.emit(self._lo, self._hi)


class InfoCard(QGroupBox):
    """Carte d'information clé/valeur."""
    def __init__(self, title: str, rows: list = None, parent=None):
        super().__init__(title, parent)
        self._grid = QGridLayout(self)
        self._grid.setColumnStretch(1, 1)
        self._widgets: dict = {}
        if rows:
            for k, v in rows:
                self.add_row(k, v)

    def add_row(self, key: str, value: str):
        row = self._grid.rowCount()
        lbl_k = QLabel(key + ":")
        lbl_k.setObjectName("cardKey")
        lbl_v = QLabel(value)
        lbl_v.setObjectName("cardVal")
        lbl_v.setWordWrap(True)
        self._grid.addWidget(lbl_k, row, 0, Qt.AlignmentFlag.AlignTop)
        self._grid.addWidget(lbl_v, row, 1, Qt.AlignmentFlag.AlignTop)
        self._widgets[key] = lbl_v

    def update_value(self, key: str, value: str):
        if key in self._widgets:
            self._widgets[key].setText(value)


def h_sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setObjectName("sep")
    return f


def section_lbl(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("sectionLabel")
    return lbl


# =============================================================================
#  FENÊTRE PRINCIPALE
# =============================================================================

class LYRAApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LYRA  ·  Laboratory YOLO Recognition & Analysis")
        self.resize(1280, 800)

        # ── État ──────────────────────────────────────────────────────────
        self._work_dir    : str  = ""
        self._model_path  : str  = ""
        self._model_info  : dict = {}
        self._scan_result : dict = {}

        # ── Workers ───────────────────────────────────────────────────────
        self._run_worker   = None
        self._probe_worker = None
        self._scan_worker  = None
        self._exp_worker   = None

        self._build_ui()
        self._apply_style()

    # =========================================================================
    #  CONSTRUCTION UI
    # =========================================================================

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_sidebar())
        outer.addWidget(self._build_tabs(), 1)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QFrame:
        sb = QFrame()
        sb.setObjectName("sidebar")
        sb.setFixedWidth(365)
        lay = QVBoxLayout(sb)
        lay.setContentsMargins(14, 18, 14, 18)
        lay.setSpacing(10)

        # Logo
        logo = QLabel()
        pix_path = os.path.join("conf", "logo.png")
        if os.path.exists(pix_path):
            pix = QPixmap(pix_path).scaled(
                300, 110,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo.setPixmap(pix)
        else:
            logo.setText("LYRA")
            logo.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(logo)
        lay.addWidget(h_sep())

        # ── Répertoire images (input) ─────────────────────────────────────
        lay.addWidget(section_lbl("INPUT IMAGES FOLDER"))
        self._wd_label = QLabel("(not selected)")
        self._wd_label.setObjectName("pathLabel")
        self._wd_label.setWordWrap(True)
        btn_wd = QPushButton("📂  Select Images Folder")
        btn_wd.clicked.connect(self._pick_work_dir)
        lay.addWidget(btn_wd)
        lay.addWidget(self._wd_label)

        # ── Nom du projet (auto = nom du dossier, éditable) ──────────────
        lay.addWidget(section_lbl("PROJECT NAME  (auto = folder name)"))
        self.proj_name_edit = QLineEdit()
        self.proj_name_edit.setPlaceholderText("auto-filled on folder selection")
        self.proj_name_edit.setToolTip(
            "Automatically set to the selected folder name.\n"
            "Changing it creates a new project subfolder."
        )
        lay.addWidget(self.proj_name_edit)

        lay.addWidget(h_sep())

        # ── Modèle ────────────────────────────────────────────────────────
        lay.addWidget(section_lbl("MODEL  (.pt)"))
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        # NB: _refresh_models() est appelé APRÈS la création de _model_card
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        btn_refresh = QPushButton("↻")
        btn_refresh.setFixedWidth(32)
        btn_refresh.setToolTip("Refresh model list")
        btn_refresh.clicked.connect(self._refresh_models)
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(btn_refresh)
        lay.addLayout(model_row)

        # Carte info modèle — doit être créée AVANT d'appeler _refresh_models
        self._model_card = InfoCard("Model info", [
            ("Task",     "—"),
            ("Classes",  "—"),
            ("Img size", "—"),
        ])
        lay.addWidget(self._model_card)

        # Maintenant on peut peupler le combo (probe déclenché en toute sécurité)
        self._refresh_models()

        lay.addWidget(h_sep())

        # ── Hardware ──────────────────────────────────────────────────────
        lay.addWidget(section_lbl("HARDWARE"))

        # Détection auto du meilleur device au démarrage
        self._best_device = self._detect_best_device()

        self._gpu_lbl = QLabel()
        self._gpu_lbl.setObjectName("gpuLabel")
        self._gpu_lbl.setWordWrap(True)
        self._update_gpu_label()
        lay.addWidget(self._gpu_lbl)

        nb_cpu = utils.compute_available_cpu()
        self.cpu_slider = LabeledSlider(
            "Max CPU", 1, nb_cpu, max(1, nb_cpu - 1),
            scale=1, fmt="{:.0f}"
        )
        lay.addWidget(self.cpu_slider)

        lay.addStretch()

        # ── Thème ─────────────────────────────────────────────────────────
        lay.addWidget(h_sep())
        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Theme:"))
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["Dark", "Light"])
        self._theme_combo.currentTextChanged.connect(self._on_theme_changed)
        theme_row.addWidget(self._theme_combo)
        lay.addLayout(theme_row)

        return sb

    # ── TABS ──────────────────────────────────────────────────────────────────

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_overview_tab(), "📁  Project")
        self.tabs.addTab(self._build_run_tab(),      "▶  Run")
        self.tabs.addTab(self._build_edition_tab(),  "✏  Edition")
        self.tabs.addTab(self._build_export_tab(),   "📤  Export")
        return self.tabs

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 0 — PROJECT OVERVIEW
    # ─────────────────────────────────────────────────────────────────────────

    def _build_overview_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(12)

        # ── Titre + bouton scan ───────────────────────────────────────────
        top = QHBoxLayout()
        self._ov_title = QLabel("Aucun projet chargé — sélectionnez un dossier pour commencer")
        self._ov_title.setObjectName("pageTitle")
        self._ov_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        top.addWidget(self._ov_title)
        top.addStretch()
        self._scan_btn = QPushButton("↻  Scan / Rafraîchir")
        self._scan_btn.setObjectName("actionBtn")
        self._scan_btn.setEnabled(False)
        self._scan_btn.clicked.connect(self._scan_project)
        top.addWidget(self._scan_btn)
        lay.addLayout(top)

        # ── Pipeline visuel (funnel) ──────────────────────────────────────
        # 5 étapes : Images → Détection → SAM → Édition → Exporté
        pipe_frame = QFrame()
        pipe_frame.setStyleSheet(
            "QFrame { background: transparent; }"
            "QLabel#pipeCount { font-size: 18px; font-weight: bold; }"
            "QLabel#pipeLabel { font-size: 10px; }"
        )
        pipe_lay = QHBoxLayout(pipe_frame)
        pipe_lay.setSpacing(0)
        pipe_lay.setContentsMargins(0, 0, 0, 0)

        self._pipe_steps: list[tuple[QLabel, QLabel]] = []
        steps = [
            ("Images",     "#89b4fa"),
            ("Détection",  "#a6e3a1"),
            ("SAM",        "#cba6f7"),
            ("Édition",    "#f5c2e7"),
            ("Exporté",    "#f9e2af"),
        ]
        for i, (label, color) in enumerate(steps):
            cell = QWidget()
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(8, 6, 8, 6)
            cl.setSpacing(2)
            cl.setAlignment(Qt.AlignmentFlag.AlignCenter)

            val_lbl = QLabel("—")
            val_lbl.setObjectName("pipeCount")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val_lbl.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")

            pct_lbl = QLabel("0 %")
            pct_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pct_lbl.setStyleSheet("color: #888; font-size: 10px;")

            txt_lbl = QLabel(label)
            txt_lbl.setObjectName("pipeLabel")
            txt_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            txt_lbl.setStyleSheet("color: #888; font-size: 10px; text-transform: uppercase;")

            cl.addWidget(val_lbl)
            cl.addWidget(pct_lbl)
            cl.addWidget(txt_lbl)

            cell.setStyleSheet(
                f"QWidget {{ border-left: 3px solid {color}; "
                f"background: transparent; border-radius: 0; }}"
            )
            pipe_lay.addWidget(cell, 1)
            self._pipe_steps.append((val_lbl, pct_lbl))

            if i < len(steps) - 1:
                arr = QLabel("›")
                arr.setStyleSheet("color: #555; font-size: 20px; padding: 0 4px;")
                arr.setAlignment(Qt.AlignmentFlag.AlignCenter)
                pipe_lay.addWidget(arr)

        lay.addWidget(pipe_frame)

        # ── Légende ───────────────────────────────────────────────────────
        legend = QHBoxLayout()
        legend.setSpacing(14)
        for dot, txt in [
            ("#6c7086", "⬜ En attente"),
            ("#89b4fa", "🔲 Détection"),
            ("#a6e3a1", "🎭 YOLO masks"),
            ("#cba6f7", "✏  SAM"),
            ("#f5c2e7", "📝 Édition"),
            ("#f9e2af", "📤 Exporté"),
        ]:
            lbl = QLabel(txt)
            lbl.setStyleSheet(f"color: {dot}; font-size: 11px;")
            legend.addWidget(lbl)
        legend.addStretch()
        lay.addLayout(legend)

        # ── Splitter principal (table images | panneau détail) ────────────
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # ── Bloc images ───────────────────────────────────────────────────
        img_box = QGroupBox("Images dans le dossier de travail")
        img_lay = QVBoxLayout(img_box)

        # Barre de filtre + actions
        flt_row = QHBoxLayout()
        flt_row.addWidget(QLabel("Filtrer :"))
        self._img_filter = QComboBox()
        self._img_filter.addItems([
            "Toutes", "En attente", "Détection", "SAM", "Édition", "Exportées"
        ])
        self._img_filter.setFixedWidth(140)
        self._img_filter.currentIndexChanged.connect(self._apply_img_filter)
        flt_row.addWidget(self._img_filter)
        flt_row.addStretch()

        self._ov_run_pending_btn = QPushButton("▶  Lancer sur sélection")
        self._ov_run_pending_btn.setObjectName("actionBtn")
        self._ov_run_pending_btn.setEnabled(False)
        self._ov_run_pending_btn.setToolTip("Bascule sur Run avec le dossier actuel")
        self._ov_run_pending_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        flt_row.addWidget(self._ov_run_pending_btn)

        self._ov_open_edit_btn = QPushButton("✏  Ouvrir dans Édition")
        self._ov_open_edit_btn.setObjectName("actionBtn")
        self._ov_open_edit_btn.setEnabled(False)
        self._ov_open_edit_btn.clicked.connect(self._ov_open_in_edition)
        flt_row.addWidget(self._ov_open_edit_btn)
        img_lay.addLayout(flt_row)

        # Splitter horizontal : table | thumbnail+info
        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Table images
        # Colonnes : Fichier | YOLO | SAM | Édition | Δ | Export | Statut | Ext
        self._img_table = QTableWidget(0, 8)
        self._img_table.setHorizontalHeaderLabels(
            ["Fichier", "YOLO", "SAM", "Édition", "Δ ann.", "Export", "Statut", "Ext"]
        )
        hh = self._img_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3, 4, 5, 6, 7):
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._img_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._img_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._img_table.setAlternatingRowColors(True)
        self._img_table.itemSelectionChanged.connect(self._on_img_selection_changed)
        self._img_table.cellDoubleClicked.connect(self._on_img_double_clicked)
        h_splitter.addWidget(self._img_table)

        # Panneau détail (thumbnail + toggle avant/après + annotations)
        detail_panel = QWidget()
        detail_panel.setMinimumWidth(240)
        detail_panel.setMaximumWidth(360)
        dp_lay = QVBoxLayout(detail_panel)
        dp_lay.setContentsMargins(8, 4, 4, 4)
        dp_lay.setSpacing(4)

        # ── Boutons de comparaison avant/après ────────────────────────────
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(2)
        self._thumb_mode_btns: dict[str, QPushButton] = {}
        for mode, label, tip in [
            ("raw",     "Original",  "Image brute sans annotation"),
            ("yolo",    "YOLO",      "Annotations YOLO (détection ou segmentation)"),
            ("sam",     "SAM",       "Masques SAM"),
            ("edition", "Édition",   "Annotations manuelles (édition)"),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(22)
            btn.setEnabled(False)
            btn.setStyleSheet(
                "QPushButton{font-size:10px;padding:0 4px;border-radius:3px;"
                "background:#313244;color:#888;border:1px solid #45475a;}"
                "QPushButton:checked{background:#1e66f5;color:#fff;border-color:#1e66f5;}"
                "QPushButton:enabled:!checked:hover{background:#45475a;color:#cdd6f4;}"
                "QPushButton:disabled{color:#555;background:#1e1e2e;}"
            )
            btn.clicked.connect(lambda checked, m=mode: self._set_thumb_mode(m))
            self._thumb_mode_btns[mode] = btn
            toggle_row.addWidget(btn)
        dp_lay.addLayout(toggle_row)

        self._thumb_lbl = QLabel()
        self._thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb_lbl.setMinimumHeight(170)
        self._thumb_lbl.setStyleSheet(
            "QLabel { background: #181825; border: 1px solid #313244; border-radius: 4px; }"
        )
        self._thumb_lbl.setText("Sélectionnez\nune image")
        dp_lay.addWidget(self._thumb_lbl)

        # Infos sur l'image sélectionnée
        self._detail_card = InfoCard("Détails", [
            ("Fichier",   "—"),
            ("Dimensions","—"),
            ("YOLO",      "—"),
            ("SAM",       "—"),
            ("Édition",   "—"),
            ("Δ ann.",    "—"),
            ("Export",    "—"),
            ("CSV source","—"),
            ("Modifié",   "—"),
        ])
        dp_lay.addWidget(self._detail_card)

        # Boutons d'action rapide pour l'image sélectionnée
        action_row = QHBoxLayout()
        action_row.setSpacing(4)
        self._ov_quick_edit_btn   = QPushButton("✏  Éditer")
        self._ov_quick_export_btn = QPushButton("📤  Exporter")
        for b in (self._ov_quick_edit_btn, self._ov_quick_export_btn):
            b.setFixedHeight(26)
            b.setEnabled(False)
            b.setObjectName("actionBtn")
            b.setStyleSheet(b.styleSheet() + "QPushButton{font-size:11px;}")
        self._ov_quick_edit_btn.clicked.connect(self._ov_open_in_edition)
        self._ov_quick_export_btn.clicked.connect(self._ov_quick_export)
        action_row.addWidget(self._ov_quick_edit_btn)
        action_row.addWidget(self._ov_quick_export_btn)
        dp_lay.addLayout(action_row)
        dp_lay.addStretch()
        h_splitter.addWidget(detail_panel)

        # État interne du panneau détail
        self._thumb_current_rd:   dict | None = None  # row_data courant
        self._thumb_current_mode: str         = "raw"

        h_splitter.setSizes([680, 260])
        img_lay.addWidget(h_splitter)
        main_splitter.addWidget(img_box)

        # ── Bloc résultats / CSVs ─────────────────────────────────────────
        res_box = QGroupBox("Résultats & exports")
        res_lay = QVBoxLayout(res_box)

        # Colonnes : Fichier | Type | Images | Objets | Classes | → JSON | → YOLO | Date | Chemin
        self._csv_table = QTableWidget(0, 9)
        self._csv_table.setHorizontalHeaderLabels(
            ["Fichier", "Type", "Images", "Objets", "Classes", "→ JSON", "→ YOLO", "Modifié", "Chemin"]
        )
        hh2 = self._csv_table.horizontalHeader()
        hh2.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3, 4, 5, 6, 7):
            hh2.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        hh2.setSectionResizeMode(8, QHeaderView.ResizeMode.Stretch)
        self._csv_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._csv_table.setAlternatingRowColors(True)
        self._csv_table.cellDoubleClicked.connect(self._on_csv_double_clicked)
        res_lay.addWidget(self._csv_table)
        main_splitter.addWidget(res_box)

        main_splitter.setSizes([460, 200])
        lay.addWidget(main_splitter, 1)

        # Buffer pour le filtre
        self._img_all_rows: list[dict] = []
        return tab


    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 1 — RUN (detection / segmentation)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_run_tab(self) -> QScrollArea:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(14)

        # ── Contexte du projet (résumé) ───────────────────────────────────
        self._run_ctx = InfoCard("Active project", [
            ("Directory",    "—"),
            ("Project name", "—"),
            ("Model",        "—"),
            ("Task",         "—"),
            ("Images found", "—"),
        ])
        lay.addWidget(self._run_ctx)

        # ── Mode : sélection du pipeline ─────────────────────────────────
        mode_box = QGroupBox("Pipeline")
        mode_lay = QVBoxLayout(mode_box)

        # Badge tâche du modèle
        badge_row = QHBoxLayout()
        badge_row.addWidget(QLabel("Model task:"))
        self._task_badge = QLabel("—")
        self._task_badge.setObjectName("taskBadge")
        self._task_badge.setFixedWidth(150)
        self._task_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge_row.addWidget(self._task_badge)
        badge_row.addStretch()
        mode_lay.addLayout(badge_row)

        # 2 boutons radio exclusifs
        radio_row = QHBoxLayout()
        self.mode_yolo = QCheckBox("YOLO  (auto detect / segment)")
        self.mode_sam  = QCheckBox("SAM Segmentation  (post-processing on prior CSV)")
        self.mode_yolo.setChecked(True)
        self.mode_yolo.toggled.connect(lambda on: self.mode_sam.setChecked(not on) if on else None)
        self.mode_sam.toggled.connect(lambda on: self.mode_yolo.setChecked(not on) if on else None)
        self.mode_yolo.toggled.connect(self._on_mode_toggled)
        self.mode_sam.toggled.connect(self._on_mode_toggled)
        radio_row.addWidget(self.mode_yolo)
        radio_row.addWidget(self.mode_sam)
        radio_row.addStretch()
        mode_lay.addLayout(radio_row)

        # Hint contextuel
        self._mode_hint = QLabel("")
        self._mode_hint.setObjectName("modeHint")
        self._mode_hint.setWordWrap(True)
        mode_lay.addWidget(self._mode_hint)

        lay.addWidget(mode_box)

        # ── Output ────────────────────────────────────────────────────────
        out_box = QGroupBox("Output directory  (results folder)")
        out_lay = QHBoxLayout(out_box)
        self._run_outdir_row = PathRow("Browse …", "(auto-set when images folder is selected)")
        self._run_outdir_row.btn.clicked.connect(
            lambda: self._pick_dir(self._run_outdir_row)
        )
        out_lay.addWidget(self._run_outdir_row)
        lay.addWidget(out_box)

        # ── Paramètres YOLO (communs detect + segment) ───────────────────
        self._det_box = QGroupBox("YOLO parameters")
        det_lay = QVBoxLayout(self._det_box)

        sliders = QHBoxLayout()
        self.conf_slider  = LabeledSlider("Confidence",    1, 100, 50)
        self.overl_slider = LabeledSlider("Overlap (IoU)", 1, 100, 30)
        sliders.addWidget(self.conf_slider)
        sliders.addWidget(self.overl_slider)
        det_lay.addLayout(sliders)

        # Overlay commun
        common_ov_row = QHBoxLayout()
        self.det_overlay_chk = QCheckBox("Save overlay images")
        self.det_overlay_chk.setChecked(True)
        self.det_bbox_chk   = QCheckBox("Bounding box")
        self.det_bbox_chk.setChecked(True)
        self.det_conf_chk   = QCheckBox("Confidence")
        self.det_labels_chk = QCheckBox("Labels")
        self.det_labels_chk.setChecked(True)
        for w in (self.det_overlay_chk, self.det_bbox_chk,
                  self.det_conf_chk, self.det_labels_chk):
            common_ov_row.addWidget(w)
        common_ov_row.addStretch()
        det_lay.addLayout(common_ov_row)

        # ── Options spécifiques segmentation (task = segment) ─────────────
        self._mask_opts_box = QGroupBox("Segmentation mask options  (segment model only)")
        mask_opts_lay = QVBoxLayout(self._mask_opts_box)

        mask_row = QHBoxLayout()
        self.det_mask_chk = QCheckBox("Show masks in overlay")
        self.det_mask_chk.setChecked(True)
        self.retina_chk   = QCheckBox("High-Res Masks (Retina)")
        self.fusion_chk   = QCheckBox("Fuse overlapping masks")
        self.fusion_chk.toggled.connect(lambda e: self._fusion_widget.setEnabled(e))
        for w in (self.det_mask_chk, self.retina_chk, self.fusion_chk):
            mask_row.addWidget(w)
        mask_row.addStretch()
        mask_opts_lay.addLayout(mask_row)

        self._fusion_widget = QWidget()
        fw_lay = QHBoxLayout(self._fusion_widget)
        fw_lay.setContentsMargins(0, 0, 0, 0)
        self.fuse_iou_sld  = LabeledSlider("Merge IoU",     1, 100, 5)
        self.phago_ioa_sld = LabeledSlider("Phagocyte IoA", 1, 100, 95)
        fw_lay.addWidget(self.fuse_iou_sld)
        fw_lay.addWidget(self.phago_ioa_sld)
        self._fusion_widget.setEnabled(False)
        mask_opts_lay.addWidget(self._fusion_widget)

        det_lay.addWidget(self._mask_opts_box)
        lay.addWidget(self._det_box)

        # ── Paramètres Segmentation ───────────────────────────────────────
        self._seg_box = QGroupBox("Segmentation parameters")
        seg_lay = QVBoxLayout(self._seg_box)
        seg_lay.addWidget(QLabel("Input CSV (output of a prior detection run):"))
        self._seg_csv_row = PathRow("Browse CSV …", "(auto-filled after Detection run)")
        self._seg_csv_row.btn.clicked.connect(
            lambda: self._pick_globinfo(self._seg_csv_row)
        )
        seg_lay.addWidget(self._seg_csv_row)
        self.seg_overlay_chk = QCheckBox("Save segmentation overlay images")
        self.seg_overlay_chk.setChecked(True)
        seg_lay.addWidget(self.seg_overlay_chk)
        self._seg_box.setVisible(False)
        lay.addWidget(self._seg_box)

        # ── Log + progress ────────────────────────────────────────────────
        self.run_progress = QProgressBar()
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(0)
        self.run_progress.setFixedHeight(12)
        lay.addWidget(self.run_progress)

        self.run_status_lbl = QLabel("")
        self.run_status_lbl.setObjectName("statusLabel")
        lay.addWidget(self.run_status_lbl)

        self._log_box = QTextEdit()
        self._log_box.setObjectName("logBox")
        self._log_box.setReadOnly(True)
        self._log_box.setFixedHeight(110)
        lay.addWidget(self._log_box)

        # ── Bouton START ──────────────────────────────────────────────────
        self.run_btn = QPushButton("▶  START")
        self.run_btn.setObjectName("runButton")
        self.run_btn.setFixedHeight(54)
        self.run_btn.clicked.connect(self._start_run)
        lay.addWidget(self.run_btn)

        lay.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 2 — EDITION
    # ─────────────────────────────────────────────────────────────────────────

    def _build_edition_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(14)

        # ── Sélection du CSV ──────────────────────────────────────────────────
        src_box = QGroupBox("Input CSV")
        src_lay = QVBoxLayout(src_box)

        # Combo rapide remplie par le scan
        self._edit_csv_combo = QComboBox()
        self._edit_csv_combo.addItem("(run a Scan / Refresh first)")
        self._edit_csv_combo.currentIndexChanged.connect(self._on_edit_csv_combo_changed)
        src_lay.addWidget(QLabel("Quick select from last scan:"))
        src_lay.addWidget(self._edit_csv_combo)

        # Ou browse manuel
        self._edit_csv_row = PathRow("Browse CSV …", "(or pick manually)")
        self._edit_csv_row.btn.clicked.connect(
            lambda: self._pick_globinfo_for_edition()
        )
        src_lay.addWidget(self._edit_csv_row)
        lay.addWidget(src_box)

        # ── Info-card sur le CSV sélectionné ──────────────────────────────────
        self._edit_info = InfoCard("Selected CSV — summary", [
            ("File",     "—"),
            ("Type",     "—"),
            ("Images",   "—"),
            ("Objects",  "—"),
        ])
        lay.addWidget(self._edit_info)

        # ── Paramètres de sortie ──────────────────────────────────────────────
        out_box = QGroupBox("Output")
        out_lay = QFormLayout(out_box)
        out_lay.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._edit_proj_edit = QLineEdit()
        self._edit_proj_edit.setPlaceholderText("auto — derived from CSV name")
        out_lay.addRow("Edition project name:", self._edit_proj_edit)

        self._edit_outdir_row = PathRow("Browse …", "(defaults to working directory)")
        self._edit_outdir_row.btn.clicked.connect(
            lambda: self._pick_dir(self._edit_outdir_row)
        )
        out_lay.addRow("Output directory:", self._edit_outdir_row)
        lay.addWidget(out_box)

        # ── Bouton de lancement ───────────────────────────────────────────────
        self.edit_run_btn = QPushButton("✏  Open Edition Window")
        self.edit_run_btn.setObjectName("runButton")
        self.edit_run_btn.setFixedHeight(46)
        self.edit_run_btn.clicked.connect(self._start_edition)

        self.edit_status_lbl = QLabel("")
        self.edit_status_lbl.setObjectName("statusLabel")
        lay.addWidget(self.edit_run_btn)
        lay.addWidget(self.edit_status_lbl)
        lay.addStretch()
        return tab

    def _pick_globinfo_for_edition(self):
        """Browse pour un CSV globinfo et l'injecte dans Edition."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select globinfo CSV", self._work_dir or "",
            "CSV Files (*_globinfo.csv *.csv)"
        )
        if path and os.path.exists(path):
            self._edit_csv_row.set_path(path)
            self._refresh_edit_info(path)
            # Auto-fill project name
            stem = os.path.splitext(os.path.basename(path))[0]
            proj = stem.replace("_globinfo", "").replace("_segmentation", "") + "_Edition"
            self._edit_proj_edit.setText(proj)

    def _on_edit_csv_combo_changed(self, _idx: int):
        """Quand l'utilisateur choisit un CSV dans le combo, met à jour l'info-card."""
        path = self._edit_csv_combo.currentData()
        if path and os.path.exists(path):
            self._edit_csv_row.set_path(path)
            self._refresh_edit_info(path)
            stem = os.path.splitext(os.path.basename(path))[0]
            proj = stem.replace("_globinfo", "").replace("_segmentation", "") + "_Edition"
            self._edit_proj_edit.setText(proj)

    def _refresh_edit_info(self, csv_path: str):
        """Relit le CSV et met à jour l'info-card Edition."""
        fname = os.path.basename(csv_path)
        is_seg = fname.endswith(CSV_SEG_SUFFIX)
        typ    = "SAM Segmentation" if is_seg else "YOLO Detection"
        n_img  = "?"
        n_obj  = "?"
        obj_kind = "?"
        try:
            df   = pd.read_csv(csv_path, comment="#", usecols=["img_id", "object_type"])
            real = df[df["object_type"].notna() & (df["object_type"] != "")]
            n_img  = str(df["img_id"].nunique())
            n_obj  = str(len(real))
            types  = real["object_type"].unique().tolist()
            obj_kind = " / ".join(str(t) for t in types) if types else "—"
        except Exception:
            pass
        self._edit_info.update_value("File",    fname)
        self._edit_info.update_value("Type",    f"{typ}  ({obj_kind})")
        self._edit_info.update_value("Images",  n_img)
        self._edit_info.update_value("Objects", n_obj)

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 3 — EXPORT
    # ─────────────────────────────────────────────────────────────────────────

    def _build_export_tab(self) -> QWidget:
        """
        Export tab — two independent sections:
          1. Annotation export  (Roboflow COCO JSON or YOLO train dataset)
          2. Metrics report     (configurable summary + aggregate CSVs)

        Both sections share the same CSV source selector at the top.
        """
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(16)

        # ═════════════════════════════════════════════════════════════════════
        #  CSV SOURCE  (shared by both sections)
        # ═════════════════════════════════════════════════════════════════════
        src_box = QGroupBox("Source CSV")
        src_lay = QVBoxLayout(src_box)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("From scan:"))
        self._exp_csv_combo = QComboBox()
        self._exp_csv_combo.addItem("(no scan yet)")
        self._exp_csv_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        src_row.addWidget(self._exp_csv_combo, 1)
        src_lay.addLayout(src_row)

        browse_row = QHBoxLayout()
        browse_row.addWidget(QLabel("Or browse:"))
        self._exp_csv_row = PathRow("Browse CSV …", "(no file selected)")
        self._exp_csv_row.btn.clicked.connect(
            lambda: self._pick_globinfo(self._exp_csv_row)
        )
        browse_row.addWidget(self._exp_csv_row, 1)
        src_lay.addLayout(browse_row)
        lay.addWidget(src_box)

        # ═════════════════════════════════════════════════════════════════════
        #  SECTION 1 — ANNOTATION EXPORT
        # ═════════════════════════════════════════════════════════════════════
        ann_box = QGroupBox("1 — Annotation export")
        ann_box.setStyleSheet(
            "QGroupBox { border-left: 3px solid #378ADD; border-radius: 0; "
            "margin-top: 12px; padding: 10px 8px 8px 12px; }"
            "QGroupBox::title { color: #378ADD; }"
        )
        ann_lay = QVBoxLayout(ann_box)

        # Format radio
        fmt_row = QHBoxLayout()
        self._exp_fmt_roboflow = QRadioButton(
            "Roboflow COCO JSON"
        )
        self._exp_fmt_roboflow.setToolTip(
            "Produces a single *_roboflow_coco.json file.\n"
            "Compatible with direct upload to Roboflow."
        )
        self._exp_fmt_yolo = QRadioButton(
            "YOLO train dataset"
        )
        self._exp_fmt_yolo.setToolTip(
            "Produces images/ + labels/ folders and data.yaml.\n"
            "Use directly with: yolo train data=data.yaml"
        )
        self._exp_fmt_roboflow.setChecked(True)
        fmt_row.addWidget(self._exp_fmt_roboflow)
        fmt_row.addWidget(self._exp_fmt_yolo)
        fmt_row.addStretch()
        ann_lay.addLayout(fmt_row)

        # YOLO output folder (hidden in Roboflow mode)
        self._exp_yolo_out = PathRow("Output folder …", "(required for YOLO dataset)")
        self._exp_yolo_out.btn.clicked.connect(lambda: self._pick_dir(self._exp_yolo_out))
        self._exp_yolo_out.setEnabled(False)
        self._exp_yolo_out.setVisible(False)
        ann_lay.addWidget(self._exp_yolo_out)

        # Train/val/test split (hidden in Roboflow mode)
        self._yolo_split_box = QGroupBox("Train / Val / Test split")
        sv = QVBoxLayout(self._yolo_split_box)
        self._yolo_split_box.setVisible(False)

        legend_row = QHBoxLayout()
        legend_row.setSpacing(16)
        for color, text in [
            (RangeSlider._COL_TRAIN, "■ Train"),
            (RangeSlider._COL_VAL,   "■ Val  (required, min 1%)"),
            (RangeSlider._COL_TEST,  "■ Test  (optional)"),
        ]:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color:{color};font-size:11px;font-weight:500;")
            legend_row.addWidget(lbl)
        legend_row.addStretch()
        sv.addLayout(legend_row)

        self._range_slider = RangeSlider(lo=70, hi=85)
        self._range_slider.setFixedHeight(38)
        sv.addWidget(self._range_slider)

        self._split_summary = QLabel("")
        self._split_summary.setObjectName("sliderLabel")
        self._split_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sv.addWidget(self._split_summary)

        self._split_n_lbl = QLabel("N images : —")  # hidden, used by callback
        self._split_n_lbl.setVisible(False)

        ann_lay.addWidget(self._yolo_split_box)

        # Export button
        self.exp_run_btn = QPushButton("▶  Export annotations")
        self.exp_run_btn.setObjectName("runButton")
        self.exp_run_btn.setFixedHeight(44)
        self.exp_run_btn.clicked.connect(self._start_export)
        ann_lay.addWidget(self.exp_run_btn)
        lay.addWidget(ann_box)

        # ── RangeSlider logic ─────────────────────────────────────────────
        def _update_split_labels(*_):
            lo, hi = self._range_slider.lo, self._range_slider.hi
            train_pct, val_pct, test_pct = lo, hi - lo, 100 - hi
            n_txt = self._split_n_lbl.text().split(":")[-1].strip()
            try:
                n = int(n_txt.split()[0])
            except Exception:
                n = 0
            n_train = round(n * train_pct / 100) if n else 0
            n_val   = round(n * val_pct   / 100) if n else 0
            n_test  = n - n_train - n_val         if n else 0
            def _fmt(pct, nb):
                return f"{pct}%" + (f"  ({nb} imgs)" if n else "")
            self._split_summary.setText(
                f"Train {_fmt(train_pct, n_train)}     "
                f"Val {_fmt(val_pct, n_val)}     "
                f"Test {_fmt(test_pct, n_test)}"
            )

        self._range_slider.changed.connect(_update_split_labels)
        _update_split_labels()
        self._update_split_labels = _update_split_labels

        def _on_fmt_toggle():
            yolo = self._exp_fmt_yolo.isChecked()
            self._exp_yolo_out.setEnabled(yolo)
            self._exp_yolo_out.setVisible(yolo)
            self._yolo_split_box.setVisible(yolo)

        self._exp_fmt_roboflow.toggled.connect(_on_fmt_toggle)
        self._exp_fmt_yolo.toggled.connect(_on_fmt_toggle)

        # ═════════════════════════════════════════════════════════════════════
        #  SECTION 2 — METRICS REPORT
        # ═════════════════════════════════════════════════════════════════════
        met_box = QGroupBox("2 — Metrics report")
        met_box.setStyleSheet(
            "QGroupBox { border-left: 3px solid #1D9E75; border-radius: 0; "
            "margin-top: 12px; padding: 10px 8px 8px 12px; }"
            "QGroupBox::title { color: #1D9E75; }"
        )
        met_lay = QVBoxLayout(met_box)

        desc = QLabel(
            "Computes per-image × per-class statistics and writes two CSV files:\n"
            "  *_summary.csv  (one row per image × class)\n"
            "  *_aggregate.csv  (one row per class across all images)"
        )
        desc.setObjectName("pathLabel")
        met_lay.addWidget(desc)

        try:
            from lyra.metric_selector_widget import MetricSelectorWidget
            self._metric_selector = MetricSelectorWidget(preset="standard")
            met_lay.addWidget(self._metric_selector)
        except ImportError:
            self._metric_selector = None
            met_lay.addWidget(QLabel(
                "(metric_selector_widget.py not found — "
                "all available groups will be computed)"
            ))

        self.exp_metrics_btn = QPushButton("📊  Generate metrics report")
        self.exp_metrics_btn.setObjectName("runButton")
        self.exp_metrics_btn.setFixedHeight(44)
        self.exp_metrics_btn.setToolTip(
            "Computes selected metric groups from the chosen CSV\n"
            "and saves *_summary.csv + *_aggregate.csv alongside it."
        )
        self.exp_metrics_btn.clicked.connect(self._start_metrics_report)
        met_lay.addWidget(self.exp_metrics_btn)
        lay.addWidget(met_box)

        # ═════════════════════════════════════════════════════════════════════
        #  SHARED PROGRESS
        # ═════════════════════════════════════════════════════════════════════
        self.exp_progress = QProgressBar()
        self.exp_progress.setRange(0, 100)
        lay.addWidget(self.exp_progress)
        self.exp_status_lbl = QLabel("")
        self.exp_status_lbl.setObjectName("statusLabel")
        lay.addWidget(self.exp_status_lbl)

        lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

    # =========================================================================
    #  HELPERS UI
    # =========================================================================

    @staticmethod
    def _make_stat_card(title: str, value: str, color: str) -> QFrame:
        card = QFrame()
        card.setObjectName("statCard")
        card.setStyleSheet(f"QFrame#statCard {{ border-top: 3px solid {color}; }}")
        vl = QVBoxLayout(card)
        val_lbl = QLabel(value)
        val_lbl.setObjectName("statValue")
        val_lbl.setFont(QFont("Arial", 26, QFont.Weight.Bold))
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ttl_lbl = QLabel(title)
        ttl_lbl.setObjectName("statTitle")
        ttl_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.addWidget(val_lbl)
        vl.addWidget(ttl_lbl)
        card._val_lbl = val_lbl
        return card

    def _pick_work_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Images Folder", ".")
        if not path:
            return
        self._work_dir = path
        folder_name    = os.path.basename(path)

        # ── Nom de projet = nom du dossier (auto) ─────────────────────────
        self.proj_name_edit.setText(folder_name)
        self.proj_name_edit.setToolTip(f"Auto-set from folder: {folder_name}")

        # ── Output = même dossier que les images ──────────────────────────
        # Structure finale : work_dir/project_id/project_id_globinfo.csv
        # Les résultats sont toujours dans le même répertoire que les images.
        self._run_outdir_row.set_path(path)

        rel = os.path.relpath(path)
        self._wd_label.setText(rel)
        self._wd_label.setToolTip(path)
        self._ov_title.setText(f"📁  {folder_name}")
        self._scan_btn.setEnabled(True)
        self._sync_ctx()
        self._scan_project()

    def _pick_dir(self, row: PathRow):
        path = QFileDialog.getExistingDirectory(self, "Select Directory", self._work_dir or ".")
        if path:
            row.set_path(path)

    def _pick_globinfo(self, row: PathRow):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select globinfo CSV",
            self._work_dir or ".",
            "CSV files (*_globinfo.csv);;All files (*.*)"
        )
        if path:
            row.set_path(path)

    def _refresh_models(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        os.makedirs("models", exist_ok=True)
        files = [f for f in os.listdir("models") if f.endswith(".pt")]
        if files:
            self.model_combo.addItems(files)
            self.model_combo.setEnabled(True)
        else:
            self.model_combo.addItem("No models found")
            self.model_combo.setEnabled(False)
        self.model_combo.blockSignals(False)
        self._on_model_changed(self.model_combo.currentText())

    def _on_mode_toggled(self):
        yolo = self.mode_yolo.isChecked()
        self._det_box.setVisible(yolo)
        self._seg_box.setVisible(not yolo)
        self._update_mode_hint()

    def _update_mode_hint(self):
        """Met à jour le badge de tâche et le texte descriptif selon modèle + mode."""
        if not hasattr(self, "_task_badge"):
            return
        task = self._model_info.get("task", "")
        yolo = self.mode_yolo.isChecked()

        if not yolo:
            self._task_badge.setText("SAM")
            self._task_badge.setStyleSheet("background:#7c3aed;color:#fff;border-radius:4px;padding:2px 6px;")
            self._mode_hint.setText(
                "SAM mode: will refine masks from an existing detection CSV using SAM."
            )
            return

        if task == "segment":
            self._task_badge.setText("YOLO  segment")
            self._task_badge.setStyleSheet("background:#059669;color:#fff;border-radius:4px;padding:2px 6px;")
            self._mode_hint.setText(
                "Segment model detected — YOLO will produce instance masks + bounding boxes. "
                "Mask options (Retina, Fusion) are enabled below."
            )
            if hasattr(self, "_mask_opts_box"):
                self._mask_opts_box.setEnabled(True)
                self._mask_opts_box.setTitle("Segmentation mask options  ✅ active")
        elif task == "detect":
            self._task_badge.setText("YOLO  detect")
            self._task_badge.setStyleSheet("background:#d97706;color:#fff;border-radius:4px;padding:2px 6px;")
            self._mode_hint.setText(
                "Detect model — YOLO will produce bounding boxes only (no masks). "
                "Mask options are disabled for this model."
            )
            if hasattr(self, "_mask_opts_box"):
                self._mask_opts_box.setEnabled(False)
                self._mask_opts_box.setTitle("Segmentation mask options  ⚠ not available (detect model)")
        else:
            self._task_badge.setText("Unknown")
            self._task_badge.setStyleSheet("background:#6b7280;color:#fff;border-radius:4px;padding:2px 6px;")
            self._mode_hint.setText("Load a model to see available options.")

    def _sync_ctx(self):
        wd    = self._work_dir or "—"
        pn    = self.proj_name_edit.text().strip() or "—"
        model = os.path.basename(self._model_path) if self._model_path else "—"
        task  = self._model_info.get("task", "—")
        n_img = str(len(self._scan_result.get("images", [])))

        self._run_ctx.update_value("Directory",    wd)
        self._run_ctx.update_value("Project name", pn)
        self._run_ctx.update_value("Model",        model)
        self._run_ctx.update_value("Task",         task)
        self._run_ctx.update_value("Images found", n_img)

    # =========================================================================
    #  MODEL PROBE
    # =========================================================================

    def _on_model_changed(self, name: str):
        if not name or name == "No models found":
            return
        path = os.path.join("models", name)
        self._model_path = path
        # Garde : _model_card peut ne pas encore exister pendant __init__
        if not hasattr(self, "_model_card"):
            return
        self._model_card.update_value("Task",     "⏳ loading…")
        self._model_card.update_value("Classes",  "—")
        self._model_card.update_value("Img size", "—")

        if self._probe_worker and self._probe_worker.isRunning():
            self._probe_worker.quit()
        self._probe_worker = ModelProbeWorker(path)
        self._probe_worker.ready.connect(self._on_model_ready)
        self._probe_worker.start()

    def _on_model_ready(self, info: dict):
        self._model_info = info
        if info.get("error"):
            self._model_card.update_value("Task", f"❌ {info['error'][:50]}")
            return
        task_icon = "🔲 detect" if info["task"] == "detect" else "🎭 segment"
        classes = list(info["classes"].values())
        cls_str = ", ".join(classes[:10])
        if len(classes) > 10:
            cls_str += f"  … +{len(classes)-10}"
        self._model_card.update_value("Task",     f"{task_icon}  ({info['nc']} classes)")
        self._model_card.update_value("Classes",  cls_str)
        self._model_card.update_value("Img size", str(info.get("imgsz", "—")))
        # Met à jour le badge de tâche + options masques dans Run tab
        self._update_mode_hint()
        self._sync_ctx()

    # =========================================================================
    #  SCAN DU PROJET
    # =========================================================================

    def _scan_project(self):
        if not self._work_dir:
            return
        self._scan_btn.setEnabled(False)
        self._scan_btn.setText("⏳  Scanning…")
        self._scan_worker = ProjectScanWorker(self._work_dir)
        self._scan_worker.done.connect(self._on_scan_done)
        self._scan_worker.start()

    def _on_scan_done(self, result: dict):
        self._scan_result = result
        self._scan_btn.setEnabled(True)
        self._scan_btn.setText("↻  Scan / Refresh")

        # ── Pipeline funnel ─────────────────────────────────────────────────
        img_info_all = result.get("img_info", {})
        n_imgs = len(result["images"])
        n_det  = sum(1 for v in img_info_all.values() if v.get("yolo"))
        n_sam  = sum(1 for v in img_info_all.values() if v.get("sam"))
        n_edit = sum(1 for v in img_info_all.values() if v.get("edition"))
        n_exp  = sum(1 for v in img_info_all.values()
                     if v.get("exported", {}).get("json") or v.get("exported", {}).get("yolo"))
        for _si, (_cnt, _tot) in enumerate([
            (n_imgs, n_imgs), (n_det, n_imgs), (n_sam, n_imgs),
            (n_edit, n_imgs), (n_exp, n_imgs),
        ]):
            if hasattr(self, "_pipe_steps") and _si < len(self._pipe_steps):
                _vl, _pl = self._pipe_steps[_si]
                _vl.setText(str(_cnt))
                _pl.setText(f"{round(100*_cnt/_tot) if _tot else 0} %")

        img_info = result.get("img_info", {})

        # ── Mise à jour split labels si disponibles ───────────────────────
        n_img = len(result["images"])
        if hasattr(self, "_split_n_lbl"):
            self._split_n_lbl.setText(f"N images : {n_img}")
        if hasattr(self, "_update_split_labels"):
            self._update_split_labels()

        # ── Table images ──────────────────────────────────────────────────
        self._img_all_rows = []
        self._img_table.setRowCount(0)

        for p in result["images"]:
            fname = os.path.basename(p)
            ext   = os.path.splitext(fname)[1].lower()
            info  = img_info.get(fname, {})
            yolo  = info.get("yolo")
            sam   = info.get("sam")
            edit  = info.get("edition")
            exp   = info.get("exported", {})

            # ── Déterminer le statut ──────────────────────────────────────
            if edit:
                status, st_color = "📝 Édité",       "#f5c2e7"
            elif sam:
                status, st_color = "✏  SAM",         "#cba6f7"
            elif yolo and yolo.get("type") == "mask":
                status, st_color = "🎭 YOLO masks",  "#a6e3a1"
            elif yolo:
                status, st_color = "🔲 YOLO boxes",  "#89b4fa"
            else:
                status, st_color = "⬜ En attente",   "#6c7086"

            # ── Colonne YOLO ──────────────────────────────────────────────
            if yolo:
                icon_y  = "🎭" if yolo.get("type") == "mask" else "🔲"
                lbl_y   = f"{yolo['count']} {icon_y}"
                col_y   = "#a6e3a1" if yolo.get("type") == "mask" else "#89b4fa"
            else:
                lbl_y, col_y = "—", "#6c7086"

            # ── Colonne SAM ───────────────────────────────────────────────
            if sam:
                lbl_s, col_s = f"{sam['count']} ✏", "#cba6f7"
            else:
                lbl_s, col_s = "—", "#6c7086"

            # ── Colonne Edition ───────────────────────────────────────────
            if edit:
                lbl_e, col_e = f"{edit['count']} 📝", "#f5c2e7"
            else:
                lbl_e, col_e = "—", "#6c7086"

            # ── Colonne Δ (différence annotations) ───────────────────────
            diff = edit.get("diff") if edit else None
            if diff is not None:
                sign   = "+" if diff > 0 else ""
                lbl_d  = f"{sign}{diff}"
                col_d  = "#a6e3a1" if diff > 0 else ("#f38ba8" if diff < 0 else "#6c7086")
            else:
                lbl_d, col_d = "—", "#6c7086"

            # ── Colonne Export ────────────────────────────────────────────
            exp_parts = []
            if exp.get("json"): exp_parts.append("JSON")
            if exp.get("yolo"): exp_parts.append("YOLO")
            lbl_exp = " · ".join(exp_parts) if exp_parts else "—"
            col_exp = "#f9e2af" if exp_parts else "#6c7086"

            # Stocker pour le filtre
            self._img_all_rows.append({
                "fname":       fname,
                "ext":         ext,
                "full_path":   p,                                           # chemin absolu image
                "yolo_csv":    yolo["csv_path"] if yolo else None,         # CSV YOLO source
                "sam_csv":     sam["csv_path"]  if sam  else None,         # CSV SAM source
                "edition_csv": edit["csv_path"] if edit else None,         # CSV édition source
                "lbl_y": lbl_y, "col_y": col_y,
                "lbl_s": lbl_s, "col_s": col_s,
                "lbl_e": lbl_e, "col_e": col_e,
                "lbl_d": lbl_d, "col_d": col_d,
                "lbl_exp": lbl_exp, "col_exp": col_exp,
                "status": status, "st_color": st_color,
                "has_yolo": bool(yolo), "has_sam": bool(sam),
                "has_edit": bool(edit), "has_exp": bool(exp_parts),
                "pending": not (yolo or sam or edit),
            })

        self._apply_img_filter()  # remplit la table selon filtre actif

        # ── Table résultats ───────────────────────────────────────────────
        # Précharger les classes par CSV
        def _csv_classes(p):
            try:
                df = pd.read_csv(p, comment="#", usecols=["name"])
                names = sorted(df["name"].dropna().unique().tolist())
                return ", ".join(names[:5]) + (f" +{len(names)-5}" if len(names) > 5 else "")
            except Exception:
                return "—"

        # Construire un index des exports disponibles (stem → chemin)
        json_stems = {
            os.path.basename(p).replace("_roboflow_coco.json", ""): p
            for p in result.get("json_exports", [])
        }
        yolo_dirs = {
            os.path.dirname(p): p
            for p in result.get("yolo_exports", [])
        }

        all_csvs = (
            [(p, "Détection YOLO")  for p in result["glob_csvs"]]
          + [(p, "Segmentation SAM") for p in result["seg_csvs"]]
          + [(p, "Édition manuelle") for p in result.get("edit_csvs", [])]
        )

        type_colors = {
            "Détection YOLO":    "#89b4fa",
            "Segmentation SAM":  "#cba6f7",
            "Édition manuelle":  "#f5c2e7",
        }

        self._csv_table.setRowCount(0)
        for p, typ in all_csvs:
            r = self._csv_table.rowCount()
            self._csv_table.insertRow(r)
            fname  = os.path.basename(p)
            rel    = os.path.relpath(os.path.dirname(p), self._work_dir)
            n_img_str  = "?"
            n_obj_str  = "?"
            try:
                df   = pd.read_csv(p, comment="#", usecols=["img_id", "object_type"])
                real = df[df["object_type"].notna() & (df["object_type"] != "")]
                n_img_str = str(df["img_id"].nunique())
                n_obj_str = str(len(real))
            except Exception:
                pass

            classes_str = _csv_classes(p)

            # JSON export correspondant ?
            stem        = fname.replace("_globinfo.csv","").replace("_segmentation_globinfo.csv","").replace("_edition_globinfo.csv","")
            has_json    = stem in json_stems
            # YOLO export dans le même dossier projet ?
            proj_dir    = os.path.dirname(p)
            has_yolo_ds = any(proj_dir in d or d.startswith(proj_dir) for d in yolo_dirs)

            item0 = QTableWidgetItem(fname)
            item0.setData(Qt.ItemDataRole.UserRole, p)
            self._csv_table.setItem(r, 0, item0)

            typ_item = QTableWidgetItem(typ)
            typ_item.setForeground(QColor(type_colors.get(typ, "#cdd6f4")))
            self._csv_table.setItem(r, 1, typ_item)

            for col, val, align in [
                (2, n_img_str,   True),
                (3, n_obj_str,   True),
                (4, classes_str, False),
            ]:
                it = QTableWidgetItem(val)
                if align:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 3 and n_obj_str.isdigit() and int(n_obj_str) > 0:
                    it.setForeground(QColor("#a6e3a1"))
                self._csv_table.setItem(r, col, it)

            # Export JSON
            json_it = QTableWidgetItem("✅ oui" if has_json else "—")
            json_it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            json_it.setForeground(QColor("#f9e2af" if has_json else "#6c7086"))
            self._csv_table.setItem(r, 5, json_it)

            # Export YOLO
            yolo_it = QTableWidgetItem("✅ oui" if has_yolo_ds else "—")
            yolo_it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            yolo_it.setForeground(QColor("#fab387" if has_yolo_ds else "#6c7086"))
            self._csv_table.setItem(r, 6, yolo_it)

            # Colonne 7 : date de modification
            try:
                import time
                mtime = os.path.getmtime(p)
                date_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
            except Exception:
                date_str = "—"
            date_it = QTableWidgetItem(date_str)
            date_it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            date_it.setForeground(QColor("#888"))
            self._csv_table.setItem(r, 7, date_it)

            # Colonne 8 : chemin relatif
            path_it = QTableWidgetItem(rel)
            path_it.setForeground(QColor("#888"))
            self._csv_table.setItem(r, 8, path_it)

        # ── Remplir les combos Edition + Export ───────────────────────────
        self._edit_csv_combo.clear()
        self._exp_csv_combo.clear()
        all_csv_paths = result["glob_csvs"] + result["seg_csvs"] + result.get("edit_csvs", [])
        if all_csv_paths:
            for p in all_csv_paths:
                label = os.path.basename(p)
                self._edit_csv_combo.addItem(label, p)
                self._exp_csv_combo.addItem(label, p)
            last = result["glob_csvs"][-1] if result["glob_csvs"] else all_csv_paths[-1]
            idx = self._edit_csv_combo.findData(last)
            if idx >= 0:
                self._edit_csv_combo.setCurrentIndex(idx)
        else:
            self._edit_csv_combo.addItem("(aucun CSV trouvé)")
            self._exp_csv_combo.addItem("(aucun CSV trouvé)")

        self._sync_ctx()

    def _apply_img_filter(self, *_):
        """Remplit la table images selon le filtre actif."""
        self._img_table.setRowCount(0)
        flt = self._img_filter.currentText() if hasattr(self, "_img_filter") else "Toutes"

        for row_data in self._img_all_rows:
            if flt == "En attente" and not row_data["pending"]:      continue
            if flt == "Détection"  and not row_data["has_yolo"]:     continue
            if flt == "SAM"        and not row_data["has_sam"]:      continue
            if flt == "Édition"    and not row_data["has_edit"]:     continue
            if flt == "Exportées"  and not row_data["has_exp"]:      continue

            r = self._img_table.rowCount()
            self._img_table.insertRow(r)

            # Stocker l'index réel dans _img_all_rows en UserRole
            # (crucial quand un filtre est actif : table row ≠ all_rows index)
            name_item = QTableWidgetItem(row_data["fname"])
            name_item.setData(Qt.ItemDataRole.UserRole, self._img_all_rows.index(row_data))
            self._img_table.setItem(r, 0, name_item)

            for col, lbl, col_hex in [
                (1, row_data["lbl_y"],   row_data["col_y"]),
                (2, row_data["lbl_s"],   row_data["col_s"]),
                (3, row_data["lbl_e"],   row_data["col_e"]),
                (4, row_data["lbl_d"],   row_data["col_d"]),
                (5, row_data["lbl_exp"], row_data["col_exp"]),
            ]:
                it = QTableWidgetItem(lbl)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                it.setForeground(QColor(col_hex))
                self._img_table.setItem(r, col, it)

            st_it = QTableWidgetItem(row_data["status"])
            st_it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            st_it.setForeground(QColor(row_data["st_color"]))
            self._img_table.setItem(r, 6, st_it)

            self._img_table.setItem(r, 7, QTableWidgetItem(row_data["ext"]))

    def _on_img_selection_changed(self):
        """Met à jour le panneau détail + thumbnail + boutons toggle."""
        rows   = self._img_table.selectionModel().selectedRows()
        has_sel = len(rows) > 0

        # Boutons d'action
        self._ov_open_edit_btn.setEnabled(has_sel)
        self._ov_run_pending_btn.setEnabled(has_sel)
        self._ov_quick_edit_btn.setEnabled(has_sel)
        self._ov_quick_export_btn.setEnabled(has_sel)

        if not has_sel:
            self._thumb_lbl.setText("Sélectionnez\nune image")
            self._thumb_lbl.setPixmap(QPixmap())
            for key in ("Fichier","Dimensions","YOLO","SAM","Édition","Δ ann.","Export","CSV source","Modifié"):
                self._detail_card.update_value(key, "—")
            for b in self._thumb_mode_btns.values():
                b.setEnabled(False); b.setChecked(False)
            self._thumb_current_rd = None
            return

        r = rows[0].row()
        item0 = self._img_table.item(r, 0)
        idx = item0.data(Qt.ItemDataRole.UserRole) if item0 else None
        if idx is None or idx >= len(self._img_all_rows):
            return
        rd = self._img_all_rows[idx]
        self._thumb_current_rd = rd

        # ── Activer les boutons toggle selon disponibilité ─────────────────
        avail = {
            "raw":     bool(rd.get("full_path")),
            "yolo":    bool(rd.get("yolo_csv")),
            "sam":     bool(rd.get("sam_csv")),
            "edition": bool(rd.get("edition_csv")),
        }
        # Sélection automatique du mode le plus avancé disponible
        best = next(
            (m for m in ("edition", "sam", "yolo", "raw") if avail.get(m)), "raw"
        )
        for mode, btn in self._thumb_mode_btns.items():
            btn.setEnabled(avail[mode])
            btn.setChecked(mode == best)
        self._thumb_current_mode = best

        # ── Rendu thumbnail ────────────────────────────────────────────────
        self._render_thumb(rd, best)

        # ── InfoCard ───────────────────────────────────────────────────────
        self._detail_card.update_value("Fichier",  rd["fname"])
        self._detail_card.update_value("YOLO",     rd["lbl_y"])
        self._detail_card.update_value("SAM",      rd["lbl_s"])
        self._detail_card.update_value("Édition",  rd["lbl_e"])
        self._detail_card.update_value("Δ ann.",   rd["lbl_d"])
        self._detail_card.update_value("Export",   rd["lbl_exp"])

        # CSV source le plus avancé + date
        csv_src = rd.get("edition_csv") or rd.get("sam_csv") or rd.get("yolo_csv")
        if csv_src:
            self._detail_card.update_value("CSV source", os.path.basename(csv_src))
            try:
                import time
                mt = os.path.getmtime(csv_src)
                self._detail_card.update_value("Modifié", time.strftime("%Y-%m-%d %H:%M", time.localtime(mt)))
            except Exception:
                self._detail_card.update_value("Modifié", "—")
        else:
            self._detail_card.update_value("CSV source", "—")
            self._detail_card.update_value("Modifié",   "—")

    def _set_thumb_mode(self, mode: str):
        """Appelé par un bouton toggle du panneau détail."""
        if self._thumb_current_rd is None:
            return
        self._thumb_current_mode = mode
        # Décocher tous les autres
        for m, btn in self._thumb_mode_btns.items():
            btn.setChecked(m == mode)
        self._render_thumb(self._thumb_current_rd, mode)

    def _render_thumb(self, rd: dict, mode: str):
        """
        Affiche la miniature dans le mode demandé :
          raw     → image brute
          yolo    → overlay annotations YOLO
          sam     → overlay masques SAM
          edition → overlay annotations d'édition
        Ajoute un bandeau coloré en bas pour indiquer le mode.
        """
        full_path = rd.get("full_path", "")
        if not full_path or not os.path.exists(full_path):
            self._thumb_lbl.setText("Image\nintrouvable")
            return
        try:
            bgr = cv2.imread(full_path)
            if bgr is None:
                self._thumb_lbl.setText("Illisible")
                return
            h, w = bgr.shape[:2]

            # Choisir le CSV source selon le mode
            csv_map = {
                "raw":     None,
                "yolo":    rd.get("yolo_csv"),
                "sam":     rd.get("sam_csv"),
                "edition": rd.get("edition_csv"),
            }
            csv_path = csv_map.get(mode)

            # Dessiner les annotations
            if csv_path:
                rd_tmp = dict(rd)  # copie pour ne pas muter
                rd_tmp["_override_csv"] = csv_path
                bgr = self._draw_overview_annotations(bgr, rd_tmp)

            # Bandeau coloré en bas indiquant le mode
            banner_colors = {
                "raw":     (80,  80,  80),
                "yolo":    (137, 180, 250),   # #89b4fa
                "sam":     (203, 166, 247),   # #cba6f7
                "edition": (245, 194, 231),   # #f5c2e7
            }
            color = banner_colors.get(mode, (80, 80, 80))
            bh = max(4, h // 40)
            bgr[-bh:, :] = color

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            thumb_w, thumb_h = 320, 210
            scale = min(thumb_w / w, thumb_h / h)
            nw, nh = int(w * scale), int(h * scale)
            rgb_small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            from PyQt6.QtGui import QImage
            img_qt = QImage(rgb_small.data, nw, nh, nw * 3, QImage.Format.Format_RGB888)
            self._thumb_lbl.setPixmap(
                QPixmap.fromImage(img_qt).scaled(
                    thumb_w, thumb_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self._detail_card.update_value("Dimensions", f"{w} × {h} px")
        except Exception as exc:
            self._thumb_lbl.setText("Erreur\nde prévisualisation")
            log.warning(f"Thumb render error: {exc}")

    def _draw_overview_annotations(self, bgr: np.ndarray, rd: dict) -> np.ndarray:
        """
        Dessine un aperçu rapide des annotations sur le thumbnail.
        Si rd contient "_override_csv", utilise ce CSV directement.
        Sinon priorité : édition > SAM > YOLO.
        Retourne une copie annotée de l'image.
        """
        img = bgr.copy()
        # Chemin CSV : override explicite (mode toggle) ou meilleur disponible
        if rd.get("_override_csv"):
            csv_path = rd["_override_csv"]
        else:
            csv_path = None
            for key in ("edition_csv", "sam_csv", "yolo_csv"):
                if rd.get(key):
                    csv_path = rd[key]
                    break
        if not csv_path or not os.path.exists(csv_path):
            return img

        try:
            df = pd.read_csv(csv_path, comment="#",
                             usecols=["img_id", "xmin", "ymin", "xmax", "ymax",
                                      "object_type", "contours"])
            fname = rd["fname"]
            sub = df[df["img_id"].apply(os.path.basename) == fname]
            h, w = img.shape[:2]
            overlay = img.copy()
            for _, row in sub.iterrows():
                ot = str(row.get("object_type", "box")).lower()
                if ot == "box":
                    x0, y0 = int(row["xmin"]), int(row["ymin"])
                    x1, y1 = int(row["xmax"]), int(row["ymax"])
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 200, 0), 1)
                elif ot in ("mask", "polygon"):
                    try:
                        clist = json.loads(row.get("contours", "[]"))
                        for pts in clist:
                            cnt = np.array(pts, dtype=np.int32)
                            cv2.polylines(overlay, [cnt], True, (150, 255, 150), 1)
                    except Exception:
                        pass
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        except Exception:
            pass
        return img

    def _on_img_double_clicked(self, row: int, _col: int):
        """Double-clic sur une image → bascule sur Édition avec le meilleur CSV."""
        item0 = self._img_table.item(row, 0)
        all_rows_idx = item0.data(Qt.ItemDataRole.UserRole) if item0 else None
        if all_rows_idx is not None and all_rows_idx < len(self._img_all_rows):
            self._open_image_in_edition(all_rows_idx)

    def _ov_open_in_edition(self):
        """Ouvre l'image sélectionnée dans l'onglet Édition avec le meilleur CSV."""
        rows = self._img_table.selectionModel().selectedRows()
        if rows:
            self._open_image_in_edition(rows[0].row())

    def _open_image_in_edition(self, row: int):
        """
        Sélectionne le meilleur CSV disponible pour la ligne donnée et bascule
        sur l'onglet Édition.
        Priorité : édition > SAM > YOLO (annotation la plus avancée).
        """
        if row >= len(self._img_all_rows):
            return
        rd = self._img_all_rows[row]

        # Trouver le meilleur CSV
        csv_path = rd.get("edition_csv") or rd.get("sam_csv") or rd.get("yolo_csv")

        if csv_path and os.path.exists(csv_path):
            # Injecter dans le combo Edition
            idx = self._edit_csv_combo.findData(csv_path)
            if idx >= 0:
                self._edit_csv_combo.setCurrentIndex(idx)
            else:
                self._edit_csv_combo.addItem(os.path.basename(csv_path), csv_path)
                self._edit_csv_combo.setCurrentIndex(self._edit_csv_combo.count() - 1)
            # Rafraîchir l'info-card Edition
            self._refresh_edit_info(csv_path)
            # Auto-fill project name
            stem = os.path.splitext(os.path.basename(csv_path))[0]
            proj = (stem.replace("_globinfo", "")
                        .replace("_segmentation", "")
                        .replace("_edition", "") + "_Edition")
            self._edit_proj_edit.setText(proj)
        else:
            # Pas de CSV — mode image vierge : juste basculer sur Edition
            # (l'utilisateur devra choisir manuellement)
            pass

        self.tabs.setCurrentIndex(2)

    def _ov_quick_export(self):
        """Export rapide depuis le panneau détail : injecte le meilleur CSV dans Export + bascule."""
        if self._thumb_current_rd is None:
            return
        rd = self._thumb_current_rd
        csv_path = rd.get("edition_csv") or rd.get("sam_csv") or rd.get("yolo_csv")
        if not csv_path or not os.path.exists(csv_path):
            QMessageBox.information(self, "Export",
                "Aucune annotation disponible pour cette image.\n"
                "Lancez d'abord une détection ou une session d'édition.")
            return
        # Injecter dans le combo Export
        idx = self._exp_csv_combo.findData(csv_path)
        if idx >= 0:
            self._exp_csv_combo.setCurrentIndex(idx)
        else:
            self._exp_csv_combo.addItem(os.path.basename(csv_path), csv_path)
            self._exp_csv_combo.setCurrentIndex(self._exp_csv_combo.count() - 1)
        self.tabs.setCurrentIndex(3)   # bascule sur Export

    def _on_csv_double_clicked(self, row: int, _col: int):
        """Double-clic sur un CSV → l'injecte dans Edition et Export, bascule sur Edition."""
        item = self._csv_table.item(row, 0)
        if not item:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path or not os.path.exists(path):
            return
        # Edition combo
        idx = self._edit_csv_combo.findData(path)
        if idx >= 0:
            self._edit_csv_combo.setCurrentIndex(idx)
        else:
            self._edit_csv_combo.addItem(os.path.basename(path), path)
            self._edit_csv_combo.setCurrentIndex(self._edit_csv_combo.count() - 1)
        # Export combo
        if self._exp_csv_combo.findData(path) == -1:
            self._exp_csv_combo.addItem(os.path.basename(path), path)
        self._refresh_edit_info(path)
        self.tabs.setCurrentIndex(2)


    # =========================================================================
    #  RUN
    # =========================================================================

    def _start_run(self):
        # Validation de base
        if not self._work_dir or not os.path.isdir(self._work_dir):
            QMessageBox.warning(self, "Error", "Please open a project folder first.")
            return

        proj_id = self.proj_name_edit.text().strip() or os.path.basename(self._work_dir)

        # Output = toujours le même dossier que les images.
        # Les résultats s'accumulent / se mettent à jour dans work_dir/proj_id/
        outdir = self._work_dir
        det_mode = self.mode_yolo.isChecked()

        if det_mode:
            # ── Validation détection ──────────────────────────────────────
            if not self._model_path or not os.path.exists(self._model_path):
                QMessageBox.warning(self, "Error", "Select a valid model (.pt).")
                return
            if not self._scan_result or not self._scan_result.get("images"):
                QMessageBox.warning(self, "No images",
                    "No images found in the working directory.\nRun a scan first.")
                return

            proj_outdir = os.path.join(outdir, proj_id)
            # Pas de dialogue "Overwrite" — le projet se met à jour sur place.
            # Le CSV est réécrit à chaque run (même comportement que detection_workflow).

            if self.retina_chk.isChecked():
                ans = QMessageBox.question(
                    self, "Memory Warning",
                    "Retina masks may require a lot of RAM.\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if ans != QMessageBox.StandardButton.Yes:
                    return

            params = {
                "input_directory":      self._work_dir,
                "output_directory":     outdir,
                "project_id":           proj_id,
                "conf_thresh":          self.conf_slider.value(),
                "overlap_thresh":       self.overl_slider.value(),
                "add_overlay":          int(self.det_overlay_chk.isChecked()),
                "model_path":           self._model_path,
                "gpu":                  self._gpu_flag(),
                "cpu":                  int(self.cpu_slider.value()),
                "show_bbox":            int(self.det_bbox_chk.isChecked()),
                "show_conf":            int(self.det_conf_chk.isChecked()),
                "show_mask":            int(self.det_mask_chk.isChecked()),
                "show_labels":          int(self.det_labels_chk.isChecked()),
                "use_retina_masks":     int(self.retina_chk.isChecked()),
                "use_fusion":           int(self.fusion_chk.isChecked()),
                "fuse_iou_thresh":      self.fuse_iou_sld.value(),
                "phagocyte_ioa_thresh": self.phago_ioa_sld.value(),
            }
            # Si le modèle est de type 'detect' (pas de masques),
            # on force les options masques à 0 pour éviter des erreurs silencieuses
            if self._model_info.get("task") == "detect":
                params["show_mask"]        = 0
                params["use_retina_masks"] = 0
                params["use_fusion"]       = 0
            mode = "detection"

        else:
            # ── Validation segmentation ───────────────────────────────────
            csv_path = self._seg_csv_row.path
            if not csv_path or not os.path.exists(csv_path):
                QMessageBox.warning(self, "Error",
                    "Select a *_globinfo.csv from a prior detection run.")
                return
            params = {
                "input_file":  csv_path,
                "add_overlay": int(self.seg_overlay_chk.isChecked()),
                "gpu":         self._gpu_flag(),
                "cpu":         int(self.cpu_slider.value()),
            }
            mode = "segmentation"

        # ── Libellé du bouton selon le mode ──────────────────────────────
        task = self._model_info.get("task", "") if det_mode else "sam"
        if task == "segment":
            run_label = "⏳  Running YOLO segmentation…"
        elif task == "detect":
            run_label = "⏳  Running YOLO detection…"
        else:
            run_label = "⏳  Running SAM segmentation…"

        # ── Lancement ─────────────────────────────────────────────────────
        self.run_btn.setEnabled(False)
        self.run_btn.setText(run_label)
        self.run_progress.setValue(0)
        self.run_status_lbl.setText("Starting…")
        self._log_box.clear()

        self._run_worker = RunWorker(mode, params)
        self._run_worker.progress.connect(lambda v: self.run_progress.setValue(int(v * 100)))
        self._run_worker.status.connect(self.run_status_lbl.setText)
        self._run_worker.log.connect(self._log_box.append)
        self._run_worker.finished.connect(self._on_run_finished)
        self._run_worker.start()

    def _on_run_finished(self, success: bool, msg: str):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶  START")
        if success:
            self.run_progress.setValue(100)
            self.run_status_lbl.setText("✅  " + msg)
            self._scan_project()          # re-scan + mise à jour tableau
            self._auto_fill_after_run()
            QMessageBox.information(self, "Success", msg)
            self.tabs.setCurrentIndex(0)  # retour sur Overview
        else:
            self.run_progress.setValue(0)
            self.run_status_lbl.setText("❌  " + msg)
            QMessageBox.critical(self, "Error", msg)

    def _auto_fill_after_run(self):
        """Injecte le CSV produit dans Edition et Segmentation."""
        outdir  = self._work_dir
        proj_id = self.proj_name_edit.text().strip() or os.path.basename(self._work_dir)
        candidate = os.path.join(outdir, proj_id, f"{proj_id}_globinfo.csv")
        if os.path.exists(candidate):
            self._edit_csv_row.set_path(candidate)
            self._seg_csv_row.set_path(candidate)
            if self._exp_csv_combo.findData(candidate) == -1:
                self._exp_csv_combo.addItem(os.path.basename(candidate), candidate)

    # =========================================================================
    #  EDITION
    # =========================================================================

    def _launch_napari(self):
        """
        Ouvre Napari avec la première image du projet et un calque Labels.

        TODO (avancé) :
          - Lire le CSV _globinfo pour reconstruire masques/bbox existants
          - Ajouter un calque Shapes pour les bounding-boxes
          - Connecter un callback de fermeture pour sauvegarder les corrections
        """
        img_dir  = self._work_dir
        csv_path = self._edit_csv_row.path

        viewer = napari.Viewer(title="LYRA — Manual Edition")

        if os.path.isdir(img_dir):
            imgs = sorted(
                f for f in os.listdir(img_dir)
                if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
            )
            if imgs:
                bgr = cv2.imread(os.path.join(img_dir, imgs[0]))
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    viewer.add_image(rgb, name=imgs[0])
                    viewer.add_labels(
                        np.zeros(rgb.shape[:2], dtype=np.int32),
                        name="Corrections",
                    )
                    # TODO: si csv_path → parser les contours et les projeter

    def _start_edition(self):
        # Résoudre le CSV : priorité au browse manuel, sinon le combo
        csv_path = self._edit_csv_row.path
        if not csv_path or not os.path.exists(csv_path):
            csv_path = self._edit_csv_combo.currentData()

        if not csv_path or not os.path.exists(csv_path):
            QMessageBox.warning(self, "Error",
                "Select a valid *_globinfo.csv — use the combo or browse manually.")
            return

        outdir = self._edit_outdir_row.path or self._work_dir
        if not outdir or not os.path.isdir(outdir):
            QMessageBox.warning(self, "Error",
                "Select a valid output directory (or open a project folder first).")
            return

        # Nom de projet : champ ou auto
        proj_id = self._edit_proj_edit.text().strip()
        if not proj_id:
            stem    = os.path.splitext(os.path.basename(csv_path))[0]
            proj_id = stem.replace("_globinfo", "").replace("_segmentation", "") + "_Edition"

        self.edit_status_lbl.setText(f"Opening edition window for  {os.path.basename(csv_path)} …")
        self.edit_run_btn.setEnabled(False)
        QApplication.processEvents()  # forcer l'affichage du message

        try:
            from lyra.edition_engine import EditionWindow
            win = EditionWindow()
            win.output_directory = os.path.join(outdir, proj_id)
            win.project_id       = proj_id
            win.display_size     = 1056
            win.load_project(csv_path)
            if self._work_dir:
                win.input_directory = Path(self._work_dir)
            win.showMaximized()

            # Garder une référence pour éviter le garbage collect
            self._edition_window = win

            # Connecter la fermeture pour réactiver le bouton
            win.closed.connect(self._on_edition_window_closed)

            self.edit_status_lbl.setText(
                f"✅  Edition window open — {proj_id}"
            )
        except Exception as exc:
            self.edit_run_btn.setEnabled(True)
            self.edit_status_lbl.setText(f"❌  {exc}")
            QMessageBox.critical(self, "Edition Error", str(exc))

    def _on_edition_window_closed(self):
        """Appelé quand l'EditionWindow est fermée."""
        self.edit_run_btn.setEnabled(True)
        self.edit_status_lbl.setText("Edition window closed.")
        self._edition_window = None
        # Rafraîchir le scan pour voir les nouveaux CSVs d'édition
        if self._work_dir:
            self._scan_project()

    # =========================================================================
    #  EXPORT
    # =========================================================================

    def _start_export(self):
        csv_path = self._exp_csv_row.path or self._exp_csv_combo.currentData()
        if not csv_path or not os.path.exists(csv_path):
            QMessageBox.warning(self, "Error", "Select a valid globinfo CSV.")
            return

        fmt = "yolo" if self._exp_fmt_yolo.isChecked() else "roboflow"

        if fmt == "yolo":
            out_dir = self._exp_yolo_out.path
            if not out_dir:
                QMessageBox.warning(self, "YOLO export",
                    "Sélectionnez un dossier de sortie pour le dataset YOLO.")
                return
            # Lire le RangeSlider : lo = frontière train|val, hi = frontière val|test
            train_pct = self._range_slider.lo
            val_pct   = self._range_slider.hi - self._range_slider.lo
            test_pct  = 100 - self._range_slider.hi
        else:
            out_dir   = ""
            train_pct = 100
            val_pct   = 0
            test_pct  = 0

        self.exp_progress.setValue(0)
        self.exp_status_lbl.setText("Démarrage…")
        self.exp_run_btn.setEnabled(False)

        self._exp_worker = ExportWorker(
            csv_path, format=fmt, output_dir=out_dir,
            train_pct=train_pct, val_pct=val_pct, test_pct=test_pct,
        )
        self._exp_worker.progress.connect(lambda v: self.exp_progress.setValue(int(v * 100)))
        self._exp_worker.status.connect(self.exp_status_lbl.setText)
        self._exp_worker.finished.connect(self._on_export_finished)
        self._exp_worker.start()

    def _on_export_finished(self, ok: bool, msg: str):
        self.exp_run_btn.setEnabled(True)
        if ok:
            self.exp_progress.setValue(100)
            self.exp_status_lbl.setText("✅  " + msg)
            QMessageBox.information(self, "Success", msg)
        else:
            self.exp_progress.setValue(0)
            self.exp_status_lbl.setText("❌  " + msg)
            QMessageBox.critical(self, "Error", msg)

    def _start_metrics_report(self):
        """
        Launches an enhanced metrics report from the selected globinfo CSV.
        Uses the metric groups selected in MetricSelectorWidget.
        Produces *_summary.csv and *_aggregate.csv.
        """
        csv_path = self._exp_csv_row.path or self._exp_csv_combo.currentData()
        if not csv_path or not os.path.exists(csv_path):
            QMessageBox.warning(self, "Error", "Select a valid globinfo CSV first.")
            return

        selected_groups = self._metric_selector.selected_groups()

        # Build image_paths dict if density metrics are requested
        image_paths: dict | None = None
        if "density" in selected_groups and self._work_dir:
            try:
                df_tmp = pd.read_csv(csv_path, comment="#", usecols=["img_id"])
                image_paths = {}
                for rel in df_tmp["img_id"].unique():
                    abs_p = os.path.join(self._work_dir, str(rel))
                    if os.path.exists(abs_p):
                        image_paths[str(rel)] = abs_p
            except Exception:
                pass

        self.exp_progress.setValue(0)
        self.exp_status_lbl.setText("Génération du rapport métriques…")
        self.exp_metrics_btn.setEnabled(False)
        self.exp_run_btn.setEnabled(False)

        self._metrics_worker = _MetricsWorker(
            csv_path, selected_groups, image_paths
        )
        self._metrics_worker.progress.connect(
            lambda v: self.exp_progress.setValue(int(v * 100))
        )
        self._metrics_worker.status.connect(self.exp_status_lbl.setText)
        self._metrics_worker.finished.connect(self._on_metrics_finished)
        self._metrics_worker.start()

    def _on_metrics_finished(self, ok: bool, msg: str):
        self.exp_metrics_btn.setEnabled(True)
        self.exp_run_btn.setEnabled(True)
        if ok:
            self.exp_progress.setValue(100)
            self.exp_status_lbl.setText("✅  " + msg)
            QMessageBox.information(self, "Metrics report", msg)
        else:
            self.exp_progress.setValue(0)
            self.exp_status_lbl.setText("❌  " + msg)
            QMessageBox.critical(self, "Error", msg)

    # =========================================================================
    #  APPARENCE
    # =========================================================================

    def _on_theme_changed(self, theme: str):
        self._apply_style(dark=(theme == "Dark"))

    def _apply_style(self, dark: bool = True):
        if dark:
            bg, bg2, bg3 = "#1e1e2e", "#181825", "#313244"
            bd, fg, fg2  = "#45475a", "#cdd6f4", "#6c7086"
            acc, acc2    = "#1e66f5", "#89b4fa"
            ok, alt      = "#a6e3a1", "#252535"
        else:
            bg, bg2, bg3 = "#f4f4f8", "#ffffff", "#e0e0e8"
            bd, fg, fg2  = "#ccccdd", "#1a1a2e", "#888899"
            acc, acc2    = "#1e66f5", "#1a56d6"
            ok, alt      = "#2d9c55", "#eaeaf2"

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {bg}; color: {fg};
                font-family: "Segoe UI", Arial, sans-serif; font-size: 13px;
            }}
            QFrame#sidebar {{ background: {bg2}; border-right: 1px solid {bd}; }}
            QLabel#sectionLabel {{
                color: {fg2}; font-size: 10px; font-weight: bold;
                letter-spacing: 1px; margin-top: 6px;
            }}
            QFrame#sep {{ background: {bd}; max-height: 1px; margin: 2px 0; }}

            QGroupBox {{
                border: 1px solid {bd}; border-radius: 7px;
                margin-top: 12px; padding: 10px 8px 8px 8px; font-weight: bold;
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 6px; color: {acc2}; }}

            QPushButton {{
                background: {bg3}; border: 1px solid {bd};
                border-radius: 5px; padding: 6px 14px; color: {fg};
            }}
            QPushButton:hover   {{ background: {bd}; }}
            QPushButton:pressed {{ background: {bg2}; }}
            QPushButton:disabled{{ color: {fg2}; }}
            QPushButton#runButton {{
                background: {acc}; border: none; color: #fff;
                font-weight: bold; font-size: 14px; border-radius: 7px;
            }}
            QPushButton#runButton:hover   {{ background: #2d79f7; }}
            QPushButton#runButton:pressed {{ background: #1550c0; }}
            QPushButton#runButton:disabled{{ background: {bg3}; color: {fg2}; }}
            QPushButton#actionBtn {{
                background: {bg3}; border: 1px solid {acc2};
                border-radius: 5px; color: {acc2}; padding: 5px 12px; font-weight: bold;
            }}
            QPushButton#actionBtn:hover {{ background: {bd}; }}

            QLineEdit, QComboBox, QTextEdit {{
                background: {bg2}; border: 1px solid {bd};
                border-radius: 4px; padding: 5px 8px; color: {fg};
            }}
            QLineEdit:focus, QComboBox:focus {{ border-color: {acc}; }}
            QComboBox::drop-down {{ border: none; width: 18px; }}

            QSlider::groove:horizontal {{
                height: 4px; background: {bg3}; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                width: 14px; height: 14px; margin: -5px 0;
                background: {acc2}; border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{ background: {acc}; border-radius: 2px; }}
            QLabel#sliderLabel {{ color: {fg2}; font-size: 11px; }}

            QTabWidget::pane {{ border: 1px solid {bd}; border-top: none; }}
            QTabBar::tab {{
                background: {bg2}; padding: 10px 22px;
                border-bottom: 2px solid transparent; color: {fg2};
            }}
            QTabBar::tab:selected {{
                background: {bg}; border-bottom: 2px solid {acc};
                color: {fg}; font-weight: bold;
            }}
            QTabBar::tab:hover {{ background: {bg3}; color: {fg}; }}

            QProgressBar {{
                border: 1px solid {bd}; border-radius: 4px;
                text-align: center; color: {fg};
            }}
            QProgressBar::chunk {{ background: {acc}; border-radius: 4px; }}

            QTableWidget {{
                background: {bg}; alternate-background-color: {alt};
                border: 1px solid {bd}; border-radius: 4px; gridline-color: {bd};
            }}
            QHeaderView::section {{
                background: {bg3}; color: {fg}; padding: 5px;
                border: none; border-bottom: 1px solid {bd}; font-weight: bold;
            }}
            QTableWidget::item:selected {{ background: {acc}; color: #fff; }}

            QLabel#pathLabel   {{ color: {fg2}; font-style: italic; font-size: 11px; }}
            QLabel#statusLabel {{ color: {ok}; font-size: 12px; }}
            QLabel#pageTitle   {{ color: {fg}; }}
            QLabel#cardKey     {{ color: {fg2}; font-size: 11px; min-width: 90px; }}
            QLabel#cardVal     {{ color: {fg}; font-size: 12px; }}
            QLabel#statValue   {{ color: {acc2}; }}
            QLabel#statTitle   {{ color: {fg2}; font-size: 11px; }}
            QLabel#taskBadge  {{
                background: #6b7280; color: #fff;
                border-radius: 4px; padding: 2px 6px;
                font-weight: bold; font-size: 12px;
            }}
            QLabel#modeHint   {{ color: {fg2}; font-size: 11px; font-style: italic; margin-top: 2px; }}

            QFrame#statCard {{
                background: {bg2}; border: 1px solid {bd};
                border-radius: 8px; padding: 10px;
            }}
            QTextEdit#logBox {{
                background: {bg2}; color: {fg2};
                font-family: "Consolas", monospace; font-size: 11px;
                border: 1px solid {bd}; border-radius: 4px;
            }}
            QCheckBox::indicator {{
                width: 16px; height: 16px;
                border: 1px solid {bd}; border-radius: 3px; background: {bg2};
            }}
            QCheckBox::indicator:checked {{ background: {acc}; border-color: {acc}; }}

            QScrollBar:vertical {{
                background: {bg}; width: 8px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {bd}; border-radius: 4px; min-height: 20px;
            }}
            QSplitter::handle {{ background: {bd}; }}

            QLabel#gpuLabel {{
                background: {bg3}; border: 1px solid {bd}; border-radius: 5px;
                padding: 6px 8px; font-size: 11px; color: {fg};
            }}
        """)

    # =========================================================================
    #  HARDWARE DETECTION
    # =========================================================================

    @staticmethod
    def _detect_best_device() -> str | list | int:
        """
        Détecte le meilleur device disponible.
        Returns:
            list[int]  si plusieurs GPU CUDA
            int (0)    si un seul GPU CUDA
            "mps"      si Apple Silicon
            "cpu"      sinon
        """
        try:
            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                if n > 1:
                    return list(range(n))
                return 0
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _update_gpu_label(self) -> None:
        """Met à jour le libellé hardware en sidebar selon _best_device."""
        d = self._best_device
        if d == "cpu":
            icon, name, sub = "🖥", "CPU only", "No GPU detected"
            color = "#f38ba8"
        elif d == "mps":
            icon, name, sub = "🍎", "Apple MPS", "Metal Performance Shaders"
            color = "#a6e3a1"
        elif d == 0:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            icon     = "⚡"
            name     = gpu_name
            sub      = f"{vram_gb:.1f} GB VRAM"
            color    = "#a6e3a1"
        else:  # liste de GPU
            n    = len(d)
            icon = "⚡⚡"
            name = f"{n}× GPU"
            names = " / ".join(torch.cuda.get_device_name(i) for i in d)
            sub  = names[:40] + ("…" if len(names) > 40 else "")
            color = "#a6e3a1"

        self._gpu_lbl.setText(
            f"<span style='font-size:16px'>{icon}</span>  "
            f"<b>{name}</b><br>"
            f"<span style='color:{color};font-size:10px'>{sub}</span>"
        )

    def _gpu_flag(self) -> int:
        """Retourne 1 si un GPU est disponible, 0 sinon (pour les params workflows)."""
        return 0 if self._best_device == "cpu" else 1


# =============================================================================
#  POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = LYRAApp()
    w.show()
    sys.exit(app.exec())