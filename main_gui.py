"""
NemaCounter GUI — PyQt6 + Napari  (v2 — mode Projet)
=====================================================
Nouvelle architecture centrée sur un répertoire de travail :

  1. Sidebar  → sélection projet + scan auto + info modèle live
  2. Onglet Overview   → tableau de bord du projet (images / résultats existants)
  3. Onglet Run        → paramètres + lancement unique (détection OU segmentation)
  4. Onglet Edition    → visualisation / correction Napari
  5. Onglet Export     → conversion Roboflow JSON

Les méthodes marquées # TODO sont à compléter pour les workflows internes.
"""

import sys
import os
import json
import csv

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
    QTextEdit, QSplitter,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QPixmap, QFont, QIcon, QColor

# ── Napari ────────────────────────────────────────────────────────────────────
import napari

# ── Modules NemaCounter ───────────────────────────────────────────────────────
import nemacounter.utils as utils
import nemacounter.common as common
# detection_workflow et edition_workflow sont importés localement dans RunWorker
# pour éviter de charger torch/ultralytics au démarrage de l'UI
from nemacounter.detection_engine import (
    NemaCounterDetection,
    NemaCounterSegmentation,
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
    """Scanne le répertoire de travail et retourne les stats du projet."""
    done = pyqtSignal(dict)

    def __init__(self, work_dir: str):
        super().__init__()
        self.work_dir = work_dir

    def run(self):
        result = {"images": [], "glob_csvs": [], "seg_csvs": []}

        # ── Images : top-level uniquement (pas de sous-dossiers) ─────────
        # Évite de ramasser les overlays produits dans MyProject/img/masks/
        for f in sorted(os.listdir(self.work_dir)):
            full = os.path.join(self.work_dir, f)
            if os.path.isfile(full):
                ext = os.path.splitext(f)[1].lower()
                if ext in IMG_EXTENSIONS:
                    result["images"].append(full)

        # ── CSVs : sous-dossiers uniquement (dossiers résultats) ─────────
        for root, dirs, files in os.walk(self.work_dir):
            if root == self.work_dir:          # ignorer la racine pour les CSV
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                continue
            dirs[:] = [d for d in dirs if d != "img"]
            for f in files:
                full = os.path.join(root, f)
                if f.endswith(CSV_SEG_SUFFIX):
                    result["seg_csvs"].append(full)
                elif f.endswith(CSV_GLOB_SUFFIX):
                    result["glob_csvs"].append(full)

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
            elif self.mode == "edition":
                self._run_edition()
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
        from nemacounter.detection_engine import detection_workflow
        detection_workflow(
            self.params,
            gui=True,
            progress_callback=self.progress.emit,
            status_callback=self.status.emit,
            log_callback=self.log.emit,
        )

    def _run_segmentation(self):
        from nemacounter.detection_engine import segmentation_workflow
        segmentation_workflow(
            self.params,
            progress_callback=self.progress.emit,
            status_callback=self.status.emit,
        )

    # _run_edition is no longer used — EditionWindow opens directly in main thread


class ExportWorker(QThread):
    """Convertit un _globinfo.csv en JSON COCO Roboflow."""
    progress = pyqtSignal(float)
    status   = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, globinfo_path: str):
        super().__init__()
        self.path = globinfo_path

    def run(self):
        try:
            self._convert()
            self.finished.emit(True, "JSON exported successfully.")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.finished.emit(False, str(e))

    def _convert(self):
        # TODO: Transcrire ici convert_to_roboflow_json() de l'ancienne GUI
        #
        # Étapes :
        #   1. Lire "# input_directory:" du CSV
        #   2. pd.read_csv(..., comment='#')
        #   3. Construire coco_json = {images:[], annotations:[], categories:[]}
        #   4. Pour chaque groupe img_id :
        #        - box     → entry bbox COCO [x,y,w,h]
        #        - mask/polygon → segmentation COCO + bbox depuis contours
        #   5. json.dump vers <globinfo_stem>_roboflow_coco.json
        #   6. Émettre self.progress(x) et self.status("...")
        raise NotImplementedError("Export Roboflow à implémenter.")


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

class NemaCounterGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NemaCounter  ·  PyQt6 & Napari")
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
        sb.setFixedWidth(265)
        lay = QVBoxLayout(sb)
        lay.setContentsMargins(14, 18, 14, 18)
        lay.setSpacing(10)

        # Logo
        logo = QLabel()
        pix_path = os.path.join("conf", "logo.png")
        if os.path.exists(pix_path):
            pix = QPixmap(pix_path).scaled(
                200, 110,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo.setPixmap(pix)
        else:
            logo.setText("NemaCounter")
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
        lay.setSpacing(14)

        # Titre + bouton scan
        top = QHBoxLayout()
        self._ov_title = QLabel("No project loaded — open a folder to start")
        self._ov_title.setObjectName("pageTitle")
        self._ov_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        top.addWidget(self._ov_title)
        top.addStretch()
        self._scan_btn = QPushButton("↻  Scan / Refresh")
        self._scan_btn.setObjectName("actionBtn")
        self._scan_btn.setEnabled(False)
        self._scan_btn.clicked.connect(self._scan_project)
        top.addWidget(self._scan_btn)
        lay.addLayout(top)

        # Cartes de statistiques
        cards = QHBoxLayout()
        cards.setSpacing(12)
        self._stat_images   = self._make_stat_card("Images found",      "0", "#89b4fa")
        self._stat_det_csv  = self._make_stat_card("Detection CSVs",    "0", "#a6e3a1")
        self._stat_seg_csv  = self._make_stat_card("Segmentation CSVs", "0", "#f9e2af")
        for c in (self._stat_images, self._stat_det_csv, self._stat_seg_csv):
            cards.addWidget(c)
        lay.addLayout(cards)

        # Splitter : table images | table CSVs
        splitter = QSplitter(Qt.Orientation.Vertical)

        img_box = QGroupBox("Images in working directory")
        img_lay = QVBoxLayout(img_box)
        self._img_table = QTableWidget(0, 5)
        self._img_table.setHorizontalHeaderLabels(
            ["Filename", "YOLO", "SAM", "Status", "Ext"]
        )
        self._img_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._img_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._img_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._img_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._img_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._img_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._img_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._img_table.setAlternatingRowColors(True)
        img_lay.addWidget(self._img_table)
        splitter.addWidget(img_box)

        csv_box = QGroupBox("Results")
        csv_lay = QVBoxLayout(csv_box)
        self._csv_table = QTableWidget(0, 5)
        self._csv_table.setHorizontalHeaderLabels(
            ["Filename", "Type", "Images", "Objects", "Path"]
        )
        self._csv_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._csv_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._csv_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._csv_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._csv_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._csv_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._csv_table.setAlternatingRowColors(True)
        # Double-click CSV → auto-fill Edition/Export
        self._csv_table.cellDoubleClicked.connect(self._on_csv_double_clicked)
        csv_lay.addWidget(self._csv_table)
        splitter.addWidget(csv_box)

        splitter.setSizes([420, 200])
        lay.addWidget(splitter, 1)
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
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(14)

        lay.addWidget(QLabel(
            "Convert a *_globinfo.csv to Roboflow COCO JSON.\n"
            "CSVs detected during project scan are listed below — or browse manually."
        ))

        quick_box = QGroupBox("Quick select (from last scan)")
        quick_lay = QVBoxLayout(quick_box)
        self._exp_csv_combo = QComboBox()
        self._exp_csv_combo.addItem("(no scan result)")
        quick_lay.addWidget(self._exp_csv_combo)
        lay.addWidget(quick_box)

        man_box = QGroupBox("Or browse manually")
        man_lay = QVBoxLayout(man_box)
        self._exp_csv_row = PathRow("Browse CSV …", "(no file selected)")
        self._exp_csv_row.btn.clicked.connect(
            lambda: self._pick_globinfo(self._exp_csv_row)
        )
        man_lay.addWidget(self._exp_csv_row)
        lay.addWidget(man_box)

        self.exp_progress = QProgressBar()
        self.exp_progress.setRange(0, 100)
        lay.addWidget(self.exp_progress)
        self.exp_status_lbl = QLabel("")
        self.exp_status_lbl.setObjectName("statusLabel")
        lay.addWidget(self.exp_status_lbl)

        self.exp_run_btn = QPushButton("Convert to Roboflow JSON")
        self.exp_run_btn.setObjectName("runButton")
        self.exp_run_btn.setFixedHeight(46)
        self.exp_run_btn.clicked.connect(self._start_export)
        lay.addWidget(self.exp_run_btn)
        lay.addStretch()
        return tab

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

        # Stats
        self._stat_images._val_lbl.setText(str(len(result["images"])))
        self._stat_det_csv._val_lbl.setText(str(len(result["glob_csvs"])))
        self._stat_seg_csv._val_lbl.setText(str(len(result["seg_csvs"])))

        # ── Lecture des CSVs pour avoir les comptes par image ────────────
        # Structure retournée : {img_basename: {"count": int, "type": "mask"|"box"|"sam"}}
        info_yolo: dict[str, dict] = {}   # depuis _globinfo.csv
        info_sam:  dict[str, dict] = {}   # depuis _segmentation_globinfo.csv

        def _load_image_info(csv_paths: list[str],
                             default_type: str = "box") -> dict[str, dict]:
            result_info: dict[str, dict] = {}
            for p in csv_paths:
                try:
                    df = pd.read_csv(p, comment="#",
                                     usecols=["img_id", "object_type"])
                    real = df[df["object_type"].notna() & (df["object_type"] != "")]
                    for img_id, grp in real.groupby("img_id"):
                        key = os.path.basename(str(img_id))
                        # Détecter si les objets sont des masques ou des boîtes
                        types = grp["object_type"].unique()
                        obj_type = "mask" if "mask" in types else default_type
                        result_info[key] = {
                            "count": result_info.get(key, {}).get("count", 0) + len(grp),
                            "type":  obj_type,
                        }
                except Exception:
                    pass
            return result_info

        info_yolo = _load_image_info(result["glob_csvs"], default_type="box")
        info_sam  = _load_image_info(result["seg_csvs"],  default_type="sam")

        # Tableau images avec résultats par ligne
        self._img_table.setRowCount(0)
        for p in result["images"]:
            r     = self._img_table.rowCount()
            fname = os.path.basename(p)
            ext   = os.path.splitext(fname)[1].lower()
            self._img_table.insertRow(r)

            yolo = info_yolo.get(fname)
            sam  = info_sam.get(fname)

            # Col 0 — nom de fichier
            self._img_table.setItem(r, 0, QTableWidgetItem(fname))

            # Col 1 — résultat YOLO  (boîtes ou masques selon object_type)
            if yolo:
                n      = yolo["count"]
                is_mask = yolo["type"] == "mask"
                label  = f"{n}  {'🎭' if is_mask else '🔲'}"
                color  = "#a6e3a1" if is_mask else "#89b4fa"
            else:
                label, color = "—", "#6c7086"
            item_y = QTableWidgetItem(label)
            item_y.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_y.setForeground(QColor(color))
            self._img_table.setItem(r, 1, item_y)

            # Col 2 — résultat SAM
            if sam:
                label_s = f"{sam['count']}  ✅"
                color_s = "#cba6f7"
            else:
                label_s, color_s = "—", "#6c7086"
            item_s = QTableWidgetItem(label_s)
            item_s.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_s.setForeground(QColor(color_s))
            self._img_table.setItem(r, 2, item_s)

            # Col 3 — statut global
            if sam:
                status, color_st = "✅ SAM done",    "#cba6f7"
            elif yolo and yolo["type"] == "mask":
                status, color_st = "🎭 YOLO masks",  "#a6e3a1"
            elif yolo:
                status, color_st = "🔲 YOLO boxes",  "#89b4fa"
            else:
                status, color_st = "⬜ pending",      "#6c7086"
            item_st = QTableWidgetItem(status)
            item_st.setForeground(QColor(color_st))
            item_st.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._img_table.setItem(r, 3, item_st)

            # Col 4 — extension
            self._img_table.setItem(r, 4, QTableWidgetItem(ext))

        # Tableau CSVs avec stats
        self._csv_table.setRowCount(0)
        for p in result["glob_csvs"] + result["seg_csvs"]:
            r     = self._csv_table.rowCount()
            self._csv_table.insertRow(r)
            fname = os.path.basename(p)
            typ   = "Segmentation" if p.endswith(CSV_SEG_SUFFIX) else "Detection"
            rel   = os.path.relpath(os.path.dirname(p), self._work_dir)

            # ── Lecture stats depuis le CSV ───────────────────────────────
            n_images  = "?"
            n_objects = "?"
            try:
                df = pd.read_csv(p, comment="#", usecols=["img_id", "object_type"])
                real = df[df["object_type"].notna() & (df["object_type"] != "")]
                n_images  = str(df["img_id"].nunique())
                n_objects = str(len(real))
            except Exception:
                pass

            item0 = QTableWidgetItem(fname)
            item0.setData(Qt.ItemDataRole.UserRole, p)
            self._csv_table.setItem(r, 0, item0)
            self._csv_table.setItem(r, 1, QTableWidgetItem(typ))

            item_img = QTableWidgetItem(n_images)
            item_img.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._csv_table.setItem(r, 2, item_img)

            item_obj = QTableWidgetItem(n_objects)
            item_obj.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            # Coloriser si des objets ont été trouvés
            if n_objects.isdigit() and int(n_objects) > 0:
                item_obj.setForeground(QColor("#a6e3a1"))
            self._csv_table.setItem(r, 3, item_obj)

            self._csv_table.setItem(r, 4, QTableWidgetItem(rel))

        # Combo Edition + Export
        self._edit_csv_combo.clear()
        self._exp_csv_combo.clear()
        all_csvs = result["glob_csvs"] + result["seg_csvs"]
        if all_csvs:
            for p in all_csvs:
                label = os.path.basename(p)
                self._edit_csv_combo.addItem(label, p)
                self._exp_csv_combo.addItem(label, p)
            # Auto-sélectionner le dernier globinfo dans Edition
            last_glob = result["glob_csvs"][-1] if result["glob_csvs"] else all_csvs[-1]
            idx = self._edit_csv_combo.findData(last_glob)
            if idx >= 0:
                self._edit_csv_combo.setCurrentIndex(idx)
        else:
            self._edit_csv_combo.addItem("(no CSVs found)")
            self._exp_csv_combo.addItem("(no CSVs found)")

        self._sync_ctx()

    def _on_csv_double_clicked(self, row: int, _col: int):
        """Double-clic sur un CSV dans la table → l'injecte dans Edition et Export."""
        item = self._csv_table.item(row, 0)
        if item:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path and os.path.exists(path):
                # Sélectionner dans le combo Edition si présent
                idx = self._edit_csv_combo.findData(path)
                if idx >= 0:
                    self._edit_csv_combo.setCurrentIndex(idx)
                else:
                    # Ajouter et sélectionner
                    self._edit_csv_combo.addItem(os.path.basename(path), path)
                    self._edit_csv_combo.setCurrentIndex(self._edit_csv_combo.count() - 1)
                # Export combo
                if self._exp_csv_combo.findData(path) == -1:
                    self._exp_csv_combo.addItem(os.path.basename(path), path)
                # Aller sur l'onglet edition
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

        viewer = napari.Viewer(title="NemaCounter — Manual Edition")

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
            from nemacounter.edition import EditionWindow
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
            win.destroyed.connect(self._on_edition_window_closed)

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
            self._run_scan()

    # =========================================================================
    #  EXPORT
    # =========================================================================

    def _start_export(self):
        csv_path = self._exp_csv_row.path or self._exp_csv_combo.currentData()
        if not csv_path or not os.path.exists(csv_path):
            QMessageBox.warning(self, "Error", "Select a valid globinfo CSV.")
            return

        self.exp_progress.setValue(0)
        self.exp_status_lbl.setText("Starting conversion…")
        self.exp_run_btn.setEnabled(False)

        self._exp_worker = ExportWorker(csv_path)
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
    w = NemaCounterGUI()
    w.show()
    sys.exit(app.exec())