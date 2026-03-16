"""
Edition Engine
==============
Module d'annotation / ré-annotation manuelle basé sur Napari.

Point d'entrée unique :
    from lyra.edition_engine import EditionWindow

    win = EditionWindow()
    win.project_id       = "MyProject_v2"
    win.output_directory = "/path/to/output"
    win.input_directory  = Path("/path/to/images")
    win.load_project(csv_path)   # OU win.load_images(image_dir)
    win.closed.connect(my_callback)
    win.showMaximized()

Deux modes d'entrée :
  load_project(csv_path)  → charge les annotations existantes d'un *_globinfo.csv
  load_images(image_dir)  → mode vierge, images sans annotations pré-existantes

Trois outils d'annotation dans le dock Napari :
  ● Box       — rectangle Napari (object_type='box')
  ● Polygone  — polygone libre  (object_type='polygon')
  ● SAM Assist— bbox prompt → SAM2/SAM3 → masque → polygone (object_type='mask')

Navigation multi-images via un QComboBox dans le dock.

Sauvegarde vers *_edition_globinfo.csv au format LYRA standard.

Corrections v2 :
  - SAMLoadWorker défini au niveau MODULE (PyQt6 interdit pyqtSignal sur les
    classes internes — c'est la cause du blocage silencieux du chargement SAM).
  - Bouton "Terminer l'édition" : sauvegarde + cache la fenêtre Napari sans
    tuer le process Qt principal.
  - La fermeture X de Napari est interceptée : dialog Sauvegarder/Ignorer/Annuler
    puis hide() au lieu de close(), ce qui évite le crash de main_gui.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QRadioButton, QVBoxLayout, QWidget,
)

import napari

log = logging.getLogger("LYRA.edition")

_OUTPUT_COLUMNS = [
    "img_id", "object_id", "xmin", "ymin", "xmax", "ymax",
    "confidence", "class", "name", "area", "contours",
    "object_type", "project_id",
]
_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_PALETTE = [
    "#ff3838", "#ff9d97", "#ff701f", "#ffb21d", "#cfd231",
    "#48f90a", "#92cc17", "#3ddb86", "#1a9334", "#00d4bb",
    "#2c99a8", "#00c2ff", "#344593", "#6473ff", "#0018ec",
    "#8438ff", "#520085", "#cb38ff", "#ff95c8", "#ff37c7",
]


# =============================================================================
#  WORKERS — définis au niveau MODULE (obligatoire pour pyqtSignal en PyQt6)
# =============================================================================

class SAMLoadWorker(QThread):
    """
    Charge SAM 2.1 dans un thread séparé.

    IMPORTANT : doit être défini au niveau module et non comme classe interne.
    PyQt6 ne permet pas d'utiliser pyqtSignal dans les classes définies
    localement (inner classes) — cela provoque un blocage silencieux sans
    exception, car les signaux ne sont jamais connectés.
    """
    ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def run(self):
        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2.1-hiera-large", device=device
            )
            self.ready.emit(predictor)
        except Exception as exc:
            log.exception("SAM load failed")
            self.error.emit(str(exc))


class SAMAssistWorker(QThread):
    """
    Lance SAM2 sur une bbox prompt en arrière-plan.
    Entrée  : image RGB (np.ndarray) + bbox [xmin,ymin,xmax,ymax]
    Sortie  : masque binaire uint8 (np.ndarray H×W)
    """
    mask_ready = pyqtSignal(object)
    error      = pyqtSignal(str)

    def __init__(self, predictor, image_rgb: np.ndarray,
                 bbox: np.ndarray, parent=None):
        super().__init__(parent)
        self.predictor = predictor
        self.image_rgb = image_rgb
        self.bbox      = bbox

    def run(self):
        try:
            import torch
            self.predictor.set_image(self.image_rgb)
            with torch.inference_mode():
                ctx = (
                    torch.autocast("cuda", dtype=torch.float16)
                    if torch.cuda.is_available() else torch.no_grad()
                )
                with ctx:
                    masks, _, _ = self.predictor.predict(
                        box=self.bbox[None, :], multimask_output=False,
                    )
            mask = (masks[0].cpu().numpy() if hasattr(masks[0], "cpu")
                    else masks[0]).astype(np.uint8)
            self.mask_ready.emit(mask)
        except Exception as exc:
            log.exception("SAM assist failed")
            self.error.emit(str(exc))


# =============================================================================
#  MODÈLE DE DONNÉES
# =============================================================================

class AnnotationStore:
    def __init__(self):
        self._store: dict[str, list[dict]] = {}
        self.image_paths: list[str] = []
        self.input_directory: str | None = None
        self.project_id: str = "edition"
        self.class_names: list[str] = []

    def load_from_csv(self, globinfo_path: str) -> None:
        self._store.clear(); self.image_paths.clear()
        self.input_directory = None
        try:
            with open(globinfo_path, "r", encoding="utf-8") as f:
                first = f.readline().strip()
                if first.startswith("# input_directory:"):
                    self.input_directory = first.split(":", 1)[1].strip()
        except Exception:
            pass
        df = pd.read_csv(globinfo_path, comment="#")
        if "img_id" not in df.columns:
            raise ValueError(f"Colonne 'img_id' absente de {globinfo_path}")
        self.class_names = sorted(
            df["name"].dropna().unique().tolist() if "name" in df.columns else []
        )
        for img_id_rel, group in df.groupby("img_id"):
            img_id_rel = str(img_id_rel)
            self.image_paths.append(img_id_rel)
            self._store[img_id_rel] = [
                {col: row.get(col, "") for col in _OUTPUT_COLUMNS}
                for _, row in group.iterrows()
            ]

    def load_from_dir(self, image_dir: str) -> None:
        self._store.clear(); self.image_paths.clear()
        self.input_directory = os.path.abspath(image_dir)
        self.class_names = []
        for fname in sorted(os.listdir(image_dir)):
            if os.path.splitext(fname)[1].lower() in _IMG_EXTENSIONS:
                self.image_paths.append(fname)
                self._store[fname] = []

    def get(self, img_id_rel: str) -> list[dict]:
        return self._store.get(img_id_rel, [])

    def set(self, img_id_rel: str, annotations: list[dict]) -> None:
        self._store[img_id_rel] = annotations
        if img_id_rel not in self.image_paths:
            self.image_paths.append(img_id_rel)
        for ann in annotations:
            name = str(ann.get("name", "")).strip()
            if name and name not in self.class_names:
                self.class_names.append(name)
                self.class_names.sort()

    def save_to_csv(self, output_path: str) -> None:
        all_rows: list[dict] = []
        for img_id_rel in self.image_paths:
            for i, ann in enumerate(self._store.get(img_id_rel, []), start=1):
                row = {col: ann.get(col, "") for col in _OUTPUT_COLUMNS}
                row["img_id"] = img_id_rel
                row["object_id"] = i
                row["project_id"] = self.project_id
                all_rows.append(row)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if self.input_directory:
                f.write(f"# input_directory: {self.input_directory}\n")
            if not all_rows:
                writer = csv.DictWriter(f, fieldnames=_OUTPUT_COLUMNS,
                                        quoting=csv.QUOTE_ALL)
                writer.writeheader()
                return
            df = pd.DataFrame(all_rows)
            for col in ("xmin", "ymin", "xmax", "ymax", "object_id"):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            for col in ("area", "confidence"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            df["class"] = pd.to_numeric(df["class"], errors="coerce").fillna(0).astype(int)
            for col in ("img_id", "name", "object_type", "project_id", "contours"):
                df[col] = df[col].astype(str).replace("<NA>", "")
            df[_OUTPUT_COLUMNS].to_csv(f, index=False, quoting=csv.QUOTE_ALL)
        log.info(f"Saved → {output_path}")


# =============================================================================
#  DOCK WIDGET
# =============================================================================

class EditionDockWidget(QWidget):
    """Panneau de contrôle injecté dans le dock Napari."""

    close_requested = pyqtSignal()   # déclenché par "Terminer l'édition"

    def __init__(self, store: AnnotationStore, viewer: napari.Viewer,
                 output_path: str, parent=None):
        super().__init__(parent)
        self._store              = store
        self._viewer             = viewer
        self._output_path        = output_path
        self._current_idx        = 0
        self._sam_predictor      = None
        self._sam_load_worker: SAMLoadWorker | None   = None
        self._sam_run_worker:  SAMAssistWorker | None = None
        self._shapes_layer       = None
        self._img_layer          = None

        self._build_ui()
        self._connect()
        if store.image_paths:
            self._load_image(0)

    # ─────────────────────────────────────────────────────────────────────────
    #  UI
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setMinimumWidth(280)
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── Navigation ────────────────────────────────────────────────────
        nav = QGroupBox("Images")
        nv  = QVBoxLayout(nav)
        self._img_combo = QComboBox()
        for p in self._store.image_paths:
            self._img_combo.addItem(os.path.basename(p), p)
        nv.addWidget(self._img_combo)
        row = QHBoxLayout()
        self._prev_btn = QPushButton("◀  Préc.")
        self._next_btn = QPushButton("Suiv.  ▶")
        row.addWidget(self._prev_btn); row.addWidget(self._next_btn)
        nv.addLayout(row)
        root.addWidget(nav)

        # ── Outils ────────────────────────────────────────────────────────
        tool = QGroupBox("Outil")
        tv   = QVBoxLayout(tool)
        self._rb_box     = QRadioButton("Box (rectangle)")
        self._rb_polygon = QRadioButton("Polygone libre")
        self._rb_sam     = QRadioButton("SAM Assist  (bbox → masque)")
        self._rb_box.setChecked(True)
        for rb in (self._rb_box, self._rb_polygon, self._rb_sam):
            tv.addWidget(rb)

        self._sam_status = QLabel("SAM : non chargé")
        self._sam_status.setStyleSheet("color:#888;font-size:11px;")
        tv.addWidget(self._sam_status)

        sr = QHBoxLayout()
        self._load_sam_btn = QPushButton("Charger SAM")
        self._run_sam_btn  = QPushButton("▶ Appliquer")
        self._run_sam_btn.setEnabled(False)
        sr.addWidget(self._load_sam_btn); sr.addWidget(self._run_sam_btn)
        tv.addLayout(sr)
        root.addWidget(tool)

        # ── Classe ────────────────────────────────────────────────────────
        cls = QGroupBox("Classe active")
        cv  = QVBoxLayout(cls)
        self._class_combo = QComboBox()
        self._class_combo.setEditable(True)
        self._class_combo.addItems(self._store.class_names or ["object"])
        cv.addWidget(self._class_combo)
        ar = QHBoxLayout()
        self._new_class_edit = QLineEdit()
        self._new_class_edit.setPlaceholderText("Nouvelle classe…")
        self._add_cls_btn = QPushButton("+ Ajouter")
        ar.addWidget(self._new_class_edit); ar.addWidget(self._add_cls_btn)
        cv.addLayout(ar)
        root.addWidget(cls)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        # ── Bouton Sauvegarder ────────────────────────────────────────────
        self._save_btn = QPushButton("💾  Sauvegarder")
        self._save_btn.setStyleSheet(
            "QPushButton{background:#1e66f5;color:#fff;border-radius:5px;"
            "padding:7px;font-weight:bold;}"
            "QPushButton:hover{background:#2d79f7;}"
        )
        root.addWidget(self._save_btn)

        # ── Bouton Terminer — cache Napari sans tuer main_gui ─────────────
        self._done_btn = QPushButton("✔  Terminer l'édition")
        self._done_btn.setStyleSheet(
            "QPushButton{background:#2d9c55;color:#fff;border-radius:5px;"
            "padding:7px;font-weight:bold;}"
            "QPushButton:hover{background:#3ab866;}"
        )
        root.addWidget(self._done_btn)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("font-size:11px;color:#888;")
        root.addWidget(self._status_lbl)
        root.addStretch()

    def _connect(self):
        self._img_combo.currentIndexChanged.connect(self._on_combo_changed)
        self._prev_btn.clicked.connect(lambda: self._navigate(self._current_idx - 1))
        self._next_btn.clicked.connect(lambda: self._navigate(self._current_idx + 1))
        self._add_cls_btn.clicked.connect(self._on_add_class)
        self._load_sam_btn.clicked.connect(self._on_load_sam)
        self._run_sam_btn.clicked.connect(self._on_run_sam)
        self._save_btn.clicked.connect(self._on_save)
        self._done_btn.clicked.connect(self._on_done)
        for rb in (self._rb_box, self._rb_polygon, self._rb_sam):
            rb.toggled.connect(self._apply_tool_mode)

    # ─────────────────────────────────────────────────────────────────────────
    #  Navigation
    # ─────────────────────────────────────────────────────────────────────────

    def _on_combo_changed(self, idx: int):
        if idx != self._current_idx:
            self._flush_current(); self._load_image(idx)

    def _navigate(self, idx: int):
        idx = max(0, min(idx, len(self._store.image_paths) - 1))
        self._flush_current()
        self._img_combo.blockSignals(True)
        self._img_combo.setCurrentIndex(idx)
        self._img_combo.blockSignals(False)
        self._load_image(idx)

    # ─────────────────────────────────────────────────────────────────────────
    #  Chargement image
    # ─────────────────────────────────────────────────────────────────────────

    def _load_image(self, idx: int):
        self._current_idx = idx
        rel = self._store.image_paths[idx]
        path = self._resolve_image(rel)
        if not path:
            self._set_status(f"⚠ Introuvable : {rel}"); return
        bgr = cv2.imread(path)
        if bgr is None:
            self._set_status(f"⚠ Illisible : {path}"); return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self._viewer.layers.clear()
        self._img_layer    = self._viewer.add_image(rgb, name=os.path.basename(rel))
        self._shapes_layer = self._viewer.add_shapes(
            name="Annotations", face_color="transparent",
            edge_color="cyan", edge_width=2,
        )
        self._shapes_layer.mode = "add_rectangle"

        for ann in self._store.get(rel):
            self._ann_to_shape(ann)
        self._apply_tool_mode()

        n = len(self._store.image_paths)
        self._prev_btn.setEnabled(idx > 0)
        self._next_btn.setEnabled(idx < n - 1)
        self._set_status(f"Image {idx+1}/{n} — {len(self._store.get(rel))} ann.")

    def _resolve_image(self, rel: str) -> str | None:
        cands = []
        if self._store.input_directory:
            cands.append(os.path.join(self._store.input_directory, rel))
        cands.append(rel)
        return next((p for p in cands if os.path.exists(p)), None)

    # ─────────────────────────────────────────────────────────────────────────
    #  Flush shapes → store
    # ─────────────────────────────────────────────────────────────────────────

    def _flush_current(self):
        if self._shapes_layer is None or not self._store.image_paths:
            return
        rel = self._store.image_paths[self._current_idx]
        self._store.set(rel, self._shapes_to_anns(rel))

    # ─────────────────────────────────────────────────────────────────────────
    #  Conversions
    # ─────────────────────────────────────────────────────────────────────────

    def _ann_to_shape(self, ann: dict):
        if self._shapes_layer is None:
            return
        ot    = str(ann.get("object_type", "box")).lower()
        name  = str(ann.get("name", "object"))
        ci    = (self._store.class_names.index(name)
                 if name in self._store.class_names else 0)
        color = _PALETTE[ci % len(_PALETTE)]

        if ot == "box":
            x0, y0 = float(ann.get("xmin", 0)), float(ann.get("ymin", 0))
            x1, y1 = float(ann.get("xmax", 0)), float(ann.get("ymax", 0))
            rect = np.array([[y0,x0],[y0,x1],[y1,x1],[y1,x0]])
            self._shapes_layer.add_rectangles(
                [rect], edge_color=[color], face_color="transparent", edge_width=2)
        elif ot in ("mask", "polygon"):
            try:
                clist = json.loads(ann.get("contours", "[]"))
            except Exception:
                clist = []
            for pts in clist:
                if len(pts) < 3:
                    continue
                poly = np.array([[p[1], p[0]] for p in pts])
                self._shapes_layer.add_polygons(
                    [poly], edge_color=[color],
                    face_color=color + "33", edge_width=2)

    def _shapes_to_anns(self, rel: str) -> list[dict]:
        if self._shapes_layer is None:
            return []
        active = self._class_combo.currentText().strip() or "object"
        props  = self._shapes_layer.properties
        anns: list[dict] = []
        for i, (data, stype) in enumerate(
            zip(self._shapes_layer.data, self._shapes_layer.shape_type)
        ):
            try:
                name = props.get("name", [active]*len(self._shapes_layer.data))[i]
            except Exception:
                name = active

            if stype == "rectangle":
                ys, xs = data[:,0], data[:,1]
                anns.append({
                    "img_id": rel,
                    "xmin": int(xs.min()), "ymin": int(ys.min()),
                    "xmax": int(xs.max()), "ymax": int(ys.max()),
                    "confidence": 1.0, "class": 0, "name": name,
                    "area": float((xs.max()-xs.min())*(ys.max()-ys.min())),
                    "contours": "[]", "object_type": "box",
                    "project_id": self._store.project_id,
                })
            elif stype in ("polygon", "path"):
                pts_xy = [[float(p[1]), float(p[0])] for p in data]
                xs2 = [p[0] for p in pts_xy]; ys2 = [p[1] for p in pts_xy]
                cnt = np.array(data[:,[1,0]], dtype=np.int32)
                anns.append({
                    "img_id": rel,
                    "xmin": int(min(xs2)), "ymin": int(min(ys2)),
                    "xmax": int(max(xs2)), "ymax": int(max(ys2)),
                    "confidence": 1.0, "class": 0, "name": name,
                    "area": float(cv2.contourArea(cnt)),
                    "contours": json.dumps([pts_xy]),
                    "object_type": "polygon",
                    "project_id": self._store.project_id,
                })
        return anns

    # ─────────────────────────────────────────────────────────────────────────
    #  Outil / classe
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_tool_mode(self, *_):
        if self._shapes_layer is None:
            return
        if self._rb_polygon.isChecked():
            self._shapes_layer.mode = "add_polygon"
        else:   # box OU SAM (SAM utilise aussi un rectangle comme prompt)
            self._shapes_layer.mode = "add_rectangle"
        self._run_sam_btn.setEnabled(
            self._rb_sam.isChecked() and self._sam_predictor is not None
        )

    def _on_add_class(self):
        name = self._new_class_edit.text().strip()
        if name and name not in self._store.class_names:
            self._store.class_names.append(name)
            self._store.class_names.sort()
            self._class_combo.addItem(name)
            self._class_combo.setCurrentText(name)
            self._new_class_edit.clear()

    # ─────────────────────────────────────────────────────────────────────────
    #  SAM
    # ─────────────────────────────────────────────────────────────────────────

    def _on_load_sam(self):
        """
        Charge SAM dans un QThread sans parent=.
        Passer un QWidget comme parent d'un QThread en PyQt6 peut provoquer la
        destruction prématurée du worker ou bloquer les signaux cross-thread.
        Le premier lancement télécharge ~2.4 GB — un QTimer pulse le label.
        """
        self._sam_status.setText(
            "SAM : chargement en cours…\n"
            "(1er lancement : téléchargement ~2.4 GB — patientez)"
        )
        self._load_sam_btn.setEnabled(False)

        # Pulse visuel toutes les 2 s pour montrer que le thread est vivant
        from PyQt6.QtCore import QTimer
        self._sam_pulse_n = 0
        self._sam_pulse_timer = QTimer(self)
        def _pulse():
            d = "." * (self._sam_pulse_n % 4 + 1)
            self._sam_status.setText(
                f"SAM : chargement{d}\n"
                "(téléchargement si 1er lancement — patience)"
            )
            self._sam_pulse_n += 1
        self._sam_pulse_timer.timeout.connect(_pulse)
        self._sam_pulse_timer.start(2000)

        # PAS de parent= ici — obligatoire pour les signaux cross-thread en PyQt6
        self._sam_load_worker = SAMLoadWorker()
        self._sam_load_worker.ready.connect(self._on_sam_loaded)
        self._sam_load_worker.error.connect(self._on_sam_error)
        self._sam_load_worker.finished.connect(self._sam_load_worker.deleteLater)
        self._sam_load_worker.start()

    def _on_sam_loaded(self, predictor):
        if hasattr(self, "_sam_pulse_timer"):
            self._sam_pulse_timer.stop()
        self._sam_predictor = predictor
        self._sam_status.setText("SAM : prêt ✅")
        self._load_sam_btn.setEnabled(True)
        if self._rb_sam.isChecked():
            self._run_sam_btn.setEnabled(True)

    def _on_sam_error(self, msg: str):
        if hasattr(self, "_sam_pulse_timer"):
            self._sam_pulse_timer.stop()
        self._sam_status.setText("SAM : erreur ❌")
        self._load_sam_btn.setEnabled(True)
        QMessageBox.critical(self, "SAM", msg)

    def _on_run_sam(self):
        if self._sam_predictor is None or self._shapes_layer is None:
            return
        # Forcer le mode "select" avant tout — évite vertex_remove qui cause IndexError
        self._shapes_layer.mode = "select"

        rects = [(i, d) for i, (d, t) in enumerate(
            zip(self._shapes_layer.data, self._shapes_layer.shape_type))
            if t == "rectangle"]
        if not rects:
            QMessageBox.information(self, "SAM",
                "Dessinez d'abord un rectangle (bbox prompt) autour de l'objet.\n"
                "Outil : sélectionnez 'SAM Assist' puis dessinez un rectangle.")
            self._shapes_layer.mode = "add_rectangle"
            return

        rect_idx, rdata = rects[-1]
        bbox = np.array([rdata[:,1].min(), rdata[:,0].min(),
                         rdata[:,1].max(), rdata[:,0].max()])

        self._run_sam_btn.setEnabled(False)
        self._set_status("SAM inférence en cours…")

        # PAS de parent= pour la même raison que SAMLoadWorker
        self._sam_run_worker = SAMAssistWorker(
            self._sam_predictor, self._img_layer.data.copy(), bbox)
        self._sam_run_worker.mask_ready.connect(
            lambda m: self._on_mask_ready(m, rect_idx))
        self._sam_run_worker.error.connect(self._on_sam_run_error)
        self._sam_run_worker.finished.connect(self._sam_run_worker.deleteLater)
        self._sam_run_worker.start()

    def _on_mask_ready(self, mask: np.ndarray, rect_idx: int):
        self._run_sam_btn.setEnabled(True)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            self._set_status("SAM : aucun masque."); return

        # ── Suppression du rectangle prompt ───────────────────────────────────
        # Sécurité : vérifier que l'index est encore valide avant toute opération
        # (le layer peut avoir changé entre _on_run_sam et ce callback).
        # On force d'abord le mode "select" pour éviter que Napari soit en
        # vertex_remove, ce qui provoque l'IndexError signalé.
        try:
            if self._shapes_layer is not None:
                self._shapes_layer.mode = "select"
                n = len(self._shapes_layer.data)
                if rect_idx < n:
                    self._shapes_layer.selected_data = {rect_idx}
                    self._shapes_layer.remove_selected()
                else:
                    log.warning(f"rect_idx {rect_idx} hors limites ({n} shapes) — prompt non supprimé")
        except Exception as exc:
            log.warning(f"Impossible de supprimer le rectangle prompt : {exc}")

        # ── Ajout des polygones résultants ─────────────────────────────────────
        active = self._class_combo.currentText().strip() or "object"
        ci     = (self._store.class_names.index(active)
                  if active in self._store.class_names else 0)
        color  = _PALETTE[ci % len(_PALETTE)]
        added  = 0
        for cnt in cnts:
            sq = cnt.squeeze()
            if sq.ndim == 1: sq = sq.reshape(1, 2)
            if sq.ndim != 2 or sq.shape[0] < 3: continue
            self._shapes_layer.add_polygons(
                [sq[:, [1, 0]].astype(float)],
                edge_color=[color], face_color=color + "33", edge_width=2)
            added += 1

        # Remettre le mode dessin après l'opération
        self._shapes_layer.mode = "add_rectangle"
        self._set_status(f"SAM : {added} masque(s) ✅")

    def _on_sam_run_error(self, msg: str):
        self._run_sam_btn.setEnabled(True)
        self._set_status(f"SAM erreur : {msg}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Sauvegarde & fermeture propre
    # ─────────────────────────────────────────────────────────────────────────

    def _on_save(self):
        self._flush_current()
        try:
            self._store.save_to_csv(self._output_path)
            n = sum(len(self._store.get(p)) for p in self._store.image_paths)
            self._set_status(f"✅ {n} annotations → {self._output_path}")
        except Exception as exc:
            log.exception("Save error")
            QMessageBox.critical(self, "Erreur", str(exc))

    def _on_done(self):
        """Sauvegarde puis émet close_requested → EditionWindow cache Napari."""
        self._flush_current()
        try:
            self._store.save_to_csv(self._output_path)
        except Exception as exc:
            ans = QMessageBox.question(
                self, "Erreur de sauvegarde",
                f"La sauvegarde a échoué :\n{exc}\n\nFermer quand même ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        self.close_requested.emit()

    def _set_status(self, msg: str):
        self._status_lbl.setText(msg)
        log.info(msg)


# =============================================================================
#  EDITION WINDOW — point d'entrée principal
# =============================================================================

class EditionWindow(QObject):
    """
    Fenêtre d'édition LYRA basée sur Napari.

    La fenêtre Napari est CACHÉE (hide) et non fermée (close) pour ne pas
    tuer le process Qt de main_gui. Le signal `closed` est émis dans les
    deux cas de sortie :
      - clic sur "Terminer l'édition" dans le dock
      - fermeture via le bouton X (après dialog Save/Discard/Cancel)
    """

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_id       : str         = "edition"
        self.output_directory : str         = ""
        self.input_directory  : Path | None = None
        self.display_size     : int         = 1056   # héritage API

        self._store       = AnnotationStore()
        self._viewer      = None
        self._dock        = None
        self._initialized = False

    # ── API publique ──────────────────────────────────────────────────────────

    def load_project(self, csv_path: str) -> None:
        self._store = AnnotationStore()
        self._store.project_id = self.project_id
        if self.input_directory:
            self._store.input_directory = str(self.input_directory)
        self._store.load_from_csv(csv_path)
        if self._initialized:
            self._refresh_dock()

    def load_images(self, image_dir: str) -> None:
        self._store = AnnotationStore()
        self._store.project_id = self.project_id
        self._store.load_from_dir(image_dir)
        if self._initialized:
            self._refresh_dock()

    def showMaximized(self) -> None:
        self._init_viewer()
        try:
            qw = self._viewer.window._qt_window
            qw.show(); qw.showMaximized()
        except Exception:
            self._viewer.window.show()

    def show(self) -> None:
        self.showMaximized()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_viewer(self):
        if self._initialized:
            # Ré-afficher si déjà créé
            try:
                qw = self._viewer.window._qt_window
                qw.show(); qw.showMaximized()
            except Exception:
                pass
            return

        self._initialized = True
        self._viewer = napari.Viewer(
            title=f"LYRA — Edition  [{self.project_id}]",
            show=False,
        )
        self._dock = EditionDockWidget(
            self._store, self._viewer, self._resolve_output_path()
        )
        self._dock.close_requested.connect(self._hide_viewer)
        self._viewer.window.add_dock_widget(
            self._dock, name="LYRA Tools", area="right"
        )

        # Intercepter la fermeture via le bouton X
        try:
            qt_win = self._viewer.window._qt_window

            def _on_close(event):
                ans = QMessageBox.question(
                    qt_win, "Quitter l'édition ?",
                    "Voulez-vous sauvegarder les annotations ?",
                    QMessageBox.StandardButton.Save
                    | QMessageBox.StandardButton.Discard
                    | QMessageBox.StandardButton.Cancel,
                )
                if ans == QMessageBox.StandardButton.Cancel:
                    event.ignore(); return
                if ans == QMessageBox.StandardButton.Save:
                    try:
                        if self._dock:
                            self._dock._flush_current()
                            self._dock._store.save_to_csv(self._resolve_output_path())
                    except Exception as exc:
                        QMessageBox.warning(qt_win, "Sauvegarde", f"Erreur : {exc}")
                # hide() au lieu de fermer → ne tue pas le process Qt
                event.ignore()
                self._hide_viewer()

            qt_win.closeEvent = _on_close
        except Exception:
            log.warning("Could not bind closeEvent on Napari window")

    def _hide_viewer(self):
        """Cache la fenêtre Napari et notifie main_gui."""
        try:
            self._viewer.window._qt_window.hide()
        except Exception:
            pass
        self.closed.emit()

    def _refresh_dock(self):
        if self._viewer is None:
            return
        try:
            self._viewer.window.remove_dock_widget(self._dock)
        except Exception:
            pass
        self._dock = EditionDockWidget(
            self._store, self._viewer, self._resolve_output_path()
        )
        self._dock.close_requested.connect(self._hide_viewer)
        self._viewer.window.add_dock_widget(
            self._dock, name="LYRA Tools", area="right"
        )

    def _resolve_output_path(self) -> str:
        if self.output_directory:
            os.makedirs(self.output_directory, exist_ok=True)
            return os.path.join(
                self.output_directory,
                f"{self.project_id}_edition_globinfo.csv"
            )
        base = self._store.input_directory or "."
        return os.path.join(base, f"{self.project_id}_edition_globinfo.csv")
