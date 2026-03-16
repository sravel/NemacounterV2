"""
Detection Engine
================
Module unifié gérant :
  - LYRADetection  : inférence YOLO (detect & segment), averaging, métriques
  - LYRASegmentation : segmentation SAM2 sur annotations existantes
  - detection_workflow()    : pipeline YOLO complet → CSV
  - segmentation_workflow() : pipeline SAM2 sur CSV de détection → CSV

Utilitaires partagés (privés)
------------------------------
  _resolve_device        : sélection CPU/GPU cohérente
  _find_image_path       : résolution multi-stratégie d'un chemin relatif
  _enforce_df_schema     : validation colonnes + types du DataFrame de sortie
  _save_results          : écriture CSV (avec commentaire metadata) + summary
  create_project_dirs_structure : arborescence de sortie
"""
from __future__ import annotations

import csv
import gc
import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

try:
    import lyra.utils as utils
    import lyra.common as common
except ImportError:
    utils = None
    common = None

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

_YOLO_COLORS: list[tuple[int, int, int]] = [
    (255, 56, 56),  (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10),   (146, 204, 23), (61, 219, 134),
    (26, 147, 52),  (0, 212, 187),   (44, 153, 168), (0, 194, 255),
    (52, 69, 147),  (100, 115, 255), (0, 24, 236),   (132, 56, 255),
    (82, 0, 133),   (203, 56, 255),  (255, 149, 200),(255, 55, 199),
]
_rng = np.random.default_rng(42)
while len(_YOLO_COLORS) < 100:
    _YOLO_COLORS.append(tuple(int(x) for x in _rng.integers(0, 255, 3)))

# Colonnes et types du schéma de sortie partagé par les deux workflows
_OUTPUT_COLUMNS = [
    "img_id", "object_id", "xmin", "ymin", "xmax", "ymax",
    "confidence", "class", "name", "area", "contours",
    "object_type", "project_id",
]
_OUTPUT_DEFAULTS: dict[str, object] = {
    "xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0, "object_id": 0,
    "confidence": 1.0, "area": np.nan, "class": 0,
    "name": "unknown", "contours": "[]", "object_type": "box",
}


# =============================================================================
#  UTILITAIRES PARTAGÉS  (privés)
# =============================================================================

def _resolve_device(gpu_if_avail: bool = True) -> str:
    """
    Retourne 'cuda:0' si disponible et demandé, sinon 'cpu'.
    Centralisé pour éviter la duplication dans chaque workflow.
    """
    if gpu_if_avail and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _find_image_path(img_path_rel: str,
                     search_dirs: list[str]) -> str | None:
    """
    Résout le chemin d'une image par ordre de priorité :
      1. Chaque répertoire de search_dirs (ex : input_directory, csv_dir)
      2. Le chemin tel quel (absolu ou relatif au CWD)

    Returns:
        Chemin absolu existant, ou None si introuvable.
    """
    candidates = [os.path.join(d, img_path_rel) for d in search_dirs if d]
    candidates.append(img_path_rel)
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _enforce_df_schema(df: pd.DataFrame, project_id: str) -> pd.DataFrame:
    """
    Garantit que le DataFrame possède toutes les colonnes requises
    avec les bons types, dans le bon ordre.

    Centralisé pour éviter le bloc identique dans detection_workflow
    et segmentation_workflow.

    Returns:
        DataFrame réordonné et typé selon _OUTPUT_COLUMNS.
    """
    # Colonnes manquantes → valeurs par défaut
    for col, default in _OUTPUT_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    if "project_id" not in df.columns:
        df["project_id"] = project_id

    # Application des types
    try:
        for col in ("xmin", "ymin", "xmax", "ymax", "object_id"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        for col in ("area", "confidence"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        df["class"] = pd.to_numeric(df["class"], errors="coerce").fillna(0).astype(int)
        for col in ("img_id", "name", "object_type", "project_id", "contours"):
            df[col] = df[col].astype(str).replace("<NA>", "")
    except Exception as e:
        logging.getLogger("LYRA").warning(f"Schema type enforcement partial: {e}")

    return df[_OUTPUT_COLUMNS]


def _save_results(df: pd.DataFrame,
                  output_path: str,
                  project_id: str,
                  input_directory: str | None = None) -> None:
    """
    Sauvegarde le DataFrame principal (avec commentaire metadata)
    et le tableau résumé.

    Centralisé pour les deux workflows qui produisent le même format.

    Args:
        df             : DataFrame validé par _enforce_df_schema
        output_path    : chemin du fichier *_globinfo.csv
        project_id     : identifiant du projet
        input_directory: répertoire source à conserver en metadata
    """
    log = logging.getLogger("LYRA.io")

    # ── globinfo ─────────────────────────────────────────────────────────
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if input_directory:
            f.write(f"# input_directory: {input_directory}\n")
        df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
    log.info(f"Globinfo saved: {output_path}")

    # ── summary ──────────────────────────────────────────────────────────
    if common:
        summary_path = output_path.replace("_globinfo.csv", "_summary.csv")
        # Exclure les lignes placeholder (object_type vide)
        df_real = df[df["object_type"].str.strip() != ""]
        if not df_real.empty:
            summary = common.create_summary_table(df_real, project_id)
            summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_ALL)
            log.info(f"Summary saved: {summary_path}")


# =============================================================================
#  DESSIN
# =============================================================================

def draw_detections_yolo_style(img: np.ndarray,
                                df: pd.DataFrame, *,
                                show_bbox: bool = True,
                                show_labels: bool = True,
                                show_conf: bool = False,
                                show_mask: bool = True) -> np.ndarray:
    """
    Dessine les détections depuis un DataFrame sur une image, style YOLO.

    Args:
        img            : image BGR (np.ndarray)
        df             : DataFrame avec colonnes xmin/ymin/xmax/ymax/class/name/confidence/contours
        show_bbox      : dessiner les bounding boxes
        show_labels    : afficher le nom de classe
        show_conf      : afficher le score de confiance
        show_mask      : remplir les masques avec transparence

    Returns:
        np.ndarray : image annotée (copie)
    """
    log = logging.getLogger("LYRA.draw")
    overlay = img.copy()
    log.debug(f"Drawing {len(df)} detections")

    # Passe 1 — masques (semi-transparents, sous les boîtes)
    if show_mask:
        mask_layer = overlay.copy()
        for idx, row in df.iterrows():
            if idx % 200 == 0 and idx > 0:
                gc.collect()
            class_id = int(row["class"]) if not pd.isna(row.get("class", float("nan"))) else 0
            color = _YOLO_COLORS[class_id % len(_YOLO_COLORS)]
            contours_raw = row.get("contours")
            if pd.notna(contours_raw) and contours_raw and contours_raw != "[]":
                try:
                    for pts in json.loads(contours_raw):
                        if len(pts) >= 3:
                            arr = np.array(pts, dtype=np.int32)
                            if arr.ndim == 2:
                                arr = arr.reshape(-1, 1, 2)
                            cv2.fillPoly(mask_layer, [arr], color)
                except (json.JSONDecodeError, ValueError):
                    pass
        overlay = cv2.addWeighted(overlay, 0.5, mask_layer, 0.5, 0)

    # Passe 2 — boîtes & labels
    if show_bbox or show_labels or show_conf:
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        for _, row in df.iterrows():
            class_id = int(row["class"]) if not pd.isna(row.get("class", float("nan"))) else 0
            color = _YOLO_COLORS[class_id % len(_YOLO_COLORS)]
            x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
            if show_bbox:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            parts = []
            if show_labels and pd.notna(row.get("name")):
                parts.append(str(row["name"]))
            if show_conf and pd.notna(row.get("confidence")):
                parts.append(f"{row['confidence']:.2f}")
            if parts:
                label = " ".join(parts)
                (tw, lh), _ = cv2.getTextSize(label, font, fs, th)
                cv2.rectangle(overlay, (x1, y1 - lh - 4), (x1 + tw + 4, y1), color, -1)
                cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                            font, fs, (255, 255, 255), th, cv2.LINE_AA)

    return overlay


# =============================================================================
#  TRAITEMENT DES MASQUES  (split / fusion)
# =============================================================================

def process_and_split_masks(results,
                             class_names: dict,
                             img_height: int,
                             img_width: int, *,
                             use_fusion: bool = False,
                             iou_thresh: float = 0.05,
                             ioa_thresh: float = 0.95) -> pd.DataFrame:
    """
    Extrait les masques YOLO-segment et les convertit en enregistrements
    DataFrame. Chaque contour connexe devient un objet indépendant (split).
    Si use_fusion=True, applique deux étapes de fusion par classe.

    Returns:
        pd.DataFrame
    """
    log = logging.getLogger("LYRA.masks")
    if not results or not results[0].boxes:
        return pd.DataFrame()

    boxes_obj = results[0].boxes
    masks_obj = results[0].masks if hasattr(results[0], "masks") else None
    n = len(boxes_obj)
    batch = 10 if n > 500 else (20 if n > 200 else (30 if n > 100 else n))
    log.debug(f"Processing {n} detections in batches of {batch}")

    all_data = []
    for start in range(0, n, batch):
        end = min(start + batch, n)
        if start > 0 and start % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for i in range(start, end):
            cls_id = int(boxes_obj.cls[i].item())
            name   = class_names.get(cls_id, "unknown")
            score  = float(boxes_obj.conf[i].item())

            if masks_obj is None or not hasattr(masks_obj, "data") or len(masks_obj.data) <= i:
                continue
            try:
                mask_t = masks_obj.data[i].cpu().numpy().squeeze()
                if mask_t.shape != (img_height, img_width):
                    mask_t = cv2.resize(mask_t, (img_width, img_height),
                                        interpolation=cv2.INTER_LINEAR)
                mask_bin = (mask_t > 0.5).astype(np.uint8)
                del mask_t
                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                del mask_bin
                for cnt in contours:
                    if cnt.shape[0] < 3:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    all_data.append({
                        "xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h,
                        "confidence":  score,
                        "class":       cls_id,
                        "name":        name,
                        "area":        float(cv2.contourArea(cnt)),
                        "contours":    json.dumps([cnt.reshape(-1, 2).tolist()]),
                        "object_type": "mask",
                    })
            except Exception as e:
                log.warning(f"Mask {i} processing failed: {e}")
                gc.collect()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not all_data:
        return pd.DataFrame()

    log.debug(f"Initial split: {len(all_data)} objects")

    if use_fusion and len(all_data) > 1:
        log.debug(f"Applying fusion (IoU={iou_thresh:.2f}, IoA={ioa_thresh:.2f})")
        by_class: dict[int, list] = defaultdict(list)
        for obj in all_data:
            by_class[obj["class"]].append(obj)
        fused = []
        for cls_id, objs in by_class.items():
            if len(objs) > 1:
                merged  = merge_highly_overlapping_masks(objs, img_height, img_width, iou_thresh)
                cleaned = cleanup_contained_masks(merged, img_height, img_width, ioa_thresh)
                fused.extend(cleaned)
            else:
                fused.extend(objs)
        log.debug(f"After fusion: {len(fused)} objects")
        all_data = fused

    gc.collect()
    return pd.DataFrame(all_data)


def merge_highly_overlapping_masks(detections: list[dict],
                                   height: int,
                                   width: int,
                                   iou_thresh: float = 0.05) -> list[dict]:
    """Stage 1 — fusionne les masques de même classe avec IoU > seuil."""
    if len(detections) < 2:
        return detections
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    is_merged = [False] * len(detections)
    result = []
    for i, det in enumerate(detections):
        if is_merged[i]:
            continue
        cur_mask = np.zeros((height, width), dtype=np.uint8)
        _fill_mask(cur_mask, det.get("contours", "[]"))
        for j in range(i + 1, len(detections)):
            if is_merged[j] or detections[j]["class"] != det["class"]:
                continue
            oth_mask = np.zeros((height, width), dtype=np.uint8)
            _fill_mask(oth_mask, detections[j].get("contours", "[]"))
            union = np.sum(cv2.bitwise_or(cur_mask, oth_mask))
            if union > 0:
                iou = np.sum(cv2.bitwise_and(cur_mask, oth_mask)) / union
                if iou > iou_thresh:
                    cur_mask     = cv2.bitwise_or(cur_mask, oth_mask)
                    is_merged[j] = True
        contours, _ = cv2.findContours(cur_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cnt.shape[0] < 3:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            result.append({**det,
                "xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h,
                "area": float(cv2.contourArea(cnt)),
                "contours": json.dumps([cnt.reshape(-1, 2).tolist()]),
            })
    return result


def cleanup_contained_masks(detections: list[dict],
                             height: int,
                             width: int,
                             ioa_thresh: float = 0.95) -> list[dict]:
    """Stage 2 — supprime les petits masques contenus dans les grands (même classe)."""
    if len(detections) < 2:
        return detections
    detections.sort(key=lambda x: x["area"], reverse=True)
    keep = [True] * len(detections)
    for i in range(len(detections)):
        if not keep[i]:
            continue
        mask_i = np.zeros((height, width), dtype=np.uint8)
        _fill_mask(mask_i, detections[i].get("contours", "[]"))
        for j in range(i + 1, len(detections)):
            if not keep[j] or detections[j]["class"] != detections[i]["class"]:
                continue
            area_j = detections[j]["area"]
            if area_j == 0:
                continue
            mask_j = np.zeros((height, width), dtype=np.uint8)
            _fill_mask(mask_j, detections[j].get("contours", "[]"))
            if np.sum(cv2.bitwise_and(mask_i, mask_j)) / area_j > ioa_thresh:
                keep[j] = False
    return [d for k, d in zip(keep, detections) if k]


def _fill_mask(canvas: np.ndarray, contours_json: str) -> None:
    """Helper interne : dessine les contours JSON sur un canvas uint8."""
    if not contours_json or contours_json == "[]":
        return
    try:
        for pts in json.loads(contours_json):
            if len(pts) >= 3:
                cv2.fillPoly(canvas, [np.array(pts, dtype=np.int32)], 1)
    except (json.JSONDecodeError, ValueError):
        pass


# =============================================================================
#  ARBORESCENCE DE SORTIE
# =============================================================================

def create_project_dirs_structure(outdir: str,
                                   project_id: str, *,
                                   display_overlay: bool = False,
                                   model_task: str = "detect") -> None:
    """Crée l'arborescence de sortie du projet."""
    log   = logging.getLogger("LYRA.io")
    dproj = os.path.join(outdir, project_id)
    if os.path.isdir(dproj):
        log.warning(f"Output dir '{dproj}' already exists — files may be overwritten.")
    else:
        os.makedirs(dproj, mode=0o755)
    if display_overlay:
        subdir = "img/bounding_boxes" if model_task == "detect" else "img/masks"
        os.makedirs(os.path.join(dproj, subdir), mode=0o755, exist_ok=True)


# =============================================================================
#  CLASSE YOLO  —  LYRADetection
# =============================================================================

class LYRADetection:
    """
    Gère le chargement du modèle YOLO, l'inférence, l'averaging temporel
    et les métriques qualité.

    Supporte automatiquement les tâches 'detect' et 'segment' :
    la tâche est détectée à partir du modèle chargé (model.task).
    """

    def __init__(self, weights_path: str | None, *,
                 conf_thresh: float = 0.5,
                 iou_thresh: float = 0.3,
                 device: str = "cpu",
                 use_retina_masks: bool = False,
                 models_dir: str | None = None):
        self.log = logging.getLogger("LYRA.detection")

        self.device          = device
        self.model           = None
        self.model_task      = "unknown"
        self.class_names     : dict[int, str] = {}
        self.models_dir      = Path(models_dir) if models_dir else Path("models").resolve()

        self.conf_thresh      = conf_thresh
        self.iou_thresh       = iou_thresh
        self.use_retina_masks = use_retina_masks

        # Averaging temporel
        self.enable_averaging      = False
        self.window_size           = 5
        self.consistency_threshold = 0.8
        self.detection_history     : list[dict] = []

        # État courant
        self.class_counts        : dict[str, int]   = {}
        self.class_averages      : dict[str, float] = {}
        self.class_avg_confidence: dict[str, float] = {}
        self.avg_confidence      : float            = 0.0

        if weights_path:
            self.load_model(weights_path, device)

    # ── Chargement ────────────────────────────────────────────────────────

    def load_model(self, model_path: str, device: str | None = None) -> bool:
        """
        Charge un modèle YOLO depuis un fichier .pt.
        Détecte automatiquement la tâche (detect / segment).

        Returns:
            True si le chargement a réussi.
        """
        from ultralytics import YOLO

        path = Path(model_path)
        if not path.is_file():
            self.log.error(f"Model file not found: {model_path}")
            return False
        try:
            self.model = YOLO(str(path))
            target = device or self.device
            self.model.to(target)
            self.device      = target
            self.model_task  = getattr(self.model, "task", "unknown")
            self.class_names = dict(self.model.names)
            self.log.info(
                f"Model loaded: {path.name}  task={self.model_task}  "
                f"classes={len(self.class_names)}  device={self.device}"
            )
            self.log.debug(f"Classes: {', '.join(self.class_names.values())}")
            return True
        except Exception as e:
            self.log.error(f"Failed to load model '{model_path}': {e}")
            self.model = None
            return False

    def get_available_models(self) -> list[str]:
        """Retourne la liste des fichiers .pt dans models_dir."""
        if not self.models_dir.exists():
            self.log.warning(f"Model directory not found: {self.models_dir}")
            return []
        files = [f.name for f in self.models_dir.glob("*.pt")]
        if not files:
            self.log.warning(f"No .pt files found in {self.models_dir}")
        return files

    # ── Inférence ─────────────────────────────────────────────────────────

    def detect_objects(self, img: np.ndarray,
                       img_path: str | None = None):
        """
        Lance l'inférence YOLO sur une image BGR.
        Gère automatiquement le fallback mémoire (retina → standard).

        Returns:
            Résultats YOLO bruts.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        imgsz = self.model.__dict__.get("overrides", {}).get("imgsz", 640)
        half  = "cuda" in str(self.device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        kwargs = dict(
            source=img, imgsz=imgsz, conf=self.conf_thresh, iou=self.iou_thresh,
            device=self.device, verbose=False, half=half, max_det=10_000,
        )
        try:
            return self.model.predict(retina_masks=self.use_retina_masks, **kwargs)
        except (RuntimeError, Exception) as e:
            if "memory" in str(e).lower():
                self.log.warning("OOM with retina masks — retrying with standard masks.")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                results = self.model.predict(retina_masks=False, **kwargs)
                self.log.info("Fallback to standard masks succeeded.")
                return results
            raise

    # ── Résultats + averaging ─────────────────────────────────────────────

    def process_detection_results(self, results) -> dict:
        """
        Traite les résultats YOLO, met à jour l'averaging et retourne
        les métriques consolidées.

        Returns:
            dict : class_counts, class_averages, avg_confidence,
                   class_confidences, stability
        """
        current_counts: dict[str, int]   = {}
        conf_sums:      dict[str, float] = {}
        conf_cnts:      dict[str, int]   = {}
        total_conf = 0.0
        n_det      = 0

        if results and len(results) > 0:
            for box in results[0].boxes:
                cls   = int(box.cls[0])
                label = self.class_names.get(cls, str(cls))
                conf  = float(box.conf[0])
                current_counts[label] = current_counts.get(label, 0) + 1
                conf_sums[label]      = conf_sums.get(label, 0.0) + conf
                conf_cnts[label]      = conf_cnts.get(label, 0) + 1
                total_conf += conf
                n_det      += 1

        self.avg_confidence      = total_conf / n_det if n_det > 0 else 0.0
        self.class_avg_confidence = {c: conf_sums[c] / conf_cnts[c] for c in conf_sums}

        if self.enable_averaging:
            self.detection_history.append(current_counts)
            if len(self.detection_history) > self.window_size:
                self.detection_history.pop(0)
            if len(self.detection_history) >= self.window_size:
                self.class_counts = self._calculate_averaged_counts()
                stability         = self._calculate_stability()
            else:
                self.class_counts   = current_counts
                self.class_averages = {k: float(v) for k, v in current_counts.items()}
                stability           = 0.5
        else:
            self.class_counts   = current_counts
            self.class_averages = {k: float(v) for k, v in current_counts.items()}
            stability           = 1.0

        return {
            "class_counts":      self.class_counts,
            "class_averages":    self.class_averages,
            "avg_confidence":    self.avg_confidence,
            "class_confidences": self.class_avg_confidence,
            "stability":         stability,
        }

    def _calculate_averaged_counts(self) -> dict[str, int]:
        """Moyenne glissante avec filtre de consistance."""
        rounded: dict[str, int]   = {}
        raw:     dict[str, float] = {}
        all_cls: set[str]         = set()
        for frame in self.detection_history:
            all_cls.update(frame.keys())
        for cls in all_cls:
            appearances = sum(1 for f in self.detection_history if cls in f)
            consistency = appearances / len(self.detection_history)
            if consistency >= self.consistency_threshold:
                total       = sum(f.get(cls, 0) for f in self.detection_history)
                avg_val     = total / len(self.detection_history)
                raw[cls]     = avg_val
                rounded[cls] = round(avg_val)
                self.log.debug(
                    f"[AVG] '{cls}': {appearances}/{len(self.detection_history)} frames  "
                    f"avg={avg_val:.2f} → {rounded[cls]}"
                )
            else:
                self.log.debug(
                    f"[FILTERED] '{cls}': consistency={consistency:.2f} < {self.consistency_threshold}"
                )
        self.class_averages = raw
        return rounded

    def _calculate_stability(self) -> float:
        """Score de stabilité [0–1] basé sur la variance du nombre total d'objets."""
        if not self.detection_history:
            return 1.0
        totals = [sum(f.values()) for f in self.detection_history]
        mean_v = sum(totals) / len(totals)
        if mean_v == 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - (max(totals) - min(totals)) / mean_v))

    # ── Qualité ───────────────────────────────────────────────────────────

    def get_quality_score(self, stability: float) -> tuple[float, str, str]:
        """
        Score qualité = 70 % confiance + 30 % stabilité.

        Returns:
            (score 0–100, label texte, couleur hex)
        """
        score = (self.avg_confidence * 0.7 + stability * 0.3) * 100
        if score > 80:   return score, "Excellent", "#4CAF50"
        elif score > 60: return score, "Good",      "#FFA500"
        elif score > 0:  return score, "Poor",      "#F44336"
        else:            return 0.0,   "Not active","#F44336"

    # ── Contrôle de l'averaging ───────────────────────────────────────────

    def set_averaging_enabled(self, enabled: bool) -> None:
        self.enable_averaging = enabled
        if not enabled:
            self.detection_history.clear()
        self.log.debug(f"Averaging {'ON' if enabled else 'OFF'}")

    def set_window_size(self, size: int) -> None:
        self.window_size = size
        self.detection_history.clear()

    # ── Export & validation ───────────────────────────────────────────────

    def extract_detection_data(self, results) -> list[dict]:
        """
        Extrait les détections sous forme de liste de dicts pour l'export.
        Calcule la surface depuis le masque si disponible (tâche segment).

        Returns:
            list[dict] : class, confidence, bounding_box, surface
        """
        detections = []
        if not results or len(results) == 0:
            return detections
        res   = results[0]
        boxes = getattr(res, "boxes", None)
        masks = getattr(res, "masks", None)
        if boxes is None:
            return detections
        for i, box in enumerate(boxes):
            cls_id  = int(box.cls[0])
            surface = float("nan")
            if masks is not None and hasattr(masks, "data") and len(masks.data) > i:
                surface = float(masks.data[i].cpu().numpy().sum())
            detections.append({
                "class":        self.class_names.get(cls_id, str(cls_id)),
                "confidence":   float(box.conf[0]),
                "bounding_box": box.xyxy[0].cpu().numpy().tolist(),
                "surface":      surface,
            })
        return detections

    @staticmethod
    def validate_detection_data(detections: list[dict]) -> list[str]:
        """
        Valide une liste de détections.

        Returns:
            list[str] : messages d'erreur (vide si tout est valide)
        """
        issues = []
        for det in detections:
            if not (0 <= det.get("confidence", 0) <= 1):
                issues.append(f"Invalid confidence: {det.get('confidence')}")
            bbox = det.get("bounding_box", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                if x1 >= x2 or y1 >= y2:
                    issues.append(f"Invalid bbox: {bbox}")
            surf = det.get("surface", 0)
            if not math.isnan(surf) and surf < 0:
                issues.append(f"Negative surface: {surf}")
        return issues


# =============================================================================
#  CLASSE SAM2  —  LYRASegmentation
# =============================================================================

class LYRASegmentation:
    """
    Segmentation SAM2 sur des annotations existantes (boxes, polygones, masques).
    Prend en entrée le CSV produit par detection_workflow et raffine les contours.
    """

    def __init__(self, device: str = "cpu"):
        self.log = logging.getLogger("LYRA.segmentation")
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-large", device=device
        )
        self.device = torch.device(device)
        self.log.info(f"SAM2 loaded on {device}")

    # ── Segmentation ──────────────────────────────────────────────────────

    def objects_segmentation(self,
                              image: np.ndarray,
                              annotations: list[dict]) -> tuple[np.ndarray, list[dict]]:
        """
        Lance SAM2 sur une image et une liste d'annotations.

        Args:
            image       : image RGB (np.ndarray)
            annotations : liste de dicts avec object_type in {'box','polygon','mask'}

        Returns:
            (masks_array N×H×W uint8, annotations filtrées et enrichies)
        """
        img_height, img_width = image.shape[:2]
        self.predictor.set_image(image)

        masks_list: list[np.ndarray] = []
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else torch.no_grad()
        )

        with torch.inference_mode(), autocast_ctx:
            for idx, ann in enumerate(annotations):
                object_type = ann.get("object_type", "unknown").lower()
                cleaned_mask = self._process_annotation(
                    ann, idx, object_type, img_height, img_width
                )
                if cleaned_mask is not None and np.sum(cleaned_mask) > 0:
                    masks_list.append(cleaned_mask.astype(np.uint8))
                    ann["_processed_mask_added"] = True
                else:
                    ann["_processed_mask_added"] = False

        processed = [a for a in annotations if a.pop("_processed_mask_added", False)]
        self.log.debug(f"Segmentation: {len(masks_list)} masks / {len(annotations)} annotations")

        if not masks_list:
            return np.array([]), []
        try:
            masks_array = np.array(masks_list)
        except ValueError as e:
            self.log.error(f"Cannot stack masks: {e}")
            return np.array([]), []

        # Sécurité alignement
        n = min(masks_array.shape[0], len(processed))
        return masks_array[:n], processed[:n]

    def _process_annotation(self,
                             ann: dict,
                             idx: int,
                             object_type: str,
                             img_height: int,
                             img_width: int) -> np.ndarray | None:
        """
        Dispatche le traitement selon le type d'annotation.
        Retourne le masque nettoyé (H×W uint8) ou None en cas d'échec.
        """
        try:
            if object_type == "box":
                return self._process_box(ann)
            elif object_type == "polygon":
                return self._process_polygon(ann, img_height, img_width)
            elif object_type == "mask":
                return self._process_existing_mask(ann, img_height, img_width)
            else:
                self.log.warning(f"Unknown annotation type '{object_type}' at index {idx}")
                return None
        except Exception as e:
            self.log.warning(f"Error processing annotation {idx} ({object_type}): {e}")
            return None

    def _process_box(self, ann: dict) -> np.ndarray | None:
        """Prédit un masque depuis une bounding box."""
        box = np.array([ann["xmin"], ann["ymin"], ann["xmax"], ann["ymax"]])
        pred_masks, _, _ = self.predictor.predict(box=box[None, :], multimask_output=False)
        mask = self._to_numpy_mask(pred_masks)
        ann["object_type"] = "mask"
        return mask

    def _process_polygon(self, ann: dict,
                          img_height: int, img_width: int) -> np.ndarray | None:
        """Prédit un masque depuis un polygone (via input mask SAM2)."""
        contours_json = ann.get("contours", "")
        if pd.isna(contours_json) or not contours_json:
            return None
        contours = json.loads(contours_json)
        if not isinstance(contours, list) or len(contours) < 3:
            return None
        pts = np.array(contours, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return None

        # Masque d'entrée → redimensionné 256×256 pour SAM2
        input_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(input_mask, [pts.reshape(-1, 1, 2)], 1)
        input_mask_256 = cv2.resize(input_mask, (256, 256),
                                     interpolation=cv2.INTER_NEAREST).astype(np.float32)
        if np.sum(input_mask_256) == 0:
            return None

        tensor = torch.from_numpy(input_mask_256).unsqueeze(0).unsqueeze(0)
        if self.device.type == "cuda":
            tensor = tensor.to(self.device)

        pred_masks, _, _ = self.predictor.predict(mask_input=tensor, multimask_output=False)
        mask = self._to_numpy_mask(pred_masks)
        if mask is None:
            return None

        # Recadrage si nécessaire
        if mask.shape != (img_height, img_width):
            mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

        # Conserver uniquement la plus grande composante connexe
        cleaned = self._keep_largest_component(mask)
        ann["object_type"] = "mask"
        return cleaned

    def _process_existing_mask(self, ann: dict,
                                img_height: int, img_width: int) -> np.ndarray | None:
        """Reconstruit un masque binaire depuis les contours JSON ou un masque existant."""
        if "mask" in ann and ann["mask"] is not None:
            m = ann["mask"]
            if isinstance(m, np.ndarray):
                if m.shape != (img_height, img_width):
                    m = cv2.resize(m, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                ann["object_type"] = "mask"
                return m
            return None

        contours_json = ann.get("contours", "")
        if pd.isna(contours_json) or not contours_json:
            return None
        try:
            contours = json.loads(contours_json)
            canvas = np.zeros((img_height, img_width), dtype=np.uint8)
            for c in (contours if isinstance(contours[0][0], list) else [contours]):
                arr = np.array(c, dtype=np.int32)
                if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 3:
                    cv2.fillPoly(canvas, [arr.reshape(-1, 1, 2)], 1)
            ann["object_type"] = "mask"
            return canvas
        except Exception:
            return None

    @staticmethod
    def _to_numpy_mask(pred_masks) -> np.ndarray | None:
        """Convertit la sortie du predictor SAM2 en numpy uint8."""
        if pred_masks is None:
            return None
        if isinstance(pred_masks, torch.Tensor):
            return pred_masks[0].cpu().numpy().astype(np.uint8)
        if isinstance(pred_masks, np.ndarray):
            return pred_masks[0].astype(np.uint8)
        return None

    @staticmethod
    def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
        """Conserve uniquement la plus grande composante connexe du masque."""
        if np.sum(mask) == 0:
            return mask
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8, ltype=cv2.CV_32S
        )
        if n_labels <= 1:
            return mask
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned = np.zeros_like(mask)
        cleaned[labels == largest] = 1
        return cleaned


# ── Overlays SAM2 ──────────────────────────────────────────────────────────────

def add_masks_on_image(masks: np.ndarray, img: np.ndarray) -> None:
    """Superpose tous les masques en rouge sur l'image (in-place)."""
    if masks is None or len(masks) == 0:
        return
    combined = np.zeros(img.shape[:2], dtype=np.uint8)
    for mask in masks:
        if mask is not None and isinstance(mask, np.ndarray) and mask.ndim == 2:
            bm = (mask > 0).astype(np.uint8)
            if bm.shape != img.shape[:2]:
                bm = cv2.resize(bm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            combined = np.logical_or(combined, bm).astype(np.uint8)
    img[combined == 1] = [0, 0, 255]


def create_multicolored_masks_image(masks: np.ndarray) -> np.ndarray:
    """Génère une image colorisée avec une couleur aléatoire par masque."""
    if masks is None or not isinstance(masks, np.ndarray) or masks.ndim != 3 or masks.shape[0] == 0:
        return np.zeros((512, 512, 3), dtype=np.uint8)
    n, h, w = masks.shape
    canvas  = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(n):
        color = np.random.randint(100, 256, size=3, dtype=np.uint8)
        bm    = (masks[i] > 0).astype(np.uint8)
        if bm.shape != (h, w):
            bm = cv2.resize(bm, (w, h), interpolation=cv2.INTER_NEAREST)
        for c in range(3):
            canvas[:, :, c][bm == 1] = color[c]
    return canvas


# =============================================================================
#  WORKFLOW DÉTECTION  (YOLO → CSV)
# =============================================================================

def detection_workflow(dct_args: dict, *,
                       gui: bool = True,
                       progress_callback=None,
                       status_callback=None,
                       log_callback=None) -> None:
    """
    Pipeline de détection complet : images → YOLO → CSV.

    Args:
        dct_args          : paramètres du pipeline (voir ci-dessous)
        gui               : conservé pour rétrocompatibilité
        progress_callback : callable(float 0–1)
        status_callback   : callable(str)
        log_callback      : callable(str)

    Clés attendues dans dct_args
    -----------------------------
    input_directory, output_directory, project_id
    model_path, conf_thresh, overlap_thresh
    gpu (0|1), cpu (int)
    add_overlay (0|1), show_bbox, show_conf, show_mask, show_labels
    use_retina_masks (0|1)
    use_fusion (0|1), fuse_iou_thresh, phagocyte_ioa_thresh
    """
    log = logging.getLogger("LYRA.workflow")

    def _status(msg: str):
        log.info(msg)
        if status_callback:
            status_callback(msg)

    def _log(msg: str):
        log.debug(msg)
        if log_callback:
            log_callback(msg)

    def _bool(key, default=0):
        val = dct_args.get(key, default)
        return bool(val) if isinstance(val, int) else val

    # ── Paramètres ────────────────────────────────────────────────────────
    add_overlay = _bool("add_overlay", 1)
    show_bbox   = _bool("show_bbox",   1)
    show_conf   = _bool("show_conf",   0)
    show_mask   = _bool("show_mask",   1)
    show_labels = _bool("show_labels", 1)
    use_retina  = _bool("use_retina_masks", 0)
    use_fusion  = _bool("use_fusion",  0)
    fuse_iou    = float(dct_args.get("fuse_iou_thresh",      0.05))
    phago_ioa   = float(dct_args.get("phagocyte_ioa_thresh", 0.95))

    if utils:
        utils.set_cpu_usage(dct_args.get("cpu", os.cpu_count()))
    device = _resolve_device(_bool("gpu", 1))

    # ── Chargement du modèle ──────────────────────────────────────────────
    _status(f"Loading model on {device}…")
    model = LYRADetection(
        dct_args["model_path"],
        conf_thresh=float(dct_args["conf_thresh"]),
        iou_thresh=float(dct_args["overlap_thresh"]),
        device=device,
        use_retina_masks=use_retina,
    )
    task = model.model_task
    log.info(f"Model task: {task}")

    if task == "detect":
        show_mask  = False
        use_retina = False
        use_fusion = False

    create_project_dirs_structure(
        dct_args["output_directory"], dct_args["project_id"],
        display_overlay=add_overlay, model_task=task,
    )

    # ── Liste des images ──────────────────────────────────────────────────
    if utils:
        lst_imgs = utils.list_image_files(dct_args["input_directory"])
    else:
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        lst_imgs = [
            str(p) for p in Path(dct_args["input_directory"]).rglob("*")
            if p.suffix.lower() in exts
        ]
    total = len(lst_imgs)
    if total == 0:
        raise ValueError(f"No images found in: {dct_args['input_directory']}")
    _status(f"Found {total} images.")

    PLACEHOLDER = {k: (np.nan if isinstance(v, float) else ("" if isinstance(v, str) else np.nan))
                   for k, v in _OUTPUT_DEFAULTS.items()}
    PLACEHOLDER["object_type"] = ""

    lst_df_final = []

    # ── Boucle principale ─────────────────────────────────────────────────
    for idx, img_path in enumerate(lst_imgs):
        if progress_callback:
            progress_callback(idx / total)
        fname = os.path.basename(img_path)
        _status(f"[{idx+1}/{total}]  {fname}")

        img = cv2.imread(img_path)
        if img is None:
            log.warning(f"Cannot read image: {img_path}")
            continue

        results = model.detect_objects(img, img_path)
        df_proc = pd.DataFrame()

        if task == "segment":
            h, w    = img.shape[:2]
            df_proc = process_and_split_masks(
                results, model.class_names, h, w,
                use_fusion=use_fusion, iou_thresh=fuse_iou, ioa_thresh=phago_ioa,
            )
        elif task == "detect" and results and results[0].boxes:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            scores  = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            df_proc = pd.DataFrame({
                "xmin":        boxes[:, 0].astype(int),
                "ymin":        boxes[:, 1].astype(int),
                "xmax":        boxes[:, 2].astype(int),
                "ymax":        boxes[:, 3].astype(int),
                "confidence":  scores,
                "class":       classes,
                "name":        [model.class_names.get(c, "unknown") for c in classes],
                "area":        np.nan,
                "contours":    "[]",
                "object_type": "box",
            })

        img_path_rel = os.path.relpath(img_path, dct_args["input_directory"])
        n_det        = 0 if df_proc.empty else int((df_proc.get("object_type", pd.Series()) != "").sum())
        _log(f"  → {n_det} objects")

        if df_proc.empty:
            df_proc = pd.DataFrame([PLACEHOLDER.copy()])
        df_proc["img_id"] = img_path_rel

        # ── Overlay ───────────────────────────────────────────────────────
        if add_overlay and n_det > 0:
            try:
                if task == "detect":
                    ann = results[0].plot(conf=show_conf, labels=show_labels,
                                          boxes=show_bbox, line_width=2, font_size=10)
                else:
                    ann = draw_detections_yolo_style(
                        img, df_proc,
                        show_bbox=show_bbox, show_labels=show_labels,
                        show_conf=show_conf, show_mask=show_mask,
                    )
                subdir = "img/bounding_boxes" if task == "detect" else "img/masks"
                fout   = os.path.join(
                    dct_args["output_directory"], dct_args["project_id"],
                    subdir, f"{dct_args['project_id']}_{fname}",
                )
                os.makedirs(os.path.dirname(fout), exist_ok=True)
                quality = 75 if n_det > 200 else 85
                cv2.imwrite(fout, ann, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                del ann
            except Exception as e:
                log.warning(f"Overlay failed for {fname}: {e}")

        lst_df_final.append(df_proc)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Consolidation & sauvegarde ────────────────────────────────────────
    _status("Saving results…")
    if progress_callback:
        progress_callback(0.98)
    if not lst_df_final:
        raise ValueError("No images were processed successfully.")

    df_global = pd.concat(lst_df_final, ignore_index=True)
    is_real   = df_global["object_type"].notna() & (df_global["object_type"] != "")
    df_global.loc[is_real, "object_id"] = (
        df_global[is_real].groupby("img_id").cumcount() + 1
    )
    df_global["object_id"] = df_global["object_id"].fillna(0)

    df_global = _enforce_df_schema(df_global, dct_args["project_id"])

    out_dir  = os.path.join(dct_args["output_directory"], dct_args["project_id"])
    out_path = os.path.join(out_dir, f"{dct_args['project_id']}_globinfo.csv")
    _save_results(df_global, out_path, dct_args["project_id"],
                  input_directory=os.path.abspath(dct_args["input_directory"]))

    _status(f"Done — {int(is_real.sum())} objects across {total} images.")
    if progress_callback:
        progress_callback(1.0)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
#  WORKFLOW SEGMENTATION  (CSV de détection → SAM2 → CSV)
# =============================================================================

def segmentation_workflow(dct_args: dict, *,
                           progress_callback=None,
                           status_callback=None) -> None:
    """
    Pipeline de segmentation : CSV de détection → SAM2 → CSV enrichi.

    Args:
        dct_args          : paramètres du pipeline
        progress_callback : callable(float 0–1)
        status_callback   : callable(str)

    Clés attendues dans dct_args
    -----------------------------
    input_file   : chemin vers le *_globinfo.csv produit par detection_workflow
    gpu (0|1), cpu (int), add_overlay (0|1)
    """
    log = logging.getLogger("LYRA.seg_workflow")

    def _status(msg: str):
        log.info(msg)
        if status_callback:
            status_callback(msg)

    def _bool(key, default=0):
        val = dct_args.get(key, default)
        return bool(val) if isinstance(val, int) else val

    # ── Paramètres ────────────────────────────────────────────────────────
    add_overlay = _bool("add_overlay")
    if utils:
        utils.set_cpu_usage(dct_args.get("cpu", os.cpu_count()))
    device = _resolve_device(_bool("gpu", 1))

    input_file = dct_args["input_file"]
    project_id = os.path.basename(input_file).replace("_globinfo.csv", "")
    csv_dir    = os.path.dirname(input_file)
    dct_args["project_id"] = project_id
    dct_args["input_dir"]  = csv_dir

    # ── Lecture du CSV (avec commentaire metadata) ────────────────────────
    input_directory: str | None = None
    try:
        with open(input_file, "r") as f:
            first = f.readline().strip()
            if first.startswith("# input_directory:"):
                input_directory = first.split(":", 1)[1].strip()
    except Exception:
        pass

    try:
        df = pd.read_csv(input_file, comment="#")
    except Exception as e:
        log.error(f"Cannot read CSV '{input_file}': {e}")
        return

    if "img_id" not in df.columns:
        log.error(f"'img_id' column missing in {input_file}. Got: {list(df.columns)}")
        return

    # ── Répertoire overlay ────────────────────────────────────────────────
    dpath_overlay = os.path.join(csv_dir, project_id, "img", "segmentation")
    if add_overlay:
        os.makedirs(dpath_overlay, exist_ok=True)

    # ── Chargement SAM2 ───────────────────────────────────────────────────
    try:
        seg_model = LYRASegmentation(device=device)
    except Exception as e:
        log.error(f"Cannot load SAM2: {e}")
        return

    lst_img_paths   = df["img_id"].unique()
    total           = len(lst_img_paths)
    all_annotations = []

    # ── Boucle principale ─────────────────────────────────────────────────
    for idx, img_path_rel in enumerate(lst_img_paths):
        if progress_callback:
            progress_callback(idx / total)
        _status(f"[{idx+1}/{total}]  {img_path_rel}")

        img_path_full = _find_image_path(
            img_path_rel,
            search_dirs=[input_directory, csv_dir],
        )
        if not img_path_full:
            log.warning(f"Image not found: '{img_path_rel}' — skipped.")
            continue

        img_bgr = cv2.imread(img_path_full)
        if img_bgr is None:
            log.warning(f"Cannot read image: {img_path_full}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Préparer les annotations de cette image
        img_df = df[df["img_id"] == img_path_rel].copy()
        for col in ("xmin", "ymin", "xmax", "ymax"):
            if col in img_df.columns:
                img_df[col] = pd.to_numeric(img_df[col], errors="coerce").fillna(0).astype(int)

        annotations_in = [
            row.to_dict() for _, row in img_df.iterrows()
            if pd.notna(row.get("object_type"))
        ]
        if not annotations_in:
            log.debug(f"No annotations for {img_path_rel} — skipped.")
            continue
        for ann in annotations_in:
            ann["object_type"] = str(ann["object_type"]).lower()

        # ── Inférence SAM2 ────────────────────────────────────────────────
        try:
            masks, processed = seg_model.objects_segmentation(img_rgb, annotations_in)
        except Exception as e:
            log.warning(f"SAM2 error on '{img_path_rel}': {e}")
            continue

        if masks is None or len(masks) == 0 or len(masks) != len(processed):
            log.warning(f"No valid masks for {img_path_rel}")
            continue

        # ── Post-traitement ───────────────────────────────────────────────
        for i, ann in enumerate(processed):
            mask = masks[i]
            ann["area"] = float(np.sum(mask))

            contours_cv, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_pts: list[np.ndarray] = []
            contours_data: list        = []
            for cnt in contours_cv:
                sq = cnt.squeeze()
                if sq.ndim == 1 and sq.shape[0] == 2:
                    sq = sq.reshape(1, 2)
                if sq.ndim == 2 and sq.shape[0] >= 3:
                    valid_pts.append(sq)
                    contours_data.append(sq.tolist())

            if valid_pts:
                all_pts     = np.vstack(valid_pts)
                ann["contours"] = json.dumps(contours_data)
                ann["xmin"]     = int(np.min(all_pts[:, 0]))
                ann["ymin"]     = int(np.min(all_pts[:, 1]))
                ann["xmax"]     = int(np.max(all_pts[:, 0]))
                ann["ymax"]     = int(np.max(all_pts[:, 1]))
            else:
                ann["contours"] = json.dumps([])
                ann["xmin"] = ann["ymin"] = ann["xmax"] = ann["ymax"] = 0

            ann["object_type"] = "mask"
            all_annotations.append(ann)

        # ── Overlay ───────────────────────────────────────────────────────
        if add_overlay:
            try:
                fname  = os.path.basename(img_path_rel)
                fstem  = os.path.splitext(fname)[0]
                over   = img_bgr.copy()
                add_masks_on_image(masks, over)
                cv2.imwrite(os.path.join(dpath_overlay, f"{project_id}_{fname}"), over)
                cv2.imwrite(
                    os.path.join(dpath_overlay, f"{project_id}_{fstem}_colored.png"),
                    create_multicolored_masks_image(masks),
                )
            except Exception as e:
                log.warning(f"Overlay error for {img_path_rel}: {e}")

        gc.collect()

    # ── Consolidation & sauvegarde ────────────────────────────────────────
    if progress_callback:
        progress_callback(0.98)

    if not all_annotations:
        log.error("No annotations processed — output not created.")
        return

    df_new = pd.DataFrame(all_annotations)
    df_new["project_id"] = project_id
    df_new["object_id"]  = df_new.groupby("img_id").cumcount() + 1
    df_new = _enforce_df_schema(df_new, project_id)

    out_path = os.path.join(csv_dir, f"{project_id}_segmentation_globinfo.csv")
    _save_results(df_new, out_path, project_id, input_directory=input_directory)

    _status(f"Done — {len(all_annotations)} masks across {total} images.")
    if progress_callback:
        progress_callback(1.0)
