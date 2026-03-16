"""
Export Engine
=============
Convertit un *_globinfo.csv LYRA vers deux formats d'export :

  export_roboflow_coco(globinfo_path, ...)
    → *_roboflow_coco.json  (COCO JSON compatible upload Roboflow)

  export_yolo_train(globinfo_path, output_dir, ...)
    → output_dir/
        images/train/   (copies des images sources)
        labels/train/   (*.txt un fichier par image)
        data.yaml       (configuration yolo train)

Format labels YOLO :
  - box     : "class_id  cx cy w h"               (coordonnées normalisées 0–1)
  - mask/polygon : "class_id  x1 y1 x2 y2 ... xn yn"  (normalisées 0–1)

Les deux fonctions sont sans dépendance GUI et acceptent des callbacks optionnels :
  progress_callback(float 0–1)
  status_callback(str)

Utilisation depuis un QThread (main_gui.py → ExportWorker) :

    from lyra.export_engine import export_roboflow_coco, export_yolo_train
    out = export_roboflow_coco(path, progress_callback=self.progress.emit,
                                     status_callback=self.status.emit)
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil

import cv2
import numpy as np
import pandas as pd

log = logging.getLogger("LYRA.export")

# Colonnes minimales requises dans le globinfo CSV
_REQUIRED_COLS = ["img_id", "xmin", "ymin", "xmax", "ymax", "name", "object_type"]


# =============================================================================
#  HELPERS INTERNES
# =============================================================================

def _read_globinfo(globinfo_path: str) -> tuple[pd.DataFrame, str | None]:
    """
    Lit un *_globinfo.csv (avec commentaire "# input_directory: ..." éventuel).

    Returns:
        (df, input_directory)  —  input_directory est None si absent du CSV.

    Raises:
        ValueError  si des colonnes requises sont manquantes.
        FileNotFoundError / pd.errors.* si le fichier est illisible.
    """
    input_directory: str | None = None
    try:
        with open(globinfo_path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
            if first.startswith("# input_directory:"):
                input_directory = first.split(":", 1)[1].strip()
    except Exception:
        pass

    df = pd.read_csv(globinfo_path, comment="#")

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    df.dropna(subset=["name"], inplace=True)
    return df, input_directory


def _resolve_image(img_id_rel: str, search_dirs: list[str]) -> str | None:
    """
    Retrouve le chemin absolu d'une image parmi plusieurs dossiers candidats.
    Essaie aussi le chemin tel quel (absolu ou relatif au CWD) en dernier recours.
    """
    candidates = [os.path.join(d, img_id_rel) for d in search_dirs if d]
    candidates.append(img_id_rel)
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _image_wh(img_path: str) -> tuple[int, int]:
    """
    Retourne (width, height) d'une image.
    Utilise cv2 (déjà disponible dans le projet).
    Retourne (0, 0) en cas d'échec.
    """
    try:
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    except Exception:
        pass
    return 0, 0


def _build_search_dirs(globinfo_path: str, input_directory: str | None) -> list[str]:
    """Construit la liste des dossiers de recherche d'images, sans doublons."""
    base_dir  = os.path.dirname(globinfo_path)
    image_dir = input_directory or os.path.dirname(base_dir)
    seen: set[str] = set()
    result: list[str] = []
    for d in [input_directory, base_dir, image_dir]:
        if d and d not in seen:
            seen.add(d)
            result.append(d)
    return result


# =============================================================================
#  EXPORT 1 — ROBOFLOW COCO JSON
# =============================================================================

def export_roboflow_coco(
    globinfo_path: str,
    *,
    progress_callback=None,
    status_callback=None,
) -> str:
    """
    Convertit un *_globinfo.csv en COCO JSON compatible Roboflow.

    Gère les deux types d'annotations :
      - 'box'           → bbox COCO [x, y, w, h]
      - 'mask'/'polygon'→ segmentation COCO (un contour par annotation)

    Le fichier JSON est écrit dans le même dossier que le CSV source,
    avec le suffixe  _roboflow_coco.json.

    Args:
        globinfo_path     : chemin absolu ou relatif du *_globinfo.csv
        progress_callback : callable(float 0–1), optionnel
        status_callback   : callable(str), optionnel

    Returns:
        Chemin absolu du fichier JSON produit.

    Raises:
        ValueError       si des colonnes sont manquantes
        IOError / OSError en cas d'erreur de lecture/écriture
    """
    def _status(msg: str):
        log.info(msg)
        if status_callback:
            status_callback(msg)

    _status("Lecture du CSV…")
    df, input_directory = _read_globinfo(globinfo_path)
    search_dirs = _build_search_dirs(globinfo_path, input_directory)

    # ── Catégories ────────────────────────────────────────────────────────────
    unique_names   = sorted(df["name"].unique())
    name_to_cat_id = {n: i + 1 for i, n in enumerate(unique_names)}

    coco: dict = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cat_id, "name": name, "supercategory": "object"}
            for name, cat_id in name_to_cat_id.items()
        ],
    }

    grouped = list(df.groupby("img_id"))
    total   = len(grouped)
    ann_id  = 1
    img_ctr = 1

    for idx, (img_id_rel, group) in enumerate(grouped):
        if progress_callback:
            progress_callback(idx / total)
        _status(f"[{idx + 1}/{total}]  {img_id_rel}")

        img_path = _resolve_image(img_id_rel, search_dirs)
        if not img_path:
            log.warning(f"Image introuvable : {img_id_rel!r} — ignorée.")
            continue

        w, h = _image_wh(img_path)
        if w == 0 or h == 0:
            log.warning(f"Impossible de lire les dimensions de {img_path} — ignorée.")
            continue

        coco["images"].append({
            "id":        img_ctr,
            "file_name": os.path.basename(img_id_rel),
            "width":     w,
            "height":    h,
        })
        current_img_id = img_ctr
        img_ctr += 1

        for _, row in group.iterrows():
            cat_id   = name_to_cat_id.get(row["name"])
            if cat_id is None:
                continue
            obj_type = str(row.get("object_type", "box")).lower()

            # ── Bounding box ─────────────────────────────────────────────────
            if obj_type == "box":
                xmin = float(row["xmin"])
                ymin = float(row["ymin"])
                xmax = float(row["xmax"])
                ymax = float(row["ymax"])
                wb, hb = xmax - xmin, ymax - ymin
                if wb <= 0 or hb <= 0:
                    continue
                coco["annotations"].append({
                    "id":          ann_id,
                    "image_id":    current_img_id,
                    "category_id": cat_id,
                    "bbox":        [xmin, ymin, wb, hb],
                    "area":        float(wb * hb),
                    "iscrowd":     0,
                })
                ann_id += 1

            # ── Masque / polygone (un contour = une annotation) ───────────────
            elif obj_type in ("mask", "polygon"):
                contours_raw = row.get("contours")
                if not pd.notna(contours_raw) or not contours_raw:
                    continue
                try:
                    contours_list = json.loads(contours_raw)
                except (json.JSONDecodeError, TypeError):
                    log.warning(f"Contours JSON invalides pour {img_id_rel}")
                    continue
                if not isinstance(contours_list, list):
                    continue

                for pts in contours_list:
                    if len(pts) < 3:
                        continue
                    cnt_np = np.array(pts, dtype=np.int32)
                    x, y, wb, hb = cv2.boundingRect(cnt_np)
                    if wb <= 0 or hb <= 0:
                        continue
                    area = float(cv2.contourArea(cnt_np))
                    coco["annotations"].append({
                        "id":            ann_id,
                        "image_id":      current_img_id,
                        "category_id":   cat_id,
                        "bbox":          [float(x), float(y), float(wb), float(hb)],
                        "area":          area,
                        "segmentation":  [cnt_np.flatten().tolist()],
                        "iscrowd":       0,
                    })
                    ann_id += 1

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    _status("Écriture du JSON…")
    import json as _json
    out_path = os.path.splitext(globinfo_path)[0] + "_roboflow_coco.json"
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(coco, f, indent=2)

    summary = (
        f"Export terminé — {ann_id - 1} annotations, "
        f"{img_ctr - 1} images\n→ {out_path}"
    )
    _status(summary)
    if progress_callback:
        progress_callback(1.0)
    return out_path


# =============================================================================
#  EXPORT 2 — YOLO TRAIN FORMAT
# =============================================================================

def export_yolo_train(
    globinfo_path: str,
    output_dir: str,
    *,
    train_pct: int = 70,
    val_pct:   int = 15,
    test_pct:  int = 15,
    copy_images: bool = True,
    progress_callback=None,
    status_callback=None,
) -> str:
    """
    Exporte au format YOLO training (ultralytics) avec répartition train/val/test.

    Structure produite :
        output_dir/
          images/{train,val,test}/
          labels/{train,val,test}/
          data.yaml

    train_pct + val_pct + test_pct doit être <= 100.
    val peut être 0. test minimum 1 image si test_pct > 0.

    data.yaml compatible avec :
        yolo train data=<output_dir>/data.yaml model=yolo11n.pt epochs=100
    """
    def _status(msg: str):
        log.info(msg)
        if status_callback:
            status_callback(msg)

    _status("Lecture du CSV…")
    df, input_directory = _read_globinfo(globinfo_path)
    search_dirs = _build_search_dirs(globinfo_path, input_directory)

    # ── Classes ───────────────────────────────────────────────────────────────
    unique_names     = sorted(df["name"].unique())
    name_to_class_id = {n: i for i, n in enumerate(unique_names)}

    grouped = list(df.groupby("img_id"))
    total   = len(grouped)

    # ── Répartition train / val / test ────────────────────────────────────────
    # Normaliser les pourcentages pour qu'ils somment à 100
    pct_sum   = train_pct + val_pct + test_pct
    if pct_sum == 0:
        pct_sum = 100
    n_train = round(total * train_pct / pct_sum)
    n_test  = max(1, round(total * test_pct / pct_sum)) if test_pct > 0 else 0
    n_val   = total - n_train - n_test   # le reste va en val (peut être 0)
    n_val   = max(0, n_val)

    import random, math
    indices  = list(range(total))
    random.shuffle(indices)
    train_idx = set(indices[:n_train])
    val_idx   = set(indices[n_train:n_train + n_val])
    # test = tout le reste

    def _split_name(i: int) -> str:
        if i in train_idx: return "train"
        if i in val_idx:   return "val"
        return "test"

    # Créer les dossiers pour les splits non vides
    splits_used = {"train"}
    if n_val  > 0: splits_used.add("val")
    if n_test > 0: splits_used.add("test")
    for sp in splits_used:
        os.makedirs(os.path.join(output_dir, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", sp), exist_ok=True)

    _status(f"Répartition : {n_train} train / {n_val} val / {n_test} test")

    for idx, (img_id_rel, group) in enumerate(grouped):
        if progress_callback:
            progress_callback(idx / total)
        sp = _split_name(idx)
        _status(f"[{idx + 1}/{total}] [{sp}]  {img_id_rel}")

        img_path = _resolve_image(img_id_rel, search_dirs)
        if not img_path:
            log.warning(f"Image introuvable : {img_id_rel!r} — ignorée.")
            continue

        w, h = _image_wh(img_path)
        if w == 0 or h == 0:
            log.warning(f"Dimensions illisibles : {img_path} — ignorée.")
            continue

        fname = os.path.basename(img_id_rel)
        stem  = os.path.splitext(fname)[0]

        if copy_images:
            dest = os.path.join(output_dir, "images", sp, fname)
            if not os.path.exists(dest):
                shutil.copy2(img_path, dest)

        lines: list[str] = []
        for _, row in group.iterrows():
            class_id = name_to_class_id.get(row["name"])
            if class_id is None:
                continue
            obj_type = str(row.get("object_type", "box")).lower()

            if obj_type == "box":
                xmin, ymin = float(row["xmin"]), float(row["ymin"])
                xmax, ymax = float(row["xmax"]), float(row["ymax"])
                cx = max(0.0, min(1.0, (xmin + xmax) / 2.0 / w))
                cy = max(0.0, min(1.0, (ymin + ymax) / 2.0 / h))
                bw = max(0.0, min(1.0, (xmax - xmin) / w))
                bh = max(0.0, min(1.0, (ymax - ymin) / h))
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            elif obj_type in ("mask", "polygon"):
                contours_raw = row.get("contours")
                if not pd.notna(contours_raw) or not contours_raw:
                    continue
                try:
                    contours_list = json.loads(contours_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                for pts in contours_list:
                    if len(pts) < 3:
                        continue
                    normalized: list[float] = []
                    for px, py in pts:
                        normalized.append(max(0.0, min(1.0, float(px) / w)))
                        normalized.append(max(0.0, min(1.0, float(py) / h)))
                    if len(normalized) < 6:
                        continue
                    lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in normalized))

        lbl_path = os.path.join(output_dir, "labels", sp, f"{stem}.txt")
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ── data.yaml ─────────────────────────────────────────────────────────────
    _status("Écriture de data.yaml…")
    abs_out = os.path.abspath(output_dir)
    yaml_lines = [
        f"path: {abs_out}",
        "train: images/train",
    ]
    if n_val  > 0: yaml_lines.append("val:   images/val")
    if n_test > 0: yaml_lines.append("test:  images/test")
    yaml_lines += [
        f"nc: {len(unique_names)}",
        "names: [" + ", ".join(unique_names) + "]",
    ]
    yaml_path = os.path.join(output_dir, "data.yaml")
    try:
        import yaml
        data_dict = {
            "path": abs_out, "train": "images/train",
            "nc":   len(unique_names), "names": unique_names,
        }
        if n_val  > 0: data_dict["val"]  = "images/val"
        if n_test > 0: data_dict["test"] = "images/test"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_dict, f, allow_unicode=True, sort_keys=False)
    except ImportError:
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yaml_lines) + "\n")

    summary = (
        f"Export YOLO — {n_train} train / {n_val} val / {n_test} test, "
        f"{len(unique_names)} classes\n→ {output_dir}"
    )
    _status(summary)
    if progress_callback:
        progress_callback(1.0)
    return yaml_path


# =============================================================================
#  EXPORT 3 — ENHANCED METRICS REPORT
# =============================================================================

def export_metrics_report(
    globinfo_path: str,
    output_dir: str | None = None,
    *,
    selected_groups: list[str] | None = None,
    image_paths: dict[str, str] | None = None,
    progress_callback=None,
    status_callback=None,
) -> dict[str, str]:
    """
    Generates enhanced metric summary CSVs from a *_globinfo.csv.

    Two output files are always created next to the globinfo (or in output_dir):
      *_summary.csv         per-image × per-class table with selected metric groups
      *_aggregate.csv       cross-image aggregate per class

    Parameters
    ----------
    globinfo_path   : source *_globinfo.csv
    output_dir      : destination directory (defaults to globinfo directory)
    selected_groups : list of group keys — see metric_config.METRIC_GROUPS.
                      None → DEFAULT_PRESET ("standard")
    image_paths     : {img_id_rel: abs_path} mapping, needed for "density" group
    progress/status : optional callbacks

    Returns
    -------
    dict with keys "summary" and "aggregate" pointing to the written file paths.
    """
    def _status(msg: str):
        log.info(msg)
        if status_callback:
            status_callback(msg)

    _status("Reading CSV…")
    df, _ = _read_globinfo(globinfo_path)

    stem = os.path.splitext(os.path.basename(globinfo_path))[0]
    out  = output_dir or os.path.dirname(globinfo_path)
    os.makedirs(out, exist_ok=True)

    try:
        from lyra.common import create_summary_table, create_aggregate_table
    except ImportError:
        raise ImportError(
            "lyra.common not found. "
            "Ensure common.py is in the lyra package."
        )

    project_id = stem.replace("_globinfo", "").replace("_segmentation_globinfo", "")

    if progress_callback:
        progress_callback(0.2)

    _status("Computing summary metrics…")
    df_summary = create_summary_table(
        df, project_id,
        selected_groups=selected_groups,
        image_paths=image_paths,
    )
    summary_path = os.path.join(out, f"{stem}_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    log.info(f"Summary saved → {summary_path}")

    if progress_callback:
        progress_callback(0.7)

    _status("Computing aggregate table…")
    df_agg = create_aggregate_table(df, project_id)
    agg_path = os.path.join(out, f"{stem}_aggregate.csv")
    df_agg.to_csv(agg_path, index=False)
    log.info(f"Aggregate saved → {agg_path}")

    _status(
        f"Metrics export done — {len(df_summary)} rows in summary, "
        f"{len(df_agg)} classes in aggregate."
    )
    if progress_callback:
        progress_callback(1.0)

    return {"summary": summary_path, "aggregate": agg_path}
