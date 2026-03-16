"""
Common utilities — LYRA
================================
Shared DataFrame helpers, summary table construction and metric computation.

Public API
----------
create_summary_table(df, project_id, selected_groups=None, image_paths=None)
    Per-image × per-class summary with configurable metric groups.
    selected_groups: list of group names from metric_config.METRIC_GROUPS.
                     None → uses DEFAULT_PRESET ("standard").

create_aggregate_table(df, project_id)
    Cross-image aggregate (one row per class across all images).
    Written to a separate *_aggregate.csv file.

get_config_info(fpath)
    Read a .ini config file.

Deprecated (kept for backward compatibility, not used internally):
    read_image, create_boxes, create_global_table,
    calculation_per_group, merge_multiple_dataframes
"""
from __future__ import annotations

import json
import logging
import os

import cv2
import numpy as np
import pandas as pd
from functools import reduce
import configparser

log = logging.getLogger("LYRA.common")

# ---------------------------------------------------------------------------
# Lazy import of metric_config (avoids hard dependency if file is missing)
# ---------------------------------------------------------------------------
try:
    from lyra.metric_config import (
        METRIC_GROUPS, DEFAULT_PRESET, groups_for_preset, columns_for_groups,
    )
    _HAS_METRIC_CONFIG = True
except ImportError:
    _HAS_METRIC_CONFIG = False
    DEFAULT_PRESET = "standard"


# =============================================================================
#  BACKWARD-COMPATIBLE HELPERS (kept, not actively used in new pipeline)
# =============================================================================

def read_image(img_path: str) -> np.ndarray:
    """Read an image as RGB numpy array."""
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_boxes(df: pd.DataFrame) -> list:
    return df[["xmin", "ymin", "xmax", "ymax"]].values.tolist()


def create_global_table(lst_df: list, project_id: str) -> pd.DataFrame:
    df = pd.concat(lst_df).reset_index(drop=True)
    df["project_id"] = project_id
    cols = ["project_id", "img_id", "object_id", "class", "name",
            "xmin", "ymin", "xmax", "ymax", "confidence"]
    return df[[c for c in cols if c in df.columns]]


def calculation_per_group(grouped_df, colname: str,
                           measurements=None, rnd: int = 3) -> pd.DataFrame:
    if measurements is None:
        measurements = ["mean", "std"]
    df = grouped_df.agg({colname: measurements})
    df.columns = [f"{colname}_{m}" for m in measurements]
    return np.round(df, rnd).reset_index()


def merge_multiple_dataframes(lst_df: list, lst_k: list,
                               how: str = "inner") -> pd.DataFrame:
    return reduce(
        lambda l, r: pd.merge(l, r, on=lst_k, how=how), lst_df
    )


def get_config_info(fpath: str):
    try:
        config = configparser.ConfigParser()
        config.read(fpath)
        return config
    except Exception:
        raise FileNotFoundError(f"Config file not found: {fpath}")


# =============================================================================
#  METRIC COMPUTATION HELPERS  (private)
# =============================================================================

def _area_detail(grp: pd.DataFrame) -> dict:
    """area_min, area_max, area_median, area_cv."""
    a = grp["area"].dropna()
    if a.empty:
        return {"area_min": np.nan, "area_max": np.nan,
                "area_median": np.nan, "area_cv": np.nan}
    cv = float(a.std() / a.mean()) if a.mean() != 0 else np.nan
    return {
        "area_min":    float(a.min()),
        "area_max":    float(a.max()),
        "area_median": float(a.median()),
        "area_cv":     round(cv, 4),
    }


def _confidence_detail(grp: pd.DataFrame) -> dict:
    """conf_min, conf_max."""
    c = grp["confidence"].dropna()
    if c.empty:
        return {"conf_min": np.nan, "conf_max": np.nan}
    return {"conf_min": float(c.min()), "conf_max": float(c.max())}


def _bbox_metrics(grp: pd.DataFrame) -> dict:
    """width/height/aspect_ratio statistics from bounding boxes."""
    widths  = (grp["xmax"] - grp["xmin"]).dropna()
    heights = (grp["ymax"] - grp["ymin"]).dropna()
    if widths.empty:
        return {k: np.nan for k in ["width_mean", "width_std",
                                     "height_mean", "height_std",
                                     "aspect_ratio_mean"]}
    # Aspect ratio: avoid divide-by-zero
    ar = (widths / heights.replace(0, np.nan)).dropna()
    return {
        "width_mean":        round(float(widths.mean()),  2),
        "width_std":         round(float(widths.std()),   2),
        "height_mean":       round(float(heights.mean()), 2),
        "height_std":        round(float(heights.std()),  2),
        "aspect_ratio_mean": round(float(ar.mean()),      4) if not ar.empty else np.nan,
    }


def _shape_metrics(grp: pd.DataFrame) -> dict:
    """
    Morphological metrics from contour data (mask/polygon only).

    Computes per-object:
      perimeter   = sum of cv2.arcLength over all contour parts
      circularity = 4π * area / perimeter²  (1 = perfect circle)
      solidity    = area / convex_hull_area  (1 = fully convex)
      eccentricity from cv2.fitEllipse (0 = circle, 1 = line)

    Returns per-group means and stds.
    Rows with missing/invalid contours are silently skipped.
    """
    perimeters, circularities, solidities, eccentricities = [], [], [], []

    for _, row in grp.iterrows():
        raw = row.get("contours")
        if not raw or not isinstance(raw, str) or raw in ("", "[]", "nan"):
            continue
        try:
            clist = json.loads(raw)
        except Exception:
            continue
        if not isinstance(clist, list) or not clist:
            continue

        area_val = float(row.get("area", 0) or 0)
        total_perim = 0.0
        all_pts: list[np.ndarray] = []

        for pts in clist:
            if len(pts) < 3:
                continue
            cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            total_perim += cv2.arcLength(cnt, True)
            sq = cnt.squeeze()
            if sq.ndim == 2 and sq.shape[0] >= 3:
                all_pts.append(sq)

        if total_perim > 0:
            perimeters.append(total_perim)
            circ = (4 * np.pi * area_val) / (total_perim ** 2) if total_perim > 0 else np.nan
            circularities.append(min(1.0, circ))  # cap at 1 for numerical noise

        if all_pts:
            all_arr = np.vstack(all_pts).astype(np.int32)
            try:
                hull      = cv2.convexHull(all_arr)
                hull_area = cv2.contourArea(hull)
                sol = area_val / hull_area if hull_area > 0 else np.nan
                solidities.append(min(1.0, max(0.0, sol)))
            except Exception:
                pass

            # Eccentricity from fitted ellipse (needs ≥ 5 points)
            if len(all_arr) >= 5:
                try:
                    (_, _), (ma, mi), _ = cv2.fitEllipse(all_arr)
                    if ma > 0:
                        ecc = np.sqrt(1 - (mi / ma) ** 2) if ma >= mi else 0.0
                        eccentricities.append(float(ecc))
                except Exception:
                    pass

    def _stat(lst):
        return (round(float(np.mean(lst)), 4) if lst else np.nan)

    return {
        "perimeter_mean":     _stat(perimeters),
        "perimeter_std":      round(float(np.std(perimeters)), 4) if perimeters else np.nan,
        "circularity_mean":   _stat(circularities),
        "solidity_mean":      _stat(solidities),
        "eccentricity_mean":  _stat(eccentricities),
    }


def _spatial_metrics(grp: pd.DataFrame) -> dict:
    """
    Centroid statistics and nearest-neighbour distance.

    centroid_x/y_mean : mean position of object centres (pixels)
    spatial_dispersion : sqrt(var_x + var_y) — spread of centroids
    nn_dist_mean       : mean distance to nearest neighbour (pixels)
                         Returns NaN if fewer than 2 objects.
    """
    cx = ((grp["xmin"] + grp["xmax"]) / 2).dropna().values
    cy = ((grp["ymin"] + grp["ymax"]) / 2).dropna().values
    n  = len(cx)

    if n == 0:
        return {k: np.nan for k in ["centroid_x_mean", "centroid_y_mean",
                                     "spatial_dispersion", "nn_dist_mean"]}

    dispersion = float(np.sqrt(np.var(cx) + np.var(cy))) if n > 1 else 0.0

    # Nearest-neighbour distances (brute force, fine for typical counts < 1000)
    nn_dist = np.nan
    if n >= 2:
        pts  = np.column_stack([cx, cy])
        dists = []
        for i in range(n):
            d = np.linalg.norm(pts - pts[i], axis=1)
            d[i] = np.inf  # exclude self
            dists.append(d.min())
        nn_dist = float(np.mean(dists))

    return {
        "centroid_x_mean":    round(float(np.mean(cx)), 2),
        "centroid_y_mean":    round(float(np.mean(cy)), 2),
        "spatial_dispersion": round(dispersion,          2),
        "nn_dist_mean":       round(nn_dist, 2) if not np.isnan(nn_dist) else np.nan,
    }


def _density_metrics(grp: pd.DataFrame,
                     img_path: str | None,
                     img_dims_cache: dict) -> dict:
    """
    Density and coverage metrics.

    density_per_mpx : count / (image_area_in_Mpx)
    coverage_pct    : sum(object_area) / image_area × 100

    img_dims_cache is a {img_path: (w, h)} dict updated in place.
    Returns NaN columns if image dimensions are unavailable.
    """
    _nan = {"density_per_mpx": np.nan, "coverage_pct": np.nan}
    if not img_path:
        return _nan

    if img_path not in img_dims_cache:
        try:
            img = cv2.imread(img_path)
            if img is None:
                img_dims_cache[img_path] = None
            else:
                h, w = img.shape[:2]
                img_dims_cache[img_path] = (w, h)
        except Exception:
            img_dims_cache[img_path] = None

    dims = img_dims_cache.get(img_path)
    if dims is None:
        return _nan

    w, h    = dims
    img_area = w * h
    if img_area == 0:
        return _nan

    count_val = len(grp)
    area_sum  = grp["area"].fillna(0).sum()

    return {
        "density_per_mpx": round(count_val / (img_area / 1_000_000), 4),
        "coverage_pct":    round(float(area_sum / img_area * 100), 4),
    }


# =============================================================================
#  MAIN PUBLIC FUNCTIONS
# =============================================================================

def create_summary_table(
    df_global: pd.DataFrame,
    project_id: str,
    selected_groups: list[str] | None = None,
    image_paths: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Build a per-image × per-class summary table with configurable metric groups.

    Parameters
    ----------
    df_global       : globinfo DataFrame (must contain at least img_id, name, confidence)
    project_id      : project identifier (added as a column)
    selected_groups : list of group keys from metric_config.METRIC_GROUPS.
                      None → DEFAULT_PRESET ("standard").
    image_paths     : {img_id_relative: absolute_path} mapping, used for density metrics.
                      Required only when "density" group is selected.

    Returns
    -------
    pd.DataFrame with columns [project_id, img_id, class_name, ...metric_columns...]
    """
    # ── Resolve groups ────────────────────────────────────────────────────────
    if selected_groups is None:
        if _HAS_METRIC_CONFIG:
            selected_groups = groups_for_preset(DEFAULT_PRESET)
        else:
            selected_groups = ["basic", "area_detail", "confidence_detail", "bbox"]

    # Ensure "basic" is always present
    if "basic" not in selected_groups:
        selected_groups = ["basic"] + list(selected_groups)

    do_area_detail   = "area_detail"    in selected_groups
    do_conf_detail   = "confidence_detail" in selected_groups
    do_bbox          = "bbox"           in selected_groups
    do_shape         = "shape"          in selected_groups
    do_spatial       = "spatial"        in selected_groups
    do_density       = "density"        in selected_groups

    # ── Ensure area column ────────────────────────────────────────────────────
    if "area" not in df_global.columns:
        df_global = df_global.copy()
        df_global["area"] = (
            (df_global["xmax"] - df_global["xmin"]) *
            (df_global["ymax"] - df_global["ymin"])
        )

    img_dims_cache: dict = {}
    rows: list[dict] = []

    for (img_id, class_name), grp in df_global.groupby(["img_id", "name"]):
        grp = grp.copy()
        row: dict = {
            "project_id": project_id,
            "img_id":     img_id,
            "class_name": class_name,
        }

        # ── Basic ──────────────────────────────────────────────────────────
        a = grp["area"].dropna()
        c = grp["confidence"].dropna()
        row["count"]      = len(grp)
        row["conf_mean"]  = round(float(c.mean()), 6) if not c.empty else np.nan
        row["conf_std"]   = round(float(c.std()),  6) if not c.empty else np.nan
        row["area_mean"]  = round(float(a.mean()), 6) if not a.empty else np.nan
        row["area_std"]   = round(float(a.std()),  6) if not a.empty else np.nan

        # ── Optional groups ────────────────────────────────────────────────
        if do_area_detail:
            row.update(_area_detail(grp))

        if do_conf_detail:
            row.update(_confidence_detail(grp))

        if do_bbox:
            row.update(_bbox_metrics(grp))

        if do_shape:
            row.update(_shape_metrics(grp))

        if do_spatial:
            row.update(_spatial_metrics(grp))

        if do_density:
            img_abs = None
            if image_paths:
                img_abs = image_paths.get(str(img_id))
            row.update(_density_metrics(grp, img_abs, img_dims_cache))

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["project_id", "img_id", "class_name"])

    df_summary = pd.DataFrame(rows)

    # ── Column order ──────────────────────────────────────────────────────────
    base_cols = ["project_id", "img_id", "class_name"]
    if _HAS_METRIC_CONFIG:
        metric_cols = columns_for_groups(selected_groups)
    else:
        metric_cols = [c for c in df_summary.columns if c not in base_cols]

    ordered = base_cols + [c for c in metric_cols if c in df_summary.columns]
    # Append any remaining columns not explicitly ordered
    ordered += [c for c in df_summary.columns if c not in ordered]
    return df_summary[ordered]


def create_aggregate_table(df_global: pd.DataFrame,
                            project_id: str) -> pd.DataFrame:
    """
    Cross-image aggregate — one row per class across all images.

    Columns: project_id, class_name, total_count, n_images,
             count_per_img_mean, count_per_img_std,
             count_per_img_min, count_per_img_max

    Written to a separate *_aggregate.csv file by the caller.
    """
    if "name" not in df_global.columns or "img_id" not in df_global.columns:
        return pd.DataFrame()

    counts_per_img = (
        df_global.groupby(["img_id", "name"]).size().reset_index(name="count")
    )
    rows = []
    for class_name, grp in counts_per_img.groupby("name"):
        c = grp["count"]
        rows.append({
            "project_id":           project_id,
            "class_name":           class_name,
            "total_count":          int(c.sum()),
            "n_images":             int(len(c)),
            "count_per_img_mean":   round(float(c.mean()), 3),
            "count_per_img_std":    round(float(c.std()),  3),
            "count_per_img_min":    int(c.min()),
            "count_per_img_max":    int(c.max()),
        })
    return pd.DataFrame(rows)
