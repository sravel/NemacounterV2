"""
Metric Configuration
====================
Defines the available metric groups for the summary output tables.
Pure Python — no Qt dependency. Imported by both common.py and the GUI widget.

Usage
-----
from lyra.metric_config import METRIC_GROUPS, PRESETS, DEFAULT_PRESET

Each entry in METRIC_GROUPS is a dict with:
  label       : human-readable group name (used in GUI)
  description : tooltip / explanation
  columns     : list of output column names this group adds
  always      : bool — if True, group cannot be deselected
  requires    : list of requirements: "contours", "image_dims" — shown as warnings in GUI
  separate    : bool — if True, written to a separate CSV file (not appended to main summary)
"""
from __future__ import annotations

# =============================================================================
#  GROUP DEFINITIONS
# =============================================================================

METRIC_GROUPS: dict[str, dict] = {

    "basic": {
        "label":       "Basic",
        "description": "Count, mean/std of confidence and object area.",
        "columns":     ["count", "conf_mean", "conf_std", "area_mean", "area_std"],
        "always":      True,
        "requires":    [],
        "separate":    False,
    },

    "area_detail": {
        "label":       "Area detail",
        "description": (
            "Extended area statistics: min, max, median and coefficient of variation "
            "(std/mean). Useful for detecting size heterogeneity within a class."
        ),
        "columns":     ["area_min", "area_max", "area_median", "area_cv"],
        "always":      False,
        "requires":    [],
        "separate":    False,
    },

    "confidence_detail": {
        "label":       "Confidence detail",
        "description": "Min and max detection confidence per image×class group.",
        "columns":     ["conf_min", "conf_max"],
        "always":      False,
        "requires":    [],
        "separate":    False,
    },

    "bbox": {
        "label":       "Bounding box geometry",
        "description": (
            "Width, height and aspect ratio (width/height) of detection boxes. "
            "Aspect ratio > 1 = wider than tall, < 1 = taller than wide."
        ),
        "columns":     ["width_mean", "width_std", "height_mean", "height_std",
                        "aspect_ratio_mean"],
        "always":      False,
        "requires":    [],
        "separate":    False,
    },

    "shape": {
        "label":       "Shape metrics",
        "description": (
            "Morphological descriptors computed from contour data. "
            "Circularity (4π·A/P²) ranges from 0 to 1, where 1 = perfect circle. "
            "Solidity (A/convex_hull_area) = 1 for fully convex objects. "
            "Eccentricity from fitted ellipse (0 = circle, 1 = line segment). "
            "Only available for mask/polygon annotations."
        ),
        "columns":     ["perimeter_mean", "perimeter_std",
                        "circularity_mean", "solidity_mean", "eccentricity_mean"],
        "always":      False,
        "requires":    ["contours"],
        "separate":    False,
    },

    "spatial": {
        "label":       "Spatial distribution",
        "description": (
            "Centroid coordinates (mean x, y of detection boxes), "
            "spatial dispersion (std of centroid positions) and "
            "mean nearest-neighbour distance between objects in each image."
        ),
        "columns":     ["centroid_x_mean", "centroid_y_mean",
                        "spatial_dispersion", "nn_dist_mean"],
        "always":      False,
        "requires":    [],
        "separate":    False,
    },

    "density": {
        "label":       "Density & coverage",
        "description": (
            "Objects per megapixel (count / image area in Mpx) and "
            "coverage percentage (sum of object areas / image area × 100). "
            "Requires reading image files to obtain pixel dimensions."
        ),
        "columns":     ["density_per_mpx", "coverage_pct"],
        "always":      False,
        "requires":    ["image_dims"],
        "separate":    False,
    },

    "aggregate": {
        "label":       "Project aggregate",
        "description": (
            "Cross-image summary per class: total object count, number of images, "
            "mean / std / min / max count per image. "
            "Written to a separate *_aggregate.csv file."
        ),
        "columns":     ["total_count", "n_images",
                        "count_per_img_mean", "count_per_img_std",
                        "count_per_img_min", "count_per_img_max"],
        "always":      False,
        "requires":    [],
        "separate":    True,  # separate output file
    },
}


# =============================================================================
#  PRESETS
# =============================================================================

PRESETS: dict[str, list[str]] = {
    "minimal":  ["basic"],
    "standard": ["basic", "area_detail", "confidence_detail", "bbox"],
    "full":     list(METRIC_GROUPS.keys()),
}

DEFAULT_PRESET: str = "standard"


# =============================================================================
#  HELPERS
# =============================================================================

def columns_for_groups(groups: list[str]) -> list[str]:
    """
    Returns the ordered list of output columns for a given set of group names.
    The 'aggregate' group is excluded (it has its own separate file).
    Keys that are not in METRIC_GROUPS are silently ignored.
    """
    cols: list[str] = []
    for g in groups:
        info = METRIC_GROUPS.get(g)
        if info and not info.get("separate"):
            for c in info["columns"]:
                if c not in cols:
                    cols.append(c)
    return cols


def groups_for_preset(preset: str) -> list[str]:
    """Returns the group list for a preset name (falls back to 'standard')."""
    return PRESETS.get(preset, PRESETS[DEFAULT_PRESET])
