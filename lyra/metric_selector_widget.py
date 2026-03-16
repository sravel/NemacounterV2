"""
Metric Selector Widget
======================
Standalone PyQt6 widget for selecting output metric groups.
Can be embedded in the Export tab or any QDialog.

Usage
-----
    selector = MetricSelectorWidget()
    selector.groups_changed.connect(my_callback)
    layout.addWidget(selector)

    # Get current selection
    groups = selector.selected_groups()   # ['basic', 'area_detail', ...]

The widget emits groups_changed(list[str]) whenever the selection changes.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QTextEdit,
    QVBoxLayout, QWidget,
)

# Lazy import — metric_config may not be installed in all environments
try:
    from lyra.metric_config import METRIC_GROUPS, PRESETS, DEFAULT_PRESET
except ImportError:
    # Fallback inline definition (minimal, no shape/spatial/density)
    METRIC_GROUPS = {
        "basic":               {"label": "Basic",              "always": True,  "requires": [], "separate": False,
                                "columns": ["count","conf_mean","conf_std","area_mean","area_std"],
                                "description": "Count, mean/std confidence and area."},
        "area_detail":         {"label": "Area detail",        "always": False, "requires": [], "separate": False,
                                "columns": ["area_min","area_max","area_median","area_cv"],
                                "description": "Min, max, median area and coefficient of variation."},
        "confidence_detail":   {"label": "Confidence detail",  "always": False, "requires": [], "separate": False,
                                "columns": ["conf_min","conf_max"],
                                "description": "Min and max confidence scores."},
        "bbox":                {"label": "Bbox geometry",      "always": False, "requires": [], "separate": False,
                                "columns": ["width_mean","width_std","height_mean","height_std","aspect_ratio_mean"],
                                "description": "Width, height and aspect ratio."},
        "shape":               {"label": "Shape metrics",      "always": False, "requires": ["contours"], "separate": False,
                                "columns": ["perimeter_mean","perimeter_std","circularity_mean","solidity_mean","eccentricity_mean"],
                                "description": "Contour-based shape descriptors (mask/polygon only)."},
        "spatial":             {"label": "Spatial distribution","always": False, "requires": [], "separate": False,
                                "columns": ["centroid_x_mean","centroid_y_mean","spatial_dispersion","nn_dist_mean"],
                                "description": "Centroid positions and nearest-neighbour distance."},
        "density":             {"label": "Density & coverage", "always": False, "requires": ["image_dims"], "separate": False,
                                "columns": ["density_per_mpx","coverage_pct"],
                                "description": "Object density and mask coverage % (reads image files)."},
        "aggregate":           {"label": "Project aggregate",  "always": False, "requires": [], "separate": True,
                                "columns": ["total_count","n_images","count_per_img_mean","count_per_img_std","count_per_img_min","count_per_img_max"],
                                "description": "Cross-image totals per class (separate CSV)."},
    }
    PRESETS = {
        "minimal":  ["basic"],
        "standard": ["basic", "area_detail", "confidence_detail", "bbox"],
        "full":     list(METRIC_GROUPS.keys()),
    }
    DEFAULT_PRESET = "standard"

# Require warning labels
_REQUIRE_LABELS = {
    "contours":    "⚠ requires mask/polygon annotations",
    "image_dims":  "⚠ reads image files (slower)",
}


class MetricSelectorWidget(QWidget):
    """
    Compact widget with preset buttons + per-group checkboxes + column preview.

    Layout:
      [ Presets: Minimal · Standard · Full ]
      ┌─────────────────────────────────────┐
      │ ☑ Basic (always)                    │
      │ ☑ Area detail    ☑ Confidence detail│
      │ ☑ Bbox geometry  □ Shape metrics ⚠ │
      │ □ Spatial        □ Density ⚠       │
      │ □ Project aggregate (→ separate CSV)│
      └─────────────────────────────────────┘
      Output columns: count, conf_mean, conf_std, area_mean, area_std,
                       area_min, ...
    """

    groups_changed = pyqtSignal(list)   # emitted on every checkbox toggle

    def __init__(self, preset: str = DEFAULT_PRESET, parent=None):
        super().__init__(parent)
        self._checkboxes: dict[str, QCheckBox] = {}
        self._build_ui(preset)

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self, preset: str):
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Preset buttons ────────────────────────────────────────────────
        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        preset_row.addWidget(QLabel("Preset :"))
        for pname, plabel in [("minimal", "Minimal"), ("standard", "Standard"), ("full", "Full")]:
            btn = QPushButton(plabel)
            btn.setFixedHeight(24)
            btn.setCheckable(True)
            btn.setChecked(pname == preset)
            btn.setStyleSheet(
                "QPushButton{font-size:11px;padding:0 10px;border-radius:3px;"
                "background:var(--color-background-secondary);"
                "color:var(--color-text-secondary);"
                "border:1px solid var(--color-border-secondary);}"
                "QPushButton:checked{background:#1e66f5;color:#fff;border-color:#1e66f5;}"
                "QPushButton:hover:!checked{background:var(--color-background-tertiary);}"
            )
            btn.clicked.connect(lambda _, p=pname: self._apply_preset(p))
            preset_row.addWidget(btn)
            # Keep reference to un-check others
            btn.setObjectName(f"preset_{pname}")
        preset_row.addStretch()
        root.addLayout(preset_row)
        self._preset_row = preset_row   # stored to iterate buttons

        # ── Group checkboxes ──────────────────────────────────────────────
        grp_box = QGroupBox("Metric groups")
        grp_lay = QVBoxLayout(grp_box)
        grp_lay.setSpacing(3)

        # Layout: two columns for the non-always groups
        # Row 0: Basic (full width, greyed out)
        # Row 1-N: pairs

        preset_groups = PRESETS.get(preset, PRESETS[DEFAULT_PRESET])
        keys = list(METRIC_GROUPS.keys())
        i = 0
        while i < len(keys):
            key  = keys[i]
            info = METRIC_GROUPS[key]

            if info.get("always"):
                # Full-width, disabled checkbox
                cb = QCheckBox(f"{info['label']}  (always included)")
                cb.setChecked(True)
                cb.setEnabled(False)
                cb.setToolTip(info.get("description", ""))
                grp_lay.addWidget(cb)
                self._checkboxes[key] = cb
                i += 1
                continue

            # Pair two per row when possible
            row_lay = QHBoxLayout()
            row_lay.setSpacing(12)
            for _ in range(2):
                if i >= len(keys):
                    break
                k    = keys[i]
                info = METRIC_GROUPS[k]
                if info.get("always"):
                    break  # don't mix "always" into a pair row

                label = info["label"]
                reqs  = info.get("requires", [])
                sep   = info.get("separate", False)

                # Compose display text
                parts = [label]
                if sep:
                    parts.append("→ separate CSV")
                warn = "  ".join(_REQUIRE_LABELS[r] for r in reqs if r in _REQUIRE_LABELS)

                cell = QVBoxLayout()
                cell.setSpacing(0)
                cb = QCheckBox("  ".join(parts))
                cb.setChecked(k in preset_groups)
                cb.setToolTip(info.get("description", ""))
                cb.stateChanged.connect(self._on_checkbox_changed)
                self._checkboxes[k] = cb
                cell.addWidget(cb)
                if warn:
                    wlbl = QLabel(warn)
                    wlbl.setStyleSheet("color:#f38ba8;font-size:10px;margin-left:20px;")
                    cell.addWidget(wlbl)
                row_lay.addLayout(cell, 1)
                i += 1

            row_lay.addStretch()
            grp_lay.addLayout(row_lay)

        root.addWidget(grp_box)

        # ── Column preview ────────────────────────────────────────────────
        prev_box = QGroupBox("Output columns preview")
        prev_lay = QVBoxLayout(prev_box)
        self._preview = QLabel("")
        self._preview.setWordWrap(True)
        self._preview.setStyleSheet(
            "font-size:11px;color:var(--color-text-secondary);"
            "font-family:var(--font-mono);"
        )
        prev_lay.addWidget(self._preview)
        root.addWidget(prev_box)

        self._update_preview()

    # ── Preset logic ──────────────────────────────────────────────────────────

    def _apply_preset(self, preset: str):
        """Apply a preset and update all checkboxes."""
        groups = PRESETS.get(preset, [])
        for k, cb in self._checkboxes.items():
            if not METRIC_GROUPS[k].get("always"):
                cb.blockSignals(True)
                cb.setChecked(k in groups)
                cb.blockSignals(False)

        # Update preset button checked states
        for btn in self._iter_preset_btns():
            pname = btn.objectName().replace("preset_", "")
            btn.setChecked(pname == preset)

        self._update_preview()
        self.groups_changed.emit(self.selected_groups())

    def _iter_preset_btns(self):
        for i in range(self._preset_row.count()):
            item = self._preset_row.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), QPushButton):
                yield item.widget()

    # ── Change handling ───────────────────────────────────────────────────────

    def _on_checkbox_changed(self, _state):
        # Deselect all preset buttons (selection is now custom)
        for btn in self._iter_preset_btns():
            btn.setChecked(False)
        # Check if current selection matches a preset
        sel = set(self.selected_groups())
        for pname, pgroups in PRESETS.items():
            if sel == set(pgroups):
                for btn in self._iter_preset_btns():
                    if btn.objectName() == f"preset_{pname}":
                        btn.setChecked(True)
        self._update_preview()
        self.groups_changed.emit(self.selected_groups())

    # ── Preview ───────────────────────────────────────────────────────────────

    def _update_preview(self):
        groups = self.selected_groups()
        cols: list[str] = []
        sep_cols: list[str] = []
        for g in groups:
            info = METRIC_GROUPS.get(g)
            if not info:
                continue
            if info.get("separate"):
                sep_cols.extend(info["columns"])
            else:
                for c in info["columns"]:
                    if c not in cols:
                        cols.append(c)
        text = "  ·  ".join(cols) or "(none)"
        if sep_cols:
            text += f"\n+ separate aggregate: {'  ·  '.join(sep_cols)}"
        self._preview.setText(text)

    # ── Public API ────────────────────────────────────────────────────────────

    def selected_groups(self) -> list[str]:
        """Returns the list of currently selected group keys (including 'basic')."""
        return [k for k, cb in self._checkboxes.items() if cb.isChecked()]

    def set_groups(self, groups: list[str]):
        """Programmatically set selected groups."""
        for k, cb in self._checkboxes.items():
            if not METRIC_GROUPS[k].get("always"):
                cb.blockSignals(True)
                cb.setChecked(k in groups)
                cb.blockSignals(False)
        self._update_preview()
