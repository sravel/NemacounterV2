# edition.py - Professional Edition Interface for NemaCounter with Enhanced Features
# FIXED VERSION - Preserves original area values for unmodified annotations

import sys
import cv2
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import time
import csv
from collections import deque
from dataclasses import dataclass
from copy import deepcopy
import gc

from PyQt6.QtCore import (Qt, QThread, pyqtSignal, QPointF, QRectF,
                          QTimer, QEvent, QPoint, QObject, pyqtSlot)
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QColor, QPen,
                         QBrush, QPolygonF, QIcon, QKeySequence, QFont,
                         QTransform, QCursor, QPainterPath, QKeyEvent)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QListWidget, QListWidgetItem, QGraphicsScene,
                             QGraphicsView, QGraphicsItem, QGraphicsPolygonItem,
                             QGraphicsRectItem, QGraphicsEllipseItem,
                             QToolBar, QDockWidget, QMessageBox,
                             QFileDialog, QProgressBar, QStatusBar, QSlider,
                             QSpinBox, QGroupBox, QCheckBox, QComboBox,
                             QSplitter, QMenu, QDialog,
                             QDialogButtonBox, QGridLayout, QGraphicsPixmapItem,
                             QGraphicsPathItem, QGraphicsProxyWidget,
                             QInputDialog)

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Warning: SAM2 not available. Smart annotation will be disabled.")
    SAM2ImagePredictor = None

# Class color palette - consistent colors for each class
CLASS_COLORS = [
    QColor(255, 0, 0, 50),  # Red - more transparent
    QColor(0, 255, 0, 50),  # Green - more transparent
    QColor(0, 0, 255, 50),  # Blue - more transparent
    QColor(255, 255, 0, 50),  # Yellow - more transparent
    QColor(255, 0, 255, 50),  # Magenta - more transparent
    QColor(0, 255, 255, 50),  # Cyan - more transparent
    QColor(255, 128, 0, 50),  # Orange - more transparent
    QColor(128, 0, 255, 50),  # Purple - more transparent
    QColor(0, 128, 255, 50),  # Light Blue - more transparent
    QColor(255, 0, 128, 50),  # Pink - more transparent
]

EDGE_CLASS_COLORS = [
    QColor(255, 0, 0),  # Red
    QColor(0, 255, 0),  # Green
    QColor(0, 0, 255),  # Blue
    QColor(255, 255, 0),  # Yellow
    QColor(255, 0, 255),  # Magenta
    QColor(0, 255, 255),  # Cyan
    QColor(255, 128, 0),  # Orange
    QColor(128, 0, 255),  # Purple
    QColor(0, 128, 255),  # Light Blue
    QColor(255, 0, 128),  # Pink
]

# Default colors for annotation types
ANNOTATION_COLORS = {
    'box': QColor(255, 0, 0, 100),
    'polygon': QColor(0, 255, 0, 100),
    'mask': QColor(0, 0, 255, 100),
    'smart': QColor(255, 255, 0, 100)
}

EDGE_COLORS = {
    'box': QColor(255, 0, 0),
    'polygon': QColor(0, 255, 0),
    'mask': QColor(0, 0, 255),
    'smart': QColor(255, 255, 0)
}


@dataclass
class AnnotationState:
    """Represents a state in the annotation history for undo/redo"""
    annotations: List
    description: str
    timestamp: float


class UndoRedoManager:
    """Manages undo/redo operations"""

    def __init__(self, max_history=50):
        self.undo_stack = deque(maxlen=max_history)
        self.redo_stack = deque(maxlen=max_history)
        self.current_state = None

    def save_state(self, annotations, description="Action"):
        """Save current state to undo stack"""
        if self.current_state is not None:
            self.undo_stack.append(self.current_state)

        # Clear redo stack when new action is performed
        self.redo_stack.clear()

        # Deep copy annotations to preserve state
        state_annotations = []
        for ann in annotations:
            ann_copy = {
                'type': ann.annotation_type,
                'class_id': ann.class_id,
                'confidence': ann.confidence,
                'position': QPointF(ann.pos()),  # SAVE POSITION
                'modified': getattr(ann, '_modified', False),  # Track if modified
                'original_area': getattr(ann, '_original_area', None)  # Preserve original area
            }

            if hasattr(ann, 'rect'):
                ann_copy['rect'] = QRectF(ann.rect)
            elif hasattr(ann, 'polygon'):
                ann_copy['polygon'] = QPolygonF(ann.polygon)
            elif hasattr(ann, 'mask'):
                ann_copy['mask'] = ann.mask.copy()
                ann_copy['offset'] = QPointF(ann.offset)

            state_annotations.append(ann_copy)

        self.current_state = AnnotationState(
            annotations=state_annotations,
            description=description,
            timestamp=time.time()
        )

    def undo(self):
        """Undo last action"""
        if self.undo_stack and self.current_state:
            self.redo_stack.append(self.current_state)
            self.current_state = self.undo_stack.pop()
            return self.current_state
        return None

    def redo(self):
        """Redo last undone action"""
        if self.redo_stack:
            if self.current_state:
                self.undo_stack.append(self.current_state)
            self.current_state = self.redo_stack.pop()
            return self.current_state
        return None

    def can_undo(self):
        """Check if undo is available"""
        return len(self.undo_stack) > 0

    def can_redo(self):
        """Check if redo is available"""
        return len(self.redo_stack) > 0

    def clear(self):
        """Clear all history"""
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.current_state = None


class SimpleAnnotationItem(QGraphicsPathItem):
    """Simplified annotation item with proper z-ordering and position tracking"""

    def __init__(self, annotation_type='box', class_id=None):
        super().__init__()
        self.annotation_type = annotation_type
        self.class_id = class_id
        self.confidence = 1.0
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)  # Track position changes
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)  # Allow focus for keyboard events

        # Set colors based on class if available
        if class_id is not None and 0 <= class_id < len(CLASS_COLORS):
            self.fill_color = CLASS_COLORS[class_id]
            self.edge_color = EDGE_CLASS_COLORS[class_id]
        else:
            self.fill_color = ANNOTATION_COLORS[annotation_type]
            self.edge_color = EDGE_COLORS[annotation_type]

        # Set appearance with thinner borders and more transparency
        self.setPen(QPen(self.edge_color, 1))  # Thinner border (1px instead of 2px)
        # Make fill more transparent
        transparent_fill = QColor(self.fill_color)
        transparent_fill.setAlpha(50)  # More transparent (50 instead of 100)
        self.setBrush(QBrush(transparent_fill))

        # Calculate initial area for z-ordering
        self._area = 0
        self._original_pos = None
        self._moved = False  # Track if item has been moved
        self._modified = False  # Track if item has been modified
        self._original_area = None  # Store original area from detection
        self._original_data = None  # Store complete original data

    def set_area(self, area):
        """Set area and update z-order (smaller objects on top)"""
        self._area = area
        # Smaller areas get higher z-values (appear on top)
        if area > 0:
            self.setZValue(1000000 / area)
        else:
            self.setZValue(0)

    def hoverEnterEvent(self, event):
        pen = self.pen()
        pen.setWidth(2)  # Thinner hover effect (2px instead of 3px)
        self.setPen(pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        pen = self.pen()
        pen.setWidth(1)  # Back to thin border (1px instead of 2px)
        self.setPen(pen)
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange:
            if value:
                pen = self.pen()
                pen.setStyle(Qt.DashLine)
                pen.setWidth(2)  # Thinner selection (2px instead of 3px)
                self.setPen(pen)
            else:
                pen = self.pen()
                pen.setStyle(Qt.SolidLine)
                pen.setWidth(1)  # Back to thin (1px instead of 2px)
                self.setPen(pen)
        elif change == QGraphicsItem.ItemPositionChange:
            # Store original position for undo
            if self._original_pos is None:
                self._original_pos = self.pos()
            self._moved = True  # Mark as moved
            self._modified = True  # Mark as modified
        elif change == QGraphicsItem.ItemPositionHasChanged:
            # Position has changed, notify the view to save state
            if self.scene() and hasattr(self.scene().views()[0], 'on_item_position_changed'):
                self.scene().views()[0].on_item_position_changed(self)
        return super().itemChange(change, value)


def create_box_item(rect, class_id=None):
    """Create a box annotation item"""
    item = SimpleAnnotationItem('box', class_id)
    path = QPainterPath()
    path.addRect(rect)
    item.setPath(path)
    item.rect = rect
    # Set area for z-ordering
    item.set_area(rect.width() * rect.height())
    return item


def create_polygon_item(points, class_id=None):
    """Create a polygon annotation item with CLOSED contour"""
    item = SimpleAnnotationItem('polygon', class_id)
    path = QPainterPath()

    # Ensure polygon is closed
    if len(points) > 0 and points[0] != points[-1]:
        points = points + [points[0]]  # Close the polygon

    polygon = QPolygonF(points)
    path.addPolygon(polygon)
    path.closeSubpath()  # Explicitly close the path
    item.setPath(path)
    item.polygon = polygon

    # Calculate area for z-ordering using shoelace formula
    area = 0
    n = len(points) - 1  # Don't count the duplicated last point
    for i in range(n):
        j = (i + 1) % n
        area += points[i].x() * points[j].y()
        area -= points[j].x() * points[i].y()
    area = abs(area) / 2.0
    item.set_area(area)
    return item


def create_mask_item(mask, offset=QPointF(0, 0), class_id=None):
    """Create a mask annotation item with CLOSED contours"""
    # Convert mask to contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    item = SimpleAnnotationItem('mask', class_id)
    path = QPainterPath()

    for contour in contours:
        if len(contour) >= 3:
            points = []
            for p in contour:
                if p.ndim == 2:
                    points.append(QPointF(p[0][0] + offset.x(), p[0][1] + offset.y()))
                else:
                    points.append(QPointF(p[0] + offset.x(), p[1] + offset.y()))

            if points:
                # Ensure contour is closed
                if points[0] != points[-1]:
                    points.append(points[0])

                polygon = QPolygonF(points)
                path.addPolygon(polygon)
                path.closeSubpath()  # Explicitly close each contour

    item.setPath(path)
    item.mask = mask
    item.offset = offset
    # Set area for z-ordering
    item.set_area(float(np.sum(mask > 0)))
    return item


class AutoSaveThread(QThread):
    """Background thread for auto-saving"""

    save_completed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_to_save = None
        self.save_path = None

    def set_data(self, data, path):
        """Set data to save"""
        self.data_to_save = data
        self.save_path = path

    def run(self):
        """Save data in background"""
        if self.data_to_save and self.save_path:
            try:
                # Save to temporary file first
                temp_path = self.save_path + '.tmp'
                self.data_to_save.to_csv(temp_path, index=False)

                # Rename to actual file
                import os
                if os.path.exists(self.save_path):
                    os.remove(self.save_path)
                os.rename(temp_path, self.save_path)

                self.save_completed.emit(f"Auto-saved to {self.save_path}")
            except Exception as e:
                self.save_completed.emit(f"Auto-save failed: {str(e)}")


class SAMPreviewThread(QThread):
    """Thread for real-time SAM preview generation with better resource management"""

    preview_ready = pyqtSignal(np.ndarray)

    def __init__(self, predictor, image):
        super().__init__()
        self.predictor = predictor
        self.image = image
        self.point_coords = None
        self.point_labels = None
        self.box = None
        self.running = True
        self._last_request_time = 0
        self._min_interval = 0.1  # Minimum interval between predictions
        self._image_set = False  # Track if image has been set

    def set_point(self, x, y):
        """Set a single point for prediction"""
        self.point_coords = np.array([[x, y]])
        self.point_labels = np.array([1])
        self.box = None
        self._last_request_time = time.time()

    def set_box(self, x1, y1, x2, y2):
        """Set a box for prediction"""
        self.box = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        self.point_coords = None
        self.point_labels = None
        self._last_request_time = time.time()

    def run(self):
        """Run continuous prediction loop with better memory management"""
        if self.predictor is None:
            return

        try:
            # Set image once at the start
            if not self._image_set:
                self.predictor.set_image(self.image)
                self._image_set = True

            while self.running:
                # Check if we have a request and enough time has passed
                current_time = time.time()
                if (current_time - self._last_request_time < self._min_interval or
                        (self.point_coords is None and self.box is None)):
                    self.msleep(50)  # Sleep 50ms
                    continue

                # Perform prediction with memory management
                with torch.no_grad():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # Clear cache before prediction

                    if self.point_coords is not None:
                        masks, _, _ = self.predictor.predict(
                            point_coords=self.point_coords,
                            point_labels=self.point_labels,
                            multimask_output=False
                        )
                    elif self.box is not None:
                        masks, _, _ = self.predictor.predict(
                            box=self.box,
                            multimask_output=False
                        )
                    else:
                        continue

                    if isinstance(masks, torch.Tensor):
                        masks = masks.cpu().numpy()

                    # Emit the mask
                    self.preview_ready.emit(masks[0])

                    # Clear GPU cache after prediction
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Clear the request
                self.point_coords = None
                self.box = None

        except Exception as e:
            print(f"SAM preview error: {e}")

    def stop(self):
        """Stop the preview thread"""
        self.running = False
        # Clear any GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ImageViewer(QGraphicsView):
    """Custom QGraphicsView for image display and annotation with undo/redo support"""

    annotation_added = pyqtSignal(object)
    annotation_removed = pyqtSignal(object)
    annotation_modified = pyqtSignal(object)
    position_changed = pyqtSignal()  # NEW: Signal for position changes

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.MinimalViewportUpdate)
        self.setFocusPolicy(Qt.StrongFocus)  # Ensure view can receive keyboard focus

        # Image and annotations
        self.pixmap_item = None
        self.current_tool = 'select'
        self.drawing = False
        self.current_points = []
        self.temp_items = []

        # Image dimensions
        self.display_width = 1056
        self.display_height = 1056
        self.original_width = 1
        self.original_height = 1
        self.scale_x = 1.0
        self.scale_y = 1.0

        # Zoom
        self.zoom_factor = 1.0
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        # Smart mode
        self.preview_mask_item = None
        self.preview_thread = None
        self.smart_box_start = None
        self.panning = False
        self.pan_start_pos = None
        self.smart_preview_enabled = False

        # Class tracking
        self.current_class_id = 0
        self.class_mapping = {}  # name -> id mapping

        # Undo/Redo manager
        self.undo_manager = UndoRedoManager()

        # Clipboard for copy/paste
        self.clipboard_annotation = None

        # Track if positions have changed
        self._positions_changed = False

    def on_item_position_changed(self, item):
        """Called when an item's position changes"""
        self._positions_changed = True
        item._modified = True  # Mark item as modified
        self.position_changed.emit()

    def set_class_mapping(self, mapping):
        """Set the class name to ID mapping"""
        self.class_mapping = mapping

    def set_current_class(self, class_id):
        """Set the current class for new annotations"""
        self.current_class_id = class_id

    def cleanup_for_new_image(self):
        """Clean up before loading a new image"""
        # Stop and clean up preview thread
        if self.preview_thread:
            self.preview_thread.stop()
            self.preview_thread.wait()
            self.preview_thread = None

        # Clear preview
        self._clear_preview()

        # Clear temp items
        self._clear_temp_items()

        # Reset drawing state
        self.drawing = False
        self.current_points = []
        self.smart_box_start = None
        self.smart_preview_enabled = False

        # Clear undo/redo history
        self.undo_manager.clear()

        # Clear clipboard
        self.clipboard_annotation = None

        # Reset position change flag
        self._positions_changed = False

        # Force garbage collection
        gc.collect()

    def set_image(self, image, display_size=1056):
        """Set the image to display"""
        # Clean up before setting new image
        self.cleanup_for_new_image()

        # Clear scene
        for item in self.scene.items():
            self.scene.removeItem(item)

        self.pixmap_item = None
        self.temp_items = []
        self.preview_mask_item = None

        if image is not None:
            h, w = image.shape[:2]
            self.original_height = h
            self.original_width = w

            # Calculate display dimensions maintaining aspect ratio
            aspect_ratio = w / h
            if w > h:
                self.display_width = display_size
                self.display_height = int(display_size / aspect_ratio)
            else:
                self.display_height = display_size
                self.display_width = int(display_size * aspect_ratio)

            # IMPORTANT: Calculate scale factors based on ACTUAL image dimensions
            self.scale_x = self.display_width / self.original_width
            self.scale_y = self.display_height / self.original_height

            # Debug print
            print(f"ImageViewer - Original: {w}x{h}, Display: {self.display_width}x{self.display_height}")
            print(f"ImageViewer - Scale factors: {self.scale_x:.3f}, {self.scale_y:.3f}")

            # Resize image for display
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize to display size
            display_image = cv2.resize(image, (self.display_width, self.display_height),
                                       interpolation=cv2.INTER_AREA)

            qimg = QImage(display_image.data, self.display_width, self.display_height,
                          3 * self.display_width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.pixmap_item.setZValue(-1)  # Put image in background
            rect = QRectF(pixmap.rect())
            self.scene.setSceneRect(rect)
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def set_tool(self, tool):
        """Set current annotation tool"""
        # Clear any leftover temporary items when switching tools
        self._clear_temp_items()

        self.current_tool = tool
        self.drawing = False
        self.current_points = []
        self.panning = False  # Reset panning state

        # Stop preview thread when not in smart mode
        if tool != 'smart' and self.preview_thread:
            self.preview_thread.stop()
            self.preview_thread.wait()
            self.preview_thread = None
            self._clear_preview()

        if tool == 'select':
            self.setDragMode(QGraphicsView.RubberBandDrag)
            self.setCursor(Qt.ArrowCursor)
        elif tool == 'pan':
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)

    def set_preview_thread(self, thread):
        """Set the SAM preview thread"""
        if self.preview_thread:
            self.preview_thread.stop()
            self.preview_thread.wait()

        self.preview_thread = thread
        if thread:
            thread.preview_ready.connect(self._update_preview_mask)
            thread.start()

    def _clear_temp_items(self):
        """Clear temporary drawing items"""
        for item in self.temp_items:
            if item and item.scene():
                try:
                    self.scene.removeItem(item)
                except:
                    pass  # Item might already be removed
        self.temp_items = []

    def _clear_preview(self):
        """Clear preview mask"""
        if self.preview_mask_item and self.preview_mask_item.scene():
            self.scene.removeItem(self.preview_mask_item)
        self.preview_mask_item = None

    def save_current_state(self, description="Action"):
        """Save current annotation state for undo"""
        annotations = self.get_annotations()
        self.undo_manager.save_state(annotations, description)

    def undo(self):
        """Undo last action"""
        state = self.undo_manager.undo()
        if state:
            self._restore_state(state)

    def redo(self):
        """Redo last undone action"""
        state = self.undo_manager.redo()
        if state:
            self._restore_state(state)

    def _restore_state(self, state):
        """Restore annotations from a saved state"""
        # Clear current annotations
        for item in list(self.scene.items()):
            if isinstance(item, SimpleAnnotationItem):
                self.scene.removeItem(item)

        # Recreate annotations from state
        for ann_data in state.annotations:
            if ann_data['type'] == 'box':
                item = create_box_item(ann_data['rect'], ann_data['class_id'])
            elif ann_data['type'] == 'polygon':
                item = create_polygon_item(list(ann_data['polygon']), ann_data['class_id'])
            elif ann_data['type'] == 'mask':
                item = create_mask_item(ann_data['mask'], ann_data['offset'], ann_data['class_id'])
            else:
                continue

            item.confidence = ann_data['confidence']
            item._modified = ann_data.get('modified', False)
            item._original_area = ann_data.get('original_area')
            # RESTORE POSITION
            if 'position' in ann_data:
                item.setPos(ann_data['position'])
            self.scene.addItem(item)

    def copy_selected(self):
        """Copy selected annotation"""
        selected = self.scene.selectedItems()
        if selected and isinstance(selected[0], SimpleAnnotationItem):
            self.clipboard_annotation = selected[0]

    def paste_annotation(self):
        """Paste copied annotation"""
        if self.clipboard_annotation:
            # Create a copy of the annotation
            if self.clipboard_annotation.annotation_type == 'box':
                new_item = create_box_item(
                    QRectF(self.clipboard_annotation.rect),
                    self.clipboard_annotation.class_id
                )
            elif self.clipboard_annotation.annotation_type == 'polygon':
                new_item = create_polygon_item(
                    list(self.clipboard_annotation.polygon),
                    self.clipboard_annotation.class_id
                )
            elif self.clipboard_annotation.annotation_type == 'mask':
                new_item = create_mask_item(
                    self.clipboard_annotation.mask.copy(),
                    QPointF(self.clipboard_annotation.offset),
                    self.clipboard_annotation.class_id
                )
            else:
                return

            new_item.confidence = self.clipboard_annotation.confidence
            new_item._modified = True  # Mark as modified since it's new
            # Offset the pasted item slightly
            new_item.setPos(20, 20)

            self.save_current_state("Paste annotation")
            self.scene.addItem(new_item)
            self.annotation_added.emit(new_item)

    def delete_selected(self):
        """Delete selected annotations"""
        selected = self.scene.selectedItems()
        if selected:
            self.save_current_state("Delete annotations")
            for item in selected:
                if isinstance(item, SimpleAnnotationItem):
                    self.scene.removeItem(item)
                    self.annotation_removed.emit(item)
        else:
            # If no items selected, try to delete items under cursor
            items_under_cursor = self.items(self.mapFromGlobal(QCursor.pos()))
            for item in items_under_cursor:
                if isinstance(item, SimpleAnnotationItem):
                    self.save_current_state("Delete annotation")
                    self.scene.removeItem(item)
                    self.annotation_removed.emit(item)
                    break  # Delete only the topmost item

    def select_all(self):
        """Select all annotations"""
        for item in self.scene.items():
            if isinstance(item, SimpleAnnotationItem):
                item.setSelected(True)

    def leaveEvent(self, event):
        """Handle mouse leaving the view"""
        # Don't update polygon preview when mouse leaves view
        if self.current_tool == 'polygon' and self.drawing:
            # Only keep the point markers and lines between actual points
            self._update_temp_polygon(None)  # Update without current position
        super().leaveEvent(event)

    def wheelEvent(self, event):
        """Handle zoom with mouse wheel"""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)
        self.zoom_factor *= zoom_factor

    def mousePressEvent(self, event):
        # ALWAYS ensure this view has focus for keyboard events
        self.setFocus(Qt.MouseFocusReason)  # Use MouseFocusReason to ensure focus

        # Handle panning
        if self.current_tool == 'pan':
            if event.button() == Qt.LeftButton:
                self.panning = True
                self.pan_start_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                # Store current scrollbar values
                self._pan_start_h = self.horizontalScrollBar().value()
                self._pan_start_v = self.verticalScrollBar().value()
            return

        if self.current_tool == 'select':
            # In select mode, handle selection normally
            super().mousePressEvent(event)
            return
        elif not self.pixmap_item:
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())

        if self.current_tool == 'box':
            if event.button() == Qt.LeftButton:
                self.drawing = True
                self.current_points = [scene_pos]
                color = self._get_current_edge_color()
                temp_rect = self.scene.addRect(
                    scene_pos.x(), scene_pos.y(), 0, 0,
                    QPen(color, 1, Qt.DashLine),  # Thinner preview (1px instead of 2px)
                    QBrush(Qt.NoBrush)
                )
                self.temp_items.append(temp_rect)

        elif self.current_tool == 'polygon':
            if event.button() == Qt.LeftButton:
                if not self.drawing:
                    self.drawing = True
                    self.current_points = [scene_pos]
                    # Add first point marker
                    marker = self.scene.addEllipse(
                        scene_pos.x() - 3, scene_pos.y() - 3, 6, 6,
                        QPen(self._get_current_edge_color(), 1),
                        QBrush(self._get_current_edge_color())
                    )
                    self.temp_items.append(marker)
                else:
                    # Check if clicking near first point to close
                    if len(self.current_points) >= 3:
                        dist = (self.current_points[0] - scene_pos).manhattanLength()
                        if dist < 10:
                            self._finish_polygon()
                            return
                    self.current_points.append(scene_pos)
                    # Add point marker
                    marker = self.scene.addEllipse(
                        scene_pos.x() - 3, scene_pos.y() - 3, 6, 6,
                        QPen(self._get_current_edge_color(), 1),
                        QBrush(self._get_current_edge_color())
                    )
                    self.temp_items.append(marker)
                    self._update_temp_polygon()

            elif event.button() == Qt.RightButton and self.drawing:
                if len(self.current_points) >= 3:
                    self._finish_polygon()
                else:
                    # Clear all temp items when canceling
                    for item in self.temp_items:
                        if item.scene():
                            self.scene.removeItem(item)
                    self.temp_items = []
                    self.drawing = False
                    self.current_points = []

        elif self.current_tool == 'smart':
            if event.button() == Qt.LeftButton:
                # Check if we're starting a box drag or applying current preview
                if event.modifiers() & Qt.ShiftModifier:
                    # Start box drag
                    self.smart_box_start = scene_pos
                    self.drawing = True
                    temp_rect = self.scene.addRect(
                        scene_pos.x(), scene_pos.y(), 1, 1,
                        QPen(EDGE_COLORS['smart'], 2, Qt.DashLine),
                        QBrush(Qt.NoBrush)
                    )
                    self.temp_items.append(temp_rect)
                else:
                    # Apply current preview mask if any
                    if self.preview_mask_item:
                        self._apply_preview_mask()

        # For non-select tools, also check if clicking on an item to select it
        if self.current_tool != 'select':
            item = self.scene.itemAt(scene_pos, self.transform())
            if isinstance(item, SimpleAnnotationItem):
                # Clear other selections
                for scene_item in self.scene.selectedItems():
                    scene_item.setSelected(False)
                # Select this item
                item.setSelected(True)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())

        # Handle panning
        if self.current_tool == 'pan' and self.panning:
            delta = event.pos() - self.pan_start_pos
            self.horizontalScrollBar().setValue(self._pan_start_h - delta.x())
            self.verticalScrollBar().setValue(self._pan_start_v - delta.y())
            return

        if self.current_tool == 'select' or not self.pixmap_item:
            super().mouseMoveEvent(event)
            return

        if self.current_tool == 'box' and self.drawing:
            if self.temp_items:  # Check if temp_items exist
                rect_item = self.temp_items[0]
                rect = QRectF(self.current_points[0], scene_pos).normalized()
                rect_item.setRect(rect)

        elif self.current_tool == 'polygon' and self.drawing:
            self._update_temp_polygon(scene_pos)
        elif self.current_tool == 'polygon' and not self.drawing:
            # Clear any leftover temp items when not in drawing mode
            pass  # Don't draw preview when not in drawing mode

        elif self.current_tool == 'smart':
            if self.drawing and self.smart_box_start:
                # Update box preview
                if self.temp_items:  # Check if temp_items exist
                    rect_item = self.temp_items[0]
                    rect = QRectF(self.smart_box_start, scene_pos).normalized()
                    rect_item.setRect(rect)

                # Send box to preview thread
                if self.preview_thread:
                    x1 = rect.x() / self.scale_x
                    y1 = rect.y() / self.scale_y
                    x2 = (rect.x() + rect.width()) / self.scale_x
                    y2 = (rect.y() + rect.height()) / self.scale_y
                    self.preview_thread.set_box(x1, y1, x2, y2)
            else:
                # Send hover point to preview thread
                if self.preview_thread and self.smart_preview_enabled:
                    orig_x = scene_pos.x() / self.scale_x
                    orig_y = scene_pos.y() / self.scale_y
                    self.preview_thread.set_point(orig_x, orig_y)

    def mouseReleaseEvent(self, event):
        # Handle panning
        if self.current_tool == 'pan':
            if event.button() == Qt.LeftButton:
                self.panning = False
                self.setCursor(Qt.OpenHandCursor)
            return

        if self.current_tool == 'select':
            # Check if any item has been moved
            if self._positions_changed:
                self.save_current_state("Move annotations")
                self._positions_changed = False
            super().mouseReleaseEvent(event)
            return

        if self.current_tool == 'box' and self.drawing:
            if event.button() == Qt.LeftButton:
                scene_pos = self.mapToScene(event.pos())
                rect = QRectF(self.current_points[0], scene_pos).normalized()
                if rect.width() > 5 and rect.height() > 5:
                    self.save_current_state("Add box")
                    self._add_box_annotation(rect)
                # Clear ALL temporary items
                for item in self.temp_items:
                    if item.scene():
                        self.scene.removeItem(item)
                self.temp_items = []
                self.drawing = False
                self.current_points = []

        elif self.current_tool == 'smart' and self.drawing:
            if event.button() == Qt.LeftButton:
                # Apply the box-generated mask
                if self.preview_mask_item:
                    self.save_current_state("Add smart mask")
                    self._apply_preview_mask()
                # Clear ALL temporary items
                for item in self.temp_items:
                    if item.scene():
                        self.scene.removeItem(item)
                self.temp_items = []
                self.drawing = False
                self.smart_box_start = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            # Clear ALL temp items completely
            for item in self.temp_items:
                if item.scene():
                    self.scene.removeItem(item)
            self.temp_items = []
            self._clear_preview()
            self.drawing = False
            self.current_points = []
            self.smart_box_start = None
        elif event.key() in [Qt.Key_Delete, Qt.Key_Backspace]:  # Handle both Delete and Backspace
            self.delete_selected()
        elif event.key() == Qt.Key_Space and self.current_tool == 'smart':
            # Toggle preview on/off
            self.smart_preview_enabled = not self.smart_preview_enabled
            if not self.smart_preview_enabled:
                self._clear_preview()
        elif event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            if event.modifiers() & Qt.ShiftModifier:
                self.redo()
            else:
                self.undo()
        elif event.key() == Qt.Key_C and event.modifiers() & Qt.ControlModifier:
            self.copy_selected()
        elif event.key() == Qt.Key_V and event.modifiers() & Qt.ControlModifier:
            self.paste_annotation()
        elif event.key() == Qt.Key_A and event.modifiers() & Qt.ControlModifier:
            self.select_all()
        else:
            super().keyPressEvent(event)

    def _get_current_edge_color(self):
        """Get the edge color for current class"""
        if 0 <= self.current_class_id < len(EDGE_CLASS_COLORS):
            return EDGE_CLASS_COLORS[self.current_class_id]
        return EDGE_COLORS.get(self.current_tool, Qt.black)

    def _get_current_fill_color(self):
        """Get the fill color for current class"""
        if 0 <= self.current_class_id < len(CLASS_COLORS):
            return CLASS_COLORS[self.current_class_id]
        return ANNOTATION_COLORS.get(self.current_tool, Qt.gray)

    def _update_temp_polygon(self, current_pos=None):
        """Update temporary polygon visualization with closing line"""
        # Remove ALL old lines and shapes except point markers
        temp_markers = []
        for item in self.temp_items:
            if isinstance(item, QGraphicsEllipseItem) and item.rect().width() <= 6:  # Keep only small point markers
                temp_markers.append(item)
            else:
                if item and item.scene():
                    self.scene.removeItem(item)
        self.temp_items = temp_markers

        # If no current position (mouse left view), just show the existing points connected
        if current_pos is None and len(self.current_points) >= 2:
            color = self._get_current_edge_color()
            # Draw lines between actual points only
            for i in range(len(self.current_points) - 1):
                line = self.scene.addLine(
                    self.current_points[i].x(), self.current_points[i].y(),
                    self.current_points[i + 1].x(), self.current_points[i + 1].y(),
                    QPen(color, 1, Qt.DashLine)
                )
                self.temp_items.append(line)
            return

        points = self.current_points.copy()
        if current_pos:
            points.append(current_pos)

        if len(points) >= 2:
            color = self._get_current_edge_color()
            # Draw lines between points
            for i in range(len(points) - 1):
                line = self.scene.addLine(
                    points[i].x(), points[i].y(),
                    points[i + 1].x(), points[i + 1].y(),
                    QPen(color, 1, Qt.DashLine)
                )
                self.temp_items.append(line)

            # Always draw closing line when we have at least 3 points
            if len(self.current_points) >= 3:
                # Draw line from current position (or last point) to first point
                last_point = points[-1]
                closing_line = self.scene.addLine(
                    last_point.x(), last_point.y(),
                    self.current_points[0].x(), self.current_points[0].y(),
                    QPen(color, 1, Qt.DotLine)  # Use dotted line for closing
                )
                self.temp_items.append(closing_line)

                # Highlight first point if mouse is near it
                if current_pos:
                    dist = (self.current_points[0] - current_pos).manhattanLength()
                    if dist < 10:
                        # Add highlight circle around first point
                        highlight = self.scene.addEllipse(
                            self.current_points[0].x() - 8, self.current_points[0].y() - 8,
                            16, 16,
                            QPen(color, 1),
                            QBrush(Qt.NoBrush)
                        )
                        self.temp_items.append(highlight)

    def _update_preview_mask(self, mask):
        """Update the preview mask display"""
        if self.current_tool != 'smart':
            return

        # Clear previous preview
        self._clear_preview()

        # Scale mask to display size
        display_h = int(mask.shape[0] * self.scale_y)
        display_w = int(mask.shape[1] * self.scale_x)

        scaled_mask = cv2.resize(
            mask.astype(np.uint8),
            (display_w, display_h),
            interpolation=cv2.INTER_NEAREST
        )

        # Create semi-transparent preview
        h, w = scaled_mask.shape
        preview_img = np.zeros((h, w, 4), dtype=np.uint8)
        preview_img[scaled_mask > 0] = [255, 255, 0, 80]  # Yellow with low opacity

        # Convert to QPixmap
        qimg = QImage(preview_img.data, w, h, 4 * w, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)

        # Add to scene
        self.preview_mask_item = self.scene.addPixmap(pixmap)
        self.preview_mask_item.setZValue(100)  # Above image but below annotations

    def _apply_preview_mask(self):
        """Convert preview mask to actual annotation"""
        if not self.preview_mask_item:
            return

        # Get the mask from preview
        pixmap = self.preview_mask_item.pixmap()
        qimg = pixmap.toImage()

        # Convert QImage to numpy array
        width = qimg.width()
        height = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

        # Extract mask (where alpha > 0)
        mask = (arr[:, :, 3] > 0).astype(np.uint8)

        # Create mask annotation with closed contours
        item = create_mask_item(mask, QPointF(0, 0), self.current_class_id)
        item._modified = True  # Mark as modified since it's new
        self.scene.addItem(item)
        self.annotation_added.emit(item)

        # Clear preview
        self._clear_preview()

    def _finish_polygon(self):
        """Finalize polygon creation with proper closing"""
        if len(self.current_points) >= 3:
            self.save_current_state("Add polygon")
            self._add_polygon_annotation(self.current_points)

        # Clear ALL temporary items from the scene
        for item in self.temp_items:
            if item.scene():
                self.scene.removeItem(item)
        self.temp_items = []
        self.drawing = False
        self.current_points = []

    def _add_box_annotation(self, rect):
        """Add box annotation to scene"""
        item = create_box_item(rect, self.current_class_id)
        item._modified = True  # Mark as modified since it's new
        self.scene.addItem(item)
        self.annotation_added.emit(item)

    def _add_polygon_annotation(self, points):
        """Add polygon annotation to scene with closed contour"""
        item = create_polygon_item(points, self.current_class_id)
        item._modified = True  # Mark as modified since it's new
        self.scene.addItem(item)
        self.annotation_added.emit(item)

    def add_mask_annotation(self, mask):
        """Add mask annotation to scene"""
        item = create_mask_item(mask, QPointF(0, 0), self.current_class_id)
        item._modified = True  # Mark as modified since it's new
        self.scene.addItem(item)
        self.annotation_added.emit(item)

    def get_annotations(self):
        """Get all annotations in the scene"""
        annotations = []
        for item in self.scene.items():
            if isinstance(item, SimpleAnnotationItem):
                annotations.append(item)
        return annotations

    def clear_annotations(self):
        """Clear all annotations"""
        self.save_current_state("Clear all")
        for item in list(self.scene.items()):
            if isinstance(item, SimpleAnnotationItem):
                self.scene.removeItem(item)


class EditionWindow(QMainWindow):
    """Main window for manual annotation with enhanced features"""

    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.current_image = None
        self.annotations = []
        self.sam_predictor = None
        self.img_height_orig = 1
        self.img_width_orig = 1

        # Class management
        self.class_names = []
        self.class_name_to_id = {}
        self.current_class_id = 0

        # Store original data for preservation
        self.original_df = None
        self.preserved_columns = []
        self.original_areas = {}  # Store original areas by image and object

        # Initialize temp_annotations
        self.temp_annotations = {}

        # Auto-save
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save)
        self.auto_save_thread = AutoSaveThread()
        self.auto_save_thread.save_completed.connect(self.on_auto_save_completed)

        self.init_ui()
        self.create_toolbar()
        self.create_dock_widgets()
        self.create_menus()
        self.setup_shortcuts()

        # Start auto-save timer (every 5 minutes)
        self.auto_save_timer.start(300000)  # 5 minutes in milliseconds

    def init_ui(self):
        """Initialize the main UI"""
        self.setWindowTitle("NemaCounter - Manual Edition Enhanced")
        self.setGeometry(100, 100, 1400, 900)

        # Central widget - Image viewer
        self.image_viewer = ImageViewer()
        self.setCentralWidget(self.image_viewer)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Connect signals
        self.image_viewer.annotation_added.connect(self.on_annotation_added)
        self.image_viewer.annotation_removed.connect(self.on_annotation_removed)
        self.image_viewer.annotation_modified.connect(self.on_annotation_modified)
        self.image_viewer.position_changed.connect(self.on_position_changed)  # NEW

    def on_position_changed(self):
        """Handle position changes - mark annotations as modified"""
        # This will be called when any annotation is moved
        # The actual saving happens in save_current_annotations
        pass

    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Undo/Redo
        undo_shortcut = QShortcut(QKeySequence.Undo, self)
        undo_shortcut.activated.connect(self.undo)

        redo_shortcut = QShortcut(QKeySequence.Redo, self)
        redo_shortcut.activated.connect(self.redo)

        # Copy/Paste
        copy_shortcut = QShortcut(QKeySequence.Copy, self)
        copy_shortcut.activated.connect(self.copy_annotation)

        paste_shortcut = QShortcut(QKeySequence.Paste, self)
        paste_shortcut.activated.connect(self.paste_annotation)

        # Select All
        select_all_shortcut = QShortcut(QKeySequence.SelectAll, self)
        select_all_shortcut.activated.connect(self.select_all)

        # Save
        save_shortcut = QShortcut(QKeySequence.Save, self)
        save_shortcut.activated.connect(self.save_annotations)

        # Navigation arrows
        left_shortcut = QShortcut(Qt.Key_Left, self)
        left_shortcut.activated.connect(self.prev_image)

        right_shortcut = QShortcut(Qt.Key_Right, self)
        right_shortcut.activated.connect(self.next_image)

        # Quick tool switches
        QShortcut(Qt.Key_V, self).activated.connect(lambda: self.set_tool('select'))
        QShortcut(Qt.Key_R, self).activated.connect(lambda: self.set_tool('box'))
        QShortcut(Qt.Key_P, self).activated.connect(lambda: self.set_tool('polygon'))
        QShortcut(Qt.Key_S, self).activated.connect(lambda: self.set_tool('smart'))
        QShortcut(Qt.Key_H, self).activated.connect(lambda: self.set_tool('pan'))

    def create_toolbar(self):
        """Create main toolbar with enhanced features"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Tool actions
        self.tool_group = QActionGroup(self)

        # Select tool
        select_action = QAction(self.create_icon('select'), "Select", self)
        select_action.setCheckable(True)
        select_action.setChecked(True)
        select_action.setShortcut('V')
        select_action.setToolTip("Select and move annotations (V)")
        select_action.triggered.connect(lambda: self.set_tool('select'))
        self.tool_group.addAction(select_action)
        toolbar.addAction(select_action)

        # Pan tool
        pan_action = QAction(self.create_icon('pan'), "Pan", self)
        pan_action.setCheckable(True)
        pan_action.setShortcut('H')
        pan_action.setToolTip("Pan/Hand tool - drag to move view (H)")
        pan_action.triggered.connect(lambda: self.set_tool('pan'))
        self.tool_group.addAction(pan_action)
        toolbar.addAction(pan_action)

        # Box tool
        box_action = QAction(self.create_icon('box'), "Rectangle", self)
        box_action.setCheckable(True)
        box_action.setShortcut('R')
        box_action.setToolTip("Draw rectangle annotations (R)")
        box_action.triggered.connect(lambda: self.set_tool('box'))
        self.tool_group.addAction(box_action)
        toolbar.addAction(box_action)

        # Polygon tool
        polygon_action = QAction(self.create_icon('polygon'), "Polygon", self)
        polygon_action.setCheckable(True)
        polygon_action.setShortcut('P')
        polygon_action.setToolTip("Draw polygon annotations (P)\nRight-click to finish")
        polygon_action.triggered.connect(lambda: self.set_tool('polygon'))
        self.tool_group.addAction(polygon_action)
        toolbar.addAction(polygon_action)

        # Smart tool
        self.smart_action = QAction(self.create_icon('smart'), "Smart", self)
        self.smart_action.setCheckable(True)
        self.smart_action.setShortcut('S')
        self.smart_action.setToolTip(
            "Smart segmentation (S)\nHover and click to segment\nShift+drag for box mode\nSpace to toggle preview")
        self.smart_action.triggered.connect(lambda: self.set_tool('smart'))
        self.tool_group.addAction(self.smart_action)
        toolbar.addAction(self.smart_action)

        toolbar.addSeparator()

        # Edit actions
        self.undo_action = QAction(self.create_icon('undo'), "Undo", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.setToolTip("Undo last action (Ctrl+Z)")
        self.undo_action.triggered.connect(self.undo)
        toolbar.addAction(self.undo_action)

        self.redo_action = QAction(self.create_icon('redo'), "Redo", self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.setToolTip("Redo last action (Ctrl+Shift+Z)")
        self.redo_action.triggered.connect(self.redo)
        toolbar.addAction(self.redo_action)

        toolbar.addSeparator()

        # Copy/Paste actions
        copy_action = QAction(self.create_icon('copy'), "Copy", self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.setToolTip("Copy selected annotation (Ctrl+C)")
        copy_action.triggered.connect(self.copy_annotation)
        toolbar.addAction(copy_action)

        paste_action = QAction(self.create_icon('paste'), "Paste", self)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.setToolTip("Paste annotation (Ctrl+V)")
        paste_action.triggered.connect(self.paste_annotation)
        toolbar.addAction(paste_action)

        toolbar.addSeparator()

        # Zoom actions
        zoom_in_action = QAction(self.create_icon('zoom_in'), "Zoom In", self)
        zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction(self.create_icon('zoom_out'), "Zoom Out", self)
        zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)

        zoom_fit_action = QAction(self.create_icon('zoom_fit'), "Fit to Window", self)
        zoom_fit_action.setShortcut('F')
        zoom_fit_action.triggered.connect(self.zoom_fit)
        toolbar.addAction(zoom_fit_action)

    def undo(self):
        """Undo last action"""
        self.image_viewer.undo()
        self.update_undo_redo_actions()
        self.update_stats()
        self.status_bar.showMessage("Undo performed", 2000)

    def redo(self):
        """Redo last undone action"""
        self.image_viewer.redo()
        self.update_undo_redo_actions()
        self.update_stats()
        self.status_bar.showMessage("Redo performed", 2000)

    def update_undo_redo_actions(self):
        """Update undo/redo action states"""
        if hasattr(self, 'undo_action'):
            self.undo_action.setEnabled(self.image_viewer.undo_manager.can_undo())
        if hasattr(self, 'redo_action'):
            self.redo_action.setEnabled(self.image_viewer.undo_manager.can_redo())

    def copy_annotation(self):
        """Copy selected annotation"""
        self.image_viewer.copy_selected()
        self.status_bar.showMessage("Annotation copied", 2000)

    def paste_annotation(self):
        """Paste annotation"""
        self.image_viewer.paste_annotation()
        self.update_stats()
        self.status_bar.showMessage("Annotation pasted", 2000)

    def delete_selected(self):
        """Delete selected annotations"""
        # Focus on the image viewer first
        self.image_viewer.setFocus()
        # Then call delete on the image viewer
        self.image_viewer.delete_selected()
        self.update_stats()
        self.status_bar.showMessage("Selected annotations deleted", 2000)

    def select_all(self):
        """Select all annotations"""
        self.image_viewer.select_all()
        self.status_bar.showMessage("All annotations selected", 2000)

    def auto_save(self):
        """Perform auto-save in background"""
        if hasattr(self, 'temp_annotations'):
            self.save_current_annotations()
            self.status_bar.showMessage("Auto-saving...", 2000)
            # Implement background save logic here if needed

    def on_auto_save_completed(self, message):
        """Handle auto-save completion"""
        self.status_bar.showMessage(message, 3000)

    def on_size_threshold_changed(self, value):
        """Handle size threshold slider change"""
        self.size_label.setText(f"{value} pixelsÂ²")

        # Update visibility of all annotations
        visible_count = 0
        total_count = 0

        for item in self.image_viewer.get_annotations():
            total_count += 1

            # Get area based on annotation type
            area = self.get_annotation_area(item)

            # Show/hide based on threshold
            if area >= value:
                item.setVisible(True)
                item.setOpacity(1.0)  # Full opacity
                visible_count += 1
            else:
                item.setVisible(True)  # Keep visible but transparent
                item.setOpacity(0.2)  # 20% opacity

        # Update filtered stats
        if value == 0:
            self.filtered_stats_label.setText("Visible: All")
        else:
            self.filtered_stats_label.setText(f"Visible: {visible_count}/{total_count}")

        # Update main stats to only count visible items
        self.update_stats()

    def get_annotation_area(self, item):
        """Get the area of an annotation item in display pixels"""
        if hasattr(item, '_area'):
            return item._area

        # Calculate area if not cached
        area = 0

        if hasattr(item, 'rect'):
            # Box area
            area = item.rect.width() * item.rect.height()
        elif hasattr(item, 'polygon'):
            # Polygon area using shoelace formula
            points = list(item.polygon)
            n = len(points) - 1 if points[0] == points[-1] else len(points)
            for i in range(n):
                j = (i + 1) % n
                area += points[i].x() * points[j].y()
                area -= points[j].x() * points[i].y()
            area = abs(area) / 2.0
        elif hasattr(item, 'mask'):
            # Mask area
            area = float(np.sum(item.mask > 0))

        # Cache the area
        item._area = area
        return area

    def create_dock_widgets(self):
        """Create dock widgets with enhanced features"""
        # Annotations list dock
        annotations_dock = QDockWidget("Annotations", self)
        annotations_widget = QWidget()
        annotations_layout = QVBoxLayout(annotations_widget)

        self.annotations_list = QListWidget()
        self.annotations_list.setAlternatingRowColors(True)
        self.annotations_list.setSelectionMode(QListWidget.ExtendedSelection)
        annotations_layout.addWidget(QLabel("Current Annotations:"))
        annotations_layout.addWidget(self.annotations_list)

        # Connect list selection to scene selection
        self.annotations_list.itemSelectionChanged.connect(self.sync_list_selection)

        # Annotation stats
        self.stats_label = QLabel("Total: 0")
        annotations_layout.addWidget(self.stats_label)

        annotations_dock.setWidget(annotations_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, annotations_dock)

        # Controls dock
        controls_dock = QDockWidget("Controls", self)
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # Class selection
        class_group = QGroupBox("Class")
        class_layout = QVBoxLayout(class_group)
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self.on_class_changed)
        class_layout.addWidget(self.class_combo)

        # Add class button
        add_class_button = QPushButton("Add New Class")
        add_class_button.clicked.connect(self.add_new_class)
        class_layout.addWidget(add_class_button)

        controls_layout.addWidget(class_group)

        # Confidence slider
        conf_group = QGroupBox("Confidence")
        conf_layout = QVBoxLayout(conf_group)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(100)
        self.conf_label = QLabel("1.00")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"{v / 100:.2f}")
        )
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        controls_layout.addWidget(conf_group)

        # Smart mode info
        smart_group = QGroupBox("Smart Mode Tips")
        smart_layout = QVBoxLayout(smart_group)
        smart_info = QLabel(
            "â€¢ Hover to preview\n"
            "â€¢ Click to apply mask\n"
            "â€¢ Shift+drag for box\n"
            "â€¢ Space to toggle preview"
        )
        smart_info.setWordWrap(True)
        smart_layout.addWidget(smart_info)
        controls_layout.addWidget(smart_group)

        # Size threshold slider
        size_group = QGroupBox("Size Filter")
        size_layout = QVBoxLayout(size_group)
        size_info_label = QLabel("Show objects larger than:")
        size_layout.addWidget(size_info_label)

        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(0, 10000)  # 0 to 10000 pixels
        self.size_slider.setValue(0)  # Show all by default
        self.size_slider.setTickPosition(QSlider.TicksBelow)
        self.size_slider.setTickInterval(1000)

        self.size_label = QLabel("0 pixelsÂ²")
        self.size_slider.valueChanged.connect(self.on_size_threshold_changed)

        # Add a reset button
        size_button_layout = QHBoxLayout()
        size_button_layout.addWidget(self.size_label)
        reset_size_button = QPushButton("Reset")
        reset_size_button.clicked.connect(lambda: self.size_slider.setValue(0))
        size_button_layout.addWidget(reset_size_button)

        size_layout.addWidget(self.size_slider)
        size_layout.addLayout(size_button_layout)

        # Stats label for filtered count
        self.filtered_stats_label = QLabel("Visible: All")
        size_layout.addWidget(self.filtered_stats_label)

        controls_layout.addWidget(size_group)

        # Navigation buttons
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout(nav_group)

        self.prev_button = QPushButton("â—€ Previous (â†)")
        self.prev_button.clicked.connect(self.prev_image)
        nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next (â†’) â–¶")
        self.next_button.clicked.connect(self.next_image)
        nav_layout.addWidget(self.next_button)

        self.save_button = QPushButton("ðŸ’¾ Save & Continue")
        self.save_button.clicked.connect(self.save_and_continue)
        self.save_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        nav_layout.addWidget(self.save_button)

        controls_layout.addWidget(nav_group)

        # Add keyboard shortcuts info
        shortcuts_group = QGroupBox("Keyboard Shortcuts")
        shortcuts_layout = QVBoxLayout(shortcuts_group)
        shortcuts_info = QLabel(
            "Ctrl+Z - Undo\n"
            "Ctrl+Shift+Z - Redo\n"
            "Ctrl+C - Copy\n"
            "Ctrl+V - Paste\n"
            "Ctrl+A - Select All\n"
            "Delete/Backspace - Delete Selected\n"
            "V - Select Tool\n"
            "R - Rectangle Tool\n"
            "P - Polygon Tool\n"
            "S - Smart Tool\n"
            "â†/â†’ - Navigate Images"
        )
        shortcuts_info.setWordWrap(True)
        shortcuts_layout.addWidget(shortcuts_info)
        controls_layout.addWidget(shortcuts_group)

        controls_layout.addStretch()

        controls_dock.setWidget(controls_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, controls_dock)

    def sync_list_selection(self):
        """Synchronize list selection with scene selection"""
        # Clear scene selection
        for item in self.image_viewer.scene.items():
            if isinstance(item, SimpleAnnotationItem):
                item.setSelected(False)

        # Select items in scene based on list selection
        for i in range(self.annotations_list.count()):
            list_item = self.annotations_list.item(i)
            if list_item.isSelected():
                scene_item = list_item.data(Qt.UserRole)
                if scene_item and scene_item.scene():
                    scene_item.setSelected(True)

    def add_new_class(self):
        """Add a new class"""
        name, ok = QInputDialog.getText(self, "Add Class", "Enter class name:")
        if ok and name and name.strip():
            name = name.strip()
            if name not in self.class_names:
                # Add to class list
                self.class_names.append(name)
                # Update class mapping
                self.class_name_to_id[name] = len(self.class_names) - 1

                # Add to combo box with color icon
                class_id = len(self.class_names) - 1
                pixmap = QPixmap(16, 16)
                if class_id < len(CLASS_COLORS):
                    pixmap.fill(CLASS_COLORS[class_id])
                else:
                    pixmap.fill(Qt.gray)
                self.class_combo.addItem(QIcon(pixmap), name)

                # Select the new class
                self.class_combo.setCurrentIndex(self.class_combo.count() - 1)
                self.current_class_id = class_id
                self.image_viewer.set_current_class(class_id)

                self.status_bar.showMessage(f"Added new class: {name}", 3000)
            else:
                self.status_bar.showMessage(f"Class '{name}' already exists", 3000)

    def create_menus(self):
        """Create menu bar with enhanced options"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open CSV", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_csv)
        file_menu.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_annotations)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        auto_save_action = QAction("Enable Auto-save", self)
        auto_save_action.setCheckable(True)
        auto_save_action.setChecked(True)
        auto_save_action.triggered.connect(self.toggle_auto_save)
        file_menu.addAction(auto_save_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")

        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()

        edit_menu.addAction("Copy", self.copy_annotation, QKeySequence.Copy)
        edit_menu.addAction("Paste", self.paste_annotation, QKeySequence.Paste)
        edit_menu.addAction("Delete", self.delete_selected, QKeySequence.Delete)
        edit_menu.addAction("Select All", self.select_all, QKeySequence.SelectAll)

    def toggle_auto_save(self, checked):
        """Toggle auto-save feature"""
        if checked:
            self.auto_save_timer.start(300000)  # 5 minutes
            self.status_bar.showMessage("Auto-save enabled", 2000)
        else:
            self.auto_save_timer.stop()
            self.status_bar.showMessage("Auto-save disabled", 2000)

    def create_icon(self, name):
        """Create a simple icon for the given tool name"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        if name == 'select':
            painter.setPen(QPen(Qt.black, 2))
            painter.drawLine(5, 5, 20, 20)
            painter.drawLine(20, 20, 15, 20)
            painter.drawLine(20, 20, 20, 15)

        elif name == 'box':
            painter.setPen(QPen(Qt.red, 2))
            painter.drawRect(5, 5, 22, 22)

        elif name == 'polygon':
            painter.setPen(QPen(Qt.green, 2))
            points = [QPointF(16, 5), QPointF(27, 12), QPointF(22, 27),
                      QPointF(10, 27), QPointF(5, 12)]
            painter.drawPolygon(QPolygonF(points))

        elif name == 'smart':
            painter.setPen(QPen(Qt.blue, 2))
            painter.setBrush(QBrush(Qt.yellow))
            painter.drawLine(8, 24, 20, 12)
            painter.drawLine(18, 5, 22, 9)
            painter.drawLine(20, 5, 20, 9)
            painter.drawLine(22, 7, 18, 7)

        elif name == 'undo':
            painter.setPen(QPen(Qt.black, 2))
            painter.drawArc(8, 8, 16, 16, 30 * 16, 270 * 16)
            painter.drawLine(8, 8, 8, 14)
            painter.drawLine(8, 8, 14, 8)

        elif name == 'redo':
            painter.setPen(QPen(Qt.black, 2))
            painter.drawArc(8, 8, 16, 16, -210 * 16, 270 * 16)
            painter.drawLine(24, 8, 24, 14)
            painter.drawLine(24, 8, 18, 8)

        elif name == 'copy':
            painter.setPen(QPen(Qt.black, 2))
            painter.drawRect(8, 8, 10, 12)
            painter.drawRect(12, 12, 10, 12)

        elif name == 'paste':
            painter.setPen(QPen(Qt.black, 2))
            painter.drawRect(10, 12, 12, 10)
            painter.drawRect(8, 8, 10, 8)
            painter.fillRect(9, 9, 8, 6, Qt.white)

        elif name == 'zoom_in':
            painter.setPen(QPen(Qt.black, 2))
            painter.drawEllipse(5, 5, 20, 20)
            painter.drawLine(15, 10, 15, 20)
            painter.drawLine(10, 15, 20, 15)

        elif name == 'zoom_out':
            painter.setPen(QPen(Qt.black, 2))
            painter.drawEllipse(5, 5, 20, 20)
            painter.drawLine(10, 15, 20, 15)

        elif name == 'zoom_fit':
            painter.setPen(QPen(Qt.black, 2))
            painter.drawRect(8, 8, 16, 16)

        elif name == 'pan':
            # Draw a hand icon
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(QBrush(Qt.lightGray))
            # Palm
            painter.drawEllipse(10, 14, 12, 10)
            # Fingers
            painter.drawRect(9, 8, 3, 8)  # thumb
            painter.drawRect(13, 6, 3, 8)  # index
            painter.drawRect(17, 5, 3, 9)  # middle
            painter.drawRect(21, 6, 3, 8)  # ring

        painter.end()
        return QIcon(pixmap)

    def set_tool(self, tool):
        """Set current annotation tool"""
        # Clean up existing preview thread before changing tool
        if hasattr(self, 'preview_thread') and self.preview_thread:
            self.preview_thread.stop()
            self.preview_thread.wait()
            self.preview_thread = None

        self.image_viewer.set_tool(tool)

        # Initialize SAM preview thread when entering smart mode
        if tool == 'smart' and self.sam_predictor and self.current_image is not None:
            # Reset SAM predictor with current image
            try:
                # Force reset the predictor with new image
                if hasattr(self.sam_predictor, 'reset_predictor'):
                    self.sam_predictor.reset_predictor()

                # Create new preview thread with fresh predictor state
                self.preview_thread = SAMPreviewThread(
                    self.sam_predictor,
                    cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                )
                self.image_viewer.set_preview_thread(self.preview_thread)
                self.image_viewer.smart_preview_enabled = True

                self.status_bar.showMessage("Smart mode: Hover to preview, click to apply, Shift+drag for box", 5000)
            except Exception as e:
                print(f"Error setting up smart mode: {e}")
                self.image_viewer.set_preview_thread(None)
        else:
            self.image_viewer.set_preview_thread(None)

    def on_class_changed(self, index):
        """Handle class selection change"""
        if 0 <= index < len(self.class_names):
            self.current_class_id = index
            self.image_viewer.set_current_class(index)

    def zoom_in(self):
        """Zoom in"""
        self.image_viewer.scale(1.2, 1.2)

    def zoom_out(self):
        """Zoom out"""
        self.image_viewer.scale(0.8, 0.8)

    def zoom_fit(self):
        """Fit image to window"""
        if self.image_viewer.pixmap_item:
            self.image_viewer.fitInView(
                self.image_viewer.scene.sceneRect(),
                Qt.KeepAspectRatio
            )

    def open_csv(self):
        """Open CSV file for editing"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Globinfo CSV",
            "",
            "CSV Files (*_globinfo.csv)"
        )
        if filename:
            self.load_project(filename)

    def load_project(self, csv_path):
        """Load project from CSV"""
        self.csv_path = csv_path
        self.project_dir = Path(csv_path).parent

        # ALWAYS Initialize temp_annotations
        if not hasattr(self, 'temp_annotations'):
            self.temp_annotations = {}

        # Try to read input directory from CSV metadata
        self.input_directory = None
        try:
            with open(csv_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('# input_directory:'):
                    self.input_directory = Path(first_line.split(':', 1)[1].strip())
        except:
            pass

        # Read CSV (skip comment lines)
        self.df = pd.read_csv(csv_path, comment='#')

        # Store original dataframe to preserve all columns
        self.original_df = self.df.copy()

        # Store original areas indexed by image and object_id
        self.original_areas = {}
        for _, row in self.df.iterrows():
            img_id = row['img_id']
            object_id = row.get('object_id', 0)
            area = row.get('area', np.nan)
            if img_id not in self.original_areas:
                self.original_areas[img_id] = {}
            self.original_areas[img_id][object_id] = area

        # Get list of all columns to preserve
        self.preserved_columns = self.df.columns.tolist()

        self.image_list = self.df['img_id'].unique().tolist()
        self.current_index = 0

        # Extract unique class names
        self._update_class_list()

        # Initialize SAM predictor
        self.init_sam_predictor()

        # Load first image
        if self.image_list:
            self.load_image(0)

    def _update_class_list(self):
        """Update class list from dataframe"""
        # Get unique class names
        if 'name' in self.df.columns:
            unique_names = self.df['name'].dropna().unique()
            self.class_names = sorted([str(name) for name in unique_names if str(name).strip()])
        else:
            self.class_names = []

        # If no classes, add a default
        if not self.class_names:
            self.class_names = ['object']

        # Create mapping
        self.class_name_to_id = {name: i for i, name in enumerate(self.class_names)}
        self.image_viewer.set_class_mapping(self.class_name_to_id)

        # Update combo box
        self.class_combo.clear()
        for i, name in enumerate(self.class_names):
            # Add colored icon
            pixmap = QPixmap(16, 16)
            if i < len(CLASS_COLORS):
                pixmap.fill(CLASS_COLORS[i])
            else:
                pixmap.fill(Qt.gray)
            self.class_combo.addItem(QIcon(pixmap), name)

        self.class_combo.setCurrentIndex(0)

    def init_sam_predictor(self):
        """Initialize SAM2 predictor with better error handling"""
        if SAM2ImagePredictor is None:
            QMessageBox.warning(
                self,
                "SAM Not Available",
                "SAM2 is not installed. Smart annotation will be disabled.\n"
                "To enable it, install sam2 package."
            )
            self.sam_predictor = None
            self.smart_action.setEnabled(False)
            return

        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.sam_predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2.1-hiera-large",
                device=device
            )

            # Add reset method if not present
            if not hasattr(self.sam_predictor, 'reset_predictor'):
                self.sam_predictor.reset_predictor = lambda: setattr(self.sam_predictor, '_features', None)

            self.status_bar.showMessage(f"SAM2 loaded on {device}", 3000)
            self.smart_action.setEnabled(True)
        except Exception as e:
            QMessageBox.warning(
                self,
                "SAM Loading Error",
                f"Failed to load SAM2 model: {str(e)}\nSmart annotation will be disabled."
            )
            self.sam_predictor = None
            self.smart_action.setEnabled(False)

    def load_image(self, index):
        """Load image at given index with better resource management"""
        if not (0 <= index < len(self.image_list)):
            return

        # Clean up before loading new image
        if hasattr(self, 'preview_thread') and self.preview_thread:
            self.preview_thread.stop()
            self.preview_thread.wait()
            self.preview_thread = None

        # Force garbage collection
        gc.collect()

        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.current_index = index
        self.current_image_path = self.image_list[index]

        # Try multiple locations to find the image
        possible_paths = []

        # 1. If we have input_directory from metadata, try there first
        if self.input_directory:
            possible_paths.append(self.input_directory / self.current_image_path)

        # 2. Relative to CSV location
        possible_paths.append(self.project_dir / self.current_image_path)

        # 3. In parent directory of CSV
        possible_paths.append(self.project_dir.parent / self.current_image_path)

        # 4. In grandparent directory
        possible_paths.append(self.project_dir.parent.parent / self.current_image_path)

        # 5. Absolute path
        possible_paths.append(Path(self.current_image_path))

        # 6. Look for image filename in common subdirectories
        img_filename = Path(self.current_image_path).name
        for subdir in ['', 'img', 'images', '..', '../..']:
            possible_paths.append(self.project_dir / subdir / img_filename)

        # Try to find the image
        full_path = None
        for path in possible_paths:
            try:
                if path.exists() and path.is_file():
                    test_img = cv2.imread(str(path))
                    if test_img is not None:
                        full_path = path
                        self.current_image = test_img
                        break
            except:
                continue

        if full_path is None:
            # Show error message
            error_msg = f"Cannot find image: {self.current_image_path}\n\n"
            error_msg += "Searched in:\n"
            for i, p in enumerate(possible_paths[:5]):
                error_msg += f"{i + 1}. {p}\n"
            error_msg += "\nPlease ensure the images are in their original location."
            QMessageBox.warning(self, "Image Not Found", error_msg)
            return

        # IMPORTANT: Set original dimensions BEFORE displaying the image
        self.img_height_orig, self.img_width_orig = self.current_image.shape[:2]

        # Display image with specified display size
        display_size = getattr(self, 'display_size', 1056)
        self.image_viewer.set_image(self.current_image, display_size)

        # Load annotations AFTER image is displayed and scale factors are set
        self.load_annotations()

        # Update UI
        self.setWindowTitle(
            f"NemaCounter - Manual Edition - {self.current_image_path} "
            f"({index + 1}/{len(self.image_list)})"
        )
        self.prev_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.image_list) - 1)

        # Update undo/redo actions
        self.update_undo_redo_actions()

        # Set focus to image viewer for keyboard shortcuts to work
        self.image_viewer.setFocus()

    def update_size_slider_range(self):
        """Update size slider range based on current annotations"""
        if not self.image_viewer.get_annotations():
            return

        # Find min and max areas
        areas = [self.get_annotation_area(item) for item in self.image_viewer.get_annotations()]
        if areas:
            max_area = max(areas)
            # Set range with some padding
            self.size_slider.setRange(0, int(max_area * 1.2))
            # Update tick interval
            self.size_slider.setTickInterval(max(1, int(max_area / 10)))

    def load_annotations(self):
        """Load annotations for current image - checks temp_annotations first"""
        self.annotations_list.clear()

        # IMPORTANT: Always get scale factors from image viewer
        scale_x = self.image_viewer.scale_x
        scale_y = self.image_viewer.scale_y

        # Clear any existing annotations first
        for item in self.image_viewer.scene.items():
            if isinstance(item, SimpleAnnotationItem):
                self.image_viewer.scene.removeItem(item)

        # Debug: Print scale factors for verification
        print(f"Loading annotations for {self.current_image_path}")
        print(f"Original size: {self.img_width_orig}x{self.img_height_orig}")
        print(f"Display size: {self.image_viewer.display_width}x{self.image_viewer.display_height}")
        print(f"Scale factors: {scale_x:.3f}, {scale_y:.3f}")

        # CHECK TEMP_ANNOTATIONS FIRST (this is the key fix!)
        if hasattr(self, 'temp_annotations') and self.current_image_path in self.temp_annotations:
            # Load from temporary saved annotations
            annotations = self.temp_annotations[self.current_image_path]
            print(f"Loading {len(annotations)} annotations from temp storage")

            for i, ann in enumerate(annotations):
                obj_type = ann.get('object_type', 'box')

                # Get class ID from name
                class_name = ann.get('name', 'object')
                class_id = self.class_name_to_id.get(class_name, 0)

                if obj_type == 'box':
                    # Scale box coordinates from original to display
                    rect = QRectF(
                        float(ann['xmin']) * scale_x,
                        float(ann['ymin']) * scale_y,
                        float(ann['xmax'] - ann['xmin']) * scale_x,
                        float(ann['ymax'] - ann['ymin']) * scale_y
                    )
                    item = create_box_item(rect, class_id)
                    item.confidence = float(ann.get('confidence', 1.0))
                    item._modified = ann.get('modified', False)
                    # <<<< START FIX: Directly use the area saved in the temp annotation.
                    item._original_area = ann.get('area')
                    # <<<< END FIX
                    item._original_data = ann.copy()
                    self.image_viewer.scene.addItem(item)
                    self.add_to_list(item)

                elif obj_type == 'polygon' and ann.get('contours'):
                    try:
                        contours = json.loads(ann['contours'])
                        if contours:
                            # Handle different contour formats
                            if isinstance(contours[0], list):
                                if len(contours[0]) > 0 and isinstance(contours[0][0], list):
                                    points = contours[0]
                                else:
                                    points = contours
                            else:
                                continue

                            # Scale points from original to display coordinates
                            qpoints = [QPointF(float(p[0]) * scale_x, float(p[1]) * scale_y)
                                       for p in points]

                            # Ensure polygon is closed
                            if qpoints and qpoints[0] != qpoints[-1]:
                                qpoints.append(qpoints[0])

                            item = create_polygon_item(qpoints, class_id)
                            item.confidence = float(ann.get('confidence', 1.0))
                            item._modified = ann.get('modified', False)
                            # <<<< START FIX: Directly use the area saved in the temp annotation.
                            item._original_area = ann.get('area')
                            # <<<< END FIX
                            item._original_data = ann.copy()
                            self.image_viewer.scene.addItem(item)
                            self.add_to_list(item)
                    except Exception as e:
                        print(f"Error loading polygon from temp: {e}")

                elif obj_type == 'mask' and ann.get('contours'):
                    try:
                        # Pass the annotation dict directly to reconstruction
                        mask = self.reconstruct_mask_from_contours(ann, scale_x, scale_y)
                        if mask is not None:
                            item = create_mask_item(mask, QPointF(0, 0), class_id)
                            item.confidence = float(ann.get('confidence', 1.0))
                            item._modified = ann.get('modified', False)
                            # <<<< START FIX: Directly use the area saved in the temp annotation.
                            item._original_area = ann.get('area')
                            # <<<< END FIX
                            item._original_data = ann.copy()
                            self.image_viewer.scene.addItem(item)
                            self.add_to_list(item)
                    except Exception as e:
                        print(f"Error loading mask from temp: {e}")
        else:
            # Fall back to loading from original dataframe
            img_df = self.df[self.df['img_id'] == self.current_image_path]

            for idx, row in img_df.iterrows():
                # Skip placeholder rows (images with 0 detections have object_id = 0)
                object_id = row.get('object_id', idx + 1)
                if object_id == 0 or pd.isna(object_id):
                    continue

                obj_type = str(row.get('object_type', 'box')).lower()

                # Skip rows with empty object_type (placeholder rows)
                if not obj_type or obj_type == 'nan':
                    continue

                # Get class ID from name
                class_name = str(row.get('name', 'object'))
                class_id = self.class_name_to_id.get(class_name, 0)

                # Get object_id and original area
                object_id = row.get('object_id', idx + 1)
                original_area = row.get('area', np.nan)

                if obj_type == 'box':
                    # Scale box coordinates from original to display
                    rect = QRectF(
                        float(row['xmin']) * scale_x,
                        float(row['ymin']) * scale_y,
                        float(row['xmax'] - row['xmin']) * scale_x,
                        float(row['ymax'] - row['ymin']) * scale_y
                    )
                    item = create_box_item(rect, class_id)
                    item.confidence = float(row.get('confidence', 1.0))
                    item._modified = False  # Not modified when loading from original
                    item._original_area = original_area
                    item._original_data = row.to_dict()
                    self.image_viewer.scene.addItem(item)
                    self.add_to_list(item)

                elif obj_type == 'polygon' and pd.notna(row.get('contours')):
                    try:
                        contours = json.loads(row['contours'])
                        if contours:
                            # Handle different contour formats
                            if isinstance(contours[0], list):
                                if len(contours[0]) > 0 and isinstance(contours[0][0], list):
                                    points = contours[0]
                                else:
                                    points = contours
                            else:
                                continue

                            # Scale points from original to display coordinates
                            qpoints = [QPointF(float(p[0]) * scale_x, float(p[1]) * scale_y)
                                       for p in points]

                            # Ensure polygon is closed
                            if qpoints and qpoints[0] != qpoints[-1]:
                                qpoints.append(qpoints[0])

                            item = create_polygon_item(qpoints, class_id)
                            item.confidence = float(row.get('confidence', 1.0))
                            item._modified = False
                            item._original_area = original_area
                            item._original_data = row.to_dict()
                            self.image_viewer.scene.addItem(item)
                            self.add_to_list(item)
                    except Exception as e:
                        print(f"Error loading polygon: {e}")

                elif obj_type == 'mask' and pd.notna(row.get('contours')):
                    try:
                        # Pass the actual scale factors
                        mask = self.reconstruct_mask_from_contours(row, scale_x, scale_y)
                        if mask is not None:
                            item = create_mask_item(mask, QPointF(0, 0), class_id)
                            item.confidence = float(row.get('confidence', 1.0))
                            item._modified = False
                            item._original_area = original_area
                            item._original_data = row.to_dict()
                            self.image_viewer.scene.addItem(item)
                            self.add_to_list(item)
                    except Exception as e:
                        print(f"Error loading mask: {e}")

        self.update_stats()
        self.update_size_slider_range()
        # Reset slider when loading new image
        self.size_slider.setValue(0)

        # Save initial state for undo
        self.image_viewer.save_current_state("Initial load")

    def reconstruct_mask_from_contours(self, data, scale_x=1.0, scale_y=1.0):
        """Reconstruct mask from contours JSON at display size with closed contours"""
        try:
            # IMPORTANT: Use the actual scale factors from image viewer
            actual_scale_x = self.image_viewer.scale_x
            actual_scale_y = self.image_viewer.scale_y

            # Create mask at display size
            h = self.image_viewer.display_height
            w = self.image_viewer.display_width
            mask = np.zeros((h, w), dtype=np.uint8)

            # Handle both dict (from temp_annotations) and pandas row
            if isinstance(data, dict):
                contours_json = data.get('contours', '')
            else:
                contours_json = data.get('contours', '')

            if not contours_json:
                return None

            contours = json.loads(contours_json)
            for contour in contours:
                pts = np.array(contour, dtype=np.float32)
                # Contours are in original image coordinates
                # Scale to display coordinates using actual scale factors
                pts[:, 0] *= actual_scale_x
                pts[:, 1] *= actual_scale_y

                # Ensure contour is closed
                if pts.ndim == 2 and pts.shape[0] >= 3:
                    if not np.array_equal(pts[0], pts[-1]):
                        pts = np.vstack([pts, pts[0:1]])  # Add first point at the end

                    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)

            return mask
        except Exception as e:
            print(f"Error reconstructing mask: {e}")
            return None

    def add_to_list(self, item):
        """Add annotation item to list"""
        type_str = item.annotation_type.capitalize()
        confidence = getattr(item, 'confidence', 1.0)

        # Get class name
        class_name = 'object'
        if hasattr(item, 'class_id') and item.class_id is not None:
            if 0 <= item.class_id < len(self.class_names):
                class_name = self.class_names[item.class_id]

        list_item = QListWidgetItem(f"{type_str} - {class_name} (conf: {confidence:.2f})")
        list_item.setData(Qt.UserRole, item)

        # Set icon color based on class
        pixmap = QPixmap(16, 16)
        if hasattr(item, 'class_id') and item.class_id is not None and 0 <= item.class_id < len(CLASS_COLORS):
            pixmap.fill(CLASS_COLORS[item.class_id])
        else:
            pixmap.fill(ANNOTATION_COLORS[item.annotation_type])
        list_item.setIcon(QIcon(pixmap))

        self.annotations_list.addItem(list_item)

    def on_annotation_added(self, item):
        """Handle new annotation"""
        item.confidence = self.conf_slider.value() / 100.0
        item.class_id = self.current_class_id
        self.add_to_list(item)
        self.update_stats()
        self.update_undo_redo_actions()

    def on_annotation_removed(self, item):
        """Handle annotation removal"""
        for i in range(self.annotations_list.count()):
            if self.annotations_list.item(i).data(Qt.UserRole) == item:
                self.annotations_list.takeItem(i)
                break
        self.update_stats()
        self.update_undo_redo_actions()

    def on_annotation_modified(self, item):
        """Handle annotation modification"""
        self.update_stats()
        self.update_undo_redo_actions()

    def update_stats(self):
        """Update annotation statistics"""
        counts = {'box': 0, 'polygon': 0, 'mask': 0}
        class_counts = {}

        # Only count visible annotations
        visible_annotations = [item for item in self.image_viewer.get_annotations()
                               if item.isVisible() and item.opacity() == 1.0]

        for item in visible_annotations:
            counts[item.annotation_type] += 1

            # Count by class
            if hasattr(item, 'class_id') and item.class_id is not None:
                if 0 <= item.class_id < len(self.class_names):
                    class_name = self.class_names[item.class_id]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

        total = sum(counts.values())

        # Build stats text
        stats_text = f"Total Visible: {total}\n"
        stats_text += f"Types: Box: {counts['box']}, Polygon: {counts['polygon']}, Mask: {counts['mask']}\n"
        if class_counts:
            stats_text += "Classes: " + ", ".join([f"{name}: {count}" for name, count in class_counts.items()])

        self.stats_label.setText(stats_text)

    def prev_image(self):
        """Load previous image"""
        if self.current_index > 0:
            # Always save current annotations before navigating
            self.save_current_annotations()
            self.load_image(self.current_index - 1)

    def next_image(self):
        """Load next image"""
        if self.current_index < len(self.image_list) - 1:
            # Always save current annotations before navigating
            self.save_current_annotations()
            self.load_image(self.current_index + 1)

    def save_and_continue(self):
        """Save and go to next image"""
        # Save current annotations
        self.save_current_annotations()

        # Show feedback
        self.status_bar.showMessage("Annotations saved!", 2000)

        # Move to next image if available
        if self.current_index < len(self.image_list) - 1:
            self.load_image(self.current_index + 1)
        else:
            QMessageBox.information(self, "Complete", "All images have been annotated!")
            self.save_annotations()

    def save_annotations(self):
        """Save all annotations to file with closed contours"""
        # Save current image first
        self.save_current_annotations()

        if hasattr(self, 'temp_annotations'):
            # Build new dataframe with all expected columns
            rows = []

            # Get the project_id from original data
            project_id = getattr(self, 'project_id', None)
            if project_id is None and self.original_df is not None and 'project_id' in self.original_df.columns:
                project_id = self.original_df.iloc[0]['project_id'] if len(self.original_df) > 0 else 'manual_edition'
            if project_id is None:
                project_id = Path(self.csv_path).stem.replace('_globinfo', '')

            for img_path, annotations in self.temp_annotations.items():
                for i, ann in enumerate(annotations):
                    # Start with a complete row structure matching detection output
                    row = {
                        'img_id': img_path,
                        'object_id': i + 1,
                        'xmin': ann.get('xmin', 0),
                        'ymin': ann.get('ymin', 0),
                        'xmax': ann.get('xmax', 0),
                        'ymax': ann.get('ymax', 0),
                        'confidence': ann.get('confidence', 1.0),
                        'class': ann.get('class', 0),
                        'name': ann.get('name', 'object'),
                        'area': ann.get('area', np.nan),
                        'contours': ann.get('contours', ''),
                        'object_type': ann.get('object_type', 'box'),
                        'project_id': project_id
                    }

                    # Preserve any additional columns from original data
                    if self.original_df is not None:
                        original_row = self.original_df[self.original_df['img_id'] == img_path]
                        if len(original_row) > 0:
                            for col in self.preserved_columns:
                                if col not in row and col in original_row.columns:
                                    # Use first value for this image as default
                                    row[col] = original_row.iloc[0][col]

                    rows.append(row)

            if rows:
                df_new = pd.DataFrame(rows)

                # Ensure all columns are present in the expected order
                expected_columns = ['img_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax',
                                    'confidence', 'class', 'name', 'area', 'contours',
                                    'object_type', 'project_id']

                # Add any additional columns from original data
                for col in self.preserved_columns:
                    if col not in expected_columns:
                        expected_columns.append(col)

                # Ensure all expected columns exist
                for col in expected_columns:
                    if col not in df_new.columns:
                        if col in ['xmin', 'ymin', 'xmax', 'ymax', 'object_id']:
                            df_new[col] = 0
                        elif col in ['confidence']:
                            df_new[col] = 1.0
                        elif col in ['area']:
                            df_new[col] = np.nan
                        elif col in ['class']:
                            df_new[col] = 0
                        elif col in ['name']:
                            df_new[col] = 'object'
                        elif col in ['contours']:
                            df_new[col] = ''
                        elif col in ['object_type']:
                            df_new[col] = 'box'
                        elif col in ['project_id']:
                            df_new[col] = project_id
                        else:
                            df_new[col] = np.nan

                # Enforce data types
                for col in ['xmin', 'ymin', 'xmax', 'ymax', 'object_id']:
                    if col in df_new.columns:
                        df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0).astype(int)

                for col in ['confidence', 'area']:
                    if col in df_new.columns:
                        df_new[col] = pd.to_numeric(df_new[col], errors='coerce')

                for col in ['class']:
                    if col in df_new.columns:
                        df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0).astype(int)

                # Reorder columns to match detection output
                df_new = df_new[expected_columns]

                # Use output_directory if available, otherwise use project directory
                output_dir = getattr(self, 'output_directory', self.project_dir)
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save with consistent naming
                output_path = output_dir / f"{project_id}_edition_globinfo.csv"

                # Write with metadata comment (same format as detection)
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    # Preserve the input directory metadata
                    if hasattr(self, 'input_directory') and self.input_directory:
                        f.write(f"# input_directory: {self.input_directory}\n")
                    df_new.to_csv(f, index=False, quoting=csv.QUOTE_ALL)

                # Create summary with same format as detection
                summary_df = self.create_summary(df_new, project_id)
                summary_path = output_dir / f"{project_id}_edition_summary.csv"
                summary_df.to_csv(summary_path, index=False, quoting=csv.QUOTE_ALL)

                QMessageBox.information(
                    self,
                    "Saved",
                    f"Annotations saved to:\n{output_path}\n\nSummary saved to:\n{summary_path}"
                )
            else:
                QMessageBox.warning(self, "No Data", "No annotations to save.")

    def create_summary(self, df, project_id):
        """Create summary dataframe matching detection format"""
        summary_rows = []

        # Group by image and class name
        for img_id in df['img_id'].unique():
            img_df = df[df['img_id'] == img_id]

            # Get unique class names in this image
            class_names = img_df['name'].unique()

            for class_name in class_names:
                class_df = img_df[img_df['name'] == class_name]

                # Calculate statistics
                areas = class_df['area'].dropna()

                summary_rows.append({
                    'project_id': project_id,
                    'img_id': img_id,
                    'class_name': class_name,
                    'count': len(class_df),
                    'conf_mean': class_df['confidence'].mean(),
                    'conf_std': class_df['confidence'].std() if len(class_df) > 1 else 0,
                    'area_mean': areas.mean() if len(areas) > 0 else np.nan,
                    'area_std': areas.std() if len(areas) > 1 else 0
                })

        return pd.DataFrame(summary_rows)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts at window level"""
        # First check if image viewer should handle it
        if self.image_viewer.hasFocus():
            # Let image viewer handle it first
            self.image_viewer.keyPressEvent(event)
            if event.isAccepted():
                return

        # Navigation keys work regardless of focus
        if event.key() == Qt.Key_Left:
            self.prev_image()
        elif event.key() == Qt.Key_Right:
            self.next_image()
        elif event.key() in [Qt.Key_Delete, Qt.Key_Backspace]:
            # If image viewer doesn't have focus, still try to delete
            self.delete_selected()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle window close event with cleanup"""
        # Stop auto-save timer
        self.auto_save_timer.stop()

        # Stop preview thread if running
        if hasattr(self, 'preview_thread') and self.preview_thread:
            self.preview_thread.stop()
            self.preview_thread.wait()

        # Clean up SAM predictor
        if self.sam_predictor is not None:
            del self.sam_predictor
            self.sam_predictor = None

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        event.accept()

    def save_current_annotations(self):
        """
        Saves the annotations for the currently displayed image into the
        self.temp_annotations dictionary. This is the crucial step for
        persisting changes between image navigation.

        FIXED: Now preserves original area values for unmodified annotations.
        FIXED: Now filters annotations based on the size threshold slider.
        """
        # Ensure there is a current image to save annotations for
        if not self.current_image_path:
            return

        # Get the scaling factors used to display the image
        scale_x = self.image_viewer.scale_x
        scale_y = self.image_viewer.scale_y

        # Prevent division by zero if image loading failed
        if scale_x == 0 or scale_y == 0:
            print("Warning: Scale factors are zero, cannot save annotations.")
            return

        # Retrieve all annotation items from the graphics scene
        current_items = self.image_viewer.get_annotations()
        temp_list_for_image = []

        saved_object_index = 0
        for item in current_items:
            # Only save items that are fully opaque (i.e., not filtered out by the size slider)
            if item.opacity() < 1.0:
                continue

            # Get common properties
            item_pos = item.pos()  # This is the offset from moving the item
            class_name = self.class_names[item.class_id] if 0 <= item.class_id < len(self.class_names) else 'object'

            ann_data = {
                'object_type': item.annotation_type,
                'confidence': item.confidence,
                'class': item.class_id,
                'name': class_name,
                'position': (item_pos.x(), item_pos.y()),  # Save the moved position
                'modified': getattr(item, '_modified', False),  # Track if modified
                'object_id': saved_object_index + 1  # Use the new counter for consistent IDs
            }

            # Check if this annotation has an original area and hasn't been modified
            if hasattr(item, '_original_area') and item._original_area is not None and not item._modified:
                # PRESERVE THE ORIGINAL AREA FOR UNMODIFIED ANNOTATIONS
                ann_data['area'] = item._original_area
                print(f"Preserving original area {item._original_area} for unmodified {item.annotation_type}")
            else:
                # For modified or new annotations, calculate the area
                # This will be done below for each type
                print(f"Calculating new area for modified/new {item.annotation_type}")

            # Convert coordinates from display back to original image size
            if item.annotation_type == 'box':
                rect = item.rect
                # Combine original rect position with the move offset, then un-scale
                xmin = (rect.x() + item_pos.x()) / scale_x
                ymin = (rect.y() + item_pos.y()) / scale_y
                xmax = (rect.right() + item_pos.x()) / scale_x
                ymax = (rect.bottom() + item_pos.y()) / scale_y

                ann_data.update({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

                # Only calculate new area if modified or no original area
                if 'area' not in ann_data:
                    ann_data['area'] = abs(xmax - xmin) * abs(ymax - ymin)

                # Recreate contours from the un-scaled box
                contours = [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]]
                ann_data['contours'] = json.dumps(contours)

            elif item.annotation_type in ['polygon', 'mask']:
                path = item.path()
                bounding_rect = path.boundingRect()

                # Un-scale the bounding box to original coordinates
                xmin = (bounding_rect.x() + item_pos.x()) / scale_x
                ymin = (bounding_rect.y() + item_pos.y()) / scale_y
                xmax = (bounding_rect.right() + item_pos.x()) / scale_x
                ymax = (bounding_rect.bottom() + item_pos.y()) / scale_y
                ann_data.update({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

                # Un-scale each point in the path to get original contours
                all_contours = []
                for sub_polygon in path.toSubpathPolygons():
                    contour_points = []
                    for p in sub_polygon:
                        orig_x = (p.x() + item_pos.x()) / scale_x
                        orig_y = (p.y() + item_pos.y()) / scale_y
                        contour_points.append([orig_x, orig_y])
                    all_contours.append(contour_points)
                ann_data['contours'] = json.dumps(all_contours)

                # Only calculate new area if modified or no original area
                if 'area' not in ann_data:
                    # To match detection.py exactly, we re-render the mask at the
                    # original resolution and sum the resulting pixels.
                    h, w = self.img_height_orig, self.img_width_orig
                    final_mask_canvas = np.zeros((h, w), dtype=np.uint8)

                    # Draw the final, original-scale contours onto this canvas.
                    drawable_contours = [np.array(c, dtype=np.int32) for c in all_contours]
                    cv2.fillPoly(final_mask_canvas, drawable_contours, 1)

                    # Calculate area by summing the pixels, just like in detection.py.
                    final_area = np.sum(final_mask_canvas)
                    ann_data['area'] = float(final_area)

            temp_list_for_image.append(ann_data)
            saved_object_index += 1  # Increment the counter for the next saved object

        # Store the processed list in the temporary dictionary
        self.temp_annotations[self.current_image_path] = temp_list_for_image
        self.status_bar.showMessage(f"'{Path(self.current_image_path).name}' annotations cached.", 2000)


# ============================================
# Backward compatibility function
# ============================================

def edition_workflow(input_file, output_directory, project_id, use_gpu=True, input_directory=None, **kwargs):
    """
    Launch professional annotation interface for editing with enhanced features.

    FIXED: Now preserves original area values for unmodified annotations!

    Features:
    - Preserves exact area values for unmodified annotations
    - Properly saves moved annotation positions
    - Closed polygons and contours
    - Undo/Redo support (Ctrl+Z / Ctrl+Shift+Z)
    - Copy/Paste annotations
    - Auto-save every 5 minutes
    - Better resource management
    - Size filtering
    - Class management
    - Keyboard shortcuts for navigation

    Args:
        input_file: Path to input CSV file
        output_directory: Directory for output files
        project_id: Project identifier
        use_gpu: Whether to use GPU for SAM
        input_directory: Optional path to directory containing original images
        **kwargs: Additional arguments
    """
    # Check if CSV file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        return

    # Create minimal Qt application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Set dark palette for better visibility
    from PyQt5.QtGui import QPalette
    from PyQt5.QtCore import Qt

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(palette)

    try:
        # Create and configure main window
        window = EditionWindow()

        # Store output directory and project_id for later use
        window.output_directory = output_directory
        window.project_id = project_id
        window.use_gpu = use_gpu

        # Get display size from kwargs if provided
        display_size = kwargs.get('input_size', 1056)
        window.display_size = display_size

        # Load project
        window.load_project(input_file)

        # Override input directory if provided explicitly
        if input_directory:
            window.input_directory = Path(input_directory)

        # Show window maximized
        window.showMaximized()

        # Run application
        app.exec_()

    except Exception as e:
        print(f"Error in edition window: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("Manual edition completed successfully!")