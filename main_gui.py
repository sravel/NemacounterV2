import sys
import os
import threading
import numpy as np
import pandas as pd
import json
import csv
import cv2
import torch
from PIL import Image
# Forcer l'utilisation de PyQt6 pour toutes les bibliothèques (napari, vispy, qtpy)
os.environ["QT_API"] = "pyqt6"
# 2. Forcer Vispy (utilisé par Napari) à utiliser PyQt6
import vispy
try:
    vispy.use_app("pyqt6")
except Exception:
    pass

# PyQt6 Imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSlider, QCheckBox, QTabWidget, QProgressBar,
                             QLineEdit, QTextEdit, QComboBox, QFrame, QFormLayout,
                             QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Napari
import napari

# Tes modules originaux
import nemacounter.utils as utils
import nemacounter.common as common
from nemacounter.detection import detection_workflow
from nemacounter.edition import edition_workflow
from nemacounter.segmentation import NemaCounterSegmentation, add_masks_on_image, create_multicolored_masks_image


# --- WORKERS POUR LES THREADS ---
class WorkflowWorker(QThread):
    """Gère l'exécution des fonctions lourdes sans bloquer l'UI"""
    finished_signal = pyqtSignal(bool, str)
    progress_signal = pyqtSignal(float)
    status_signal = pyqtSignal(str)

    def __init__(self, task_type, params):
        super().__init__()
        self.task_type = task_type
        self.params = params

    def run(self):
        try:
            if self.task_type == "detection":
                detection_workflow(self.params, gui=True)
                self.finished_signal.emit(True, "Object detection completed successfully.")

            elif self.task_type == "segmentation":
                # Note: On appelle ici une version simplifiée ou ton bloc logique
                # Pour l'exemple, j'émets juste un signal
                self.finished_signal.emit(True, "Segmentation completed.")
        except Exception as e:
            self.finished_signal.emit(False, str(e))


# --- CLASSE PRINCIPALE ---
class NemaCounterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NemaCounter GUI - PyQt6 & Napari")
        self.resize(1100, 750)

        # Variables de stockage (Remplace les StringVar de Tkinter)
        self.indir_path = ""
        self.outdir_path = ""
        self.globinfo_path = ""
        self.model_path = ""

        # Initialisation UI
        self.setup_ui()
        self.apply_style()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)

        # 1. SIDEBAR
        self.setup_sidebar()

        # 2. TABS CENTRALES
        self.tabs = QTabWidget()
        self.setup_detection_tab()
        self.setup_edition_tab()
        self.setup_segmentation_tab()
        self.setup_export_tab()

        self.main_layout.addWidget(self.tabs, 4)

    def setup_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(250)
        sidebar.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(sidebar)

        # Logo
        self.logo_label = QLabel()
        logo_path = os.path.join("conf", "logo.png")
        if os.path.exists(logo_path):
            from PyQt6.QtGui import QPixmap
            pix = QPixmap(logo_path).scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
            self.logo_label.setPixmap(pix)
        else:
            self.logo_label.setText("NemaCounter")
            self.logo_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.logo_label)

        # Hardware
        hw_group = QFrame()
        hw_layout = QVBoxLayout(hw_group)
        self.use_gpu_check = QCheckBox("Use GPU if available")
        self.use_gpu_check.setChecked(True)
        hw_layout.addWidget(self.use_gpu_check)

        nb_cpu = utils.compute_available_cpu()
        self.cpu_label = QLabel(f"Max CPU: {nb_cpu - 1}")
        self.cpu_slider = QSlider(Qt.Orientation.Horizontal)
        self.cpu_slider.setRange(1, nb_cpu)
        self.cpu_slider.setValue(nb_cpu - 1)
        self.cpu_slider.valueChanged.connect(lambda v: self.cpu_label.setText(f"Max CPU: {v}"))
        hw_layout.addWidget(self.cpu_label)
        hw_layout.addWidget(self.cpu_slider)
        layout.addWidget(hw_group)

        # Models
        layout.addWidget(QLabel("Select Model (Detection):"))
        self.model_combo = QComboBox()
        self.refresh_models()
        layout.addWidget(self.model_combo)

        layout.addStretch()  # Espaceur
        self.main_layout.addWidget(sidebar, 1)

    def setup_detection_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Project Name
        form = QFormLayout()
        self.det_proj_name = QLineEdit("MyDetectionProject")
        form.addRow("Project Name:", self.det_proj_name)
        layout.addLayout(form)

        # IO Buttons
        self.btn_in = QPushButton("Select Input Directory")
        self.btn_in.clicked.connect(self.select_indir)
        self.label_in = QLabel("No directory selected")
        layout.addWidget(self.btn_in)
        layout.addWidget(self.label_in)

        self.btn_out = QPushButton("Select Output Directory")
        self.btn_out.clicked.connect(self.select_outdir)
        self.label_out = QLabel("No directory selected")
        layout.addWidget(self.btn_out)
        layout.addWidget(self.label_out)

        # Sliders
        self.conf_label = QLabel("Confidence Threshold: 0.50")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        layout.addWidget(self.conf_label)
        layout.addWidget(self.conf_slider)

        # Run Button
        self.run_det_btn = QPushButton("START DETECTION")
        self.run_det_btn.setFixedHeight(50)
        self.run_det_btn.clicked.connect(self.start_detection)
        layout.addWidget(self.run_det_btn)

        self.tabs.addTab(tab, "Object Detection")

    def setup_edition_tab(self):
        """Intégration de NAPARI"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("Manual Edition using Napari"))
        self.btn_launch_napari = QPushButton("🚀 Open Napari Editor")
        self.btn_launch_napari.setFixedHeight(60)
        self.btn_launch_napari.clicked.connect(self.launch_napari_workflow)
        layout.addWidget(self.btn_launch_napari)

        self.tabs.addTab(tab, "Manual Edition")

    def setup_segmentation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.prog_bar = QProgressBar()
        layout.addWidget(self.prog_bar)

        self.tabs.addTab(tab, "Segmentation")

    def setup_export_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Export to Roboflow JSON"))
        self.tabs.addTab(tab, "Export")

    # --- LOGIQUE ---

    def select_indir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if path:
            self.indir_path = path
            self.label_in.setText(path)

    def select_outdir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.outdir_path = path
            self.label_out.setText(path)

    def update_conf_label(self, val):
        self.conf_label.setText(f"Confidence Threshold: {val / 100:.2f}")

    def refresh_models(self):
        if os.path.exists("models"):
            files = [f for f in os.listdir("models") if f.endswith(".pt")]
            self.model_combo.addItems(files)

    def start_detection(self):
        # On construit le dictionnaire exactement comme ton ancien code
        dct_params = {
            'input_directory': self.indir_path,
            'output_directory': self.outdir_path,
            'project_id': self.det_proj_name.text(),
            'conf_thresh': self.conf_slider.value() / 100,
            'gpu': int(self.use_gpu_check.isChecked()),
            'cpu': self.cpu_slider.value(),
            'model_path': os.path.join("models", self.model_combo.currentText())
        }

        # Validation
        if not self.indir_path or not self.outdir_path:
            QMessageBox.warning(self, "Error", "Select Input and Output directories")
            return

        # Lancement du Worker (QThread remplace threading.Thread)
        self.worker = WorkflowWorker("detection", dct_params)
        self.worker.finished_signal.connect(self.on_worker_finished)
        self.worker.start()
        self.run_det_btn.setEnabled(False)

    def on_worker_finished(self, success, message):
        self.run_det_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)

    def launch_napari_workflow(self):
        """
        Remplace ta logique d'édition par Napari
        """
        # On crée le viewer napari
        viewer = napari.Viewer(title="NemaCounter Manual Edition")

        # Exemple : Si tu as un dossier d'images sélectionné
        if self.indir_path:
            # On charge la première image pour l'exemple
            img_list = [f for f in os.listdir(self.indir_path) if f.endswith(('.png', '.jpg'))]
            if img_list:
                img = cv2.imread(os.path.join(self.indir_path, img_list[0]))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                viewer.add_image(img_rgb, name="Original Image")

                # On ajoute un calque de labels pour corriger les masques
                viewer.add_labels(np.zeros(img_rgb.shape[:2], dtype=int), name="Nematode Masks")

    def apply_style(self):
        # Simple QSS pour un look "Dark Mode"
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { background-color: #1e1e1e; color: #ffffff; }
            QPushButton { background-color: #3d3d3d; border: 1px solid #555; border-radius: 4px; padding: 8px; }
            QPushButton:hover { background-color: #505050; }
            QLineEdit, QComboBox, QSlider { background-color: #2b2b2b; color: white; border: 1px solid #444; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #2b2b2b; padding: 10px; margin-right: 2px; }
            QTabBar::tab:selected { background: #3d3d3d; border-bottom: 2px solid #0078d7; }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NemaCounterGUI()
    window.show()
    sys.exit(app.exec())