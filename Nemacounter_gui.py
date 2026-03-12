# Nemacounter_gui.py

import tkinter as tk
import customtkinter as CTK
from PIL import Image as PILimage
from tkinter import filedialog, messagebox
import os
import numpy as np
import pandas as pd
import sys
import threading
import torch
import cv2
import json
import csv

# Assuming nemacounter package structure is correct
import nemacounter.utils as utils
import nemacounter.common as common
from nemacounter.detection import detection_workflow
from nemacounter.edition import edition_workflow
# Import the specific class and functions needed from segmentation
from nemacounter.segmentation import NemaCounterSegmentation, add_masks_on_image, create_multicolored_masks_image

CTK.set_appearance_mode("Dark")
CTK.set_default_color_theme("blue")


class NemaCounterGUI:

    def __init__(self):
        self.root = CTK.CTk()
        self.nb_wanted_cpu = CTK.IntVar()
        self.use_GPU = CTK.IntVar(value=1)
        self.scaling_var = CTK.StringVar(value="110%")
        self.theme_var = CTK.StringVar(value="Dark")
        self.tab_var = CTK.StringVar(value="Object Detection")
        self.model_var = CTK.StringVar()
        self.set_main_window()

    def open_directory(self, var, label_obj):
        dpath = filedialog.askdirectory(initialdir='.', title='Select directory')
        if dpath:
            var.set(dpath)
            label_obj.configure(text=os.path.relpath(dpath))
        else:
            var.set('')  # Clear if dialog cancelled

    def get_globinfo_fpath(self, var, label_obj):
        fpath = filedialog.askopenfilename(initialdir='.', title='Select a globinfo file',
                                           filetypes=[('csv files', '*_globinfo.csv')])
        if fpath:
            var.set(fpath)
            label_obj.configure(text=os.path.relpath(fpath))
        else:
            var.set('')  # Clear if dialog cancelled

    def change_appearance_mode_event(self, _):
        CTK.set_appearance_mode(self.theme_var.get())

    def change_scaling_event(self, _):
        try:
            new_scaling_float = int(self.scaling_var.get().replace("%", "")) / 100
            CTK.set_widget_scaling(new_scaling_float)
        except ValueError:
            print("Invalid scaling value selected.")

    def set_main_window(self):
        self.root.title("NemaCounter GUI")
        self.root.geometry(f"{1100}x{700}")
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.set_side_bar()
        self.set_central_tabview()
        self.root.mainloop()

    def display_cpu_number(self, val):
        self.label_cpu_slider.configure(text=f"Max. number of CPU: {int(val)}")

    def display_confidence(self, val):
        self.label_conf_slider.configure(text=f"Confidence Threshold: {np.round(val, 2)}")

    def display_overlap(self, val):
        self.label_overl_slider.configure(text=f"Overlap Threshold: {np.round(val, 2)}")

    def display_fuse_iou(self, val):
        if hasattr(self, 'fuse_iou_label'):
            self.fuse_iou_label.configure(text=f"Merge IoU: {np.round(val, 2)}")

    def display_phagocyte_ioa(self, val):
        if hasattr(self, 'phagocyte_ioa_label'):
            self.phagocyte_ioa_label.configure(text=f"Phagocyte IoA: {np.round(val, 2)}")

    def toggle_fusion_controls(self):
        if hasattr(self, 'fuse_iou_slider'):
            state = "normal" if self.use_fusion_var.get() else "disabled"
            self.fuse_iou_slider.configure(state=state)
            self.phagocyte_ioa_slider.configure(state=state)
            self.fuse_iou_label.configure(state=state)
            self.phagocyte_ioa_label.configure(state=state)

    def set_side_bar(self):
        sidebar_frame = CTK.CTkFrame(master=self.root, width=140, corner_radius=0)
        sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        sidebar_frame.grid_rowconfigure(8, weight=1)

        logo_frame = CTK.CTkFrame(master=sidebar_frame, fg_color="transparent")
        logo_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        try:
            logo_path = os.path.join("conf", "logo.png")
            if not os.path.exists(logo_path): logo_path = "conf/logo.png"
            nemacounter_logo = CTK.CTkImage(PILimage.open(logo_path), size=(200, 200))
            image_label = CTK.CTkLabel(master=logo_frame, image=nemacounter_logo, text='')
            image_label.pack(pady=10)
        except Exception as e:
            print(f"Error loading logo: {e}")
            logo_label = CTK.CTkLabel(master=logo_frame, text="Nemacounter", font=CTK.CTkFont(size=20, weight="bold"))
            logo_label.pack(pady=10)

        hardware_frame = CTK.CTkFrame(master=sidebar_frame, fg_color="transparent")
        hardware_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        switch_GPU = CTK.CTkSwitch(master=hardware_frame, variable=self.use_GPU, onvalue=1, offvalue=0,
                                   text=f"Use GPU if available")
        switch_GPU.pack(pady=5, anchor="w")

        nb_avail_cpu = utils.compute_available_cpu()
        self.label_cpu_slider = CTK.CTkLabel(master=hardware_frame, text=f"Max. number of CPU: {nb_avail_cpu - 1}",
                                             anchor="w")
        self.label_cpu_slider.pack(pady=(10, 0), anchor="w")
        slider_cpu = CTK.CTkSlider(master=hardware_frame, from_=1, to=nb_avail_cpu,
                                   number_of_steps=max(1, nb_avail_cpu - 1), variable=self.nb_wanted_cpu,
                                   command=self.display_cpu_number)
        slider_cpu.set(max(1, nb_avail_cpu - 1))
        slider_cpu.pack(pady=5, fill="x", expand=True)

        modelspath_frame = CTK.CTkFrame(master=sidebar_frame, fg_color="transparent")
        modelspath_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        model_label = CTK.CTkLabel(master=modelspath_frame, text="Select Model (Detection):", anchor="w")
        model_label.pack(pady=(5, 0), anchor="w")

        try:
            model_dir = "models"
            if not os.path.isdir(model_dir): os.makedirs(model_dir)
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
            if not model_files:
                model_files = ["No Models Found"]
                self.model_var.set(model_files[0])
                model_menu = CTK.CTkOptionMenu(master=modelspath_frame, values=model_files, variable=self.model_var,
                                               state="disabled")
            else:
                self.model_var.set(model_files[0])
                model_menu = CTK.CTkOptionMenu(master=modelspath_frame, values=model_files, variable=self.model_var)
            model_menu.pack(pady=5, fill="x", expand=True)
        except Exception as e:
            print(f"Error listing models: {e}")
            CTK.CTkLabel(master=modelspath_frame, text="Error loading models").pack()

        displparams_frame = CTK.CTkFrame(master=sidebar_frame, fg_color="transparent")
        displparams_frame.grid(row=9, column=0, padx=20, pady=(20, 20), sticky="sew")
        appearance_mode_label = CTK.CTkLabel(master=displparams_frame, text="Appearance Mode:", anchor="w")
        appearance_mode_label.pack(pady=(5, 0), anchor="w")
        appearance_mode_optionemenu = CTK.CTkOptionMenu(master=displparams_frame, values=["Dark", "Light", "System"],
                                                        command=self.change_appearance_mode_event,
                                                        variable=self.theme_var)
        appearance_mode_optionemenu.pack(pady=5, fill="x", expand=True)
        scaling_label = CTK.CTkLabel(master=displparams_frame, text="UI Scaling:", anchor="w")
        scaling_label.pack(pady=(5, 0), anchor="w")
        scaling_optionemenu = CTK.CTkOptionMenu(master=displparams_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                command=self.change_scaling_event, variable=self.scaling_var)
        scaling_optionemenu.pack(pady=5, fill="x", expand=True)

    def start_detection(self, indir_var, outdir_var, projid_entry, confslid_var, overslid_var, add_overlay_var,
                        show_bbox_var, show_conf_var, show_mask_var, show_labels_var, retina_masks_var):
        """Callback function to start the detection workflow."""
        selected_model = self.model_var.get()
        if not selected_model or selected_model == "No Models Found":
            messagebox.showwarning("Warning", "You must select a valid model from the 'models' directory.")
            return

        model_path = os.path.join("models", selected_model)
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Selected model file not found: {model_path}")
            return

        dct_var_detection = {
            'input_directory': indir_var.get(),
            'output_directory': outdir_var.get(),
            'project_id': projid_entry.get("1.0", tk.END).strip(),
            'conf_thresh': confslid_var.get(),
            'overlap_thresh': overslid_var.get(),
            'add_overlay': add_overlay_var.get(),
            'model_path': model_path,
            'gpu': self.use_GPU.get(),
            'cpu': self.nb_wanted_cpu.get(),
            'show_bbox': show_bbox_var.get(),
            'show_conf': show_conf_var.get(),
            'show_mask': show_mask_var.get(),
            'show_labels': show_labels_var.get(),
            'use_retina_masks': retina_masks_var.get(),
            'use_fusion': self.use_fusion_var.get(),
            'fuse_iou_thresh': self.fuse_iou_var.get(),
            'phagocyte_ioa_thresh': self.phagocyte_ioa_var.get()
        }

        if not dct_var_detection['input_directory'] or not os.path.isdir(dct_var_detection['input_directory']):
            messagebox.showwarning("Input Error", "Please select a valid input directory.")
            return
        if not dct_var_detection['output_directory'] or not os.path.isdir(dct_var_detection['output_directory']):
            messagebox.showwarning("Input Error", "Please select a valid output directory.")
            return
        if not dct_var_detection['project_id']:
            messagebox.showwarning("Input Error", "Please enter a project name.")
            return

        project_outdir = os.path.join(dct_var_detection['output_directory'], dct_var_detection['project_id'])
        if os.path.exists(project_outdir):
            if not messagebox.askyesno("Warning",
                                       f"The output project folder '{project_outdir}' already exists. Overwrite?"):
                return

        if dct_var_detection['use_retina_masks']:
            if not messagebox.askyesno("Memory Warning",
                                       "High-resolution masks (Retina) are enabled.\nThis may still cause memory issues on systems with low RAM.\n\nContinue?"):
                return

        print("Starting detection workflow...")
        thread = threading.Thread(target=self._run_detection_thread, args=(dct_var_detection,))
        thread.start()

    def _run_detection_thread(self, dct_params):
        """Worker function for detection thread."""
        try:
            detection_workflow(dct_params, gui=True)
            messagebox.showinfo("Success", "Object detection process completed successfully.")
        except MemoryError as e:
            print(f"Memory error during detection: {e}")
            messagebox.showerror("Memory Error", "Detection failed due to insufficient memory.")
        except Exception as e:
            print(f"Error during detection workflow: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred during detection:\n{e}")

    def set_central_tabview(self):
        tabview = CTK.CTkTabview(master=self.root, fg_color="transparent")
        tabview.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
        tabview.add("Object Detection")
        tabview.add("Manual Edition")
        tabview.add("Object Segmentation")
        tabview.add("Export to Roboflow")
        self.detection_tab = tabview.tab("Object Detection")
        self.edition_tab = tabview.tab("Manual Edition")
        self.segmentation_tab = tabview.tab("Object Segmentation")
        self.export_tab = tabview.tab("Export to Roboflow")
        self.set_detection_tab()
        self.set_edition_tab()
        self.set_segmentation_tab()
        self.set_export_tab()

    def set_detection_tab(self):
        """Populates the Object Detection tab with all controls."""
        for widget in self.detection_tab.winfo_children(): widget.destroy()

        self.detection_tab.grid_columnconfigure(0, weight=1)
        self.detection_tab.grid_rowconfigure(2, weight=1)

        projid_frame = CTK.CTkFrame(master=self.detection_tab, fg_color="transparent")
        projid_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        projid_text = CTK.CTkLabel(master=projid_frame, text='Project Name:')
        projid_text.pack(side=tk.LEFT, padx=(0, 10))
        self.projid_entry_detect = CTK.CTkTextbox(master=projid_frame, height=20)
        self.projid_entry_detect.insert("1.0", "MyDetectionProject")
        self.projid_entry_detect.pack(side=tk.LEFT, fill="x", expand=True)

        parameters_frame = CTK.CTkFrame(master=self.detection_tab)
        parameters_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        parameters_frame.grid_columnconfigure(0, weight=1)

        # --- Input/Output Frames ---
        indir_var = CTK.StringVar()
        indir_frame = CTK.CTkFrame(master=parameters_frame, fg_color="transparent")
        indir_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        indir_button = CTK.CTkButton(master=indir_frame, width=100, text="Input Dir",
                                     command=lambda: self.open_directory(indir_var, indir_label))
        indir_button.pack(side=tk.LEFT)
        indir_label = CTK.CTkLabel(master=indir_frame, text='(No directory selected)', anchor="w", text_color="gray")
        indir_label.pack(side=tk.LEFT, padx=10, fill="x", expand=True)

        outdir_var = CTK.StringVar()
        outdir_frame = CTK.CTkFrame(master=parameters_frame, fg_color="transparent")
        outdir_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        outdir_button = CTK.CTkButton(master=outdir_frame, width=100, text="Output Dir",
                                      command=lambda: self.open_directory(outdir_var, outdir_label))
        outdir_button.pack(side=tk.LEFT)
        outdir_label = CTK.CTkLabel(master=outdir_frame, text='(No directory selected)', anchor="w", text_color="gray")
        outdir_label.pack(side=tk.LEFT, padx=10, fill="x", expand=True)

        # --- Primary Sliders ---
        sliders_frame = CTK.CTkFrame(master=parameters_frame, fg_color="transparent")
        sliders_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        sliders_frame.grid_columnconfigure((0, 1), weight=1)

        confslid_var = CTK.DoubleVar(value=0.5)
        overslid_var = CTK.DoubleVar(value=0.3)
        self.label_conf_slider = CTK.CTkLabel(master=sliders_frame, text=f"Confidence: {confslid_var.get():.2f}",
                                              anchor="w")
        self.label_conf_slider.grid(row=0, column=0, padx=(0, 5), pady=(5, 0), sticky="w")
        confidence_slider = CTK.CTkSlider(master=sliders_frame, from_=0.01, to=1.0, variable=confslid_var,
                                          command=self.display_confidence)
        confidence_slider.grid(row=1, column=0, padx=(0, 5), pady=(0, 10), sticky="ew")
        self.label_overl_slider = CTK.CTkLabel(master=sliders_frame, text=f"Overlap (IoU): {overslid_var.get():.2f}",
                                               anchor="w")
        self.label_overl_slider.grid(row=0, column=1, padx=(5, 0), pady=(5, 0), sticky="w")
        overlap_slider = CTK.CTkSlider(master=sliders_frame, from_=0.01, to=1.0, variable=overslid_var,
                                       command=self.display_overlap)
        overlap_slider.grid(row=1, column=1, padx=(5, 0), pady=(0, 10), sticky="ew")

        # --- Overlay Options ---
        overlay_opts_frame = CTK.CTkFrame(master=parameters_frame, fg_color="transparent")
        overlay_opts_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        add_overlay_var = CTK.IntVar(value=1)
        overlay_switch = CTK.CTkSwitch(master=overlay_opts_frame, variable=add_overlay_var, text='Save Overlay Images')
        overlay_switch.pack(side=tk.LEFT, padx=(0, 20))
        show_bbox_var = CTK.IntVar(value=1)
        show_conf_var = CTK.IntVar(value=0)
        show_mask_var = CTK.IntVar(value=1)
        show_labels_var = CTK.IntVar(value=1)
        show_bbox_check = CTK.CTkCheckBox(overlay_opts_frame, text="Show Box", variable=show_bbox_var);
        show_bbox_check.pack(side=tk.LEFT, padx=5)
        show_conf_check = CTK.CTkCheckBox(overlay_opts_frame, text="Show Conf", variable=show_conf_var);
        show_conf_check.pack(side=tk.LEFT, padx=5)
        show_mask_check = CTK.CTkCheckBox(overlay_opts_frame, text="Show Mask", variable=show_mask_var);
        show_mask_check.pack(side=tk.LEFT, padx=5)
        show_labels_check = CTK.CTkCheckBox(overlay_opts_frame, text="Show Labels", variable=show_labels_var);
        show_labels_check.pack(side=tk.LEFT, padx=5)

        # --- Advanced Options Frame ---
        advanced_frame = CTK.CTkFrame(master=parameters_frame)
        advanced_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
        advanced_frame.grid_columnconfigure(1, weight=1)

        advanced_label = CTK.CTkLabel(master=advanced_frame, text="Advanced Segmentation Options:", anchor="w",
                                      font=CTK.CTkFont(weight="bold"))
        advanced_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")

        retina_masks_var = CTK.IntVar(value=0)
        retina_masks_switch = CTK.CTkSwitch(master=advanced_frame, variable=retina_masks_var,
                                            text='High-Res Masks (Retina)', onvalue=1, offvalue=0)
        retina_masks_switch.grid(row=1, column=0, pady=(5, 10), sticky="w", padx=10)

        # --- Fusion Controls ---
        self.use_fusion_var = CTK.IntVar(value=0)
        fusion_switch = CTK.CTkSwitch(master=advanced_frame, variable=self.use_fusion_var,
                                      text='Fusion of overlapping masks', command=self.toggle_fusion_controls)
        fusion_switch.grid(row=2, column=0, pady=(5, 10), sticky="w", padx=10)

        fusion_sliders_frame = CTK.CTkFrame(master=advanced_frame, fg_color="transparent")
        fusion_sliders_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10)
        fusion_sliders_frame.grid_columnconfigure((0, 1), weight=1)

        self.fuse_iou_var = CTK.DoubleVar(value=0.05)
        self.phagocyte_ioa_var = CTK.DoubleVar(value=0.95)

        self.fuse_iou_label = CTK.CTkLabel(master=fusion_sliders_frame,
                                           text=f"Merge IoU: {self.fuse_iou_var.get():.2f}", anchor="w")
        self.fuse_iou_label.grid(row=0, column=0, padx=(0, 5), pady=(5, 0), sticky="w")
        self.fuse_iou_slider = CTK.CTkSlider(master=fusion_sliders_frame, from_=0.01, to=1.0,
                                             variable=self.fuse_iou_var, command=self.display_fuse_iou)
        self.fuse_iou_slider.grid(row=1, column=0, padx=(0, 5), pady=(0, 10), sticky="ew")

        self.phagocyte_ioa_label = CTK.CTkLabel(master=fusion_sliders_frame,
                                                text=f"Phagocyte IoA: {self.phagocyte_ioa_var.get():.2f}", anchor="w")
        self.phagocyte_ioa_label.grid(row=0, column=1, padx=(5, 0), pady=(5, 0), sticky="w")
        self.phagocyte_ioa_slider = CTK.CTkSlider(master=fusion_sliders_frame, from_=0.01, to=1.0,
                                                  variable=self.phagocyte_ioa_var, command=self.display_phagocyte_ioa)
        self.phagocyte_ioa_slider.grid(row=1, column=1, padx=(5, 0), pady=(0, 10), sticky="ew")

        # --- Start Button ---
        start_button_frame = CTK.CTkFrame(master=self.detection_tab, fg_color="transparent")
        start_button_frame.grid(row=3, column=0, padx=20, pady=20, sticky="ew")
        start_button_frame.grid_columnconfigure(0, weight=1)
        start_button = CTK.CTkButton(
            master=start_button_frame, text="Start Detection", height=40,
            command=lambda: self.start_detection(
                indir_var, outdir_var, self.projid_entry_detect, confslid_var, overslid_var,
                add_overlay_var, show_bbox_var, show_conf_var, show_mask_var,
                show_labels_var, retina_masks_var
            )
        )
        start_button.grid(row=0, column=0, pady=10)

        self.toggle_fusion_controls()

    # ... (The rest of the GUI file is unchanged)

    # --- Edition Tab Setup ---
    def set_edition_tab(self):
        """Populates the Manual Edition tab."""
        for widget in self.edition_tab.winfo_children(): widget.destroy()

        self.edition_tab.grid_columnconfigure(0, weight=1)
        self.edition_tab.grid_rowconfigure(1, weight=1)  # Allow input frame to expand if needed

        # Project Name
        projid_frame = CTK.CTkFrame(master=self.edition_tab, fg_color="transparent")
        projid_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        projid_text = CTK.CTkLabel(master=projid_frame, text='Project Name:')
        projid_text.pack(side=tk.LEFT, padx=(0, 10))
        self.projid_entry_edit = CTK.CTkTextbox(master=projid_frame, height=20)
        self.projid_entry_edit.insert("1.0", "MyEditionProject")
        self.projid_entry_edit.pack(side=tk.LEFT, fill="x", expand=True)

        # Input/Output Frame
        io_frame = CTK.CTkFrame(master=self.edition_tab)
        io_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        io_frame.grid_columnconfigure(0, weight=1)

        # Input File
        infile_var = CTK.StringVar()
        infile_frame = CTK.CTkFrame(master=io_frame, fg_color="transparent")
        infile_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        infile_button = CTK.CTkButton(master=infile_frame, width=100, text="Input CSV",
                                      command=lambda: self.get_globinfo_fpath(infile_var, infile_label))
        infile_button.pack(side=tk.LEFT)
        infile_label = CTK.CTkLabel(master=infile_frame, text='(No *_globinfo.csv selected)', anchor="w",
                                    text_color="gray")
        infile_label.pack(side=tk.LEFT, padx=10, fill="x", expand=True)

        # Output Directory
        outdir_var = CTK.StringVar()
        outdir_frame = CTK.CTkFrame(master=io_frame, fg_color="transparent")
        outdir_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        outdir_button = CTK.CTkButton(master=outdir_frame, width=100, text="Output Dir",
                                      command=lambda: self.open_directory(outdir_var, outdir_label))
        outdir_button.pack(side=tk.LEFT)
        outdir_label = CTK.CTkLabel(master=outdir_frame, text='(No directory selected)', anchor="w", text_color="gray")
        outdir_label.pack(side=tk.LEFT, padx=10, fill="x", expand=True)

        # Start Button Frame
        start_button_frame = CTK.CTkFrame(master=self.edition_tab, fg_color="transparent")
        start_button_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")
        start_button_frame.grid_columnconfigure(0, weight=1)
        start_button = CTK.CTkButton(master=start_button_frame, text="Start Manual Edition", height=40,
                                     command=lambda: self.start_manual_edition(self.projid_entry_edit, infile_var,
                                                                               outdir_var))
        start_button.grid(row=0, column=0, pady=10)

    def start_manual_edition(self, projid_entry, infile_var, outdir_var):
        """Callback to start the manual edition workflow."""
        fpath_globinfo = infile_var.get()
        dpath_out = outdir_var.get()
        project_id = projid_entry.get("1.0", tk.END).strip()

        # Validation
        if not fpath_globinfo or not os.path.exists(fpath_globinfo):
            messagebox.showwarning("Input Error", "Please select a valid input *_globinfo.csv file.")
            return
        if not dpath_out or not os.path.isdir(dpath_out):
            messagebox.showwarning("Input Error", "Please select a valid output directory.")
            return
        if not project_id:
            messagebox.showwarning("Input Error", "Please enter a project name.")
            return

        # Output directory for this specific project's edition results
        # Place it INSIDE the selected output directory
        edition_output_directory = os.path.join(dpath_out, project_id + "_Edition")

        # Run in thread
        print("Starting manual edition workflow...")
        # Pass necessary args, including potential SAM model paths if needed by edition_workflow kwargs
        thread = threading.Thread(target=self._run_edition_thread,
                                  args=(fpath_globinfo, edition_output_directory, project_id))
        thread.start()

    def _run_edition_thread(self, input_csv, output_dir, proj_id):
        """Worker function for edition thread."""
        try:
            # Call edition_workflow, passing necessary parameters
            # Assuming edition_workflow handles its own SAM loading based on **kwargs if needed
            edition_workflow(
                input_file=input_csv,
                output_directory=output_dir,  # Pass the specific dir for this run
                project_id=proj_id,
                use_gpu=self.use_GPU.get(),
                # Add kwargs if needed, e.g.:
                # sam_model_checkpoint=self.sam_checkpoint_var.get(),
                # sam_model_config=self.sam_config_var.get()
            )
            # No messagebox here as edition_workflow likely prints success/failure
            print(f"Manual edition thread finished for project {proj_id}.")
        except Exception as e:
            print(f"Error during edition workflow: {e}")
            # Avoid messagebox in thread, just print
            # messagebox.showerror("Error", f"An error occurred during manual edition:\n{e}")

    # --- Segmentation Tab Setup ---
    def set_segmentation_tab(self):
        """Populates the Object Segmentation tab."""
        for widget in self.segmentation_tab.winfo_children(): widget.destroy()

        self.segmentation_tab.grid_columnconfigure(0, weight=1)
        self.segmentation_tab.grid_rowconfigure(1, weight=1)  # Allow frame to expand

        # Input/Options Frame
        param_frame = CTK.CTkFrame(master=self.segmentation_tab)
        param_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        param_frame.grid_columnconfigure(0, weight=1)

        # Input File
        infile_var = CTK.StringVar()
        infile_frame = CTK.CTkFrame(master=param_frame, fg_color="transparent")
        infile_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        infile_button = CTK.CTkButton(master=infile_frame, width=100, text="Input CSV",
                                      command=lambda: self.get_globinfo_fpath(infile_var, infile_label))
        infile_button.pack(side=tk.LEFT)
        infile_label = CTK.CTkLabel(master=infile_frame, text='(No *_globinfo.csv selected)', anchor="w",
                                    text_color="gray")
        infile_label.pack(side=tk.LEFT, padx=10, fill="x", expand=True)

        # Overlay Option
        overlay_frame = CTK.CTkFrame(master=param_frame, fg_color="transparent")
        overlay_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        add_overlay_var = CTK.IntVar(value=1)
        overlay_switch = CTK.CTkSwitch(master=overlay_frame, variable=add_overlay_var,
                                       text='Save Segmentation Overlay Images')
        overlay_switch.pack(anchor="w")

        # Start Button Frame
        start_button_frame = CTK.CTkFrame(master=self.segmentation_tab, fg_color="transparent")
        start_button_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        start_button_frame.grid_columnconfigure(0, weight=1)
        start_button = CTK.CTkButton(master=start_button_frame, text="Start Segmentation", height=40,
                                     command=lambda: self.start_segmentation(infile_var, add_overlay_var))
        start_button.grid(row=0, column=0, pady=10)

        # Progress Bar Frame (at the bottom)
        self.progress_frame = CTK.CTkFrame(master=self.segmentation_tab, fg_color="transparent")
        self.progress_frame.grid(row=3, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.progress_bar = CTK.CTkProgressBar(master=self.progress_frame)
        self.progress_bar.pack(fill="x", expand=True, pady=(0, 5))
        self.progress_bar.set(0)  # Initial state

        self.processing_label = CTK.CTkLabel(master=self.progress_frame, text="", anchor="w")  # Start empty
        self.processing_label.pack(fill="x", expand=True)

    def start_segmentation(self, infile_var, add_overlay_var):
        """Callback to start the segmentation workflow."""
        input_csv_path = infile_var.get()

        if not input_csv_path or not os.path.exists(input_csv_path):
            messagebox.showwarning("Input Error", "Please select a valid input *_globinfo.csv file.")
            return

        dct_var_segmentation = {
            'input_file': input_csv_path,
            'add_overlay': add_overlay_var.get(),
            'gpu': self.use_GPU.get(),
            'cpu': self.nb_wanted_cpu.get()
        }

        # Update UI before starting thread
        self.processing_label.configure(text="Starting segmentation...")
        self.progress_bar.set(0)

        # Run in thread
        thread = threading.Thread(target=self.run_segmentation_workflow, args=(dct_var_segmentation,))
        thread.start()

    def run_segmentation_workflow(self, dct_var_segmentation):
        """Worker thread for segmentation."""
        project_id_from_file = os.path.basename(dct_var_segmentation['input_file']).replace('_globinfo.csv', '')
        input_dir_from_file = os.path.dirname(dct_var_segmentation['input_file'])

        # Add derived paths to the dictionary
        dct_var_segmentation['project_id'] = project_id_from_file
        dct_var_segmentation['input_dir'] = input_dir_from_file

        # --- Device Selection ---
        gpu_if_avail = utils.get_bool(dct_var_segmentation['gpu'])
        add_overlay = utils.get_bool(dct_var_segmentation['add_overlay'])
        utils.set_cpu_usage(dct_var_segmentation['cpu'])
        try:
            cuda_available = torch.cuda.is_available()
        except (AssertionError, RuntimeError):
            cuda_available = False
        if not cuda_available: gpu_if_avail = False
        device = torch.device('cuda:0' if cuda_available and gpu_if_avail else 'cpu')
        print(f"Segmentation using device: {device}")

        # --- READ METADATA FROM CSV TO GET ACTUAL INPUT DIRECTORY ---
        actual_input_directory = None
        try:
            with open(dct_var_segmentation['input_file'], 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('# input_directory:'):
                    actual_input_directory = first_line.split(':', 1)[1].strip()
                    print(f"Found input directory from metadata: {actual_input_directory}")
        except Exception as e:
            print(f"Warning: Could not read metadata from CSV: {e}")

        # --- Prepare Output Path ---
        if add_overlay:
            dpath_overlay = os.path.join(input_dir_from_file, project_id_from_file, 'img', 'segmentation')
            os.makedirs(dpath_overlay, exist_ok=True)

        # --- Main Processing Logic ---
        segmentation_success = False  # Flag for final message
        try:
            if utils.check_file_existence(dct_var_segmentation['input_file']):
                self.update_processing_label("Reading input CSV...")
                # Read CSV with comment='#' to skip metadata lines
                df = pd.read_csv(dct_var_segmentation['input_file'], comment='#')
                lst_img_paths_rel = df['img_id'].unique()

                self.update_processing_label(f"Initializing model on {device}...")
                segmentation_model = NemaCounterSegmentation(device=device)

                all_final_annotations = []  # Store results for final DF
                total_images = len(lst_img_paths_rel)

                for idx, img_path_rel in enumerate(lst_img_paths_rel):
                    current_progress = idx / total_images
                    self.update_progress(current_progress)
                    self.update_processing_label(
                        f"Processing image {idx + 1}/{total_images}: {os.path.basename(img_path_rel)}")

                    # Try multiple locations to find the image
                    img_path_full = None
                    possible_paths = []

                    # PRIORITY 1: Use the actual input directory from metadata if available
                    if actual_input_directory:
                        possible_paths.append(os.path.join(actual_input_directory, img_path_rel))

                    # PRIORITY 2: Try relative to CSV location (for backward compatibility)
                    possible_paths.append(os.path.join(input_dir_from_file, img_path_rel))

                    # PRIORITY 3: Try in parent directory of CSV
                    possible_paths.append(os.path.join(os.path.dirname(input_dir_from_file), img_path_rel))

                    # PRIORITY 4: Try absolute path
                    possible_paths.append(img_path_rel)

                    # Find the first existing path
                    for path in possible_paths:
                        if os.path.exists(path):
                            img_path_full = path
                            break

                    if not img_path_full:
                        print(f"Warning: Image not found '{img_path_rel}'")
                        print(f"  Searched in:")
                        for i, p in enumerate(possible_paths[:3]):
                            print(f"    {i + 1}. {p}")
                        continue

                    try:
                        img_bgr = cv2.imread(img_path_full)
                        if img_bgr is None:
                            raise IOError("Failed to read image")
                        # Note: common.read_image returns RGB, but cv2.imread returns BGR
                        # So we need to convert BGR to RGB for SAM
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        print(f"Warning: Failed to read/convert image '{img_path_rel}': {e}, skipping.")
                        continue

                    img_df = df[df['img_id'] == img_path_rel].reset_index(drop=True)

                    # --- Prepare annotations for THIS image ---
                    annotations_for_sam = []
                    for _, row in img_df.iterrows():
                        ann = row.to_dict()
                        obj_type = str(ann.get('object_type', '')).lower()

                        if obj_type == 'box':
                            # Ensure coords are int
                            try:
                                ann['xmin'] = int(ann['xmin'])
                                ann['ymin'] = int(ann['ymin'])
                                ann['xmax'] = int(ann['xmax'])
                                ann['ymax'] = int(ann['ymax'])
                                annotations_for_sam.append(ann)
                            except (ValueError, TypeError):
                                print(f"Warn: Invalid box coords for {img_path_rel}, skipping.")
                        elif obj_type == 'mask':
                            annotations_for_sam.append(ann)  # Pass mask annotation as is
                        elif obj_type == 'polygon':
                            # Ensure 'contours' exists
                            if 'contours' in ann and pd.notna(ann['contours']):
                                annotations_for_sam.append(ann)
                            else:
                                print(f"Warn: Polygon annotation missing contours for {img_path_rel}, skipping.")

                    if not annotations_for_sam:
                        print(f"No valid annotations prepared for image {img_path_rel}. Skipping.")
                        continue

                    # --- Run SAM segmentation ---
                    try:
                        # Pass RGB image to segmentation
                        masks, updated_annotations = segmentation_model.objects_segmentation(img_rgb,
                                                                                             annotations_for_sam)
                    except Exception as e:
                        print(f"Error during segmentation call for '{img_path_rel}': {e}")
                        continue

                    if masks is None or len(masks) != len(updated_annotations):
                        print(f"Warn: Mask/Annotation mismatch for {img_path_rel}, skipping results.")
                        continue

                    # --- Process results (area, contours, bbox update) ---
                    records_for_df = []
                    for ann, mask_data in zip(updated_annotations, masks):
                        if not isinstance(ann, dict):
                            continue

                        try:
                            area = np.sum(mask_data > 0)
                            ann['area'] = float(area)

                            contours_list, _ = cv2.findContours(mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            valid_contours = [c.squeeze().tolist() for c in contours_list if c.shape[0] >= 3]

                            if valid_contours:
                                ann['contours'] = json.dumps(valid_contours)
                                all_points = np.vstack([np.array(c) for c in valid_contours])
                                ann['xmin'] = int(np.min(all_points[:, 0]))
                                ann['ymin'] = int(np.min(all_points[:, 1]))
                                ann['xmax'] = int(np.max(all_points[:, 0]))
                                ann['ymax'] = int(np.max(all_points[:, 1]))
                            else:
                                ann['contours'] = json.dumps([])
                                ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax'] = 0, 0, 0, 0

                            ann['object_type'] = 'mask'
                            records_for_df.append(ann)
                        except Exception as e:
                            print(f"Error post-processing annotation for {img_path_rel}: {e}")

                    all_final_annotations.extend(records_for_df)

                    # --- Save Overlays ---
                    if add_overlay and len(masks) > 0:
                        try:
                            overlay_img_bgr = img_bgr.copy()
                            add_masks_on_image(masks, overlay_img_bgr)
                            fname = os.path.basename(img_path_rel)
                            fpath_out_img = os.path.join(dpath_overlay, f"{project_id_from_file}_{fname}")
                            cv2.imwrite(fpath_out_img, overlay_img_bgr)

                            multicolored_img = create_multicolored_masks_image(masks)
                            fstem = os.path.splitext(fname)[0]
                            fpath_out_multi = os.path.join(dpath_overlay, f"{project_id_from_file}_{fstem}_colored.png")
                            cv2.imwrite(fpath_out_multi, multicolored_img)
                        except Exception as e:
                            print(f"Error saving overlay images for {img_path_rel}: {e}")

                # --- End of Image Loop ---

                if not all_final_annotations:
                    raise ValueError("No annotations were successfully processed.")

                self.update_processing_label("Finalizing output files...")
                df_new = pd.DataFrame(all_final_annotations)
                df_new['project_id'] = project_id_from_file
                df_new['object_id'] = df_new.groupby('img_id').cumcount() + 1

                # Define/Reorder columns, ensure defaults
                expected_columns = ['img_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class',
                                    'name', 'area', 'contours', 'object_type', 'project_id']
                for col in expected_columns:
                    if col not in df_new.columns:
                        if col in ['xmin', 'ymin', 'xmax', 'ymax', 'object_id', 'area']:
                            default_val = 0
                        elif col == 'confidence':
                            default_val = np.nan
                        else:
                            default_val = pd.NA
                        df_new[col] = default_val

                # Enforce types
                try:
                    for col in ['xmin', 'ymin', 'xmax', 'ymax', 'object_id']:
                        df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0).astype(int)
                    for col in ['area', 'confidence']:
                        df_new[col] = pd.to_numeric(df_new[col], errors='coerce').astype(float)
                    for col in ['img_id', 'name', 'object_type', 'project_id', 'class']:
                        df_new[col] = df_new[col].astype(str).replace('<NA>', '')
                except Exception as e:
                    print(f"Warn: Type enforcement failed: {e}")

                df_new = df_new[expected_columns]

                # Save outputs WITH METADATA
                output_globinfo = os.path.join(input_dir_from_file, f"{project_id_from_file}_segmentation_globinfo.csv")

                # Write with metadata comment (same format as detection)
                with open(output_globinfo, 'w', newline='', encoding='utf-8') as f:
                    # Preserve the input directory metadata
                    if actual_input_directory:
                        f.write(f"# input_directory: {actual_input_directory}\n")
                    df_new.to_csv(f, index=False, quoting=csv.QUOTE_ALL)

                df_summary = common.create_summary_table(df_new, project_id_from_file)
                output_summary = os.path.join(input_dir_from_file, f"{project_id_from_file}_segmentation_summary.csv")
                df_summary.to_csv(output_summary, index=False, quoting=csv.QUOTE_ALL)

                segmentation_success = True

            else:
                raise FileNotFoundError(f"Input file not found: {dct_var_segmentation['input_file']}")

        except Exception as e:
            print(f"Error during segmentation workflow: {e}")
            self.update_progress(0)
            self.update_processing_label(f"Error: {e}")

        finally:
            if segmentation_success:
                self.update_progress(1)
                self.update_processing_label("Segmentation complete.")
                messagebox.showinfo("Success", "Object segmentation process completed successfully.")
            else:
                self.update_processing_label("Segmentation failed.")

    def update_progress(self, progress):
        """Safely update progress bar from thread."""
        if hasattr(self, 'progress_bar'):  # Check if widget exists
            self.root.after(0, self.progress_bar.set, progress)

    def update_processing_label(self, text):
        """Safely update processing label from thread."""
        if hasattr(self, 'processing_label'):  # Check if widget exists
            # Ensure label is visible when updating text
            # self.root.after(0, self.processing_label.grid) # Might cause layout issues if called repeatedly
            self.root.after(0, self.processing_label.configure, {"text": text})

    # --- Export Tab Setup ---
    def set_export_tab(self):
        """Populates the Export to Roboflow tab."""
        for widget in self.export_tab.winfo_children(): widget.destroy()

        self.export_tab.grid_columnconfigure(0, weight=1)
        self.export_tab.grid_rowconfigure(2, weight=1)  # Push button down

        master_frame = CTK.CTkFrame(master=self.export_tab, fg_color="transparent")
        master_frame.grid(row=0, column=0, padx=20, pady=20, sticky='nsew')
        master_frame.grid_columnconfigure(0, weight=1)

        instruction_label = CTK.CTkLabel(master=master_frame,
                                         text='Select a *_globinfo.csv file to convert to Roboflow COCO JSON format:')
        instruction_label.grid(row=0, column=0, pady=(0, 10), sticky='w')

        # File Selection Frame
        file_frame = CTK.CTkFrame(master=master_frame, fg_color="transparent")
        file_frame.grid(row=1, column=0, pady=5, sticky='ew')
        globinfo_var = CTK.StringVar()
        select_button = CTK.CTkButton(master=file_frame, width=100, text="Select CSV",
                                      command=lambda: self.get_globinfo_fpath(globinfo_var, selected_file_label))
        select_button.pack(side=tk.LEFT)
        selected_file_label = CTK.CTkLabel(master=file_frame, text="(No file selected)", anchor="w", text_color="gray")
        selected_file_label.pack(side=tk.LEFT, padx=10, fill="x", expand=True)

        # Convert Button Frame
        button_frame = CTK.CTkFrame(master=self.export_tab, fg_color="transparent")
        button_frame.grid(row=3, column=0, padx=20, pady=20, sticky="ew")
        button_frame.grid_columnconfigure(0, weight=1)  # Center button
        convert_button = CTK.CTkButton(master=button_frame, text="Convert to Roboflow JSON", height=40,
                                       command=lambda: self.convert_to_roboflow_json(globinfo_var))
        convert_button.grid(row=0, column=0, pady=10)

    def convert_to_roboflow_json(self, globinfo_var):
        """
        Converts the selected globinfo CSV to Roboflow COCO JSON.
        FIXED: Handles multi-part annotations by creating a separate annotation for each part.
        """
        globinfo_path = globinfo_var.get()

        if not globinfo_path or not os.path.exists(globinfo_path):
            messagebox.showerror("Error", "Please select a valid globinfo CSV file.")
            return

        output_json_path = os.path.splitext(globinfo_path)[0] + '_roboflow_coco.json'
        base_folder = os.path.dirname(os.path.dirname(globinfo_path))
        input_folder_metadata = None

        try:
            with open(globinfo_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('# input_directory:'):
                    input_folder_metadata = first_line.split(':', 1)[1].strip()

            df = pd.read_csv(globinfo_path, comment='#')
            image_base_path = input_folder_metadata if input_folder_metadata else base_folder

            required_columns = ['img_id', 'xmin', 'ymin', 'xmax', 'ymax', 'name', 'object_type']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if not col in df.columns]
                raise ValueError(f"Missing required columns in CSV: {', '.join(missing)}")

            coco_json = {"images": [], "annotations": [], "categories": []}
            annotation_id_counter = 1
            image_id_map = {}
            image_id_counter = 1

            df.dropna(subset=['name'], inplace=True)
            unique_names = sorted(df['name'].unique())
            name_to_cat_id = {name: i + 1 for i, name in enumerate(unique_names)}

            for name, cat_id in name_to_cat_id.items():
                coco_json["categories"].append({"id": cat_id, "name": name, "supercategory": "object"})

            grouped = df.groupby('img_id')
            total_imgs = len(grouped)
            processed_imgs = 0
            for img_id_rel, group in grouped:
                processed_imgs += 1
                self.update_progress(processed_imgs / total_imgs)
                self.update_processing_label(f"Converting image {processed_imgs}/{total_imgs}")

                img_path_full = os.path.join(image_base_path, img_id_rel)
                if not os.path.exists(img_path_full):
                    print(f"Warning: Image file '{img_path_full}' not found, skipping.")
                    continue

                try:
                    img = cv2.imread(img_path_full)
                    if img is None: raise IOError("imread failed")
                    height, width = img.shape[:2]
                except Exception as e:
                    print(f"Warning: Could not read image '{img_path_full}': {e}, skipping.")
                    continue

                current_image_id = image_id_counter
                image_id_map[img_id_rel] = current_image_id
                coco_json["images"].append({
                    "id": current_image_id, "file_name": os.path.basename(img_id_rel),
                    "width": width, "height": height
                })
                image_id_counter += 1

                for _, row in group.iterrows():
                    class_name = row['name']
                    if class_name not in name_to_cat_id: continue
                    category_id = name_to_cat_id[class_name]
                    obj_type = str(row.get('object_type', 'box')).lower()

                    if obj_type == 'box':
                        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                        w_box, h_box = xmax - xmin, ymax - ymin
                        if w_box <= 0 or h_box <= 0: continue

                        annotation = {
                            "id": annotation_id_counter, "image_id": current_image_id,
                            "category_id": category_id, "bbox": [float(xmin), float(ymin), float(w_box), float(h_box)],
                            "area": float(w_box * h_box), "iscrowd": 0
                        }
                        coco_json["annotations"].append(annotation)
                        annotation_id_counter += 1

                    elif (obj_type == 'mask' or obj_type == 'polygon') and pd.notna(row.get('contours')):
                        try:
                            contours_list = json.loads(row['contours'])
                            if not isinstance(contours_list, list) or not contours_list: continue

                            for contour_points in contours_list:
                                if len(contour_points) < 3: continue
                                contour_np = np.array(contour_points, dtype=np.int32)

                                x, y, w, h = cv2.boundingRect(contour_np)
                                if w <= 0 or h <= 0: continue

                                area = cv2.contourArea(contour_np)
                                segmentation = [contour_np.flatten().tolist()]

                                annotation = {
                                    "id": annotation_id_counter, "image_id": current_image_id,
                                    "category_id": category_id, "bbox": [float(x), float(y), float(w), float(h)],
                                    "area": float(area), "segmentation": segmentation, "iscrowd": 0
                                }
                                coco_json["annotations"].append(annotation)
                                annotation_id_counter += 1

                        except (json.JSONDecodeError, TypeError):
                            print(f"Warning: Invalid contour JSON for {img_id_rel}, skipping row.")
                            continue

            self.update_processing_label("Saving JSON file...")
            with open(output_json_path, 'w') as f:
                json.dump(coco_json, f, indent=4)

            self.update_progress(1)
            self.update_processing_label("Conversion complete.")
            messagebox.showinfo("Success", f"Roboflow COCO JSON file created at:\n{output_json_path}")

        except Exception as e:
            self.update_progress(0)
            self.update_processing_label(f"Error: {e}")
            print(f"Error during Roboflow conversion: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred during conversion:\n{e}")


# --- Main Execution ---
if __name__ == "__main__":
    # Set high DPI awareness for Windows (optional, improves scaling)
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass  # Ignore if platform is not Windows or ctypes fails

    app = NemaCounterGUI()  # This starts the mainloop