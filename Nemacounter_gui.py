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

import nemacounter.utils as utils
import nemacounter.common as common
from nemacounter.detection import detection_workflow
from nemacounter.edition import edition_workflow
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
        self.fpath_segany = ""
        self.set_main_window()

    def open_directory(self, var, label_obj):
        dpath = filedialog.askdirectory(initialdir='.', title='Select directory')
        if dpath:
            var.set(dpath)
            label_obj.configure(text=os.path.relpath(dpath))
        else:
            var.set('')

    def get_globinfo_fpath(self, var, label_obj):
        fpath = filedialog.askopenfilename(initialdir='.', title='Select a globinfo file',
                                           filetypes=[('csv files', '*_globinfo.csv')])
        if fpath:
            var.set(fpath)
            label_obj.configure(text=os.path.relpath(fpath))
        else:
            var.set('')

    def change_appearance_mode_event(self, _):
        CTK.set_appearance_mode(self.theme_var.get())

    def change_scaling_event(self, _):
        new_scaling_float = int(self.scaling_var.get().replace("%", "")) / 100
        CTK.set_widget_scaling(new_scaling_float)

    def set_main_window(self):
        self.root.title("Nemacounter by Thomas Baum lab")
        self.root.geometry(f"{1100}x{700}")
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure((0, 1, 2), weight=1)
        self.set_side_bar()
        self.set_central_tabview()
        self.root.mainloop()

    def display_cpu_number(self, val):
        self.label_cpu_slider.configure(text=f"Max. number of CPU: {int(val)}")

    def display_confidence(self, val):
        self.label_conf_slider.configure(text=f"Confidence Threshold: {np.round(val, 2)}")

    def display_overlap(self, val):
        self.label_overl_slider.configure(text=f"Overlap Threshold: {np.round(val, 2)}")

    def set_side_bar(self):
        sidebar_frame = CTK.CTkFrame(master=self.root, width=140, corner_radius=0)
        sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")

        logo_frame = CTK.CTkFrame(master=sidebar_frame, fg_color="transparent")
        logo_frame.grid(row=0, column=0, rowspan=4)
        nemacounter_logo = CTK.CTkImage(PILimage.open(os.path.relpath("conf/logo.png")), size=(300, 300))
        image_label = CTK.CTkLabel(master=logo_frame, image=nemacounter_logo, text='')
        image_label.grid(row=0, column=0)

        hardware_frame = CTK.CTkFrame(master=sidebar_frame, fg_color="transparent")
        hardware_frame.grid(row=5, column=0, rowspan=2)
        switch_GPU = CTK.CTkSwitch(master=hardware_frame, variable=self.use_GPU,
                                   onvalue=1, offvalue=0, text=f"Use GPU if available")
        switch_GPU.grid(row=0, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        nb_avail_cpu = utils.compute_available_cpu()
        self.label_cpu_slider = CTK.CTkLabel(master=hardware_frame, text=f"Max. number of CPU: {nb_avail_cpu - 1}",
                                             anchor="w")
        self.label_cpu_slider.grid(row=1, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        slider_cpu = CTK.CTkSlider(master=hardware_frame, from_=1, to=nb_avail_cpu,
                                   number_of_steps=nb_avail_cpu, variable=self.nb_wanted_cpu,
                                   command=self.display_cpu_number)
        slider_cpu.set(nb_avail_cpu - 1)
        slider_cpu.grid(row=2, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        displparams_frame = CTK.CTkFrame(master=sidebar_frame, fg_color="transparent")
        displparams_frame.grid(row=7, column=0, rowspan=2)
        appearance_mode_label = CTK.CTkLabel(master=displparams_frame, text="Appearance Mode:", anchor="w")
        appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        appearance_mode_optionemenu = CTK.CTkOptionMenu(master=displparams_frame,
                                                        values=["Dark", "Light", "System"],
                                                        command=self.change_appearance_mode_event,
                                                        variable=self.theme_var)
        appearance_mode_optionemenu.grid(row=7, column=0, padx=20, pady=(10, 10))

        scaling_label = CTK.CTkLabel(master=displparams_frame, text="UI Scaling:", anchor="w")
        scaling_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        scaling_optionemenu = CTK.CTkOptionMenu(master=displparams_frame,
                                                values=["80%", "90%", "100%", "110%", "120%"],
                                                command=self.change_scaling_event,
                                                variable=self.scaling_var)
        scaling_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 20))

        modelspath_frame = CTK.CTkFrame(master=sidebar_frame, fg_color="transparent")
        modelspath_frame.grid(row=9, column=0, rowspan=2)

        model_label = CTK.CTkLabel(master=modelspath_frame, text="Select Model:", anchor="w")
        model_label.grid(row=0, column=0, padx=20, pady=(10, 0), sticky='w')

        model_files = [f for f in os.listdir("models") if f.endswith('.pt')]
        if not model_files:
            message = "No model files found in the models directory."
            tk.messagebox.showwarning("Warning", message)
            sys.exit()

        self.model_var.set(model_files[0])  # Select the first model by default
        model_menu = CTK.CTkOptionMenu(master=modelspath_frame, values=model_files, variable=self.model_var)
        model_menu.grid(row=1, column=0, padx=20, pady=(10, 10))

        fpath_conf = 'conf/config.ini'
        if os.path.exists(fpath_conf):
            config = common.get_config_info(os.path.abspath(fpath_conf))
            self.sam2_checkpoint_path = os.path.relpath(config['Models']['sam2_model_checkpoint_path'])
            self.sam2_config_path = os.path.relpath(config['Models']['sam2_model_config_path'])

        else:
            message = "No configuration file found"
            tk.messagebox.showwarning("Warning", message)
            sys.exit()

    def start_detection(self, indir_var, outdir_var, projid_entry, confslid_var, overslid_var, add_overlay_var,
                        show_bbox_var, show_conf_var, show_mask_var):
        if not self.model_var.get():
            messagebox.showwarning("Warning", "You must select a model before launching the analysis.")
            return

        dct_var_detection = {
            'input_directory': indir_var.get(),
            'output_directory': outdir_var.get(),
            'project_id': projid_entry.get(),
            'conf_thresh': confslid_var.get(),
            'overlap_thresh': overslid_var.get(),
            'add_overlay': add_overlay_var.get(),
            'model_path': os.path.join("models", self.model_var.get()),
            'gpu': self.use_GPU.get(),
            'cpu': self.nb_wanted_cpu.get(),
            # Additional toggles for detection overlay:
            'show_bbox': show_bbox_var.get(),
            'show_conf': show_conf_var.get(),
            'show_mask': show_mask_var.get()
        }

        if (dct_var_detection['input_directory'] != '') and (dct_var_detection['output_directory'] != ''):
            project_outdir = os.path.join(dct_var_detection['output_directory'], dct_var_detection['project_id'])
            if os.path.exists(project_outdir):
                message = f"The folder '{project_outdir}' already exists."
                messagebox.showwarning("Warning", message)
            else:
                execution_log = detection_workflow(dct_var_detection, gui=True)
                if execution_log is None:
                    message = "Detection executed successfully."
                    messagebox.showinfo("Info", message)
                else:
                    message = "An error occurred during program execution:"
                    messagebox.showerror("Error", message)
        else:
            message = "You must select an input and an output directory before launching the analysis"
            messagebox.showwarning("Warning", message)

    def tabview_callback(self, value):
        selected_tab = value.get()
        if selected_tab == "Object Detection":
            self.set_detection_tab()
        elif selected_tab == "Manual Edition":
            self.set_edition_tab()
        elif selected_tab == "Object Segmentation":
            self.set_segmentation_tab()
        elif selected_tab == "Export to Roboflow":
            self.set_export_tab()

    def set_central_tabview(self):
        tabview = CTK.CTkTabview(master=self.root,
                                 command=lambda: self.tabview_callback(tabview),
                                 fg_color="transparent")
        tabview.grid(row=0, column=1, rowspan=4,
                     padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.detection_tab = tabview.add("Object Detection")
        self.edition_tab = tabview.add("Manual Edition")
        self.segmentation_tab = tabview.add("Object Segmentation")
        self.export_tab = tabview.add("Export to Roboflow")  # New tab for exporting

        # Set up the tabs
        self.set_detection_tab()
        self.set_edition_tab()
        self.set_segmentation_tab()
        self.set_export_tab()  # Set up the new tab

    def set_detection_tab(self):
        for widget in self.detection_tab.winfo_children():
            widget.destroy()
        projid_frame = CTK.CTkFrame(master=self.detection_tab,
                                    fg_color="transparent",
                                    width=500,
                                    height=50)
        projid_frame.grid(row=0, column=0, pady=(40, 10))
        projid_text = CTK.CTkLabel(master=projid_frame, text='Enter a project name')
        projid_text.grid(row=0, column=0, sticky='')
        projid_var = CTK.StringVar(value="MyProject")
        projid_entry = CTK.CTkEntry(master=projid_frame, placeholder_text=projid_var.get(),
                                    textvariable=projid_var)
        projid_entry.grid(row=1, column=0, sticky='')

        parameters_frame = CTK.CTkFrame(master=self.detection_tab,
                                        fg_color="transparent", width=500, height=400)
        parameters_frame.grid(row=1, column=0, pady=5)

        indir_frame = CTK.CTkFrame(master=parameters_frame,
                                   fg_color="transparent")
        indir_frame.grid(row=1, column=0, padx=20, pady=(20, 10), sticky="w")
        indir_var = CTK.StringVar()
        indir_text = CTK.CTkLabel(master=indir_frame, text='Select Input Image Directory:')
        indir_text.grid(row=0, column=0, sticky="w")
        indir_button = CTK.CTkButton(master=indir_frame,
                                     text="Select",
                                     command=lambda: self.open_directory(indir_var, indir_label))
        indir_button.grid(row=1, column=0, padx=20)
        indir_label = CTK.CTkLabel(master=indir_frame, text='', anchor="w")
        indir_label.grid(row=1, column=2)

        outdir_frame = CTK.CTkFrame(master=parameters_frame,
                                    fg_color="transparent")
        outdir_frame.grid(row=2, column=0, padx=20, pady=(5, 20), sticky="w")
        outdir_var = CTK.StringVar()
        outdir_text = CTK.CTkLabel(master=outdir_frame, text='Select Output Directory:')
        outdir_text.grid(row=0, column=0, sticky="w")
        outdir_button = CTK.CTkButton(master=outdir_frame,
                                      text="Select",
                                      command=lambda: self.open_directory(outdir_var, outdir_label))
        outdir_button.grid(row=1, column=0, padx=20)
        outdir_label = CTK.CTkLabel(master=outdir_frame, text='', anchor="w")
        outdir_label.grid(row=1, column=2)

        sliders_frame = CTK.CTkFrame(master=parameters_frame,
                                     fg_color="transparent")
        sliders_frame.grid(row=3, column=0, columnspan=4, padx=20, pady=20)
        confslid_var = CTK.DoubleVar()
        overslid_var = CTK.DoubleVar()
        default_conf_thresh = 0.5
        self.label_conf_slider = CTK.CTkLabel(master=sliders_frame,
                                              text=f"Confidence Threshold: {np.round(default_conf_thresh, 2)}",
                                              anchor="w")
        self.label_conf_slider.grid(row=0, column=0, columnspan=2, padx=(20, 10), pady=(10, 10), sticky="ew")
        confidence_slider = CTK.CTkSlider(master=sliders_frame, from_=0, to=1,
                                          number_of_steps=100, variable=confslid_var,
                                          command=self.display_confidence)
        confidence_slider.grid(row=1, column=0, columnspan=2, padx=(20, 10), pady=(10, 10), sticky="ew")
        confidence_slider.set(default_conf_thresh)

        default_overlap_val = 0.3
        self.label_overl_slider = CTK.CTkLabel(master=sliders_frame,
                                               text=f"Overlap Threshold: {np.round(default_overlap_val, 2)}",
                                               anchor="w")
        self.label_overl_slider.grid(row=0, column=2, columnspan=2, padx=(20, 10), pady=(10, 10), sticky="w")
        overlap_slider = CTK.CTkSlider(master=sliders_frame, from_=0, to=1,
                                       number_of_steps=100, variable=overslid_var,
                                       command=self.display_overlap)
        overlap_slider.grid(row=1, column=2, columnspan=2, padx=(20, 10), pady=(10, 10), sticky="w")
        overlap_slider.set(default_overlap_val)

        overlay_frame = CTK.CTkFrame(master=parameters_frame, fg_color="transparent")
        overlay_frame.grid(row=4, column=0, columnspan=4, padx=20, pady=(10, 10))
        overlay_switch_text = 'Save images copies with overlay'
        add_overlay_var = CTK.IntVar(value=1)
        overlay_switch = CTK.CTkSwitch(master=overlay_frame, variable=add_overlay_var,
                                       onvalue=1, offvalue=0, text=overlay_switch_text)
        overlay_switch.grid(row=0, column=0, padx=(20, 10), pady=(10, 10), sticky="w")

        # Additional toggles for bounding box, confidence, and mask (for detection only)
        toggles_frame = CTK.CTkFrame(master=parameters_frame, fg_color="transparent")
        toggles_frame.grid(row=5, column=0, columnspan=4, padx=20, pady=(10, 20), sticky="w")

        show_bbox_var = CTK.IntVar(value=1)
        show_conf_var = CTK.IntVar(value=0)
        show_mask_var = CTK.IntVar(value=0)
        show_bbox_check = CTK.CTkCheckBox(toggles_frame, text="Show Bounding Box", variable=show_bbox_var)
        show_bbox_check.grid(row=0, column=0, padx=(20, 5), pady=(5, 5))
        show_conf_check = CTK.CTkCheckBox(toggles_frame, text="Show Confidence", variable=show_conf_var)
        show_conf_check.grid(row=0, column=1, padx=(5, 5), pady=(5, 5))
        show_mask_check = CTK.CTkCheckBox(toggles_frame, text="Show Mask (seg. model only)", variable=show_mask_var)
        show_mask_check.grid(row=0, column=2, padx=(5, 5), pady=(5, 5))

        start_button_frame = CTK.CTkFrame(master=self.detection_tab,
                                          fg_color="transparent", width=500, height=50)
        start_button_frame.grid(row=2, column=0, pady=10)

        start_button = CTK.CTkButton(master=start_button_frame,
                                     text="Start Detection",
                                     command=lambda: self.start_detection(indir_var,
                                                                          outdir_var,
                                                                          projid_entry,
                                                                          confslid_var,
                                                                          overslid_var,
                                                                          add_overlay_var,
                                                                          show_bbox_var,
                                                                          show_conf_var,
                                                                          show_mask_var))
        start_button.grid(row=0, column=0)

        self.detection_tab.grid_columnconfigure(0, weight=1)

    def set_edition_tab(self):
        for widget in self.edition_tab.winfo_children():
            widget.destroy()
        master_frame = CTK.CTkFrame(master=self.edition_tab, fg_color="transparent")
        master_frame.grid(row=0, column=0)
        projid_frame = CTK.CTkFrame(master=master_frame, fg_color="transparent")
        projid_frame.grid(row=0, column=0)
        projid_text = CTK.CTkLabel(master=projid_frame, text='Enter a project name')
        projid_text.grid(row=0, column=0, sticky='')
        projid_entry = CTK.CTkEntry(master=projid_frame, placeholder_text="MyProject")
        projid_entry.grid(row=1, column=0, sticky='')

        input_frame = CTK.CTkFrame(master=master_frame,
                                   fg_color="transparent",
                                   width=400, height=300)
        input_frame.grid(row=1, column=0, padx=20, pady=(20, 10), sticky='w')
        infile_var = CTK.StringVar()
        infile_text = CTK.CTkLabel(master=input_frame, text='Select *.globinfo.csv file:')
        infile_text.grid(row=0, column=0, sticky="w")
        infile_button = CTK.CTkButton(master=input_frame,
                                      text="Select",
                                      command=lambda: self.get_globinfo_fpath(infile_var, infile_label))
        infile_button.grid(row=1, column=0, padx=20)
        infile_label = CTK.CTkLabel(master=input_frame, text='', anchor="w")
        infile_label.grid(row=1, column=2)

        outdir_var = CTK.StringVar()
        outdir_text = CTK.CTkLabel(master=input_frame, text='Select Output Directory:')
        outdir_text.grid(row=2, column=0, sticky="w")
        outdir_button = CTK.CTkButton(master=input_frame,
                                      text="Select",
                                      command=lambda: self.open_directory(outdir_var, outdir_label))
        outdir_button.grid(row=3, column=0, padx=20)
        outdir_label = CTK.CTkLabel(master=input_frame, text='', anchor="w")
        outdir_label.grid(row=3, column=2)

        start_button_frame = CTK.CTkFrame(master=master_frame,
                                          fg_color="transparent", width=500, height=50)
        start_button_frame.grid(row=3, column=0, pady=10)

        start_button = CTK.CTkButton(master=start_button_frame,
                                     text="Start Manual Edition",
                                     command=lambda: self.start_manual_edition(projid_entry, infile_var, outdir_var))
        start_button.grid(row=0, column=0, padx=175)

        self.edition_tab.grid_rowconfigure(0, weight=1)
        self.edition_tab.grid_columnconfigure(0, weight=1)

    def start_manual_edition(self, projid_entry, infile_var, outdir_var):
        fpath_globinfo = infile_var.get()
        dpath_parent = outdir_var.get()
        project_id = projid_entry.get()
        dpath_outdir = os.path.join(dpath_parent, project_id)

        if os.path.exists(fpath_globinfo) and dpath_outdir:
            if not os.path.exists(dpath_outdir):
                os.makedirs(dpath_outdir)
            try:
                # Pass the SAM2 model paths
                edition_workflow(
                    input_file=fpath_globinfo,
                    output_directory=dpath_outdir,
                    project_id=project_id,
                    sam_model_checkpoint=self.sam2_checkpoint_path,
                    sam_model_config=self.sam2_config_path,
                    use_gpu=self.use_GPU.get()
                )
                messagebox.showinfo("Success", "Manual edition process completed successfully.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            if not os.path.exists(fpath_globinfo):
                messagebox.showerror("Error", "Please provide an input *globinfo.csv file.")
            if not dpath_outdir:
                messagebox.showerror("Error", "Please provide an output directory.")

    def set_segmentation_tab(self):
        for widget in self.segmentation_tab.winfo_children():
            widget.destroy()
        parameters_frame = CTK.CTkFrame(master=self.segmentation_tab,
                                        fg_color="transparent", width=500, height=400)
        parameters_frame.grid(row=0, column=0, pady=5)

        input_frame = CTK.CTkFrame(master=parameters_frame,
                                   fg_color="transparent")
        input_frame.grid(row=1, column=0, padx=20, pady=(20, 10))
        infile_var = CTK.StringVar()
        infile_text = CTK.CTkLabel(master=input_frame, text='Select *.globinfo.csv file:')
        infile_text.grid(row=0, column=0, sticky="w")
        infile_button = CTK.CTkButton(master=input_frame,
                                      text="Select",
                                      command=lambda: self.get_globinfo_fpath(infile_var, infile_label))
        infile_button.grid(row=1, column=0, padx=20)
        infile_label = CTK.CTkLabel(master=input_frame, text='', anchor="w")
        infile_label.grid(row=1, column=2)

        overlay_frame = CTK.CTkFrame(master=parameters_frame, fg_color="transparent")
        overlay_frame.grid(row=4, column=0, columnspan=4, padx=20, pady=20)
        overlay_switch_text = 'Save images copies with the segmentation overlay'
        add_overlay_var = CTK.IntVar(value=1)
        overlay_switch = CTK.CTkSwitch(master=overlay_frame, variable=add_overlay_var,
                                       onvalue=1, offvalue=0, text=overlay_switch_text)
        overlay_switch.grid(row=0, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        start_button_frame = CTK.CTkFrame(master=self.segmentation_tab,
                                          fg_color="yellow", width=500, height=50)
        start_button_frame.grid(row=1, column=0, pady=10)

        start_button = CTK.CTkButton(master=start_button_frame,
                                     text="Start Segmentation",
                                     command=lambda: self.start_segmentation(infile_var, add_overlay_var))
        start_button.grid(row=0, column=0)

        self.progress_frame = CTK.CTkFrame(master=self.segmentation_tab,
                                           fg_color="transparent", width=500, height=50)
        self.progress_frame.grid(row=2, column=0, pady=10)
        self.progress_bar = CTK.CTkProgressBar(master=self.progress_frame)
        self.progress_bar.grid(row=0, column=0, padx=20, pady=(10, 10), sticky="ew")
        self.progress_bar.set(0)

        # Add the processing label
        self.processing_label = CTK.CTkLabel(master=self.progress_frame, text="Processing...", anchor="w")
        self.processing_label.grid(row=1, column=0, padx=20, pady=(10, 10), sticky="ew")
        self.processing_label.grid_remove()

        self.segmentation_tab.grid_columnconfigure(0, weight=1)

    def start_segmentation(self, infile_var, add_overlay_var):
        dct_var_segmentation = {
            'input_file': infile_var.get(),
            'add_overlay': add_overlay_var.get(),
            'segany': self.sam2_checkpoint_path,
            'sam2_config': self.sam2_config_path,
            'gpu': self.use_GPU.get(),
            'cpu': self.nb_wanted_cpu.get()
        }

        if dct_var_segmentation['input_file'] == '':
            message = f"No *_globinfo.csv file selected. Please select a file before launching the analysis."
            messagebox.showwarning("Warning", message)
        else:
            self.processing_label.grid()
            threading.Thread(target=self.run_segmentation_workflow, args=(dct_var_segmentation,)).start()

    def run_segmentation_workflow(self, dct_var_segmentation):
        dct_var_segmentation['project_id'] = os.path.basename(dct_var_segmentation['input_file']).replace(
            '_globinfo.csv', '')
        dct_var_segmentation['input_dir'] = os.path.dirname(dct_var_segmentation['input_file'])

        gpu_if_avail = utils.get_bool(dct_var_segmentation['gpu'])
        add_overlay = utils.get_bool(dct_var_segmentation['add_overlay'])
        utils.set_cpu_usage(dct_var_segmentation['cpu'])

        # Safely check if CUDA is available
        cuda_available = False
        try:
            cuda_available = torch.cuda.is_available()
        except (AssertionError, RuntimeError):
            cuda_available = False

        # If CUDA is not available, ensure gpu_if_avail is False
        if not cuda_available:
            gpu_if_avail = False

        device = torch.device('cuda:0' if cuda_available and gpu_if_avail else 'cpu')

        print(f"cuda_available: {cuda_available}")
        print(f"gpu_if_avail: {gpu_if_avail}")
        print(f"Using device: {device}")

        if add_overlay:
            dpath_overlay = os.path.join(dct_var_segmentation['input_dir'], dct_var_segmentation['project_id'],
                                         'img',
                                         'segmentation')
            os.makedirs(dpath_overlay, exist_ok=True)

        if utils.check_file_existence(dct_var_segmentation['input_file']):
            df = pd.read_csv(dct_var_segmentation['input_file'])
            lst_img_paths = df['img_id'].unique()
            segmentation_model = NemaCounterSegmentation(
                device=device
            )

            all_annotations = []
            total_images = len(lst_img_paths)
            processed_images = 0

            for img_path in lst_img_paths:
                img = common.read_image(img_path)
                img_df = df[df['img_id'] == img_path].reset_index(drop=True)

                annotations = []
                for idx, row in img_df.iterrows():
                    ann = row.to_dict()
                    if ann['object_type'] == 'box':
                        ann['xmin'] = int(ann['xmin'])
                        ann['ymin'] = int(ann['ymin'])
                        ann['xmax'] = int(ann['xmax'])
                        ann['ymax'] = int(ann['ymax'])
                        annotations.append(ann)
                    elif ann['object_type'] == 'mask':
                        # Reconstruct mask from contours
                        contours_json = ann['contours']
                        if pd.isna(contours_json) or len(contours_json) == 0:
                            continue
                        contours = json.loads(contours_json)
                        if len(contours) == 0:
                            continue
                        contours_np = np.array(contours, dtype=np.int32)
                        if contours_np.ndim == 2:
                            contours_np = contours_np.reshape((-1, 1, 2))
                        elif contours_np.ndim == 3:
                            pass
                        else:
                            continue  # Invalid contour shape
                        # Create mask
                        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                        cv2.fillPoly(mask, [contours_np], 1)
                        ann['mask'] = mask
                        annotations.append(ann)

                if not annotations:
                    print(f"No valid annotations for image {img_path}. Skipping.")
                    continue

                # Pass annotations to objects_segmentation
                masks, updated_annotations = segmentation_model.objects_segmentation(img, annotations)

                # Update DataFrame with new annotations
                records = []
                for ann, mask in zip(updated_annotations, masks):
                    # Extract contours from mask
                    contours_list, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours_list) > 0:
                        # Take the largest contour
                        contour = max(contours_list, key=cv2.contourArea)
                        area = cv2.contourArea(contour)
                        ann['area'] = area
                        ann['contours'] = json.dumps(contour.squeeze().tolist())
                        x_min = int(np.min(contour[:, 0, 0]))
                        y_min = int(np.min(contour[:, 0, 1]))
                        x_max = int(np.max(contour[:, 0, 0]))
                        y_max = int(np.max(contour[:, 0, 1]))
                        ann['xmin'] = x_min
                        ann['ymin'] = y_min
                        ann['xmax'] = x_max
                        ann['ymax'] = y_max
                    else:
                        ann['area'] = 0
                        ann['contours'] = np.nan
                    records.append(ann)

                all_annotations.extend(records)

                if add_overlay:
                    # Overlay masks on the image
                    overlay_img = img.copy()
                    add_masks_on_image(masks, overlay_img)
                    fpath_out_img = os.path.join(dpath_overlay,
                                                 f"{dct_var_segmentation['project_id']}_{os.path.basename(img_path)}")
                    cv2.imwrite(fpath_out_img, overlay_img)

                    multicolored_img = create_multicolored_masks_image(masks)
                    fpath_out_multi = os.path.join(dpath_overlay,
                                                   f"{dct_var_segmentation['project_id']}_{os.path.splitext(os.path.basename(img_path))[0]}_colored.png")
                    cv2.imwrite(fpath_out_multi, multicolored_img)

                processed_images += 1
                progress = processed_images / total_images
                self.update_progress(progress)

            # Create the new globinfo DataFrame
            df_new = pd.DataFrame(all_annotations)
            df_new['project_id'] = dct_var_segmentation['project_id']
            df_new['object_id'] = df_new.groupby('img_id').cumcount() + 1

            expected_columns = ['img_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax',
                                'confidence', 'class', 'name', 'area', 'contours', 'object_type', 'project_id']
            df_new = df_new[expected_columns]

            # Save the updated globinfo.csv
            output_globinfo = os.path.join(dct_var_segmentation['input_dir'],
                                           f"{dct_var_segmentation['project_id']}_segmentation_globinfo.csv")
            df_new.to_csv(output_globinfo, index=False)

            # Create summary table
            df_summary = common.create_summary_table(df_new, dct_var_segmentation['project_id'])
            output_summary = os.path.join(dct_var_segmentation['input_dir'],
                                          f"{dct_var_segmentation['project_id']}_segmentation_summary.csv")
            df_summary.to_csv(output_summary, index=False)

            self.update_progress(1)  # Ensure progress bar is full
            self.processing_label.grid_remove()  # Hide processing label
            messagebox.showinfo("Success", "Object segmentation process completed successfully.")
        else:
            messagebox.showerror("Error", "The specified input file does not exist.")
    def update_progress(self, progress):
        self.progress_bar.set(progress)
        self.root.update_idletasks()

    def set_export_tab(self):
        for widget in self.export_tab.winfo_children():
            widget.destroy()
        master_frame = CTK.CTkFrame(master=self.export_tab, fg_color="transparent")
        master_frame.grid(row=0, column=0, padx=20, pady=20, sticky='nsew')

        instruction_label = CTK.CTkLabel(master=master_frame, text='Select a globinfo CSV file to convert to Roboflow JSON:')
        instruction_label.grid(row=0, column=0, sticky='w')

        file_selection_frame = CTK.CTkFrame(master=master_frame, fg_color="transparent")
        file_selection_frame.grid(row=1, column=0, pady=(10, 0), sticky='w')

        globinfo_var = CTK.StringVar()

        select_file_button = CTK.CTkButton(master=file_selection_frame,
                                           text="Select globinfo CSV File",
                                           command=lambda: self.get_globinfo_fpath(globinfo_var, selected_file_label))
        select_file_button.grid(row=0, column=0, padx=10)

        selected_file_label = CTK.CTkLabel(master=file_selection_frame, text='', anchor="w")
        selected_file_label.grid(row=0, column=1, padx=10)

        convert_button = CTK.CTkButton(master=master_frame,
                                       text="Convert to Roboflow JSON",
                                       command=lambda: self.convert_to_roboflow_json(globinfo_var))
        convert_button.grid(row=2, column=0, pady=(20, 0))

        self.export_tab.grid_rowconfigure(0, weight=1)
        self.export_tab.grid_columnconfigure(0, weight=1)

    def convert_to_roboflow_json(self, globinfo_var):
        globinfo_path = globinfo_var.get()

        if not globinfo_path or not os.path.exists(globinfo_path):
            messagebox.showerror("Error", "Please select a valid globinfo CSV file.")
            return

        try:
            # Read the globinfo CSV file
            df = pd.read_csv(globinfo_path)

            # Check if required columns are present
            required_columns = ['img_id', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'object_type']
            for col in required_columns:
                if col not in df.columns:
                    messagebox.showerror("Error", f"Missing required column: {col}")
                    return

            # Determine the input folder (directory containing images)
            input_folder = os.path.dirname(globinfo_path)

            # Prepare the COCO JSON structure
            coco_json = {
                "images": [],
                "annotations": [],
                "categories": []
            }

            # Create category mapping
            categories = df['class'].unique()
            category_mapping = {int(cat): idx + 1 for idx, cat in enumerate(categories)}
            for cat in categories:
                category_entry = {
                    "id": category_mapping[int(cat)],
                    "name": str(cat),
                    "supercategory": "none"
                }
                coco_json["categories"].append(category_entry)

            # Initialize counters
            annotation_id = 1
            image_id_mapping = {}
            image_id_counter = 1

            # Group annotations by image
            grouped = df.groupby('img_id')

            for img_id, group in grouped:
                # Get image path and dimensions
                img_path = img_id
                if not os.path.isabs(img_path):
                    img_path = os.path.join(input_folder, img_id)
                if not os.path.exists(img_path):
                    messagebox.showerror("Error", f"Image file does not exist: {img_path}")
                    return
                img = cv2.imread(img_path)
                height, width = img.shape[:2]

                # Create image entry
                image_entry = {
                    "id": image_id_counter,
                    "file_name": os.path.basename(img_id),
                    "width": width,
                    "height": height
                }
                coco_json["images"].append(image_entry)
                image_id_mapping[img_id] = image_id_counter
                image_id_counter += 1

                for _, row in group.iterrows():
                    category_id = category_mapping[int(row['class'])]
                    if row['object_type'] == 'box':
                        # For bounding boxes
                        xmin = row['xmin']
                        ymin = row['ymin']
                        xmax = row['xmax']
                        ymax = row['ymax']
                        width_bbox = xmax - xmin
                        height_bbox = ymax - ymin
                        area = width_bbox * height_bbox
                        bbox = [xmin, ymin, width_bbox, height_bbox]

                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id_mapping[img_id],
                            "category_id": category_id,
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0
                        }
                        annotation_id += 1
                    elif row['object_type'] == 'mask':
                        # For segmentation masks
                        if pd.isna(row['contours']):
                            continue
                        contours = json.loads(row['contours'])
                        segmentation = [np.array(contours).flatten().tolist()]

                        area = row['area'] if not pd.isna(row['area']) else 0
                        bbox = [row['xmin'], row['ymin'], row['xmax'] - row['xmin'], row['ymax'] - row['ymin']]

                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id_mapping[img_id],
                            "category_id": category_id,
                            "segmentation": segmentation,
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0
                        }
                        annotation_id += 1
                    else:
                        continue  # Skip unknown object types

                    coco_json["annotations"].append(annotation)

            # Output file path
            output_json_path = os.path.join(input_folder, 'roboflow_annotations.json')

            # Save the JSON file
            with open(output_json_path, 'w') as f:
                json.dump(coco_json, f, indent=4)

            messagebox.showinfo("Success", f"Roboflow JSON file created at: {output_json_path}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


if __name__ == "__main__":
    app = NemaCounterGUI()
