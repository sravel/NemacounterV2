from ultralytics import YOLO
import torch
import os
import cv2
import pandas as pd
import numpy as np
import json  # To store contours as JSON strings
import sys
from pathlib import Path  # For handling file paths
import csv
import nemacounter.utils as utils
import nemacounter.common as common

# Define a global color list
colors_list = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (255, 255, 255), # White
    (0, 128, 255),   # Orange
    (255, 0, 127),   # Pink
    (127, 0, 255)    # Purple
]
num_colors = len(colors_list)


class NemaCounterDetection:

    def __init__(self, weights_path, conf_thresh=0.5, iou_thresh=0.3, device='cpu'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.custom_model = YOLO(weights_path)
        self.custom_model.to(self.device)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model_task = self.custom_model.task  # 'detect' or 'segment'
        self.class_names = self.custom_model.names

    def detect_objects(self, img, img_path):
        img_height, img_width = img.shape[:2]
        input_size = self.custom_model.__dict__["overrides"]["imgsz"]

        if 'cuda' in self.device:
            half = True
        else:
            half = False
        img = cv2.imread(img_path)
        results = self.custom_model.predict(
            source=img,
            imgsz=input_size,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
            retina_masks=True,
            half=half,
            max_det=100000
        )

        adjusted_size = results[0].orig_shape[:2]  # (height, width)

        if not results or not results[0].boxes:
            df_columns = ['img_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax',
                          'confidence', 'class', 'name', 'area', 'contours', 'object_type']
            df = pd.DataFrame(columns=df_columns)
            return df, None

        scale_x = img_width / adjusted_size[1]
        scale_y = img_height / adjusted_size[0]

        if self.model_task == 'detect':
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            names = [self.class_names[class_id] for class_id in classes]

            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y

            df = pd.DataFrame()
            df['xmin'] = boxes[:, 0].astype(int)
            df['ymin'] = boxes[:, 1].astype(int)
            df['xmax'] = boxes[:, 2].astype(int)
            df['ymax'] = boxes[:, 3].astype(int)
            df['confidence'] = scores
            df['class'] = classes
            df['name'] = names
            df['contours'] = np.nan
            df['object_type'] = 'box'
            df['area'] = np.nan
            return df, None

        elif self.model_task == 'segment':
            boxes_list = []
            scores_list = []
            classes_list = []
            names_list = []
            masks_list = []
            areas_list = []
            contours_json_list = []

            num_detections = len(results[0].boxes)
            for idx in range(num_detections):
                class_id = int(results[0].boxes.cls[idx].cpu().numpy().squeeze())
                class_name = self.class_names[class_id]
                score = results[0].boxes.conf[idx].cpu().numpy().squeeze()
                box = results[0].boxes.xyxy[idx].cpu().numpy().squeeze()
                box[0] *= scale_x
                box[1] *= scale_y
                box[2] *= scale_x
                box[3] *= scale_y

                mask_tensor = results[0].masks.data[idx]
                mask = mask_tensor.cpu().numpy().squeeze()
                mask = (mask * 255).astype(np.uint8)

                area = np.sum(mask > 0)
                contours, hierarchy = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                contours = [cnt for cnt in contours if cnt.size > 0]

                if not contours:
                    continue

                adjusted_contours = []
                for contour in contours:
                    contour = contour.reshape(-1, 2).astype(int)
                    adjusted_contours.append(contour.tolist())

                contours_json = json.dumps(adjusted_contours)

                boxes_list.append(box)
                scores_list.append(score)
                classes_list.append(class_id)
                names_list.append(class_name)
                masks_list.append(mask)
                areas_list.append(area)
                contours_json_list.append(contours_json)

            if boxes_list:
                boxes_array = np.array(boxes_list)
                df = pd.DataFrame()
                df['xmin'] = boxes_array[:, 0].astype(int)
                df['ymin'] = boxes_array[:, 1].astype(int)
                df['xmax'] = boxes_array[:, 2].astype(int)
                df['ymax'] = boxes_array[:, 3].astype(int)
                df['confidence'] = scores_list
                df['class'] = classes_list
                df['name'] = names_list
                df['contours'] = contours_json_list
                df['object_type'] = 'mask'
                df['area'] = areas_list
                masks = np.array(masks_list)
                return df, masks
            else:
                df_columns = ['img_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax',
                              'confidence', 'class', 'name', 'area', 'contours', 'object_type']
                df = pd.DataFrame(columns=df_columns)
                return df, None


def add_boxes_on_image(boxes, img, classes, color_map, show_conf=False, confidences=None):
    for i, (box, class_id) in enumerate(zip(boxes, classes)):
        xmin, ymin, xmax, ymax = box
        color = color_map[class_id]
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        if show_conf and confidences is not None:
            conf_text = f"{confidences[i]:.2f}"
            cv2.putText(img, conf_text, (int(xmin), int(ymin) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def add_masks_on_image(masks, img, classes, color_map, show_conf=False, confidences=None):
    img_height, img_width = img.shape[:2]
    overlay = img.copy()

    for i, (mask, class_id) in enumerate(zip(masks, classes)):
        color = color_map[class_id]
        if mask.shape != (img_height, img_width):
            mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask > 0).astype(np.uint8)

        colored_mask = np.zeros_like(img, dtype=np.uint8)
        colored_mask[mask_binary == 1] = color

        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

        # If show_conf is True, write confidence near the mask's bounding box
        # For that, we might need a bounding box from the mask:
        if show_conf and confidences is not None:
            ys, xs = np.where(mask_binary == 1)
            if len(xs) > 0 and len(ys) > 0:
                xmin, ymin = np.min(xs), np.min(ys)
                conf_text = f"{confidences[i]:.2f}"
                cv2.putText(overlay, conf_text, (int(xmin), int(ymin) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    img[:] = overlay


def create_project_dirs_structure(dpath_outdir, project_id, display_overlay=False, model_task='detect'):
    dpath_project = os.path.join(dpath_outdir, project_id)
    if os.path.isdir(dpath_project):
        print('Output directory already exists, program will stop.')
        sys.exit()
    else:
        os.makedirs(dpath_project, mode=0o755)
        if display_overlay:
            if model_task == 'detect':
                os.makedirs(os.path.join(dpath_project, 'img', 'bounding_boxes'), mode=0o755, exist_ok=True)
            elif model_task == 'segment':
                os.makedirs(os.path.join(dpath_project, 'img', 'masks'), mode=0o755, exist_ok=True)


def detection_workflow(dct_args, gui=True):
    gpu_if_avail = utils.get_bool(dct_args['gpu'])
    add_overlay = utils.get_bool(dct_args['add_overlay'])
    show_bbox = utils.get_bool(dct_args.get('show_bbox', 1))
    show_conf = utils.get_bool(dct_args.get('show_conf', 0))
    show_mask = utils.get_bool(dct_args.get('show_mask', 0))

    utils.set_cpu_usage(dct_args['cpu'])
    device = 'cuda:0' if torch.cuda.is_available() and gpu_if_avail else 'cpu'

    lst_img_paths = utils.list_image_files(dct_args['input_directory'])

    detection_model = NemaCounterDetection(
        os.path.relpath(dct_args['model_path']),
        conf_thresh=dct_args['conf_thresh'],
        iou_thresh=dct_args['overlap_thresh'],
        device=device
    )

    model_task = detection_model.model_task
    class_names = detection_model.class_names
    # Create a stable color_map for all classes based on class_id
    # Class IDs are 0-based. Use them directly:
    color_map = {}
    for class_id in range(len(class_names)):
        color_map[class_id] = colors_list[class_id % num_colors]

    create_project_dirs_structure(dct_args['output_directory'], dct_args['project_id'],
                                  display_overlay=add_overlay, model_task=model_task)

    lst_df = []
    for img_path in lst_img_paths:
        img = common.read_image(img_path)
        df, masks = detection_model.detect_objects(img, img_path)

        df['img_id'] = img_path
        df['object_id'] = df.index.values + 1

        df = df[['img_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax',
                 'confidence', 'class', 'name', 'area', 'contours', 'object_type']]

        if add_overlay:
            classes = df['class'].values
            confidences = df['confidence'].values

            if model_task == 'detect':
                # Show bbox if toggled
                if show_bbox:
                    boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values
                    add_boxes_on_image(boxes, img, classes, color_map, show_conf=show_conf, confidences=confidences)

                # If the model is 'detect', we have no masks to draw unless it's actually a seg model
                # show_mask is ignored if no masks
                output_subdir = 'img/bounding_boxes'

            elif model_task == 'segment':
                # If user wants mask and masks exist
                if show_mask and masks is not None:
                    add_masks_on_image(masks, img, classes, color_map, show_conf=show_conf, confidences=confidences)
                # If user wants bbox too (segment models also have bbox info)
                if show_bbox:
                    boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values
                    add_boxes_on_image(boxes, img, classes, color_map, show_conf=show_conf, confidences=confidences)
                output_subdir = 'img/masks'

            fpath_out_img = os.path.join(
                dct_args['output_directory'],
                dct_args['project_id'],
                output_subdir,
                f"{dct_args['project_id']}_{os.path.basename(img_path)}"
            )
            os.makedirs(os.path.dirname(fpath_out_img), exist_ok=True)
            if not cv2.imwrite(fpath_out_img, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80]):
                raise Exception("Could not write image")

        lst_df.append(df)

    dpath_stats = os.path.join(dct_args['output_directory'], dct_args['project_id'])
    df_global = pd.concat(lst_df, ignore_index=True)

    expected_columns = ['img_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax',
                        'confidence', 'class', 'name', 'area', 'contours', 'object_type']
    for col in expected_columns:
        if col not in df_global.columns:
            df_global[col] = np.nan
    df_global = df_global[expected_columns]

    df_global.to_csv(
        os.path.join(dpath_stats, f"{dct_args['project_id']}_globinfo.csv"),
        index=False,
        quoting=csv.QUOTE_ALL
    )

    df_summary = common.create_summary_table(df_global, dct_args['project_id'])
    df_summary.to_csv(
        os.path.join(dpath_stats, f"{dct_args['project_id']}_summary.csv"),
        index=False,
        quoting=csv.QUOTE_ALL
    )
    return None
