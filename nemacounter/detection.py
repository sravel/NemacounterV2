# detection.py
# COMPLETE VERSION: Supports both simple splitting and two-stage fusion modes
# Mode controlled by 'use_fusion' parameter from GUI
# Optimized for high memory usage with retina masks

from ultralytics import YOLO
import torch
import os
import cv2
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import csv
import gc  # For garbage collection
from collections import defaultdict

# Import utils and common
try:
    import nemacounter.utils as utils
    import nemacounter.common as common
except ImportError:
    print("Warning: 'nemacounter' package not found. Using mock objects for utils and common.")


def draw_detections_yolo_style(img, df, show_bbox=True, show_labels=True,
                               show_conf=False, show_mask=True):
    """
    Draw ALL detections from a dataframe, mimicking YOLO's aesthetic.
    Used for all segmentation overlays to ensure the overlay matches the final data.
    """
    overlay = img.copy()
    height, width = overlay.shape[:2]

    # YOLO-style colors (vibrant, same as ultralytics uses)
    yolo_colors = [
        (255, 56, 56),  # Red
        (255, 157, 151),  # Light red
        (255, 112, 31),  # Orange
        (255, 178, 29),  # Yellow-orange
        (207, 210, 49),  # Yellow-green
        (72, 249, 10),  # Green
        (146, 204, 23),  # Light green
        (61, 219, 134),  # Turquoise
        (26, 147, 52),  # Dark green
        (0, 212, 187),  # Cyan
        (44, 153, 168),  # Teal
        (0, 194, 255),  # Light blue
        (52, 69, 147),  # Blue
        (100, 115, 255),  # Purple-blue
        (0, 24, 236),  # Dark blue
        (132, 56, 255),  # Purple
        (82, 0, 133),  # Dark purple
        (203, 56, 255),  # Magenta
        (255, 149, 200),  # Pink
        (255, 55, 199),  # Hot pink
    ]

    # Extend colors if needed
    while len(yolo_colors) < 100:
        yolo_colors.append((
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        ))

    print(f"Drawing {len(df)} detections with YOLO-style aesthetic...")

    # First pass: Draw all masks (underneath)
    if show_mask:
        mask_overlay = overlay.copy()

        for idx, row in df.iterrows():
            if idx % 100 == 0 and idx > 0:
                print(f"  Processing mask {idx}/{len(df)}...")
                # Periodic memory cleanup for large datasets
                if idx % 200 == 0:
                    gc.collect()

            class_id = int(row['class']) if not pd.isna(row['class']) else 0
            color = yolo_colors[class_id % len(yolo_colors)]

            # Draw mask using contours
            if pd.notna(row.get('contours')):
                try:
                    contours_json = row['contours']
                    if contours_json and contours_json != '[]':
                        contours = json.loads(contours_json)

                        for contour in contours:
                            if isinstance(contour, list) and len(contour) >= 3:
                                pts = np.array(contour, dtype=np.int32)
                                if pts.ndim == 2:
                                    pts = pts.reshape((-1, 1, 2))
                                # Fill with semi-transparent color
                                cv2.fillPoly(mask_overlay, [pts], color)
                except:
                    pass

        # Blend masks with original image (YOLO uses 0.5 alpha for masks)
        overlay = cv2.addWeighted(overlay, 0.5, mask_overlay, 0.5, 0)

    # Second pass: Draw all boxes and labels (on top)
    if show_bbox:
        for idx, row in df.iterrows():
            if idx % 100 == 0 and idx > 0:
                print(f"  Drawing boxes {idx}/{len(df)}...")

            class_id = int(row['class']) if not pd.isna(row['class']) else 0
            color = yolo_colors[class_id % len(yolo_colors)]

            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            # Draw rectangle with YOLO's line width (2 pixels)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Add label (YOLO style)
            if show_labels or show_conf:
                label_parts = []
                if show_labels and pd.notna(row.get('name')):
                    label_parts.append(str(row['name']))
                if show_conf and pd.notna(row.get('confidence')):
                    label_parts.append(f"{row['confidence']:.2f}")

                if label_parts:
                    label = ' '.join(label_parts)

                    # YOLO uses specific font settings
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5  # YOLO default
                    thickness = 1

                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )

                    # Draw background rectangle (YOLO style - filled rectangle above box)
                    cv2.rectangle(overlay,
                                  (x1, y1 - text_height - 4),  # 4 pixel padding
                                  (x1 + text_width + 4, y1),
                                  color, -1)

                    # Draw text in white (YOLO default)
                    cv2.putText(overlay, label,
                                (x1 + 2, y1 - 4),  # 2 pixel padding from left, 4 from bottom
                                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    print(f"Successfully drew all {len(df)} detections")
    return overlay


class NemaCounterDetection:

    def __init__(self, weights_path, conf_thresh=0.5, iou_thresh=0.3, device='cpu', use_retina_masks=False):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.custom_model = YOLO(weights_path)
        self.custom_model.to(self.device)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model_task = self.custom_model.task  # 'detect' or 'segment'
        self.class_names = self.custom_model.names
        self.use_retina_masks = use_retina_masks

    def detect_objects(self, img, img_path):
        """
        Detect objects and return raw results for further processing.
        Keeps retina masks when selected, with better memory management.
        """
        img_height, img_width = img.shape[:2]
        input_size = self.custom_model.__dict__.get("overrides", {}).get("imgsz", 640)

        half = 'cuda' in self.device

        # Clear memory before processing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            # Try to run detection with requested settings
            results = self.custom_model.predict(
                source=img,
                imgsz=input_size,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                device=self.device,
                verbose=False,
                retina_masks=self.use_retina_masks,
                half=half,
                max_det=10000
            )

            if self.use_retina_masks and results and results[0].boxes:
                num_detections = len(results[0].boxes)
                if num_detections > 100:
                    print(f"  Processing {num_detections} objects with retina masks (high memory usage expected)...")

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "not enough memory" in str(e) or "out of memory" in str(e).lower():
                print(f"  Memory error with retina masks. Attempting with standard masks...")

                # Clear memory and try again with standard masks
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                results = self.custom_model.predict(
                    source=img,
                    imgsz=input_size,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    device=self.device,
                    verbose=False,
                    retina_masks=False,  # Fall back to standard masks only on error
                    half=half,
                    max_det=10000
                )
                print(f"  Successfully processed with standard masks.")
            else:
                raise  # Re-raise if it's not a memory error

        return results


def merge_highly_overlapping_masks(detections, height, width, iou_thresh=0.05):
    """
    STAGE 1: Greedily fuses masks that have an IoU above the threshold.
    Only merges masks of the SAME CLASS.
    """
    if len(detections) < 2:
        return detections

    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    is_merged = [False] * len(detections)
    merged_detections = []

    for i in range(len(detections)):
        if is_merged[i]:
            continue

        current_det = detections[i]
        current_mask = np.zeros((height, width), dtype=np.uint8)

        # Draw current detection's mask
        contours_json = current_det.get('contours', '[]')
        if contours_json and contours_json != '[]':
            contours = json.loads(contours_json)
            for c in contours:
                if len(c) >= 3:
                    cv2.fillPoly(current_mask, [np.array(c, dtype=np.int32)], 1)

        # Try to merge with other detections of the same class
        for j in range(i + 1, len(detections)):
            if is_merged[j]:
                continue

            other_det = detections[j]

            # Only merge if same class
            if other_det['class'] != current_det['class']:
                continue

            other_mask = np.zeros((height, width), dtype=np.uint8)
            other_contours_json = other_det.get('contours', '[]')
            if other_contours_json and other_contours_json != '[]':
                other_contours = json.loads(other_contours_json)
                for c in other_contours:
                    if len(c) >= 3:
                        cv2.fillPoly(other_mask, [np.array(c, dtype=np.int32)], 1)

            # Calculate IoU
            intersection = np.sum(cv2.bitwise_and(current_mask, other_mask))
            union = np.sum(cv2.bitwise_or(current_mask, other_mask))
            iou = intersection / union if union > 0 else 0

            # Merge if IoU exceeds threshold
            if iou > iou_thresh:
                current_mask = cv2.bitwise_or(current_mask, other_mask)
                is_merged[j] = True

        # Create final detection(s) from merged mask
        # The merged mask might have multiple components, so we split them
        final_contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in final_contours:
            if cnt.shape[0] < 3:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            merged_detections.append({
                'xmin': x,
                'ymin': y,
                'xmax': x + w,
                'ymax': y + h,
                'confidence': current_det['confidence'],
                'class': current_det['class'],
                'name': current_det['name'],
                'area': area,
                'contours': json.dumps([cnt.reshape(-1, 2).tolist()]),
                'object_type': 'mask'
            })

    return merged_detections


def cleanup_contained_masks(detections, height, width, ioa_thresh=0.95):
    """
    STAGE 2: Removes smaller masks that are almost entirely contained within larger masks.
    Only removes if both masks are of the SAME CLASS.
    """
    if len(detections) < 2:
        return detections

    # Sort by area (largest first)
    detections.sort(key=lambda x: x['area'], reverse=True)
    keep = [True] * len(detections)

    for i in range(len(detections)):
        if not keep[i]:
            continue

        mask_i = np.zeros((height, width), dtype=np.uint8)
        contours_i_json = detections[i].get('contours', '[]')
        if contours_i_json and contours_i_json != '[]':
            contours_i = json.loads(contours_i_json)
            for c in contours_i:
                if len(c) >= 3:
                    cv2.fillPoly(mask_i, [np.array(c, dtype=np.int32)], 1)

        for j in range(i + 1, len(detections)):
            if not keep[j]:
                continue

            # Only remove if same class
            if detections[j]['class'] != detections[i]['class']:
                continue

            mask_j = np.zeros((height, width), dtype=np.uint8)
            contours_j_json = detections[j].get('contours', '[]')
            if contours_j_json and contours_j_json != '[]':
                contours_j = json.loads(contours_j_json)
                for c in contours_j:
                    if len(c) >= 3:
                        cv2.fillPoly(mask_j, [np.array(c, dtype=np.int32)], 1)

            # Calculate intersection over area of j (smaller mask)
            intersection = np.sum(cv2.bitwise_and(mask_i, mask_j))
            area_j = detections[j]['area']
            if area_j == 0:
                continue

            ioa = intersection / area_j

            # Remove smaller mask if it's mostly contained in larger mask
            if ioa > ioa_thresh:
                keep[j] = False

    return [det for i, det in enumerate(detections) if keep[i]]


def process_and_split_masks(results, class_names, img_height, img_width,
                            use_fusion=False, iou_thresh=0.05, ioa_thresh=0.95):
    """
    Processes segmentation results with optional two-stage fusion.
    Now with aggressive batch processing for better memory management.

    Args:
        results: YOLO detection results
        class_names: Dictionary of class names
        img_height: Image height
        img_width: Image width
        use_fusion: If True, applies two-stage fusion process. If False, just splits masks.
        iou_thresh: IoU threshold for merging overlapping masks (Stage 1)
        ioa_thresh: IoA threshold for removing contained masks (Stage 2)
    """
    if not results or not results[0].boxes:
        return pd.DataFrame()

    all_data = []
    num_detections = len(results[0].boxes)

    # Determine batch size based on number of detections
    # More aggressive batching for better memory management
    if num_detections > 500:
        batch_size = 10  # Very small batches for many objects
    elif num_detections > 200:
        batch_size = 20  # Small batches
    elif num_detections > 100:
        batch_size = 30  # Medium batches
    else:
        batch_size = num_detections  # Process all at once for few objects

    print(f"  Processing {num_detections} detections in batches of {batch_size}...")

    # Process masks in batches to avoid memory issues
    for batch_start in range(0, num_detections, batch_size):
        batch_end = min(batch_start + batch_size, num_detections)

        if batch_start > 0 and batch_start % 50 == 0:
            print(f"    Processed {batch_start}/{num_detections} detections...")
            # Clear memory more frequently
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Process current batch
        for i in range(batch_start, batch_end):
            # Get detection info
            class_id = int(results[0].boxes.cls[i].item())
            class_name = class_names.get(class_id, "unknown")
            score = results[0].boxes.conf[i].item()

            # Process mask if available
            if results[0].masks is None:
                continue

            try:
                # Extract mask data with immediate memory cleanup
                mask_tensor = results[0].masks.data[i].cpu()
                mask = mask_tensor.numpy().squeeze()
                del mask_tensor

                # Force garbage collection for very large datasets
                if num_detections > 300 and i % 10 == 0:
                    gc.collect()

                # Resize mask if needed
                if mask.shape != (img_height, img_width):
                    mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

                # Convert to binary
                mask = (mask > 0.5).astype(np.uint8)

                # Find all contours (separate parts)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Immediately delete the mask to free memory
                del mask

                if not contours:
                    continue

                # Create a detection for EACH contour (splitting multi-part masks)
                for cnt in contours:
                    if cnt.shape[0] < 3:  # Must be a valid polygon
                        continue

                    x, y, w, h = cv2.boundingRect(cnt)
                    area = cv2.contourArea(cnt)

                    # Contour format for JSON: list containing one contour
                    contour_list_for_json = [cnt.reshape(-1, 2).tolist()]

                    all_data.append({
                        'xmin': x,
                        'ymin': y,
                        'xmax': x + w,
                        'ymax': y + h,
                        'confidence': score,
                        'class': class_id,
                        'name': class_name,
                        'area': area,
                        'contours': json.dumps(contour_list_for_json),
                        'object_type': 'mask'
                    })

            except Exception as e:
                print(f"    Warning: Failed to process mask {i}: {e}")
                # Try to recover memory even on error
                gc.collect()
                continue

    # Final memory cleanup after processing all masks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not all_data:
        return pd.DataFrame()

    print(f"  Initial split resulted in {len(all_data)} objects")

    # Apply fusion if enabled
    if use_fusion and len(all_data) > 1:
        print(f"  Applying two-stage fusion (IoU={iou_thresh:.2f}, IoA={ioa_thresh:.2f})...")

        # Group objects by class for per-class processing
        objects_by_class = defaultdict(list)
        for obj in all_data:
            objects_by_class[obj['class']].append(obj)

        all_final_objects = []

        # Process each class separately
        for class_id, class_objects in objects_by_class.items():
            class_name = class_names.get(class_id, "unknown")
            print(f"    Processing class '{class_name}': {len(class_objects)} objects")

            if len(class_objects) > 1:
                # Stage 1: Merge highly overlapping masks
                merged_objects = merge_highly_overlapping_masks(
                    class_objects, img_height, img_width, iou_thresh
                )
                print(f"      After merge stage: {len(merged_objects)} objects")

                # Stage 2: Remove contained masks
                final_class_objects = cleanup_contained_masks(
                    merged_objects, img_height, img_width, ioa_thresh
                )
                print(f"      After cleanup stage: {len(final_class_objects)} objects")

                all_final_objects.extend(final_class_objects)
            else:
                # Only one object in this class, no fusion needed
                all_final_objects.extend(class_objects)

        print(f"  Final count after fusion: {len(all_final_objects)} objects")
        all_data = all_final_objects

    gc.collect()
    return pd.DataFrame(all_data)


def detection_workflow(dct_args, gui=True):
    """
    Main detection workflow with optional two-stage fusion.
    """
    # --- Get Parameters ---
    gpu_if_avail = utils.get_bool(dct_args['gpu'])
    add_overlay = utils.get_bool(dct_args['add_overlay'])
    show_bbox = utils.get_bool(dct_args.get('show_bbox', 1))
    show_conf = utils.get_bool(dct_args.get('show_conf', 0))
    show_mask = utils.get_bool(dct_args.get('show_mask', 1))
    show_labels = utils.get_bool(dct_args.get('show_labels', 1))
    use_retina_masks = utils.get_bool(dct_args.get('use_retina_masks', 0))

    # Fusion parameters
    use_fusion = utils.get_bool(dct_args.get('use_fusion', 0))
    fuse_iou_thresh = float(dct_args.get('fuse_iou_thresh', 0.05))
    phagocyte_ioa_thresh = float(dct_args.get('phagocyte_ioa_thresh', 0.95))

    # --- Setup ---
    utils.set_cpu_usage(dct_args.get('cpu', os.cpu_count()))
    device = 'cuda:0' if torch.cuda.is_available() and gpu_if_avail else 'cpu'

    lst_img_paths = utils.list_image_files(dct_args['input_directory'])

    detection_model = NemaCounterDetection(
        dct_args['model_path'],
        conf_thresh=dct_args['conf_thresh'],
        iou_thresh=dct_args['overlap_thresh'],
        device=device,
        use_retina_masks=use_retina_masks
    )
    model_task = detection_model.model_task

    create_project_dirs_structure(
        dct_args['output_directory'],
        dct_args['project_id'],
        display_overlay=add_overlay,
        model_task=model_task
    )

    # Display mode information
    if model_task == 'segment':
        if use_fusion:
            print(f"\n=== FUSION MODE ENABLED ===")
            print(f"  Merge IoU threshold: {fuse_iou_thresh:.2f}")
            print(f"  Phagocyte IoA threshold: {phagocyte_ioa_thresh:.2f}")
        else:
            print(f"\n=== SIMPLE SPLITTING MODE (No Fusion) ===")

    lst_df_final = []

    for img_path in lst_img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        print(f"\nProcessing image: {os.path.basename(img_path)}")

        # Detect objects
        results = detection_model.detect_objects(img, img_path)
        df_processed = pd.DataFrame()

        # Process results based on model type
        if model_task == 'segment':
            h, w = img.shape[:2]
            # Process masks with optional fusion
            df_processed = process_and_split_masks(
                results,
                detection_model.class_names,
                h, w,
                use_fusion=use_fusion,
                iou_thresh=fuse_iou_thresh,
                ioa_thresh=phagocyte_ioa_thresh
            )

        elif model_task == 'detect' and results and results[0].boxes:
            # Simple detection (no segmentation masks)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            df_processed = pd.DataFrame({
                'xmin': boxes[:, 0].astype(int),
                'ymin': boxes[:, 1].astype(int),
                'xmax': boxes[:, 2].astype(int),
                'ymax': boxes[:, 3].astype(int),
                'confidence': scores,
                'class': classes,
                'name': [detection_model.class_names.get(c, "unknown") for c in classes],
                'area': np.nan,
                'contours': '[]',
                'object_type': 'box'
            })

        # Add image path (always, even for empty detections)
        img_path_rel = os.path.relpath(img_path, dct_args['input_directory'])

        if df_processed.empty:
            # Create a placeholder row for images with 0 detections
            # This ensures the image appears in globinfo and can be opened in manual edit mode
            print(f"  No objects detected - creating placeholder entry for manual editing")
            df_processed = pd.DataFrame([{
                'xmin': np.nan,
                'ymin': np.nan,
                'xmax': np.nan,
                'ymax': np.nan,
                'confidence': np.nan,
                'class': np.nan,
                'name': '',
                'area': np.nan,
                'contours': '',
                'object_type': ''
            }])

        df_processed['img_id'] = img_path_rel

        # Create overlay if requested (only for actual detections, not placeholders)
        has_real_detections = not df_processed['object_type'].isna().all() and (df_processed['object_type'] != '').any()
        if add_overlay and has_real_detections:
            try:
                print(f"  Creating overlay for {len(df_processed)} final objects...")

                if model_task == 'detect':
                    # For simple detection, YOLO's plot is fine
                    annotated_img = results[0].plot(
                        conf=show_conf,
                        labels=show_labels,
                        boxes=show_bbox,
                        line_width=2,
                        font_size=10
                    )
                else:
                    # For segmentation, use our custom drawer with the final dataframe
                    annotated_img = draw_detections_yolo_style(
                        img, df_processed,
                        show_bbox=show_bbox,
                        show_labels=show_labels,
                        show_conf=show_conf,
                        show_mask=show_mask
                    )

                # Save overlay
                output_subdir = 'img/bounding_boxes' if model_task == 'detect' else 'img/masks'
                fpath_out_img = os.path.join(
                    dct_args['output_directory'],
                    dct_args['project_id'],
                    output_subdir,
                    f"{dct_args['project_id']}_{os.path.basename(img_path)}"
                )
                os.makedirs(os.path.dirname(fpath_out_img), exist_ok=True)

                quality = 75 if len(df_processed) > 200 else 85
                cv2.imwrite(fpath_out_img, annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                print(f"  Saved overlay.")

                del annotated_img

            except Exception as e:
                print(f"  Warning: Could not create/save overlay for {img_path}: {e}")

        lst_df_final.append(df_processed)

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Consolidate and Save Final CSVs ---
    dpath_stats = os.path.join(dct_args['output_directory'], dct_args['project_id'])

    if not lst_df_final:
        df_global = pd.DataFrame()
    else:
        df_global = pd.concat(lst_df_final, ignore_index=True)

    # Add object_id after final concatenation
    # Placeholder rows (no detections) get object_id = 0
    # Real detections get object_id = 1, 2, 3, etc.
    if not df_global.empty:
        # Identify placeholder rows (object_type is empty or NaN)
        is_placeholder = df_global['object_type'].isna() | (df_global['object_type'] == '')

        # Assign object_id = 0 for placeholders
        df_global.loc[is_placeholder, 'object_id'] = 0

        # For real detections, assign sequential object_ids per image (1, 2, 3, ...)
        real_detections = ~is_placeholder
        if real_detections.any():
            df_global.loc[real_detections, 'object_id'] = (
                    df_global.loc[real_detections].groupby('img_id').cumcount() + 1
            )

    df_global['project_id'] = dct_args['project_id']

    # Ensure all expected columns are present
    expected_columns = [
        'img_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax',
        'confidence', 'class', 'name', 'area', 'contours',
        'object_type', 'project_id'
    ]

    for col in expected_columns:
        if col not in df_global.columns:
            if col in ['xmin', 'ymin', 'xmax', 'ymax', 'object_id']:
                df_global[col] = 0
            elif col in ['confidence']:
                df_global[col] = 1.0
            elif col in ['area']:
                df_global[col] = np.nan
            elif col in ['class']:
                df_global[col] = 0
            elif col in ['name']:
                df_global[col] = 'unknown'
            elif col in ['contours']:
                df_global[col] = '[]'
            elif col in ['object_type']:
                df_global[col] = 'box'
            elif col in ['project_id']:
                df_global[col] = dct_args['project_id']
            else:
                df_global[col] = pd.NA

    df_global = df_global[expected_columns]

    # Save globinfo CSV with metadata
    fpath_globinfo = os.path.join(dpath_stats, f"{dct_args['project_id']}_globinfo.csv")
    with open(fpath_globinfo, 'w', newline='', encoding='utf-8') as f:
        abs_input_path = os.path.abspath(dct_args['input_directory'])
        f.write(f"# input_directory: {abs_input_path}\n")
        df_global.to_csv(f, index=False, quoting=csv.QUOTE_ALL)

    # Save summary if data exists
    if not df_global.empty:
        df_summary = common.create_summary_table(df_global, dct_args['project_id'])
        df_summary.to_csv(
            os.path.join(dpath_stats, f"{dct_args['project_id']}_summary.csv"),
            index=False,
            quoting=csv.QUOTE_ALL
        )

    # Final report
    total_objects = len(df_global)
    total_images = len(lst_img_paths)

    if use_fusion:
        print(f"\nDetection complete! (Fusion Mode)")
    else:
        print(f"\nDetection complete! (Simple Split Mode)")

    print(f"Processed {total_objects} total objects across {total_images} images.")

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return None


def create_project_dirs_structure(dpath_outdir, project_id, display_overlay=False, model_task='detect'):
    """Create project directory structure."""
    dpath_project = os.path.join(dpath_outdir, project_id)

    if os.path.isdir(dpath_project):
        print(f"Warning: Output directory '{dpath_project}' already exists. Files may be overwritten.")
    else:
        os.makedirs(dpath_project, mode=0o755)

    if display_overlay:
        if model_task == 'detect':
            os.makedirs(os.path.join(dpath_project, 'img', 'bounding_boxes'), mode=0o755, exist_ok=True)
        elif model_task == 'segment':
            os.makedirs(os.path.join(dpath_project, 'img', 'masks'), mode=0o755, exist_ok=True)