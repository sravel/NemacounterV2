# --- START OF FILE segmentation.py ---
import torch
import cv2
import os
import numpy as np
import pandas as pd
import json
import csv  # Added for consistent CSV writing
import sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import nemacounter.utils as utils
import nemacounter.common as common


class NemaCounterSegmentation:

    def __init__(self, device='cpu'):
        print(f"Initializing NemaCounterSegmentation with device: {device}")
        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large", device=device)
        self.device = device

    def objects_segmentation(self, image, annotations, batch_size=1):
        img = image.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img
        img_height, img_width = img_rgb.shape[:2]
        print(f"--- Setting image in predictor for shape: {img_rgb.shape} ---")
        self.predictor.set_image(img_rgb)
        masks_list = []
        num_annotations = len(annotations)
        print(f"--- objects_segmentation received {num_annotations} annotations ---")

        with torch.inference_mode():
            if self.device.type == 'cuda':
                try:
                    autocast_context = torch.autocast(device_type='cuda', dtype=torch.float16)
                except Exception:
                    print("Autocast unavailable, using no_grad."); autocast_context = torch.no_grad()
            else:
                autocast_context = torch.no_grad()

            with autocast_context:
                for idx, ann in enumerate(annotations):
                    mask = None
                    cleaned_mask = None
                    object_type = ann.get('object_type', 'unknown').lower()
                    img_id_info = ann.get('img_id', 'N/A')
                    print(
                        f"--- Processing ann {idx + 1}/{num_annotations}, type: {object_type}, img: {img_id_info} ---")

                    try:
                        if object_type == 'box':
                            x_min, y_min, x_max, y_max = ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']
                            box = np.array([x_min, y_min, x_max, y_max])
                            pred_masks, _, _ = self.predictor.predict(box=box[None, :], multimask_output=False)
                            if isinstance(pred_masks, torch.Tensor):
                                mask = pred_masks[0].cpu().numpy().astype(np.uint8)
                            elif isinstance(pred_masks, np.ndarray):
                                mask = pred_masks[0].astype(np.uint8)
                            else:
                                print(f"Warn: Unexpected mask type box: {type(pred_masks)}"); continue
                            cleaned_mask = mask
                            ann['mask'] = cleaned_mask
                            ann['object_type'] = 'mask'
                            print(f"--- Box processed successfully for ann {idx + 1} ---")

                        elif object_type == 'polygon':
                            print(f"--- Entering Polygon processing for ann {idx + 1} ---")
                            contours_json = ann.get('contours', '')
                            if pd.isna(contours_json) or not contours_json:
                                print(f"Warn: Skip polygon - missing/empty contours: {img_id_info}");
                                continue
                            try:
                                contours = json.loads(contours_json)
                                if not isinstance(contours, list) or len(contours) < 3:
                                    print(f"Warn: Skip polygon - invalid contour data: {img_id_info}");
                                    continue
                                polygon_points = np.array(contours, dtype=np.int32)
                                if polygon_points.ndim != 2 or polygon_points.shape[1] != 2:
                                    print(f"Warn: Skip polygon - invalid shape {polygon_points.shape}: {img_id_info}");
                                    continue
                                print(f"--- Polygon points shape: {polygon_points.shape} ---")

                                # Create Input Mask
                                input_mask_orig_np = np.zeros((img_height, img_width), dtype=np.uint8)
                                cv2.fillPoly(input_mask_orig_np, [polygon_points.reshape(-1, 1, 2)], 1)

                                # Resize Mask for SAM input
                                target_mask_size = (256, 256)
                                input_mask_resized_np = cv2.resize(input_mask_orig_np, target_mask_size,
                                                                   interpolation=cv2.INTER_NEAREST).astype(np.float32)
                                print(
                                    f"--- Input mask created (orig sum: {np.sum(input_mask_orig_np)}), resized to {target_mask_size} (resized sum: {np.sum(input_mask_resized_np)}) ---")
                                if np.sum(input_mask_resized_np) == 0:
                                    print(f"Warn: Resized input mask empty for polygon ann {idx + 1}. Skipping.");
                                    continue

                                # Convert to Tensor
                                if self.device.type == 'cuda':
                                    input_mask_tensor = torch.from_numpy(input_mask_resized_np).to(
                                        self.device).unsqueeze(0).unsqueeze(0)
                                else:
                                    input_mask_tensor = torch.from_numpy(input_mask_resized_np).unsqueeze(0).unsqueeze(
                                        0)
                                print(f"--- Input mask tensor created, shape: {input_mask_tensor.shape} ---")

                                # Predict using Input Mask
                                print(f"--- Calling predictor.predict with RESIZED mask_input for ann {idx + 1} ---")
                                pred_masks, scores, logits = self.predictor.predict(mask_input=input_mask_tensor,
                                                                                    multimask_output=False)
                                print(f"--- Predict returned: type={type(pred_masks)}, score={scores} ---")

                                # Process Output Mask
                                if pred_masks is None or (
                                        isinstance(pred_masks, (np.ndarray, torch.Tensor)) and pred_masks.size == 0):
                                    print(f"Warn: Predict returned None/empty mask polygon ann {idx + 1}.");
                                    continue
                                elif isinstance(pred_masks, torch.Tensor):
                                    print(f"   Output Tensor shape={pred_masks.shape}");
                                    mask = pred_masks[0].cpu().numpy().astype(np.uint8)
                                elif isinstance(pred_masks, np.ndarray):
                                    print(f"   Output Numpy shape={pred_masks.shape}");
                                    mask = pred_masks[0].astype(np.uint8)
                                else:
                                    print(f"Warn: Unexpected mask type polygon: {type(pred_masks)}");
                                    continue

                                # Ensure output mask matches original image size
                                if mask.shape[0] != img_height or mask.shape[1] != img_width:
                                    print(
                                        f"Warn: Output mask shape {mask.shape} doesn't match image {img_height, img_width}. Resizing output.")
                                    mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

                                print(f"--- Raw mask received from SAM, shape: {mask.shape}, sum: {np.sum(mask)} ---")

                                # POST-PROCESSING: Keep only the largest connected component
                                if np.sum(mask) > 0:
                                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask,
                                                                                                            connectivity=8,
                                                                                                            ltype=cv2.CV_32S)

                                    if num_labels > 1:
                                        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                                        print(
                                            f"--- Found {num_labels - 1} components. Largest label: {largest_label}, Area: {stats[largest_label, cv2.CC_STAT_AREA]} ---")
                                        cleaned_mask = np.zeros_like(mask)
                                        cleaned_mask[labels == largest_label] = 1
                                    else:
                                        print(
                                            f"--- No components found in SAM mask for ann {idx + 1}. Using raw mask. ---")
                                        cleaned_mask = mask
                                else:
                                    print(f"--- SAM mask is empty for ann {idx + 1}. No cleaning needed. ---")
                                    cleaned_mask = mask

                                print(
                                    f"--- Cleaned mask generated, shape: {cleaned_mask.shape}, sum: {np.sum(cleaned_mask)} ---")
                                ann['mask'] = cleaned_mask
                                ann['object_type'] = 'mask'
                                print(f"--- Polygon processed successfully (with cleaning) for ann {idx + 1} ---")

                            except json.JSONDecodeError:
                                print(f"Warn: Skip polygon - invalid JSON: {img_id_info}");
                                continue
                            except Exception as e:
                                print(f"Error processing polygon (mask input/cleaning): {img_id_info}. Error: {e}");
                                continue

                        elif object_type == 'mask':
                            # Handle Existing Mask
                            if 'mask' not in ann or ann['mask'] is None:
                                contours_json = ann.get('contours', '')
                                if pd.isna(contours_json) or not contours_json:
                                    continue
                                try:
                                    contours = json.loads(contours_json)
                                    assert contours
                                    contours_np_list = []
                                    if isinstance(contours, list) and contours:
                                        if isinstance(contours[0], list) and contours[0]:
                                            if isinstance(contours[0][0], (int, float)):
                                                cont = np.array(contours, dtype=np.int32)
                                                if cont.ndim == 2 and cont.shape[1] == 2 and cont.shape[0] >= 3:
                                                    contours_np_list.append(cont.reshape(-1, 1, 2))
                                            elif isinstance(contours[0][0], list):
                                                for c in contours:
                                                    cont = np.array(c, dtype=np.int32)
                                                    if cont.ndim == 2 and cont.shape[1] == 2 and cont.shape[0] >= 3:
                                                        contours_np_list.append(cont.reshape(-1, 1, 2))
                                    if not contours_np_list:
                                        continue
                                    mask_reconstructed = np.zeros((img_height, img_width), dtype=np.uint8)
                                    cv2.fillPoly(mask_reconstructed, contours_np_list, 1)
                                    cleaned_mask = mask_reconstructed
                                    ann['mask'] = cleaned_mask
                                except Exception as e:
                                    print(f"Error reconstructing mask: {e}");
                                    continue
                            else:
                                mask_existing = ann['mask']
                                if isinstance(mask_existing, np.ndarray):
                                    if mask_existing.shape[0] != img_height or mask_existing.shape[1] != img_width:
                                        cleaned_mask = cv2.resize(mask_existing, (img_width, img_height),
                                                                  interpolation=cv2.INTER_NEAREST)
                                    else:
                                        cleaned_mask = mask_existing
                                    ann['mask'] = cleaned_mask
                                else:
                                    continue
                            ann['object_type'] = 'mask'

                        else:
                            print(f"Unknown type: {object_type}");
                            continue

                        # Append valid CLEANED mask
                        if cleaned_mask is not None and np.sum(cleaned_mask) > 0:
                            print(f"--- Appending mask derived from ann {idx + 1} (type: {object_type}) ---")
                            masks_list.append(cleaned_mask.astype(np.uint8))
                            ann['_processed_mask_added'] = True
                        else:
                            print(
                                f"--- Mask is None or empty for ann {idx + 1} AFTER processing/cleaning, not appending ---")
                            ann['_processed_mask_added'] = False

                    except KeyError as e:
                        print(f"Error key: {e}");
                        ann['_processed_mask_added'] = False;
                        continue
                    except Exception as e:
                        print(f"Unexpected error processing ann {idx + 1}: {e}");
                        ann['_processed_mask_added'] = False;
                        continue

        # Filter annotations based on the flag
        processed_annotations = [ann for ann in annotations if ann.get('_processed_mask_added', False)]
        for ann in processed_annotations:
            if '_processed_mask_added' in ann:
                del ann['_processed_mask_added']

        print(
            f"--- objects_segmentation returning {len(masks_list)} masks and {len(processed_annotations)} annotations ---")

        if not masks_list:
            return np.array([]), []
        try:
            masks_array = np.array(masks_list)
        except ValueError as e:
            print(f"Error stacking masks: {e}");
            return np.array([]), []
        if masks_array.shape[0] != len(processed_annotations):
            print(f"Error: Mismatch masks/annots count ({masks_array.shape[0]} vs {len(processed_annotations)})")
            min_len = min(masks_array.shape[0], len(processed_annotations))
            return masks_array[:min_len], processed_annotations[:min_len]
        return masks_array, processed_annotations


def add_masks_on_image(masks, img):
    if masks is None or len(masks) == 0:
        return
    img_overlay = img.copy()
    img_shape = img_overlay.shape[:2]
    combined_mask = np.zeros(img_shape, dtype=np.uint8)
    print(f"--- add_masks_on_image received {len(masks)} masks ---")
    for i, mask in enumerate(masks):
        if mask is None or not isinstance(mask, np.ndarray) or mask.ndim != 2:
            print(f"--- Skipping invalid mask {i} in drawing ---")
            continue
        binary_mask = (mask > 0).astype(np.uint8)
        print(f"--- Drawing mask {i}, shape: {binary_mask.shape}, sum: {np.sum(binary_mask)} ---")
        if binary_mask.shape != img_shape:
            print(f"--- Resizing mask {i} for drawing ---")
            binary_mask = cv2.resize(binary_mask, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
        combined_mask = np.logical_or(combined_mask, binary_mask).astype(np.uint8)
    img_overlay[combined_mask == 1] = [0, 0, 255]
    img[:] = img_overlay
    print(f"--- add_masks_on_image completed ---")


def create_multicolored_masks_image(masks):
    if masks is None or not isinstance(masks, np.ndarray) or masks.ndim != 3 or masks.shape[0] == 0:
        print("Warning: Cannot create multicolored mask image.")
        height, width = 512, 512
        return np.zeros((height, width, 3), dtype=np.uint8)
    num_masks, height, width = masks.shape
    black_image = np.zeros((height, width, 3), dtype=np.uint8)
    print(f"--- create_multicolored_masks_image received {num_masks} masks ---")
    for i in range(num_masks):
        color = np.random.randint(100, 256, size=3, dtype=np.uint8)
        mask_layer = masks[i, :, :]
        assert mask_layer.ndim == 2
        binary_mask_layer = (mask_layer > 0).astype(np.uint8)
        if binary_mask_layer.shape != (height, width):
            binary_mask_layer = cv2.resize(binary_mask_layer, (width, height), interpolation=cv2.INTER_NEAREST)
        for c in range(3):
            black_image[:, :, c][binary_mask_layer == 1] = color[c]
    print(f"--- create_multicolored_masks_image completed ---")
    return black_image


def segmentation_workflow(dct_args):
    """
    Main segmentation workflow that reads detection output and performs segmentation.
    FIXED: Now properly handles CSV files with comment lines and outputs correct format.
    """
    # Set up project ID and input directory
    dct_args['project_id'] = os.path.basename(dct_args['input_file']).replace('_globinfo.csv', '')
    dct_args['input_dir'] = os.path.dirname(dct_args['input_file'])

    # Setup device and parameters
    gpu_if_avail = utils.get_bool(dct_args['gpu'])
    add_overlay = utils.get_bool(dct_args['add_overlay'])
    utils.set_cpu_usage(dct_args['cpu'])

    try:
        if torch.cuda.is_available() and gpu_if_avail:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        print(f"Using device: {device}")
    except Exception as e:
        print(f"Torch device check error: {e}. Defaulting CPU.")
        device = torch.device('cpu')

    # Create overlay directory if needed
    if add_overlay:
        dpath_overlay = os.path.join(dct_args['input_dir'], dct_args['project_id'], 'img', 'segmentation')
        os.makedirs(dpath_overlay, exist_ok=True)

    # Read metadata from CSV if present
    input_directory = None
    try:
        with open(dct_args['input_file'], 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('# input_directory:'):
                input_directory = first_line.split(':', 1)[1].strip()
    except:
        pass

    if utils.check_file_existence(dct_args['input_file']):
        try:
            # FIXED: Read CSV with comment='#' to skip metadata lines
            df = pd.read_csv(dct_args['input_file'], comment='#')
        except Exception as e:
            print(f"Error reading CSV '{dct_args['input_file']}': {e}")
            return

        # Check if required columns exist
        if 'img_id' not in df.columns:
            print(f"Error: 'img_id' column not found in {dct_args['input_file']}")
            print(f"Available columns: {list(df.columns)}")
            return

        lst_img_paths = df['img_id'].unique()

        try:
            segmentation_model = NemaCounterSegmentation(device=device)
        except Exception as e:
            print(f"Error initializing segmentation model: {e}")
            return

        all_processed_annotations_out = []
        total_images = len(lst_img_paths)

        for idx, img_path_rel in enumerate(lst_img_paths):
            print(f"\n======= Processing image {idx + 1}/{total_images}: {img_path_rel} =======")

            # Try to find image using multiple strategies
            img_path_full = None
            possible_paths = []

            # If we have input_directory from metadata, try there first
            if input_directory:
                possible_paths.append(os.path.join(input_directory, img_path_rel))

            # Try relative to CSV location
            possible_paths.append(os.path.join(dct_args['input_dir'], img_path_rel))

            # Try absolute path
            possible_paths.append(img_path_rel)

            # Find the first existing path
            for path in possible_paths:
                if os.path.exists(path):
                    img_path_full = path
                    break

            if not img_path_full:
                print(f"Skip: Image not found: '{img_path_rel}'.")
                continue

            try:
                img_bgr = cv2.imread(img_path_full)
                assert img_bgr is not None
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Skip: Error reading/converting image '{img_path_full}': {e}.")
                continue

            # Prepare annotations for this image
            img_df = df[df['img_id'] == img_path_rel].copy()

            # Ensure coordinates are integers
            try:
                for col in ['xmin', 'ymin', 'xmax', 'ymax']:
                    if col in img_df.columns:
                        img_df[col] = pd.to_numeric(img_df[col], errors='coerce').fillna(0).astype(int)
            except Exception as e:
                print(f"Warn: Coord processing issue {img_path_rel}: {e}")

            current_image_annotations_in = [
                row.to_dict() for _, row in img_df.iterrows()
                if 'object_type' in row and pd.notna(row['object_type'])
            ]

            if not current_image_annotations_in:
                print(f"Skip: No annotations for {img_path_rel}.")
                continue

            for ann in current_image_annotations_in:
                ann['object_type'] = str(ann['object_type']).lower()

            # Run segmentation
            try:
                masks, processed_annotations = segmentation_model.objects_segmentation(img_rgb,
                                                                                       current_image_annotations_in)
            except Exception as e:
                print(f"Skip: Segmentation error '{img_path_rel}': {e}")
                continue

            print(
                f"--- segmentation_workflow received {len(masks)} masks and {len(processed_annotations)} annotations for {img_path_rel} ---")

            if masks is not None and len(masks) > 0:
                print(f"--- Masks array shape: {masks.shape} ---")

            # Post-process results
            if masks is not None and len(masks) > 0 and len(processed_annotations) == len(masks):
                for i, ann in enumerate(processed_annotations):
                    mask = masks[i]
                    ann['area'] = float(np.sum(mask))

                    # Extract contours
                    contours_list, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid_contours_data, all_points_list = [], []

                    if contours_list:
                        for c in contours_list:
                            if c.shape[0] >= 3:
                                sq_c = c.squeeze()
                                if sq_c.ndim == 1 and sq_c.shape[0] == 2:
                                    all_points_list.append(sq_c.reshape(1, 2))
                                    valid_contours_data.append(sq_c.tolist())
                                elif sq_c.ndim == 2:
                                    all_points_list.append(sq_c)
                                    valid_contours_data.append(sq_c.tolist())

                        if all_points_list:
                            ann['contours'] = json.dumps(valid_contours_data)
                            all_pts = np.vstack(all_points_list)
                            ann['xmin'] = int(np.min(all_pts[:, 0]))
                            ann['ymin'] = int(np.min(all_pts[:, 1]))
                            ann['xmax'] = int(np.max(all_pts[:, 0]))
                            ann['ymax'] = int(np.max(all_pts[:, 1]))
                        else:
                            ann['contours'] = json.dumps([])
                            ann['xmin'] = ann['ymin'] = ann['xmax'] = ann['ymax'] = 0
                    else:
                        ann['contours'] = json.dumps([])
                        ann['xmin'] = ann['ymin'] = ann['xmax'] = ann['ymax'] = 0

                    ann['object_type'] = 'mask'
                    all_processed_annotations_out.append(ann)

                # Generate overlays if requested
                if add_overlay:
                    print(f"--- Generating overlay for {img_path_rel} with {len(masks)} masks ---")
                    try:
                        overlay_img_bgr = img_bgr.copy()
                        add_masks_on_image(masks, overlay_img_bgr)
                        fname = os.path.basename(img_path_rel)
                        fstem = os.path.splitext(fname)[0]
                        cv2.imwrite(os.path.join(dpath_overlay, f"{dct_args['project_id']}_{fname}"), overlay_img_bgr)
                        print(f"--- Saved combined overlay for {img_path_rel} ---")
                        cv2.imwrite(os.path.join(dpath_overlay, f"{dct_args['project_id']}_{fstem}_colored.png"),
                                    create_multicolored_masks_image(masks))
                        print(f"--- Saved multicolored overlay for {img_path_rel} ---")
                    except Exception as e:
                        print(f"Error saving overlays {img_path_rel}: {e}")
            else:
                print(f"Warn: No masks/mismatch for {img_path_rel}, skipping post-processing & overlay.")

        # Create final dataframe and save
        if not all_processed_annotations_out:
            print("Error: No annotations processed. Outputs not created.")
            return

        df_new = pd.DataFrame(all_processed_annotations_out)
        df_new['project_id'] = dct_args['project_id']
        df_new['object_id'] = df_new.groupby('img_id').cumcount() + 1

        # Ensure all expected columns are present with correct types
        expected_columns = ['img_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax',
                            'confidence', 'class', 'name', 'area', 'contours',
                            'object_type', 'project_id']

        for col in expected_columns:
            if col not in df_new.columns:
                if col in ['xmin', 'ymin', 'xmax', 'ymax', 'object_id']:
                    default_val = 0
                elif col == 'confidence':
                    default_val = 1.0
                elif col == 'area':
                    default_val = np.nan
                elif col == 'class':
                    default_val = 0
                elif col == 'name':
                    default_val = 'object'
                elif col == 'contours':
                    default_val = '[]'
                else:
                    default_val = ''
                df_new[col] = default_val

        # Enforce data types
        try:
            for col in ['xmin', 'ymin', 'xmax', 'ymax', 'object_id']:
                df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0).astype(int)
            for col in ['area', 'confidence']:
                df_new[col] = pd.to_numeric(df_new[col], errors='coerce').astype(float)
            df_new['class'] = pd.to_numeric(df_new['class'], errors='coerce').fillna(0).astype(int)
            for col in ['img_id', 'name', 'object_type', 'project_id']:
                df_new[col] = df_new[col].astype(str).replace('<NA>', '')
        except Exception as e:
            print(f"Warn: Type enforcement failed: {e}")

        # Reorder columns
        df_new = df_new[expected_columns]

        # Save output files with proper format
        output_globinfo = os.path.join(dct_args['input_dir'], f"{dct_args['project_id']}_segmentation_globinfo.csv")
        output_summary = os.path.join(dct_args['input_dir'], f"{dct_args['project_id']}_segmentation_summary.csv")

        try:
            # Write globinfo with metadata comment (same format as detection)
            with open(output_globinfo, 'w', newline='', encoding='utf-8') as f:
                # Preserve the input directory metadata if we have it
                if input_directory:
                    f.write(f"# input_directory: {input_directory}\n")
                df_new.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
            print(f"Saved annotations: {output_globinfo}")

            # Create and save summary
            summary_df = common.create_summary_table(df_new, dct_args['project_id'])
            summary_df.to_csv(output_summary, index=False, quoting=csv.QUOTE_ALL)
            print(f"Saved summary: {output_summary}")

            print("Success: Segmentation workflow completed.")
        except Exception as e:
            print(f"Error saving output files: {e}")
    else:
        print(f"Error: Input file not found: {dct_args['input_file']}")

# --- END OF FILE segmentation.py ---