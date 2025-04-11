import torch
import cv2
import os
import numpy as np
import pandas as pd
import json
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
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(img_rgb)
        masks_list = []
        num_annotations = len(annotations)

        with torch.inference_mode():
            if self.device.type == 'cuda':
                autocast_context = torch.autocast(device_type='cuda', dtype=torch.float16)
            else:
                autocast_context = torch.no_grad()

            with autocast_context:
                for i in range(0, num_annotations, batch_size):
                    batch_annotations = annotations[i:i + batch_size]
                    batch_masks = []
                    for ann in batch_annotations:
                        if ann['object_type'] == 'box':
                            # Use the box coordinates to generate a mask
                            x_min, y_min, x_max, y_max = ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']
                            box = np.array([x_min, y_min, x_max, y_max])
                            masks, _, _ = self.predictor.predict(
                                box=box[None, :],
                                multimask_output=False
                            )
                            mask = masks[0].astype(np.uint8)
                            ann['mask'] = mask
                            ann['object_type'] = 'mask'  # Update object type to 'mask'
                            batch_masks.append(mask)
                        elif ann['object_type'] == 'mask':
                            # Reconstruct mask from contours if needed
                            if 'mask' not in ann or ann['mask'] is None:
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
                            else:
                                mask = ann['mask']
                            batch_masks.append(mask)
                        else:
                            print(f"Unknown annotation type: {ann['object_type']}")
                    masks_list.extend(batch_masks)

        masks = np.array(masks_list)
        masks = masks.astype('uint8')
        return masks, annotations  # Return updated annotations with masks


def add_masks_on_image(masks, img):
    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
    img[combined_mask == 1] = [0, 0, 255]  # Change color to red


def create_multicolored_masks_image(masks):
    if masks.ndim != 3:
        raise ValueError("Expected a 3D array of masks")

    # Create a black background image
    black_image = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)

    # Loop over each mask layer
    for i in range(masks.shape[0]):
        color = np.random.randint(100, 256, size=3)  # Generate a bright color
        mask_layer = masks[i, :, :]  # Select the i-th mask layer
        # Apply color to pixels where mask_layer is 1
        for c in range(3):  # Apply to each color channel
            black_image[:, :, c][mask_layer == 1] = color[c]

    return black_image


def segmentation_workflow(dct_args):
    dct_args['project_id'] = os.path.basename(dct_args['input_file']).replace('_globinfo.csv', '')
    dct_args['input_dir'] = os.path.dirname(dct_args['input_file'])

    gpu_if_avail = utils.get_bool(dct_args['gpu'])
    add_overlay = utils.get_bool(dct_args['add_overlay'])
    utils.set_cpu_usage(dct_args['cpu'])
    device = torch.device('cuda:0' if torch.cuda.is_available() and gpu_if_avail else 'cpu')

    if add_overlay:
        dpath_overlay = os.path.join(dct_args['input_dir'], dct_args['project_id'], 'img', 'segmentation')
        os.makedirs(dpath_overlay, exist_ok=True)

    if utils.check_file_existence(dct_args['input_file']):
        df = pd.read_csv(dct_args['input_file'])
        lst_img_paths = df['img_id'].unique()
        segmentation_model = NemaCounterSegmentation(
            checkpoint_path=dct_args['segany'],
            model_cfg_path=dct_args['sam2_config'],
            device=device
        )

        all_annotations = []

        for img_path in lst_img_paths:
            img = common.read_image(img_path)
            img_df = df[df['img_id'] == img_path]
            annotations = []

            for _, row in img_df.iterrows():
                ann = row.to_dict()
                if ann['object_type'] == 'box':
                    # Ensure keys are correct
                    ann['xmin'] = int(ann['xmin'])
                    ann['ymin'] = int(ann['ymin'])
                    ann['xmax'] = int(ann['xmax'])
                    ann['ymax'] = int(ann['ymax'])
                    annotations.append(ann)
                elif ann['object_type'] == 'mask':
                    # Keep the existing mask annotation
                    annotations.append(ann)
                else:
                    # Unknown object_type
                    print(f"Unknown object_type: {ann['object_type']}")

            if not annotations:
                print(f"No valid annotations for image {img_path}. Skipping.")
                continue

            masks, updated_annotations = segmentation_model.objects_segmentation(img, annotations)
            # Calculate the surface area for each mask
            surfaces = np.sum(masks.reshape(masks.shape[0], -1), axis=1)

            # Update annotations with area and contours
            for ann, mask in zip(updated_annotations, masks):
                ann['area'] = np.sum(mask)
                # Extract contours from mask
                contours_list, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours_list) > 0:
                    # Take the largest contour
                    contour = max(contours_list, key=cv2.contourArea)
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
                    ann['contours'] = np.nan
                all_annotations.append(ann)

            if add_overlay:
                # Overlay masks on the image
                overlay_img = img.copy()
                add_masks_on_image(masks, overlay_img)
                fpath_out_img = os.path.join(dpath_overlay, f"{dct_args['project_id']}_{os.path.basename(img_path)}")
                cv2.imwrite(fpath_out_img, overlay_img)

                # Create multicolored masks image
                multicolored_img = create_multicolored_masks_image(masks)
                fpath_out_multi = os.path.join(
                    dpath_overlay,
                    f"{dct_args['project_id']}_{os.path.splitext(os.path.basename(img_path))[0]}_colored.png"
                )
                cv2.imwrite(fpath_out_multi, multicolored_img)

        # Create the new globinfo DataFrame
        df_new = pd.DataFrame(all_annotations)
        df_new['project_id'] = dct_args['project_id']
        df_new['object_id'] = df_new.groupby('img_id').cumcount() + 1
        # Reorder columns to match the expected structure
        expected_columns = ['img_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax',
                            'confidence', 'class', 'name', 'area', 'contours', 'object_type', 'project_id']
        df_new = df_new[expected_columns]
        # Save the updated globinfo.csv
        output_globinfo = os.path.join(dct_args['input_dir'], f"{dct_args['project_id']}_segmentation_globinfo.csv")
        df_new.to_csv(output_globinfo, index=False)

        # Create summary table
        df_summary = common.create_summary_table(df_new, dct_args['project_id'])
        output_summary = os.path.join(dct_args['input_dir'], f"{dct_args['project_id']}_segmentation_summary.csv")
        df_summary.to_csv(output_summary, index=False)
        print("Success: Segmentation completed and results saved.")
    else:
        print("Error: The specified input file does not exist.")
