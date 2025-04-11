import sam2
import cv2
import pandas as pd
import numpy as np
import os
import torch
import json

from sam2.sam2_image_predictor import SAM2ImagePredictor

# Global variables
drawing = False
adding_object = False
positive_points = []
negative_points = []
selected_annotation = -1
new_annotations = []
deleted_annotations = []
img = None
img_original = None
annotations = []
mode = 'classic'  # Modes: 'classic', 'smart'
sam_predictor = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_size = 1056  # Should match the model's input size

def edit_image(event, x, y, flags, param):
    global drawing, adding_object, new_annotations, img, selected_annotation, positive_points, negative_points, mode
    if mode == 'classic':
        handle_classic_mode(event, x, y, flags, param)
    elif mode == 'smart':
        handle_smart_mode(event, x, y, flags, param)

def handle_classic_mode(event, x, y, flags, param):
    global drawing, new_annotations, selected_annotation
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        new_annotations.append({'type': 'rectangle', 'points': [(x, y), (x, y)], 'confidence': 1.0})
        selected_annotation = len(annotations) + len(new_annotations) - 1
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            new_annotations[-1]['points'][1] = (x, y)
            redraw_image()
        else:
            highlight_annotation(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            redraw_image()

def handle_smart_mode(event, x, y, flags, param):
    global adding_object, positive_points, negative_points, selected_annotation
    if adding_object:
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add a positive point
            positive_points.append((x, y))
            redraw_image()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Add a negative point
            negative_points.append((x, y))
            redraw_image()
    else:
        if event == cv2.EVENT_MOUSEMOVE:
            highlight_annotation(x, y)

def generate_mask_from_points():
    global positive_points, negative_points, new_annotations, sam_predictor, img, device
    if sam_predictor is not None and len(positive_points) > 0:
        try:
            input_point = np.array(positive_points + negative_points)
            input_label = np.array([1]*len(positive_points) + [0]*len(negative_points))

            # Set the image in the predictor
            sam_predictor.set_image(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))

            with torch.no_grad():
                masks, scores, logits = sam_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False
                )

            mask = masks[0].astype(np.uint8)

            # Check the area of the mask
            mask_area = np.sum(mask > 0)
            image_area = img.shape[0] * img.shape[1]
            if mask_area / image_area <= 0.8:  # Allow masks covering up to 80% of the image
                # Add the mask to annotations
                new_annotations.append({'type': 'mask', 'mask': mask, 'confidence': 1.0})
            else:
                print("Mask covers more than 80% of the image, ignoring.")
        except Exception as e:
            print(f"Error during SAM prediction: {e}")

def highlight_annotation(x, y):
    global selected_annotation, annotations, new_annotations
    selected_annotation = -1
    all_annotations = annotations + new_annotations
    for i, ann in enumerate(all_annotations):
        if ann['type'] == 'rectangle':
            (x1, y1), (x2, y2) = ann['points']
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            if x_min <= x <= x_max and y_min <= y <= y_max:
                selected_annotation = i
                break
        elif ann['type'] == 'mask' and 'mask' in ann:
            mask = ann['mask']
            if mask.shape != img.shape[:2]:
                # Resize mask to match image size if necessary
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                ann['mask'] = mask
            if mask[y, x] > 0:
                selected_annotation = i
                break
    redraw_image()

def redraw_image():
    global img, img_original, annotations, new_annotations, selected_annotation, adding_object, positive_points, negative_points
    if img_original is not None:
        img = img_original.copy()
        all_annotations = annotations + new_annotations
        # Draw masks first
        for i, ann in enumerate(all_annotations):
            if ann['type'] == 'mask':
                mask = ann['mask']
                if mask.shape != img.shape[:2]:
                    # Resize mask to match image size if necessary
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    ann['mask'] = mask
                # Create a colored mask
                colored_mask = np.zeros_like(img)
                color = (0, 255, 0)  # Green color for masks
                if i == selected_annotation:
                    color = (255, 0, 0)  # Highlighted color
                colored_mask[mask > 0] = color
                img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)
        # Draw rectangles
        for i, ann in enumerate(all_annotations):
            if ann['type'] == 'rectangle':
                color = (0, 0, 255)  # Default color
                if i == selected_annotation:
                    color = (255, 0, 0)  # Highlighted color
                cv2.rectangle(img, ann['points'][0], ann['points'][1], color, 2)
        # Draw positive points
        for pt in positive_points:
            cv2.circle(img, pt, 5, (0, 255, 0), -1)
            cv2.putText(img, '+', (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        # Draw negative points
        for pt in negative_points:
            cv2.circle(img, pt, 5, (0, 0, 255), -1)
            cv2.putText(img, '-', (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        # Draw the legend on the image
        draw_legend_on_image()

def draw_legend_on_image():
    global img, mode, adding_object
    instructions = [
        f"Mode: {mode.capitalize()} (Press 'M' to switch)",
    ]
    if mode == 'classic':
        instructions.append("Press 'R' to remove selected annotation")
    elif mode == 'smart':
        if adding_object:
            instructions.append("Adding new object: Left-click '+' point, Right-click '-' point")
            instructions.append("Press 'G' to generate mask")
        else:
            instructions.append("Press 'N' to add new object")
            instructions.append("Press 'R' to remove selected annotation")
    instructions.append("Press 'S' to save and exit")
    instructions.append("Press 'ESC' to exit without saving")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 1
    line_height = 25
    x, y0 = 10, 20
    for i, text in enumerate(instructions):
        y = y0 + i * line_height
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def remove_selected_annotation():
    global annotations, new_annotations, selected_annotation, deleted_annotations
    if selected_annotation != -1:
        all_annotations = annotations + new_annotations
        ann = all_annotations[selected_annotation]
        deleted_annotations.append(ann)
        if selected_annotation < len(annotations):
            annotations.pop(selected_annotation)
        else:
            new_annotations.pop(selected_annotation - len(annotations))
        selected_annotation = -1
        redraw_image()

def run_manual_annotation(img_path, annotations_input, sam_predictor):
    global img, img_original, annotations, new_annotations, deleted_annotations, mode, adding_object, positive_points, negative_points, selected_annotation, img_height_orig, img_width_orig
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image at {img_path}")
        return None
    img_height_orig, img_width_orig = img.shape[:2]
    # Resize image to input_size x input_size
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img_original = img.copy()
    annotations = annotations_input.copy()
    new_annotations = []
    deleted_annotations = []
    adding_object = False
    positive_points = []
    negative_points = []
    selected_annotation = -1
    cv2.namedWindow('Annotation Tool', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Annotation Tool', input_size, input_size)
    cv2.setMouseCallback('Annotation Tool', edit_image)
    redraw_image()  # Draw initial image with annotations

    dct_res = {}
    while True:
        cv2.imshow('Annotation Tool', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key to exit without saving
            new_annotations = []
            deleted_annotations = []
            break
        elif k == ord('s'):  # Save and exit
            dct_res = {
                'img_id': img_path,
                'annotations': annotations + new_annotations,
                'deleted_annotations': deleted_annotations
            }
            new_annotations = []
            deleted_annotations = []
            break
        elif k == ord('r'):  # Remove selected annotation
            remove_selected_annotation()
        elif k == ord('m'):  # Switch mode
            mode = 'smart' if mode == 'classic' else 'classic'
            redraw_image()
        elif k == ord('n'):  # Start adding new object
            if mode == 'smart':
                adding_object = True
                positive_points = []
                negative_points = []
                redraw_image()
        elif k == ord('g'):  # Generate mask from points
            if adding_object:
                generate_mask_from_points()
                adding_object = False
                positive_points = []
                negative_points = []
                redraw_image()
    cv2.destroyAllWindows()
    return dct_res

def build_annotations_dataframe(annotations, img_path):
    records = []
    for ann in annotations:
        if ann['type'] == 'rectangle':
            (x1, y1), (x2, y2) = ann['points']
            # Scale coordinates back to original image size
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            x_min = int(x_min * img_width_orig / input_size)
            x_max = int(x_max * img_width_orig / input_size)
            y_min = int(y_min * img_height_orig / input_size)
            y_max = int(y_max * img_height_orig / input_size)
            records.append({
                'img_id': img_path,
                'xmin': x_min,
                'ymin': y_min,
                'xmax': x_max,
                'ymax': y_max,
                'confidence': ann.get('confidence', 1.0),
                'class': np.nan,
                'name': '',
                'area': np.nan,
                'contours': np.nan,
                'object_type': 'box'
            })
        elif ann['type'] == 'mask':
            mask = ann['mask']
            # No need to resize mask since it's already in input_size x input_size
            # Extract contours from mask
            contours_list, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours_list:
                if len(contour) < 3:
                    continue  # Skip invalid contours
                # Scale contour coordinates back to original image size
                contour_scaled = contour.squeeze().astype(np.float32)
                contour_scaled[:, 0] *= img_width_orig / input_size
                contour_scaled[:, 1] *= img_height_orig / input_size
                # Recalculate bounding box and area
                x_min = int(np.min(contour_scaled[:, 0]))
                y_min = int(np.min(contour_scaled[:, 1]))
                x_max = int(np.max(contour_scaled[:, 0]))
                y_max = int(np.max(contour_scaled[:, 1]))
                area = cv2.contourArea(contour_scaled.astype(np.int32))
                records.append({
                    'img_id': img_path,
                    'xmin': x_min,
                    'ymin': y_min,
                    'xmax': x_max,
                    'ymax': y_max,
                    'confidence': ann.get('confidence', 1.0),
                    'class': np.nan,
                    'name': '',
                    'area': area,
                    'contours': json.dumps(contour_scaled.tolist()),
                    'object_type': 'mask'
                })
    df = pd.DataFrame(records)
    return df

def edition_workflow(input_file, output_directory, project_id,  use_gpu):
    global sam_predictor, device, img_height_orig, img_width_orig, input_size
    if use_gpu == 0:
        device = 'cpu'
    if os.path.exists(input_file):
        # Initialize SAM2 predictor
        sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large", device=device)

        df_input = pd.read_csv(input_file)
        lst_img_paths = df_input['img_id'].unique()
        all_annotations = []

        for img_path in lst_img_paths:
            full_img_path = os.path.join(os.path.dirname(input_file), img_path)
            if not os.path.exists(full_img_path):
                full_img_path = img_path  # Try absolute path
                if not os.path.exists(full_img_path):
                    print(f"Error: Image {img_path} not found.")
                    continue
            # Load existing annotations
            annotation_data = df_input[df_input['img_id'] == img_path]
            annotations_list = []
            img_orig = cv2.imread(full_img_path)
            if img_orig is None:
                print(f"Error: Unable to read image at {full_img_path}")
                continue
            img_height_orig, img_width_orig = img_orig.shape[:2]
            # Resize image to input_size x input_size
            img = cv2.resize(img_orig, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            for _, row in annotation_data.iterrows():
                if row['object_type'] == 'box':
                    # Scale box coordinates to input_size
                    x_min = int(row['xmin'] * input_size / img_width_orig)
                    y_min = int(row['ymin'] * input_size / img_height_orig)
                    x_max = int(row['xmax'] * input_size / img_width_orig)
                    y_max = int(row['ymax'] * input_size / img_height_orig)
                    annotations_list.append({
                        'type': 'rectangle',
                        'points': [(x_min, y_min), (x_max, y_max)],
                        'confidence': row['confidence']
                    })
                elif row['object_type'] == 'mask':
                    # Contours are in input_size coordinate system, reconstruct mask
                    contours_json = row['contours']
                    if pd.isna(contours_json) or len(contours_json) == 0:
                        continue
                    contours = json.loads(contours_json)
                    if len(contours) == 0:
                        continue
                    mask = np.zeros((input_size, input_size), dtype=np.uint8)
                    for contour in contours:
                        contour_np = np.array(contour, dtype=np.float32)
                        # Scale contour coordinates
                        contour_np[:, 0] = contour_np[:, 0] * input_size / img_width_orig
                        contour_np[:, 1] = contour_np[:, 1] * input_size / img_height_orig
                        contour_np = contour_np.astype(np.int32)
                        contour_np = contour_np.reshape((-1, 1, 2))
                        # Draw contour on mask
                        cv2.fillPoly(mask, [contour_np], 1)
                    annotations_list.append({
                        'type': 'mask',
                        'mask': mask,
                        'confidence': row['confidence']
                    })
            # Run manual annotation
            result = run_manual_annotation(full_img_path, annotations_list, sam_predictor)
            # Handle the results
            if result:
                df_ann = build_annotations_dataframe(result['annotations'], img_path)
                all_annotations.append(df_ann)
            else:
                print(f"No annotations were made for image {img_path}")
        # Combine all annotations
        if all_annotations:
            df_all = pd.concat(all_annotations, ignore_index=True)
            df_all['project_id'] = project_id
            df_all['object_id'] = df_all.groupby('img_id').cumcount() + 1
            # Reorder columns to match the expected structure
            expected_columns = ['img_id', 'object_id', 'xmin', 'ymin', 'xmax', 'ymax',
                                'confidence', 'class', 'name', 'area', 'contours', 'object_type', 'project_id']
            df_all = df_all[expected_columns]
            # Save the updated globinfo.csv
            output_globinfo = os.path.join(output_directory, f"{project_id}_manualedition_globinfo.csv")
            df_all.to_csv(output_globinfo, index=False)
            # Create summary table
            df_summary = create_summary_table(df_all, project_id)
            output_summary = os.path.join(output_directory, f"{project_id}_manualedition_summary.csv")
            df_summary.to_csv(output_summary, index=False)
            print("Success: Annotations saved successfully.")
        else:
            print("No annotations were made.")
    else:
        print("Error: The specified file does not exist.")


def create_summary_table(df, project_id):
    summary_records = []
    for img_id, group in df.groupby('img_id'):
        count = len(group)
        conf_mean = group['confidence'].mean()
        conf_std = group['confidence'].std()
        # Check if there are any masks in the group
        mask_areas = group[group['object_type'] == 'mask']['area']
        if len(mask_areas) > 0:
            area_mean = mask_areas.mean()
            area_std = mask_areas.std()
        else:
            area_mean = np.nan
            area_std = np.nan
        summary_records.append({
            'project_id': project_id,
            'img_id': img_id,
            'count': count,
            'conf_mean': conf_mean,
            'conf_std': conf_std,
            'area_mean': area_mean,
            'area_std': area_std
        })
    df_summary = pd.DataFrame(summary_records)
    return df_summary
