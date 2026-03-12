import cv2
import numpy as np
import pandas as pd
from functools import reduce
import configparser

def read_image(img_path):
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

def create_boxes(df):
    return df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

def create_global_table(lst_df, project_id):
    """
    Concatenates a list of DataFrames (each presumably bounding box detections).
    Keeps 'class' (and optionally 'name') so we can group by class in summary.
    """
    df = pd.concat(lst_df).reset_index(drop=True)
    df['project_id'] = project_id

    # Optionally remove just 'name' if you don't care about textual labels,
    # but DO NOT remove 'class' if you want separate lines per class later.
    # df.drop(columns=['name'], inplace=True)   # if you don’t need the name text

    # Reorder columns as desired; here we keep both 'class' and 'name'.
    df = df[
        [
            'project_id',
            'img_id',
            'object_id',
            'class',
            'name',
            'xmin',
            'ymin',
            'xmax',
            'ymax',
            'confidence'
        ]
    ]

    return df

def calculation_per_group(grouped_df, colname, measurements=['mean', 'std'], rnd=3):
    """
    Compute measures per group for a given column
    """
    df = grouped_df.agg({colname: measurements})
    df.columns = [f'{colname}_{m}' for m in measurements]
    df = np.round(df, rnd)
    df = df.reset_index()
    return df

def merge_multiple_dataframes(lst_df, lst_k, how='inner'):
    """
    Merge multiple dataframes contained in a list (on a common list of keys)
    """
    from functools import reduce
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=lst_k, how=how), lst_df)
    return df_merged

def create_summary_table(df_global, project_id):
    # If 'area' doesn’t exist, compute it:
    if 'area' not in df_global.columns:
        df_global['area'] = (df_global['xmax'] - df_global['xmin']) * (df_global['ymax'] - df_global['ymin'])

    # Group by (img_id × name) so we get one row per image-class_name pair
    grouped = df_global.groupby(['img_id', 'name'], as_index=False)

    # Count detections
    counts = grouped.size().rename(columns={'size': 'count'})

    # Mean and std of confidence
    conf_stats = grouped['confidence'].agg(['mean', 'std']).reset_index()
    conf_stats.rename(columns={'mean': 'conf_mean', 'std': 'conf_std'}, inplace=True)

    # Mean and std of area
    area_stats = grouped['area'].agg(['mean', 'std']).reset_index()
    area_stats.rename(columns={'mean': 'area_mean', 'std': 'area_std'}, inplace=True)

    # Merge all stats
    df_summary = pd.merge(counts, conf_stats, on=['img_id', 'name'], how='left')
    df_summary = pd.merge(df_summary, area_stats, on=['img_id', 'name'], how='left')

    # Add project ID column
    df_summary['project_id'] = project_id

    # Reorder columns (optionally rename 'name' to 'class_name')
    df_summary.rename(columns={'name': 'class_name'}, inplace=True)
    df_summary = df_summary[
        [
            'project_id',
            'img_id',
            'class_name',
            'count',
            'conf_mean',
            'conf_std',
            'area_mean',
            'area_std'
        ]
    ]

    return df_summary


def get_config_info(fpath):
    try:
        config = configparser.ConfigParser()
        config.read(fpath)
        return config
    except:
        raise FileNotFoundError(f"The config file {fpath} does not exists")
