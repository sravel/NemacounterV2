import os
import psutil
import torch


def list_image_files(dir_path):
    """
    List images with the '.jpg', '.jpeg', '.png' extensions
    in a given directory or subdirectories
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']  # Add more extensions if needed
    image_files = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                image_files.append(os.path.join(root, file))

    return image_files


def check_file_existence(path):
    """
    Check is a path exists and if it points to a file (and not a dir)
    """
    return os.path.isfile(path)


def get_bool(arg):
    if arg == 1:
        return True
    else:
        return False


def compute_available_cpu():
    """
    Returns rounding to the lower integer, with a guaranteed minimum of 1.
    """
    # Calculate available CPUs based on current usage
    avail = int(psutil.cpu_count() * (1 - psutil.cpu_percent() / 100))
    # --- FIX: Ensure the function never returns less than 1 ---
    return max(1, avail)


def set_cpu_usage(n):
    """
    Set the number of CPU to be used by pytorch.

    If -1 (default value), all the available cpus can be
    used by torch. Else, use the given number of CPU max.
    """
    # The fix in compute_available_cpu ensures avail_cpu is always >= 1
    avail_cpu = compute_available_cpu()

    if n != -1:
        # If user requests a number, use the smaller of their request and what's available
        num_to_set = min(n, avail_cpu)
        # Also ensure the user's number is at least 1
        torch.set_num_threads(max(1, num_to_set))
    else:
        # If user wants all available, use our safe calculated value
        torch.set_num_threads(avail_cpu)