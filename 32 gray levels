import nibabel as nib
import numpy as np
import os


def normalize_to_gray_levels(voxel_data, num_levels=32):
    """
    Normalize the gray levels of the VOI region to the specified number of gray levels.
    :param voxel_data: Input 3D image data (VOI region)
    :param num_levels: Target number of gray levels (default is 32 levels)
    :return: Normalized image data, or None (if the number of unique gray levels is less than 32)
    """
    # Find the minimum and maximum values of the image
    min_val = np.min(voxel_data)
    max_val = np.max(voxel_data)

    # If the minimum and maximum values are the same, the image has no range
    if min_val == max_val:
        print("Warning: The voxel data has no range. Skipping normalization.")
        return None

    # Linearly map the gray values to the range 0 to num_levels-1
    normalized_data = (voxel_data - min_val) / (max_val - min_val) * (num_levels - 1)

    # Convert to integer gray values
    normalized_data = np.round(normalized_data).astype(np.uint8)

    # Ensure the number of unique gray levels does not exceed num_levels
    unique_gray_levels = np.unique(normalized_data)
    if len(unique_gray_levels) < num_levels:
        print(f"Warning: Image has only {len(unique_gray_levels)} unique gray levels. Skipping image.")
        return None

    return normalized_data


def process_vois(input_vois_dir, output_vois_dir, num_levels=32):
    """
    Batch process all VOIs in the folder, normalizing their gray levels to the specified number of gray levels.
    :param input_vois_dir: Input folder containing all VOI images to process (in NIfTI format)
    :param output_vois_dir: Output folder to save the processed VOI images
    :param num_levels: Target number of gray levels
    """
    os.makedirs(output_vois_dir, exist_ok=True)

    # Get all NIfTI files in the input folder
    nifti_files = [f for f in os.listdir(input_vois_dir) if f.endswith('.nii.gz')]

    for nifti_file in nifti_files:
        input_path = os.path.join(input_vois_dir, nifti_file)

        # Load the NIfTI image
        img = nib.load(input_path)
        voxel_data = img.get_fdata()

        # Normalize gray levels
        normalized_data = normalize_to_gray_levels(voxel_data, num_levels)

        # If the image has insufficient gray levels (less than 32), skip it
        if normalized_data is None:
            continue

        # Create a new NIfTI image and save it
        normalized_img = nib.Nifti1Image(normalized_data, img.affine)
        output_path = os.path.join(output_vois_dir, nifti_file)
        nib.save(normalized_img, output_path)

        print(f"Saved normalized VOI: {output_path}")


# Configure paths
input_vois_dir = r"E:\Radiolung\extract_voi"  # Path to the input VOI folder
output_vois_dir = r"E:\Radiolung\normalized_voi"  # Path to the output processed VOI folder

# Execute batch processing
process_vois(input_vois_dir, output_vois_dir, num_levels=32)
