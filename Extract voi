import os
import nibabel as nib
import numpy as np
from glob import glob


def parse_acsv(file_path):
    """Parse ACSV file to extract the center and offset of the nodule."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract point coordinates
    point_lines = [line.strip() for line in lines if line.startswith('point|')]
    if len(point_lines) < 2:
        raise ValueError(f"Not enough point entries in {file_path}")

    # Calculate the world coordinates range of the nodule
    center = np.array([float(i) for i in point_lines[0].split('|')[1:4]])
    offset = np.array([float(i) for i in point_lines[1].split('|')[1:4]])
    p1 = center - offset
    p2 = center + offset
    return p1, p2


def world_to_voxel(world_coords, affine):
    """Convert world coordinates to voxel coordinates."""
    world_coords = np.append(world_coords, 1.0)
    voxel_coords = np.linalg.inv(affine) @ world_coords
    return np.round(voxel_coords[:3]).astype(int)


def extract_voi(nifti_path, acsv_path, output_path):
    """Extract VOI (volume of interest) from NIfTI and ACSV files."""
    # Load NIfTI file using nibabel
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine

    try:
        # Parse ACSV file and get the nodule range
        world_p1, world_p2 = parse_acsv(acsv_path)
        voxel_p1 = world_to_voxel(world_p1, affine)
        voxel_p2 = world_to_voxel(world_p2, affine)

        # Calculate VOI region indices
        z1, y1, x1 = np.minimum(voxel_p1, voxel_p2)
        z2, y2, x2 = np.maximum(voxel_p1, voxel_p2)

        z1, y1, x1 = np.maximum([z1, y1, x1], 0)
        z2 = min(z2, data.shape[0])
        y2 = min(y2, data.shape[1])
        x2 = min(x2, data.shape[2])

        # Extract VOI
        voi = data[z1:z2, y1:y2, x1:x2]
        voi_img = nib.Nifti1Image(voi, affine=np.eye(4))  # Use identity matrix as affine (since VOI is a local region)
        nib.save(voi_img, output_path)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to extract VOI from {nifti_path} and {acsv_path}: {e}")


def batch_extract_voi(ct_root_dir, output_dir):
    """Batch process the directory to extract VOI from all subfolders."""
    os.makedirs(output_dir, exist_ok=True)

    # Traverse all subfolders (recursively)
    for root, dirs, files in os.walk(ct_root_dir):
        # Find NIfTI files and ACSV files
        nii_files = glob(os.path.join(root, "*.nii.gz"))
        acsv_files = glob(os.path.join(root, "*.acsv"))

        for acsv_path in acsv_files:
            acsv_name = os.path.splitext(os.path.basename(acsv_path))[0]

            # If a NIfTI file exists, process it
            if nii_files:
                ct_path = nii_files[0]  # Assume only one .nii.gz file per folder
                output_filename = f"{os.path.basename(root)}_{acsv_name}_VOI.nii.gz"
                output_path = os.path.join(output_dir, output_filename)
                extract_voi(ct_path, acsv_path, output_path)
            else:
                print(f"[WARN] No NIfTI file found in {root}")


# ==== Configure paths ====
ct_root_dir = r"E:\Radiolung\dataverse_files\CT"  # Replace with the actual path
output_dir = r"E:\Radiolung\extract_voi"  # Replace with the actual output directory

# ==== Execute batch processing ====
batch_extract_voi(ct_root_dir, output_dir)
