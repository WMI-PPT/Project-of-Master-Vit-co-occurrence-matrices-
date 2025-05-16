import numpy as np
import nibabel as nib
import os
import pandas as pd
from skimage.feature import greycomatrix

def compute_glcm_matrix(image_slice, levels=32):
    """
    Compute the GLCM (Gray Level Co-occurrence Matrix) for a given image slice, keeping matrices from all angles.
    Returns shape: (levels, levels, 4) → 4 directions
    """
    image_slice = np.uint8(image_slice)  # Must be uint8 type, values range from [0, levels-1]

    glcm = greycomatrix(
        image_slice,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True
    )

    return glcm[:, :, 0, :]  # Extract shape (levels, levels, 4)


def process_3d_vois(vois_image, levels=32, target_slices=14):
    """
    Process 3D VOI image: Slice → Compute GLCM for each slice (4 directions)
    Output uniform shape: (target_slices, 32, 32, 4)
    """
    glcm_list = []

    for i in range(vois_image.shape[2]):  # Traverse along the Z-axis
        slice_image = vois_image[:, :, i]

        # Skip slices that are all zero
        if np.any(slice_image):
            glcm = compute_glcm_matrix(slice_image, levels=levels)
            glcm_list.append(glcm)

    if len(glcm_list) == 0:
        glcm_list = [np.zeros((32, 32, 4), dtype=np.float32)]

    glcm_tensor = np.stack(glcm_list, axis=0)  # [N, 32, 32, 4]

    # Uniform length to target_slices
    current_slices = glcm_tensor.shape[0]
    if current_slices < target_slices:
        padding = np.zeros((target_slices - current_slices, 32, 32, 4), dtype=np.float32)
        glcm_tensor = np.concatenate([glcm_tensor, padding], axis=0)
    elif current_slices > target_slices:
        glcm_tensor = glcm_tensor[:target_slices]

    return glcm_tensor  # shape: [target_slices, 32, 32, 4]


def save_glcm_to_npz(glcm_tensor, output_path):
    """
    Save as .npz format, flattening into a fixed-length vector
    """
    flat_tensor = glcm_tensor.flatten()  # shape: (target_slices * 32 * 32 * 4,)
    np.savez_compressed(output_path, glcm=flat_tensor)
    print(f"✅ GLCM tensor saved to: {output_path}, shape={flat_tensor.shape}")


def batch_process_3d_vois(input_folder, output_folder, metadata_file, levels=32, target_slices=14):
    """
    Batch process all NIfTI files in the VOI image folder, compute GLCM and save as .npz files
    Also generate a CSV file containing paths and labels
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read metadata file
    metadata_df = pd.read_excel(metadata_file)

    # Path for the CSV file to store the data
    csv_file = os.path.join(output_folder, 'glcm_vit_data.csv')
    data_entries = []

    # Traverse all NIfTI files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".nii.gz"):
            patient_id_str = filename.split("_")[1]  # Extract patient ID
            patient_id = int(patient_id_str)

            # Find the corresponding label
            patient_info = metadata_df[metadata_df['patient_id'] == patient_id]
            if not patient_info.empty:
                nodule_type = patient_info['type'].values[0]
                label = 1 if nodule_type == 'Malignant' else 0
            else:
                label = -1  # Unknown label

            # Load VOI image
            file_path = os.path.join(input_folder, filename)
            img = nib.load(file_path)
            vois_image = img.get_fdata()
            vois_image = np.clip(vois_image, 0, 31).astype(np.uint8)  # Normalize gray values

            # Extract and standardize GLCM features
            glcm_tensor = process_3d_vois(vois_image, levels=levels, target_slices=target_slices)

            # Save as .npz
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_glcm_tensor.npz")
            save_glcm_to_npz(glcm_tensor, output_path)

            # Add data entry
            data_entries.append({"path": output_path, "label": label})

    # Save CSV
    df = pd.DataFrame(data_entries)
    df.to_csv(csv_file, index=False)
    print(f"✅ Batch processing completed. CSV saved to: {csv_file}")


# ==== Main Program Entry ====
if __name__ == "__main__":
    input_folder = r"E:\Radiolung\normalized_voi"
    output_folder = r"E:\Radiolung\co-occurrence matrices"
    metadata_file = r"E:\Radiolung\dataverse_files\11112024_BDMetaData.xlsx"

    # Execute batch processing
    batch_process_3d_vois(input_folder, output_folder, metadata_file, levels=32, target_slices=14)
