import numpy as np
import nibabel as nib
from itertools import product
import os

def load_voi_nii_gz(filepath):
    """
    Load VOI from a .nii.gz file and return as a NumPy array.
    """
    nii = nib.load(filepath)
    data = nii.get_fdata()
    return data.astype(np.int32)

def quantize_gray_levels(volume, num_levels=32):
    """
    Quantize 3D volume to fixed number of gray levels using histogram equalization.
    """
    flat = volume.flatten()
    hist, bin_edges = np.histogram(flat, bins=num_levels)
    bin_indices = np.digitize(flat, bin_edges[:-1], right=True)
    quantized = bin_indices.reshape(volume.shape)
    return np.clip(quantized - 1, 0, num_levels - 1)

def compute_3d_glcm(volume, num_levels=32):
    """
    Compute 3D GLCMs in 13 directions and normalize them.
    """
    offsets = [(dz, dy, dx) for dz, dy, dx in product([-1, 0, 1], repeat=3)
               if (dz, dy, dx) != (0, 0, 0)]
    offsets = offsets[:13]

    D, H, W = volume.shape
    glcm_matrices = [np.zeros((num_levels, num_levels), dtype=np.uint32) for _ in range(13)]

    for idx, (dz, dy, dx) in enumerate(offsets):
        for z in range(max(0, -dz), min(D, D - dz)):
            for y in range(max(0, -dy), min(H, H - dy)):
                for x in range(max(0, -dx), min(W, W - dx)):
                    i = volume[z, y, x]
                    j = volume[z + dz, y + dy, x + dx]
                    glcm_matrices[idx][i, j] += 1

    glcm_normalized = [mat / mat.sum() if mat.sum() != 0 else mat for mat in glcm_matrices]
    return np.stack(glcm_normalized, axis=-1)

def main(input_path, output_path, num_levels=32):
    voi = load_voi_nii_gz(input_path)
    quantized = quantize_gray_levels(voi, num_levels)
    glcm_tensor = compute_3d_glcm(quantized, num_levels)
    np.savez_compressed(output_path, glcm=glcm_tensor)
    print(f"Saved GLCM tensor to: {output_path}")

if __name__ == "__main__":
    # === Replace with your actual file paths ===
    input_nii = r"E:\Radiolung\normalized_voi\Paciente_1_TC_9_R_1_VOI.nii.gz"
    output_npz = "E:/Radiolung/co-occurrence matrices/glcm_output_3D.npz"
    main(input_nii, output_npz)
