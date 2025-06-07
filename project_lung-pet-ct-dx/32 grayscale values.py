import os
import nibabel as nib
import numpy as np

# Input and output paths
input_folder = r"E:\LUNG DATA\lung\voi_ct"          # ← Replace with the path to your voi_ct folder
output_folder = r"E:\LUNG DATA\normalized_voi"       # ← Output path, can be different from input

# Create output folder
os.makedirs(output_folder, exist_ok=True)

def normalize_to_32_levels(image_data):
    # Remove extreme values (0.5% - 99.5%), optional
    p_low, p_high = np.percentile(image_data, (0.5, 99.5))
    image_data = np.clip(image_data, p_low, p_high)

    # Normalize to range 0~1
    norm = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)

    # Convert to integer grayscale levels 0~31 (32 levels total)
    gray32 = np.floor(norm * 32).astype(np.uint8)
    gray32 = np.clip(gray32, 0, 31)

    return gray32

# Iterate through all .nii.gz files
for filename in os.listdir(input_folder):
    if filename.endswith(".nii.gz"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load NIfTI image
        img = nib.load(input_path)
        data = img.get_fdata()

        # Normalize to 32 grayscale levels
        gray_data = normalize_to_32_levels(data)

        # Save new image using original affine and header
        new_img = nib.Nifti1Image(gray_data, affine=img.affine, header=img.header)
        nib.save(new_img, output_path)

        print(f"Processed: {filename}")

print("All files processed and normalized to 32 grayscale levels.")
