import os
import shutil

# Set input and output paths
input_folder = r"E:\LUNG DATA\LUNG-PET-CT-DX"     # ← Change to your input folder path
output_root = r"E:\LUNG DATA\lung"                # ← Change to the path where you want to save classified results

# Define classification subfolders
output_dirs = {
    'mask': os.path.join(output_root, 'mask'),
    'voi_ct': os.path.join(output_root, 'voi_ct'),
    'voi_mask': os.path.join(output_root, 'voi_mask'),
}

# Create output subfolders (if they don't exist)
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.nii.gz'):
        src_path = os.path.join(input_folder, filename)

        # Classify based on keywords in filename
        if '_voi_ct' in filename:
            dst_path = os.path.join(output_dirs['voi_ct'], filename)
        elif '_voi_mask' in filename:
            dst_path = os.path.join(output_dirs['voi_mask'], filename)
        elif '_mask' in filename:  # Note: must check after _voi_mask
            dst_path = os.path.join(output_dirs['mask'], filename)
        else:
            continue  # Skip files that do not match any category

        # Copy file to the target folder
        shutil.copy2(src_path, dst_path)

print("File classification completed and saved to:", output_root)
