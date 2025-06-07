# extract_glcm_batch_modified_with_label.py
import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from custom_glcm import CustomGLCM

# Set directories
voi_dir = r"E:\LUNG DATA\normalized_voi"
mask_dir = r"E:\LUNG DATA\lung\voi_mask"
output_dir = r"E:\LUNG DATA\output"
os.makedirs(output_dir, exist_ok=True)

# Parameters for feature extraction
params = {
    'binWidth': 1,
    'force2D': False,
    'normalize': True,
    'normalizeScale': 100,
    'geometryTolerance': 1e-6
}

# Label mapping
label_map = {'A': 1, 'B': 2, 'G': 3}

# Get all VOI files with new naming: *_voi_ct.nii.gz
voi_files = glob.glob(os.path.join(voi_dir, "*_voi_ct.nii.gz"))
voi_files = sorted(voi_files)

records = []

for voi_path in voi_files:
    base_name = os.path.basename(voi_path)
    base_id = base_name.replace("_voi_ct.nii.gz", "")
    mask_name = base_id + "_voi_mask.nii.gz"
    mask_path = os.path.join(mask_dir, mask_name)

    print(f"✅ Processing {base_id}")

    if not os.path.exists(mask_path):
        print(f"❌ Mask not found for {base_id}, skipping.")
        continue

    try:
        image = sitk.ReadImage(voi_path)
        mask = sitk.ReadImage(mask_path)

        extractor = CustomGLCM(image, mask, **params)
        extractor.enableAllFeatures()

        features = extractor.execute()

        glcm = extractor.glcm_raw_matrix
        if glcm is None:
            print(f"❌ GLCM matrix not calculated for {base_id}, skipping.")
            continue

        # Remove first dimension: (1, 32, 32, 13) → (32, 32, 13)
        glcm = np.squeeze(glcm, axis=0)
        # Transpose to (13, 32, 32) for ViT
        glcm = np.transpose(glcm, (2, 0, 1))

        # Only save data with shape (13, 32, 32)
        if glcm.shape != (13, 32, 32):
            print(f"❌ Unexpected GLCM shape {glcm.shape} for {base_id}, skipping save.")
            continue

        print(f"GLCM shape for {base_id}: {glcm.shape}")

        save_path = os.path.join(output_dir, f"GLCM_{base_id}.npy")
        np.save(save_path, glcm)

        records.append({
            "id": base_id,
            "voi_path": voi_path,
            "mask_path": mask_path,
            "glcm_path": save_path,
            "shape": glcm.shape
        })

    except Exception as e:
        import traceback
        print(f"❌ Failed for {base_id}: {e}")
        traceback.print_exc()

# Map labels based on the first letter of the file name and filter unsupported labels
filtered_records = []
for record in records:
    # glcm_path format: .../GLCM_A0001_1.npy, extract first letter of id
    patient_letter = record["id"][0]
    if patient_letter in label_map:
        record["label"] = label_map[patient_letter]
        filtered_records.append(record)
    else:
        print(f"⚠️ Skipping {record['id']} due to unsupported label '{patient_letter}'")

# Save CSV with labels
df = pd.DataFrame(filtered_records)
csv_path = os.path.join(output_dir, "glcm_index_with_label.csv")
df.to_csv(csv_path, index=False)
print(f"\n📄 Done! Labelled index CSV saved to: {csv_path}")
