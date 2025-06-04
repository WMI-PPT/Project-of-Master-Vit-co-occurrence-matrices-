# extract_glcm_batch.py
import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from custom_glcm import CustomGLCM

# Set directories
voi_dir = r"E:\Radiolung\normalized_voi"
mask_dir = r"E:\Radiolung\mask"
output_dir = r"E:\Radiolung\output"
os.makedirs(output_dir, exist_ok=True)

# Parameters for feature extraction
params = {
    'binWidth': 1,
    'force2D': False,
    'normalize': True,
    'normalizeScale': 100,
    'geometryTolerance': 1e-6
}

# Get all VOI files
voi_files = glob.glob(os.path.join(voi_dir, "Paciente_*_VOI.nii.gz"))
voi_files = sorted(voi_files)

records = []

for voi_path in voi_files:
    # Parse corresponding MASK file path based on VOI file name
    base_name = os.path.basename(voi_path)
    # Example: Paciente_1_TC_9_R_1_VOI.nii.gz -> Paciente_1_TC_9_R_1_mask.nii.gz
    base_id = base_name.replace("_VOI.nii.gz", "")
    mask_name = base_id + "_mask.nii.gz"
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

        features = extractor.execute()  # Must capture the return value, don't treat as bool
        # Debug print to confirm features output
        print(f"Features keys: {list(features.keys())[:5]} ...")

        glcm = extractor.glcm_raw_matrix
        if glcm is None:
            print(f"❌ GLCM matrix not calculated for {base_id}, skipping.")
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

# Save index CSV
df = pd.DataFrame(records)
csv_path = os.path.join(output_dir, "glcm_index.csv")
df.to_csv(csv_path, index=False)
print(f"\n📄 Done! Index CSV saved to: {csv_path}")
