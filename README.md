# Project-of-Master-Vit-co-occurrence-matrices-
1. For the Extract voi:
Description:
This script extracts 3D Volume of Interest (VOI) regions from the Radiolung dataset using .acsv annotation files. It reads the center coordinates and size from the annotation and crops the corresponding region from the CT scan.
Input:
CT scan files in .nii.gz format
Annotation files in .acsv format (defining VOI location and size)
Output:
Extracted VOI images saved as .nii.gz in the extract_voi/ folder
(e.g., Paciente_<id>_voi.nii.gz)

2. For the 32 gray levels :
Description:
This script normalizes the gray levels of the VOI images extracted in step 1. It rescales voxel intensity values to a fixed range of 0–31 (32 levels), preparing the data for GLCM extraction.
Input:
VOI images from extract_voi/
Output:
Gray-level normalized VOI images saved in the normalized_voi/ folder as .nii.gz

3. For the Calculate the co-occurrence matrix :
Description:
This script computes the GLCM (Gray Level Co-occurrence Matrix) features for each axial slice in the normalized VOIs. It extracts 4-direction GLCMs, flattens them into fixed-length vectors, and saves them in .npz format. A corresponding CSV is also generated for training input.
Input:
Normalized VOI images (normalized_voi/)
Metadata Excel file with patient_id and type columns
Output:
.npz files: Each VOI’s GLCM tensor, stored in the co-occurrence matrices/ folder
glcm_vit_data.csv: A CSV file listing .npz file paths and associated labels (0 for benign, 1 for malignant)
