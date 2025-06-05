# Project-of-Master-Vit-co-occurrence-matrices-
1. For the Extract voi.py:
Description:
This script extracts 3D Volume of Interest (VOI) regions from the Radiolung dataset using .acsv annotation files. It reads the center coordinates and size from the annotation and crops the corresponding region from the CT scan.
Input:
CT scan files in .nii.gz format
Annotation files in .acsv format (defining VOI location and size)
Output:
Extracted VOI images saved as .nii.gz in the extract_voi/ folder
(e.g., Paciente_<id>_voi.nii.gz)

2. For the 32 gray levels.py :
Description:
This script normalizes the gray levels of the VOI images extracted in step 1. It rescales voxel intensity values to a fixed range of 0–31 (32 levels), preparing the data for GLCM extraction.
Input:
VOI images from extract_voi/
Output:
Gray-level normalized VOI images saved in the normalized_voi/ folder as .nii.gz

3_1. For the Calculate the co-occurrence matrix of each 2D slice.py :
Description:
This script computes the GLCM (Gray Level Co-occurrence Matrix) features for each axial slice in the normalized VOIs. It extracts 4-direction GLCMs, flattens them into fixed-length vectors, and saves them in .npz format. A corresponding CSV is also generated for training input.
Input:
Normalized VOI images (normalized_voi/)
Metadata Excel file with patient_id and type columns
Output:
.npz files: Each VOI’s GLCM tensor, stored in the co-occurrence matrices/ folder
glcm_vit_data.csv: A CSV file listing .npz file paths and associated labels (0 for benign, 1 for malignant)

3_2. For the Calculate 3D the co-occurrence matrix

① custom_glcm.py

Extends PyRadiomics RadiomicsGLCM class.
Overrides _calculateMatrix method to save the raw GLCM matrix (self.glcm_raw_matrix) during feature extraction.

② extract_glcm_batch.py

Loads VOI and mask pairs from specified directories.
Extracts GLCM features and raw matrices using CustomGLCM.
Saves raw GLCM matrix to .npy files.
Generates a summary CSV for all processed data.

These two codes are based on PyRadiomics and implement a custom extraction of 3D gray-level co-occurrence matrix (GLCM) features. They support extracting the raw GLCM matrix from 3D medical images and batch processing multiple volumes of interest (VOIs) along with their corresponding masks.
They inherit PyRadiomics’ GLCM class and extend it to extract and save the raw GLCM matrix.
Support batch processing of VOI and mask files in NIfTI format.
Save the extracted raw GLCM matrices as .npy files for convenient further analysis.
Automatically generate an index CSV file that contains paths and results for all processed files.
Support 3D GLCM extraction, with the option to switch between 2D or 3D modes as needed.
All output results have the shape (13，32，32), where 32 is the number of gray levels, and 13 corresponds to the 13 directions.

