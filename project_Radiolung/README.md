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

3_2. For the Calculate 3D the co-occurrence matrix.py
This repository provides a simple Python script to extract 3D Gray-Level Co-occurrence Matrix (GLCM) features from volumetric VOI (Volume of Interest) data stored in .nii.gz files. The script uses nibabel to load 3D medical image volumes in NIfTI format, quantizes grayscale values into a fixed number of levels (default is 32), and computes 3D GLCMs in 13 spatial directions to capture texture relationships. The normalized GLCM tensor is then saved as a compressed .npz file for further analysis. The script requires Python 3.7+ and the libraries numpy and nibabel, which can be installed via pip install numpy nibabel. To use, prepare your input .nii.gz VOI file, modify the input/output paths in the script or call the main(input_path, output_path) function directly, and run the script. The output is a .npz file containing a 3D GLCM tensor of shape (num_levels, num_levels, 13), where num_levels is the number of quantization levels (default 32) and 13 is the number of 3D directions. The current implementation uses simple nested loops and may be slow on large volumes; future improvements could include vectorization or parallelization for better performance.
