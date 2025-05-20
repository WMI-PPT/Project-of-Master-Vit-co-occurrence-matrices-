Dataset Description (LNDb v4)
This project is based on the LNDb v4 (Lung Nodule Database), which contains 294 chest CT scans collected retrospectively at the Centro Hospitalar e Universitário de São João (CHUSJ) in Porto, Portugal, between 2016 and 2018. All data was acquired with ethical approval from the CHUSJ Ethical Committee and anonymized before analysis, retaining only the patient’s birth year and gender.

🔗 Download the dataset from Zenodo:
👉 https://zenodo.org/records/6613714

👨‍⚕️ Radiologist Annotations
Each CT scan was reviewed by at least one experienced radiologist;

A total of 5 radiologists participated in the annotation process, each with at least 4 years of experience and reviewing up to 30 CTs per week;

All annotations were performed independently (single-read), with no consensus or double-reading;

Annotation guidelines were adapted from the LIDC-IDRI protocol.

Each radiologist labeled three categories of findings:

Nodule ≥3mm: Lesions considered nodules with a longest in-plane dimension ≥3mm;

Nodule <3mm: Lesions considered nodules with a longest in-plane dimension <3mm;

Non-nodule: Lesions not considered nodules but showing nodule-like features.

Annotation details:

Nodules ≥3mm: Fully segmented in 3D and rated on 9 characteristics (subtlety, texture, margin, sphericity, malignancy likelihood, etc.);

Nodules <3mm: Marked by centroid and qualitatively assessed;

Non-nodules: Only centroid coordinates provided.

⚠️ Note: 58 of the 294 CT scans, annotated by at least two radiologists, are withheld as a private test set and are not publicly available.

📂 Data Format
CT Scans: Provided in MetaImage format (.mhd/.raw) with filenames like LNDb-XXXX.mhd;

Radiologist Annotations (trainNodules.csv):

Each row represents one finding by a single radiologist and includes: CT ID, radiologist number, finding ID, (x, y, z) coordinates in world space, nodule label (1/0), volume, and texture rating (1–5; 0 for non-nodules);

Merged Annotations (trainNodules_gt.csv):

Merged findings across radiologists, including consensus level, unified finding ID, and average texture score;

Segmentation Masks:

Files like LNDbXXXX_radR.mhd contain the segmentation masks for nodules marked by radiologist R on scan XXXX. Each voxel value corresponds to a finding ID in trainNodules.csv;

Fleischner Scores (trainFleischner.csv):

Contains one line per scan with the CT ID and its corresponding Fleischner classification for clinical risk assessment.
