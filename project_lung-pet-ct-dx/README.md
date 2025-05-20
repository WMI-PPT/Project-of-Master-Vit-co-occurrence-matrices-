Lung-pet-ct-dx dataset introduction
The Lung-PET-CT-Dx dataset contains CT and PET-CT DICOM images of patients with suspected lung cancer, along with XML annotation files that mark tumor locations using bounding boxes. All images were retrospectively acquired from patients who underwent standard-of-care PET/CT and lung biopsy, with the final diagnosis based on histopathological tissue analysis.

🧬 Diagnostic Label Mapping
Patients are categorized into four lung cancer types based on their ID, with corresponding numeric labels for classification tasks:

'A' – Adenocarcinoma (label 0)

'B' – Small Cell Carcinoma (label 1)

'E' – Large Cell Carcinoma (label 2)

'G' – Squamous Cell Carcinoma (label 3)

🖼 Imaging Protocol
CT Scans: Acquired with mediastinal (WW=350 HU, WL=40 HU) and lung (WW=1400 HU, WL=–700 HU) window settings, with slice thickness ranging from 0.625 mm to 5 mm. Reconstruction used 2 mm thickness in lung settings.

PET Scans:

Whole-body scans acquired 60 minutes after 18F-FDG injection (4.44 MBq/kg).

FDG dose: 168.72–468.79 MBq (avg. 295.8 ± 64.8 MBq)

Uptake time: 27–171 min (avg. 70.4 ± 24.9 min)

Patients fasted ≥6 hours and had blood glucose <11 mmol/L.

Image Fusion & Reconstruction:

One CT volume, one PET volume, and a fused PET-CT volume per patient.

CT resolution: 512×512 @ 1×1 mm

PET resolution: 200×200 @ 4.07×4.07 mm

Slice thickness: 1 mm for both modalities

PET images reconstructed using TrueX TOF

🧠 Annotation Details
Tumor annotations were created and verified by a team of five experienced thoracic radiologists, two with over 15 years of experience and three with over 5 years. The annotation process included:

Initial labeling by one radiologist

Cross-verification by four others

All annotations saved in PASCAL VOC (XML) format using LabelImg

Compatible with pascal-voc-tools

Python tools are available to overlay bounding boxes on DICOM images for visualization.

🧪 Benchmarking & Model Performance
Two deep learning researchers used this dataset to train and evaluate several well-known object detection models, achieving a mean average precision (mAP) of approximately 0.87 on the validation set.

📂 Data Access
Current Version: v5 (Updated 2020-12-22)

Subjects: 355 (after excluding 8 cases pending further diagnosis)

Clinical Data: Available for all included subjects

Download Link: https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/

Access and download the dataset from The Cancer Imaging Archive (TCIA) via the link above.
