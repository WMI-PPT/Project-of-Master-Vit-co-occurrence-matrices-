import os
import xml.etree.ElementTree as ET
import numpy as np
import SimpleITK as sitk
import csv
from tqdm import tqdm

# Parse bounding box from a single XML file
def parse_bbox_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bbox = root.find("object").find("bndbox")
    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)
    return xmin, ymin, xmax, ymax

# (Optional utility) Map SOPInstanceUID to slice index
def get_sop_instance_uid_map(image_series):
    sop_uid_map = {}
    for idx, meta in enumerate(image_series.GetMetaDataKeys()):
        if "0008|0018" in meta:  # SOPInstanceUID tag
            sop_uid = image_series.GetMetaData(meta)
            sop_uid_map[sop_uid] = idx
    return sop_uid_map

# Build a binary mask image from all XML annotations for a given patient
def build_mask_from_xml_annotations(xml_dir, ct_image, sop_uid_to_index):
    arr_shape = sitk.GetArrayFromImage(ct_image).shape  # (Z, Y, X)
    mask_array = np.zeros(arr_shape, dtype=np.uint8)

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue
        sop_uid = os.path.splitext(xml_file)[0]
        if sop_uid not in sop_uid_to_index:
            print(f"Warning: SOP UID {sop_uid} not found in CT slices")
            continue
        z = sop_uid_to_index[sop_uid]
        xmin, ymin, xmax, ymax = parse_bbox_from_xml(os.path.join(xml_dir, xml_file))
        mask_array[z, ymin:ymax, xmin:xmax] = 1

    return sitk.GetImageFromArray(mask_array)

# Extract VOI region from image and mask
def extract_voi(image, mask):
    mask_np = sitk.GetArrayFromImage(mask)
    coords = np.argwhere(mask_np)
    if coords.size == 0:
        return None, None
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0) + 1

    image_np = sitk.GetArrayFromImage(image)
    voi_np = image_np[zmin:zmax, ymin:ymax, xmin:xmax]
    mask_voi_np = mask_np[zmin:zmax, ymin:ymax, xmin:xmax]

    voi_img = sitk.GetImageFromArray(voi_np)
    mask_img = sitk.GetImageFromArray(mask_voi_np)

    voi_img.SetSpacing(image.GetSpacing())
    mask_img.SetSpacing(image.GetSpacing())

    return voi_img, mask_img

# Get the label (class) from patient ID prefix
def get_label_from_id(patient_id):
    type_map = {'A': 0, 'B': 1, 'E': 2, 'G': 3}
    return type_map.get(patient_id[0].upper(), -1)  # -1 if unknown

# Recursively find all DICOM series (leaf folders containing .dcm files)
def find_dicom_series(dicom_root):
    dicom_series = []
    for root, dirs, files in os.walk(dicom_root):
        series_files = [os.path.join(root, f) for f in files if f.lower().endswith(".dcm")]
        if series_files:
            dicom_series.append(series_files)
    return dicom_series

# Main batch processing function
def batch_extract_vois(ct_root, xml_root, output_dir, csv_file):
    os.makedirs(output_dir, exist_ok=True)

    # Write CSV header
    fieldnames = ['patient_id', 'dicom_sequence', 'label', 'voi_ct_file', 'voi_mask_file', 'mask_file']
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    patient_dirs = [d for d in os.listdir(ct_root) if d.startswith("Lung_Dx-")]

    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_id = patient_dir.split("-")[1]
        ct_patient_path = os.path.join(ct_root, patient_dir)

        xml_dir = os.path.join(xml_root, patient_id)
        if not os.path.isdir(xml_dir):
            continue

        # Find all DICOM series in the patient folder
        dicom_series = find_dicom_series(ct_patient_path)
        if not dicom_series:
            continue

        # Iterate over each DICOM series
        for seq_id, series_files in enumerate(dicom_series, start=1):
            try:
                # Read the DICOM image series
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(series_files)
                ct_image = reader.Execute()

                # Build SOPInstanceUID to slice index mapping
                sop_uid_to_index = {}
                for i, fname in enumerate(series_files):
                    meta_reader = sitk.ImageFileReader()
                    meta_reader.SetFileName(fname)
                    meta_reader.LoadPrivateTagsOn()
                    meta_reader.ReadImageInformation()
                    sop_uid = meta_reader.GetMetaData("0008|0018")
                    sop_uid_to_index[sop_uid] = i

                # Skip if no matching XML file
                if not any(os.path.splitext(f)[0] in sop_uid_to_index for f in os.listdir(xml_dir)):
                    continue

                # Build mask and extract VOI
                mask_img = build_mask_from_xml_annotations(xml_dir, ct_image, sop_uid_to_index)
                voi_ct, voi_mask = extract_voi(ct_image, mask_img)
                if voi_ct is None:
                    continue

                # Save NIfTI outputs
                base_name = f"{patient_id}_{seq_id}"
                ct_path = os.path.join(output_dir, f"{base_name}_voi_ct.nii.gz")
                mask_path = os.path.join(output_dir, f"{base_name}_mask.nii.gz")
                voi_mask_path = os.path.join(output_dir, f"{base_name}_voi_mask.nii.gz")

                sitk.WriteImage(mask_img, mask_path)
                sitk.WriteImage(voi_ct, ct_path)
                sitk.WriteImage(voi_mask, voi_mask_path)

                label = get_label_from_id(patient_id)

                # Append entry to CSV
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writerow({
                        'patient_id': patient_id,
                        'dicom_sequence': seq_id,
                        'label': label,
                        'voi_ct_file': ct_path,
                        'voi_mask_file': voi_mask_path,
                        'mask_file': mask_path
                    })

                print(f"Processed {patient_id}_{seq_id} with label {label}")

            except Exception as e:
                print(f"[!] Error processing {patient_id} sequence {seq_id}: {e}")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    ct_root = r"E:\Dataset\lung cancer 1\manifest-1608669183333\Lung-PET-CT-Dx"
    xml_root = r"E:/Dataset/lung cancer 1/manifest-1608669183333/XML/Annotation"
    output_dir = r"E:\LUNG-PET-CT-DX"
    csv_file = r"E:\LUNG-PET-CT-DX\patient_data.csv"

    batch_extract_vois(ct_root, xml_root, output_dir, csv_file)
