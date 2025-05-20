import SimpleITK as sitk  # Third-party library, install with pip install SimpleITK if not already installed
# image_path = r'E:\04jianzhi\lung_CT\New_data\LNDb\LNDb-0001.mhd'
# save_path = r'E:\04jianzhi\lung_CT\New_data\data_nii\LNDb-0001.nii.gz'
mask_path = r"E:\DATA\LNDb Dataset\data0\LNDb-0002.mhd"  # Path to the .mhd file to convert
save_path = r"E:\DATA\OUTput\LNDb-0002.nii.gz"  # Path to save the converted file
# Read the .mhd file
image = sitk.ReadImage(mask_path)

# Save as .nii.gz format
sitk.WriteImage(image, save_path)

print("Conversion completed!")
