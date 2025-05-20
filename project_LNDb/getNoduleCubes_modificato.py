import numpy as np
import copy
from matplotlib import pyplot as plt
from utils import readMhd, readCsv, getImgWorldTransfMats, convertToImgCoord, extractCube
from readNoduleList import nodEqDiam
import SimpleITK as sitk
dispFlag = False
save_nii = True

# Read nodules csv
csvlines = readCsv(r"E:\DATA\LNDb Dataset\trainset_csv\trainNodules_gt.csv")  # Modify this to the path of your trainNodules_gt.csv
header = csvlines[0]
nodules = csvlines[1:]

lndloaded = -1
for n in nodules:
    vol = float(n[header.index('Volume')])
    if nodEqDiam(vol) > 3:  # only get nodule cubes for nodules > 3mm
        ctr = np.array([float(n[header.index('x')]), float(n[header.index('y')]), float(n[header.index('z')])])
        lnd = int(n[header.index('LNDbID')])
        rads = list(map(int, list(n[header.index('RadID')].split(','))))
        radfindings = list(map(int, list(n[header.index('RadFindingID')].split(','))))
        finding = int(n[header.index('FindingID')])

        print(lnd, finding, rads, radfindings)

        # Read scan
        if lnd != lndloaded:
            [scan, spacing, origin, transfmat] = readMhd(r'C:\Users\jiaoj\Desktop\LNDb\LNDb-{:04}.mhd'.format(lnd))  # Modify to the path of your mhd file
            transfmat_toimg, transfmat_toworld = getImgWorldTransfMats(spacing, transfmat)
            lndloaded = lnd

        # Convert coordinates to image
        ctr = convertToImgCoord(ctr, origin, transfmat_toimg)

        for rad, radfinding in zip(rads, radfindings):
            # Read segmentation mask
            [mask, _, _, _] = readMhd(r'E:\DATA\LNDb Dataset\masks\masks\LNDb-{:04}_rad{}.mhd'.format(lnd, rad))  # Modify to your mask mhd file path

            # Extract cube around nodule
            scan_cube = extractCube(scan, spacing, ctr)
            masknod = copy.copy(mask)
            masknod[masknod != radfinding] = 0
            masknod[masknod > 0] = 1
            mask_cube = extractCube(masknod, spacing, ctr)

            # Display mid slices from resampled scan/mask
            if dispFlag:  # Default is False, you can set it to True to visualize
                fig, axs = plt.subplots(2, 3)
                axs[0, 0].imshow(scan_cube[int(scan_cube.shape[0] / 2), :, :])
                axs[1, 0].imshow(mask_cube[int(mask_cube.shape[0] / 2), :, :])
                axs[0, 1].imshow(scan_cube[:, int(scan_cube.shape[1] / 2), :])
                axs[1, 1].imshow(mask_cube[:, int(mask_cube.shape[1] / 2), :])
                axs[0, 2].imshow(scan_cube[:, :, int(scan_cube.shape[2] / 2)])
                axs[1, 2].imshow(mask_cube[:, :, int(mask_cube.shape[2] / 2)])
                plt.show()

            if save_nii:  # Save as NIfTI format
                scan_cube_sitk = sitk.GetImageFromArray(scan_cube.astype(np.float32))

                # Set spatial metadata (important step!)
                scan_cube_sitk.SetSpacing(spacing)  # voxel spacing
                scan_cube_sitk.SetOrigin(origin)  # world coordinate origin
                # If direction matrix is not identity, you should also set it:
                # scan_cube_sitk.SetDirection(direction.flatten().tolist())

                # Save as compressed NIfTI
                sitk.WriteImage(scan_cube_sitk, f"cube_nii\\scan_cube_LNDb-{lnd:04}_finding{finding}_rad{rad}.nii.gz")

                # ---------------------------------------------------------------------
                # Save mask_cube as .nii.gz
                # ---------------------------------------------------------------------
                mask_cube_sitk = sitk.GetImageFromArray(mask_cube.astype(np.uint8))  # binary mask uses uint8
                mask_cube_sitk.SetSpacing(spacing)
                mask_cube_sitk.SetOrigin(origin)
                sitk.WriteImage(mask_cube_sitk, f"cube_nii\\mask_cube_LNDb-{lnd:04}_finding{finding}_rad{rad}.nii.gz")

            # Save mask cubes as .npy file, you can comment out this if not needed
            np.save('mask_cubes/LNDb-{:04d}_finding{}_rad{}.npy'.format(lnd, finding, rad), mask_cube)

