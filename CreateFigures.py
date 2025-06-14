import numpy as np
import nibabel as nib
import pandas as pd
import os

def create_nifti(muse_csv_path, nifti_template_path, data):
    """
    Generate a NIfTI image with data values mapped to brain ROIs.

    Parameters:
    - muse_csv_path (str): Path to the MUSE ROI CSV file.
    - nifti_template_path (str): Path to the template NIfTI file (.nii or .nii.gz).
    - data (DataFrame): DataFrame with data values, must contain 'ROI_Index' and their values.

    Returns:
    - nifti image
    """
    muse_roi = pd.read_csv(muse_csv_path)
    merged_muse = pd.merge(muse_roi, data, on='ROI_Index')
    roi_to_shap = dict(zip(merged_muse['ROI_Index'], merged_muse['ROI_value']))
    
    template_nifti = nib.load(nifti_template_path)
    template_data = np.round(template_nifti.get_fdata())
    roi_image = np.zeros_like(template_data)
    
    for roi_val, shap_val in roi_to_shap.items():
        voxel_indices = np.argwhere(template_data == roi_val)
        for x, y, z in voxel_indices:
            roi_image[x, y, z] = shap_val

    roi_nifti = nib.Nifti1Image(roi_image, affine=template_nifti.affine, header=template_nifti.header)
    return roi_nifti