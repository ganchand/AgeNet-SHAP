import os
import numpy as np
import pandas as pd
import pandas as pd
import os 
import numpy as np
from scipy.stats import pearsonr, spearmanr,kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series
from numpy import var, mean, sqrt
from scipy import stats
import nibabel as nib
from statsmodels.stats.multitest import multipletests

from CreateFigures import create_nifti

def calculate_spearman_corr_pval(df, target_column):
    """Calculates Spearman's correlation and p-value for all columns with the target column."""

    result = pd.DataFrame(columns=['Column', 'Correlation', 'P-value'])

    for column in df.columns:
        if column != target_column:
            corr, pval = spearmanr(df[target_column], df[column])
            result = result.append({'Column': column, 'Correlation': corr, 'P-value': pval}, ignore_index=True)

    return result

cn_AvgSHAP_CDR_brainage = pd.read_csv('SHAP values of ROIs belonging to CN category.csv')  # format should be CDRSB, ROI1, ROI2, ..., ROIlast
mci_AvgSHAP_CDR_brainage = pd.read_csv('SHAP values of ROIs belonging to MCI category.csv')  # format should be CDRSB, ROI1, ROI2, ..., ROIlast
ad_AvgSHAP_CDR_brainage = pd.read_csv('SHAP values of ROIs belonging to AD category.csv')  # format should be CDRSB, ROI1, ROI2, ..., ROIlast
all_avg = pd.read_csv('SHAP values of ROIs belonging to All diagnositic categories.csv')  # format should be CDRSB, ROI1, ROI2, ..., ROIlast

target_column = 'CDRSB'

cn_result = calculate_spearman_corr_pval(cn_AvgSHAP_CDR_brainage, target_column)
mci_result = calculate_spearman_corr_pval(mci_AvgSHAP_CDR_brainage, target_column)
ad_result = calculate_spearman_corr_pval(ad_AvgSHAP_CDR_brainage, target_column)
all_result = calculate_spearman_corr_pval(all_avg, target_column)

print('CN')
cn_result = cn_result[cn_result['P-value']<0.05]
print(cn_result.sort_values(by='Correlation', ascending=False))
print('\n')

print('MCI')
mci_result = mci_result[mci_result['P-value']<0.05]
print(mci_result.sort_values(by='Correlation', ascending=False))
print('\n')

print('AD')
ad_result = ad_result[ad_result['P-value']<0.05]
print(ad_result.sort_values(by='Correlation', ascending=False))
print('\n')

print('ALL')
all_result = all_result[all_result['P-value']<0.05]
print(all_result.sort_values(by='Correlation', ascending=False))
print('\n')

cn_result['ROI_Index'] = cn_result['Column'].str.replace('R', '', regex=False).astype(int)
cn_result['ROI_value'] = cn_result['Correlation'].copy()

mci_result['ROI_Index'] = mci_result['Column'].str.replace('R', '', regex=False).astype(int)
mci_result['ROI_value'] = mci_result['Correlation'].copy()

ad_result['ROI_Index'] = ad_result['Column'].str.replace('R', '', regex=False).astype(int)
ad_result['ROI_value'] = ad_result['Correlation'].copy()

all_result['ROI_Index'] = all_result['Column'].str.replace('R', '', regex=False).astype(int)
all_result['ROI_value'] = all_result['Correlation'].copy()

# create nifti figures
muse_csv_path = 'path to muse csv file'
nifti_template_path = 'path to nifti template .nii/.nii.gz'

cn_cdrsb_nifti = create_nifti(muse_csv_path, nifti_template_path, cn_result)  # save this image to get the CDR-SB correlation with SHAP values image for CN data 

mci_cdrsb_nifti = create_nifti(muse_csv_path, nifti_template_path, mci_result)  # save this image to get the CDR-SB correlation with SHAP values image for MCI data 

ad_cdrsb_nifti = create_nifti(muse_csv_path, nifti_template_path, ad_result)  # save this image to get the CDR-SB correlation with SHAP values image for AD data 

all_cdrsb_nifti = create_nifti(muse_csv_path, nifti_template_path, all_result)  # save this image to get the CDR-SB correlation with SHAP values image for entire data 