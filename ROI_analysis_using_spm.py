import numpy as np
import pandas as pd
import os
from nilearn.input_data import NiftiMasker
from nilearn.image import load_img
from scipy.stats import ttest_ind
import glob
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
#warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')

#function to extract mean values from the SPM con file
def extract_mean_spmT_values(subjects, spmT_dir, roi_mask, spm_file):
    mean_spmT_values = []
    masker = NiftiMasker(mask_img=roi_mask)

    for subject in subjects:
            #depending on the contrast that needs to be analyzed
        spmT_file = spmT_dir + f"{subject}" + "/" + f"{spm_file}"
            #compute the mean within the mask
        spmT_img = load_img(spmT_file)
        mean_spmT_value = masker.fit_transform(spmT_img).mean(axis=1)
        mean_spmT_values.append(mean_spmT_value)
        
    return np.array(mean_spmT_values).flatten()

#function to perform ROI analysis for the ROI nifti masks
def perform_roi_analysis(control_subjects, cannabis_subjects, spmT_dir, mask_list, cannabis_age, control_age):

    #Create a data frame to store pvalue, ts cores for each ROI mask
    #results_df = pd.DataFrame(columns=['ROI','f-stat','p-value'])
    results = []
    mean_values_dict = {'ROI': [], 'Mean_SPMT':[], 'Group': [], 'Age':[]}
    contrast = 'con_0001'
    spm_file = contrast + '.nii'
#Run for each ROI mask
    for mask in mask_list:
        roi_mask = f"/Volumes/LaCie/arush/analysis/uow/uow_MIDT/new_analysis_17042024/masks_from_em/{mask}"
        masker = NiftiMasker(mask_img=roi_mask, smoothing_fwhm=None)
        print(f"Processing mask: {roi_mask}")
        control_mean_spmT_values = extract_mean_spmT_values(control_subjects, spmT_dir, roi_mask, spm_file)
        cannabis_mean_spmT_values = extract_mean_spmT_values(cannabis_subjects, spmT_dir, roi_mask, spm_file)

        roi_data = pd.DataFrame({
            'Mean_SPMT': np.concatenate((control_mean_spmT_values, cannabis_mean_spmT_values)),
            'Group': ['Control'] * len(control_mean_spmT_values) + ['Cannabis'] * len(cannabis_mean_spmT_values),
            'Age': np.concatenate((control_age,cannabis_age))
        })
        model = smf.ols('Mean_SPMT ~ Group + Age', data=roi_data).fit()
        ancova_results = sm.stats.anova_lm(model, type=2)
        print(f'ANCOVA results for {mask}:\n', ancova_results)

        #Extract P values, F-stat for the group effect
        f_stat = ancova_results.loc['Group', 'F']
        p_value_f = ancova_results.loc['Group', 'PR(>F)']

        t_stat = model.tvalues['Group[T.Control]']
        p_value_t = model.pvalues['Group[T.Control]']
        #t_stat, p_value = ttest_ind(cannabis_mean_spmT_values, control_mean_spmT_values)
        #print(f'T-test results: t-statistic = {t_stat}, p-value = {p_value}')

        results.append({'ROI': mask, 'f-stat':f_stat, 'p-value_f':p_value_f, 't-stat':t_stat, 'p-value_t':p_value_t})

        # Interpretation
        #if p_value < 0.05:
           #print("Significant difference between cannabis users and controls in the ROI mask:", roi_mask)
        #else:
            #print("No significant difference between cannabis users and controls in the ROI mask:", roi_mask)
            
        mean_values_dict['ROI'].extend([mask] * len(control_mean_spmT_values))
        mean_values_dict['Mean_SPMT'].extend(control_mean_spmT_values)
        mean_values_dict['Group'].extend(['Control'] * len(control_mean_spmT_values))
        mean_values_dict['Age'].extend(control_age)
        
            
        mean_values_dict['ROI'].extend([mask] * len(cannabis_mean_spmT_values))
        mean_values_dict['Mean_SPMT'].extend(cannabis_mean_spmT_values)
        mean_values_dict['Group'].extend(['Cannabis'] * len(cannabis_mean_spmT_values))
        mean_values_dict['Age'].extend(cannabis_age)
        
    results_df = pd.DataFrame(results)
    return results_df, mean_values_dict, contrast

def main():
    # Directory for all the frist level outputs from SPM
    spmT_dir = 'specify_path'
    out_dir = 'specify_path'
    #List of all the control subjects
    control_subjects = ['sub-001','sub-004','sub-037','sub-039','sub-040','sub-041','sub-042','sub-043',
    'sub-057','sub-069','sub-071','sub-072','sub-077','sub-078','sub-079','sub-080','sub-081','sub-087',
    'sub-102','sub-116','sub-127','sub-128','sub-129','sub-130','sub-131','sub-132','sub-133','sub-135',
    'sub-136']

    #List of all the cannabis subjects
    cannabis_subjects = ['sub-002','sub-006','sub-007','sub-009','sub-014','sub-016','sub-018','sub-019','sub-021',
    'sub-023','sub-024','sub-025','sub-026','sub-027','sub-028','sub-029','sub-031','sub-032','sub-033','sub-034','sub-035','sub-036',
                         'sub-038','sub-046','sub-047','sub-048','sub-049','sub-050','sub-051','sub-052','sub-053','sub-055','sub-056','sub-060','sub-062','sub-064','sub-065','sub-066','sub-067',
    'sub-074','sub-076','sub-083','sub-086','sub-088','sub-089','sub-090','sub-093','sub-094','sub-095',
    'sub-097','sub-098','sub-099','sub-100','sub-101','sub-103','sub-104','sub-107','sub-110','sub-112',
    'sub-115','sub-117','sub-118','sub-119','sub-120','sub-122','sub-123']
    
    control_age = np.array([22.39, 20.73, 25.97, 21.92, 21.38, 20.43, 19.57, 27.06, 19.16, 22.02, 20.01, 18.58,
    21.79, 28.67, 24.99, 20.31, 21.57, 29.41, 25.49, 21.43, 21.14, 20.03, 20.3, 22.81, 22.44, 27.74, 28.58, 17, 13.5,])

    cannabis_age = np.array([24.91, 19.86, 21.39, 26.39, 22.2, 29.85, 22.23, 23.81, 22.05, 21.68, 21.6, 26.22,
    20.85, 19.88, 19.93, 28.44, 23.74, 21.89, 20.35, 20.94, 21.71, 24.48, 27.4, 21.08, 24.09, 22.41,18.89,
    21.76, 22.77,28.51, 19.96, 30.77, 20.3, 19.66, 23.92, 18.47, 20.6, 22.46, 24.95, 18.71, 24.49, 20.61, 25.02, 25.07, 21.1, 20.45, 25.34, 30.31, 22.88, 27.59,
    21.87, 26, 20.15, 30.59, 19.99, 22.75, 28.3, 23.88, 27.72, 22.24, 22.54, 24.63, 32.36, 26.88, 20.65, 20.09,])

    #enter .nii masks for which ROI analysis needs to be performed
    mask_list = ['roi_putamen_l.nii', 'roi_putamen_r.nii', 'roi_insula_l.nii', 'roi_insula_r.nii', 'roi_caudate_l_dor.nii', 'roi_caudate_l_ven.nii', 'roi_ofc.nii', 'roi_cingulum_ant_bil.nii', 'roi_caudate_r_dor.nii', 'roi_caudate_r_ven.nii']
    #mask_list = ['roi_putamen_l.nii']

    #calling fucntion to perfrom ROI analysis
    results_df, mean_values_dict, contrast = perform_roi_analysis(control_subjects, cannabis_subjects, spmT_dir, mask_list, cannabis_age, control_age)
    
    print(results_df)
    #test_float = np.float64(mean_values_dict)
    mean_values_df = pd.DataFrame(mean_values_dict)
    results_df.to_csv(f'specify/path_{contrast}.csv', index=False)  
    plt.figure(figsize=(10,6))
    sns.boxplot(data = mean_values_df, x='ROI', y='Mean_SPMT', hue='Group')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(out_dir, f'{contrast}_boxplot.png'), bbox_inches='tight')
    plt.show()
    #print (mean_values_df)
    return mean_values_df, mean_values_dict
if __name__ == '__main__':
	mean_values_df = main()
