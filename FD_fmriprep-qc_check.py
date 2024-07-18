#!/usr/bin/env python
#This code reads the TSV file from the outputs of fmriprep and evaluates Frame-wise dispalcement value based on Linden et al - https://pubmed.ncbi.nlm.nih.gov/29278773/

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#Specify the derivatives folder path
DataDir = '/Path/'


#output fodler for the .csv file
OutDir = '/Path/'


#enter number of sessions
session = ['1']
#enter subject IDS
subID = ['sub-006','sub-007','sub-009','sub-020','sub-021','sub-025','sub-032','sub-074','sub-088']
Percentage = np.zeros((len(subID), len(session)))
meanFD =  np.zeros((len(subID), len(session)))
GrtThan = np.zeros((len(subID), len(session)), dtype=int)
All_FD = [[None]*len(session) for _ in range(len(subID))]
IDs = [[None]*len(session) for _ in range(len(subID))]



for i in range(len(subID)):
    session = ['1']
    for j in range(len(session)):
        # Read in .tsv file
        #file_path = os.path.join(DataDir, f"{subID[i]}/ses-{session[j]}/func/{subID[i]}_ses-{session[j]}_task-CR_desc-confounds_timeseries.tsv")
        file_path = os.path.join(DataDir, f"{subID[i]}", f"ses-{session[j]}", "func", f"{subID[i]}_ses-1_task-MIDT_desc-confounds_timeseries.tsv")
        df = pd.read_csv(file_path, sep='\t')
        # Get FD traces
        FD_noNA = df['framewise_displacement'][1:]

        # Calc mean FD. Remove if mean FD > 0.25 mm
        #print (i,j)
        meanFD[i,j] = sum(FD_noNA) / len(FD_noNA)

        # Determine % of FD values more than 20%.
        # Remove if more than 20% of the FDs are above 0.2 mm
        Percentage[i,j] = (sum(fd > 0.2 for fd in FD_noNA) / len(FD_noNA)) * 100

        # Count no. of FD. Remove if any FDs are greater than 5 mm
        GrtThan[i,j] = sum(fd > 5 for fd in FD_noNA)

        All_FD[i][j] = FD_noNA

        num_rows = len(df)

        print("Number of rows:", num_rows)
        IDs[i][j] = subID[i]

data = {'subID': [item for sublist in IDs for item in sublist],
        'meanFD': [item for sublist in meanFD for item in sublist],
        'PercentageAbove02': [item for sublist in Percentage for item in sublist],
        'FDMoreThan5': [item for sublist in GrtThan for item in sublist]}
Table = pd.DataFrame(data)

# Save DataFrame as CSV
Table.to_csv(os.path.join(OutDir, 'FD_Stats_session1_v2.csv'), index=False)

# Visualisation of the FD results
# Create a box plot
plt.figure(figsize=(8, 6))
plt.boxplot(meanFD, vert=False)
plt.xlabel('Mean FD')
plt.title('Box plot of Mean FD')
plt.yticks([])
plt.grid(True)
#plt.savefig(os.path.join(OutDir, 'box_plot.png'), bbox_inches='tight')
plt.show()

# Create the violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(meanFD, inner='quartile')
plt.xlabel('Mean FD Value')
plt.ylabel('Density')
plt.title('Violin Plot of Mean FD Values')
plt.grid(True)
plt.show()

# Create the ridgeline plot
plt.figure(figsize=(10, 6))
sns.histplot(meanFD, kde=True, stat="density", linewidth=0)
plt.xlabel('Mean FD Value')
plt.ylabel('Density')
plt.title('Ridgeline Plot of Mean FD Values')
plt.grid(True)
#plt.savefig(os.path.join(OutDir, 'ridgeline_plot.png'), bbox_inches='tight')
plt.show()
