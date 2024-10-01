import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import math

folders = ['output_graded_onychrom_100_f1', 'output_graded_onychrom_100_f2', 'output_graded_onychrom_100_f3']
bins = list(range(0, 105, 5))  # 0-100 with bin width 5

grades_dict = {}
samples_dict = {}

for folder in folders:
    grades = []
    samples = {}
    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg') and len(filename) >= 4 and filename[:3].isdigit():
            grade = int(filename[:3])
            if 0 <= grade <= 100:
                grades.append(grade)
                bin_index = grade // 5
                if bin_index not in samples and bin_index < 20:
                    samples[bin_index] = os.path.join(folder, filename)
    grades_dict[folder] = grades
    samples_dict[folder] = samples

# Plot Histograms
fig_hist, axes_hist = plt.subplots(len(folders), 1, figsize=(10, 5 * len(folders)))

for i, folder in enumerate(folders):
    axes_hist[i].hist(grades_dict[folder], bins=bins, edgecolor='black')
    axes_hist[i].set_title(f'Histogram of {folder}')
    axes_hist[i].set_xlabel('Grade')
    axes_hist[i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('histograms.png')
plt.close(fig_hist)