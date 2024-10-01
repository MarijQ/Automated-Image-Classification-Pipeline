import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os

# Define the Excel file names
excel_files = {
    'F1': 'simulate_max_results_F1.xlsx',
    'F2': 'simulate_max_results_F2.xlsx',
    'F3': 'simulate_max_results_F3.xlsx'
}

# Initialize a dictionary to store ROC data and AUC scores
roc_data = {}
auc_scores = {}

# Iterate over each file and extract ROC data
for approach, file in excel_files.items():
    if not os.path.exists(file):
        print(f"Error: File '{file}' not found.")
        continue

    # Read the "ROC Curve Data" sheet
    try:
        df = pd.read_excel(file, sheet_name='ROC Curve Data')
    except ValueError:
        print(f"Error: 'ROC Curve Data' sheet not found in '{file}'.")
        continue

    # Ensure the necessary columns are present
    if not {'False Positive Rate', 'True Positive Rate'}.issubset(df.columns):
        print(f"Error: Required columns not found in 'ROC Curve Data' of '{file}'.")
        continue

    # Extract FPR and TPR
    fpr = df['False Positive Rate'].astype(float)
    tpr = df['True Positive Rate'].astype(float)

    # Compute AUC
    roc_auc = auc(fpr, tpr)
    auc_scores[approach] = roc_auc

    # Store the ROC data
    roc_data[approach] = (fpr, tpr)

# Plotting the ROC curves
plt.figure(figsize=(10, 8))
colors = {'F1': 'blue', 'F2': 'green', 'F3': 'red'}
for approach, (fpr, tpr) in roc_data.items():
    plt.plot(fpr, tpr, color=colors[approach], lw=2,
             label=f'{approach} (AUC = {auc_scores[approach]:.3f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

# Configure plot aesthetics
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curves for Filtering Approaches', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)

# Save the plot to a file
plt.savefig('roc_curves.png')
plt.close()

print("ROC curves have been successfully plotted and saved as 'roc_curves.png'.")
