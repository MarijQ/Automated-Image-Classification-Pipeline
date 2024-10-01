import os
import pandas as pd
import matplotlib.pyplot as plt

data_dir = '../test_single_cnn_results_'
files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]

# Initialize data structures
training_histories = {}
metrics = []
confusion_matrices = []
roc_curves = {}

for file in files:
    filepath = os.path.join(data_dir, file)
    # Extract method and dataset from filename
    filename_parts = file.replace('.xlsx','').split('_')
    method_parts = filename_parts[4:-1]
    method = '_'.join(method_parts)  # 'simple_cnn', 'transfer_learning', 'tuned_cnn'
    dataset = 'P' + filename_parts[-1]
    approach = method

    # Read Excel sheets
    df_history = pd.read_excel(filepath, sheet_name='Training History')
    df_metrics = pd.read_excel(filepath, sheet_name='Metrics')
    df_cm = pd.read_excel(filepath, sheet_name='Confusion Matrix', index_col=0)
    df_roc = pd.read_excel(filepath, sheet_name='ROC Curve')

    # Store data
    key = method + '_' + dataset
    training_histories[key] = df_history
    metrics_data = {
        'Approach': method,
        'Dataset': dataset,
        'Accuracy': df_metrics['Accuracy'].values[0],
        'Precision': df_metrics['Precision'].values[0],
        'Recall': df_metrics['Recall'].values[0],
        'F1-Score': df_metrics['F1 Score'].values[0],
        'Training Time': df_metrics['Training Time (s)'].values[0],
        'Number of Parameters': None  # Update with actual values if available
    }
    metrics.append(metrics_data)
    confusion_matrices.append({'Method': method, 'Dataset': dataset, 'Confusion Matrix': df_cm})
    roc_curves[key] = df_roc

# Ensure charts directory exists
charts_dir = './charts'
if not os.path.exists(charts_dir):
    os.makedirs(charts_dir)

# Plot training and validation accuracy and loss over epochs for methods with P2
methods = ['simple_cnn', 'tuned_cnn', 'transfer_learning']
dataset = 'P2'

for method in methods:
    key = method + '_' + dataset
    if key in training_histories:
        df_history = training_histories[key]
        epochs = range(1, len(df_history) + 1)
        plt.figure(figsize=(12, 5))
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, df_history['accuracy'], label='Training Accuracy')
        plt.plot(epochs, df_history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Accuracy over Epochs for {method} on {dataset}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, df_history['loss'], label='Training Loss')
        plt.plot(epochs, df_history['val_loss'], label='Validation Loss')
        plt.title(f'Loss over Epochs for {method} on {dataset}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        # Save the figure
        filename = f'training_validation_{method}_{dataset}.png'
        plt.savefig(os.path.join(charts_dir, filename))
        plt.close()

# Create confusion matrices Excel file
output_cm_filename = 'confusion_matrices.xlsx'
with pd.ExcelWriter(output_cm_filename) as writer:
    for cm_data in confusion_matrices:
        method = cm_data['Method']
        dataset = cm_data['Dataset']
        df_cm = cm_data['Confusion Matrix']
        sheet_name = f'{method}_{dataset}'
        df_cm.to_excel(writer, sheet_name=sheet_name)

# Plot ROC curves for methods with P2
plt.figure()
for method in methods:
    key = method + '_' + dataset
    if key in roc_curves:
        df_roc = roc_curves[key]
        plt.plot(df_roc['False Positive Rate'], df_roc['True Positive Rate'], label=method)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curves for methods on {dataset}')
plt.legend()
filename = f'roc_curves_methods_on_{dataset}.png'
plt.savefig(os.path.join(charts_dir, filename))
plt.close()

# Consolidate performance metrics into an Excel file
df_metrics_summary = pd.DataFrame(metrics)
df_metrics_summary = df_metrics_summary[['Approach', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Time', 'Number of Parameters']]
metrics_filename = 'performance_metrics_summary.xlsx'
df_metrics_summary.to_excel(metrics_filename, index=False)

# Plot ROC curves for transfer_learning method across datasets
datasets = ['P1', 'P2', 'P3']
method = 'transfer_learning'
plt.figure()
for dataset in datasets:
    key = method + '_' + dataset
    if key in roc_curves:
        df_roc = roc_curves[key]
        plt.plot(df_roc['False Positive Rate'], df_roc['True Positive Rate'], label=dataset)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curves for {method} across datasets')
plt.legend()
filename = f'roc_curves_{method}_across_datasets.png'
plt.savefig(os.path.join(charts_dir, filename))
plt.close()
