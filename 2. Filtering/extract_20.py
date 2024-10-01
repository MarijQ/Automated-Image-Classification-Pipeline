import shutil
import random
import os

folders = ['output_graded_onychrom_100_f1', 'output_graded_onychrom_100_f2', 'output_graded_onychrom_100_f3']
bins = list(range(0, 105, 5))  # 0-100 with bin width 5
samples_dict = {}

def select_and_copy_images(folder, samples_dict, total=20):
    selected = list(samples_dict.get(folder, {}).values())
    if len(selected) < total:
        remaining = set(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.jpg') and len(f) >= 3 and f[:3].isdigit() and 0 <= int(f[:3]) <= 100) - set(selected)
        selected += random.sample(remaining, total - len(selected)) if len(remaining) >= (total - len(selected)) else list(remaining)
    filter_folder = f'filter_{folder}'
    os.makedirs(filter_folder, exist_ok=True)
    for img_path in selected[:total]:
        shutil.copy(img_path, filter_folder)
for folder in folders:
    select_and_copy_images(folder, samples_dict)
