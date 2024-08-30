import os
import random

def randomize_filenames(input_dir):
    """Randomly assigns scores to images and renames them accordingly."""
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Generate random scores for each filter (between 0 and 100)
            ocr_score = random.randint(0, 100)
            color_score = random.randint(0, 100)
            quality_score = random.randint(0, 100)
            edge_score = random.randint(0, 100)

            # Create a new filename with the random scores
            new_filename = f"image_ocr_{ocr_score}_color_{color_score}_quality_{quality_score}_edges_{edge_score}{os.path.splitext(filename)[1]}"

            # Rename the file
            old_file_path = os.path.join(input_dir, filename)
            new_file_path = os.path.join(input_dir, new_filename)
            os.rename(old_file_path, new_file_path)

            print(f"Renamed {filename} to {new_filename}")

if __name__ == "__main__":
    input_dir = 'input'
    randomize_filenames(input_dir)
