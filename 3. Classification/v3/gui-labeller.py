import os
import shutil
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class ImageLabeller:
	def __init__(self, root, image_dir):
		self.root = root
		self.root.title("image-labeller-3000")

		# Set a fixed window size and disable resizing
		self.window_width = 800
		self.window_height = 600
		self.root.geometry(f"{self.window_width}x{self.window_height}")
		self.root.resizable(False, False)

		self.image_scale = 1.0  # No scaling — display the image as-is

		self.image_dir = image_dir

		# Get list of images
		self.image_filenames = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
		self.current_image_index = 0

		# Create UI elements
		self.init_ui()

		# Load the first image
		self.load_image(self.current_image_index)

	def init_ui(self):
		# Define an area to display the image
		self.image_frame = tk.Frame(self.root, width=self.window_width, height=self.window_height - 100)
		self.image_frame.grid(row=0, column=0, columnspan=3, padx=(0, 0), pady=(0, 0), sticky="nsew")
		self.image_frame.grid_propagate(False)

		# Image label inside the fixed area
		self.image_label = tk.Label(self.image_frame)
		self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Center the label in the frame

		# Create the buttons
		self.healthy_button = tk.Button(self.root, text="HEALTHY", padx=10, pady=10, command=self.on_health_clicked)
		self.unhealthy_button = tk.Button(self.root, text="UNHEALTHY", padx=10, pady=10, command=self.on_unhealthy_clicked)
		self.discard_button = tk.Button(self.root, text="DISCARD", padx=10, pady=10, command=self.on_discard_clicked)
		self.back_button = tk.Button(self.root, text="<", padx=10, pady=10, command=self.backward)
		self.forward_button = tk.Button(self.root, text=">", padx=10, pady=10, command=self.forward)
		self.exit_button = tk.Button(self.root, text="EXIT", padx=10, pady=10, command=self.root.quit)

		# Place the first row of buttons (option selection)
		self.healthy_button.grid(row=1, column=0, padx=(10, 10), pady=(10, 10), sticky="ew")
		self.unhealthy_button.grid(row=1, column=1, padx=(10, 10), pady=(10, 10), sticky="ew")
		self.discard_button.grid(row=1, column=2, padx=(10, 10), pady=(10, 10), sticky="ew")

		# Place the first row of buttons (navigation)
		self.back_button.grid(row=2, column=0, padx=(10, 10), pady=(10, 10), sticky="ew")
		self.forward_button.grid(row=2, column=1, padx=(10, 10), pady=(10, 10), sticky="ew")
		self.exit_button.grid(row=2, column=2, padx=(10, 10), pady=(10, 10), sticky="ew")
		
		# Grid autoconfiguration
		self.root.grid_rowconfigure(0, weight=1)
		self.root.grid_columnconfigure(0, weight=1)
		self.root.grid_columnconfigure(1, weight=1)
		self.root.grid_columnconfigure(2, weight=1)

	def load_image(self, index):
		# Load an image

		if index >= len(self.image_filenames):
			return  # Safety check
		
		image_path = os.path.join(self.image_dir, self.image_filenames[index])
		image = Image.open(image_path)
		self.photo = ImageTk.PhotoImage(image)

		# Update the label with the new image, centered in its frame
		self.image_label.config(image=self.photo)
		self.image_label.image = self.photo

	def forward(self):
		# Move to the next image, wrapping around if at the last image
		self.current_image_index += 1
		if self.current_image_index >= len(self.image_filenames):
			self.current_image_index = 0  # Wrap to the first image
		self.load_image(self.current_image_index)

	def backward(self):
		# Move to the previous image, wrapping around if at the first image
		self.current_image_index -= 1
		if self.current_image_index < 0:
			self.current_image_index = len(self.image_filenames) - 1  # Wrap to the last image
		self.load_image(self.current_image_index)


	def on_health_clicked(self):
		# Get image path
		image_path = os.path.join(self.image_dir, self.image_filenames[self.current_image_index])
		
		# Re-label image
		base_name = os.path.basename(image_path)
		new_filename = f"healthy_{base_name.split('_')[1]}"
		
		# Select output dir
		output_dir = os.path.join(os.path.dirname(__file__), "output_labelled")
		os.makedirs(output_dir, exist_ok=True)

		# Rename (move) the file to the new location with the new name
		new_path = os.path.join(output_dir, new_filename)

		shutil.copy(image_path, new_path)

		# Optional confirmation message
		print(f"Image '{base_name}' copied as '{new_filename}' into '{output_dir}'.")

		# Automatically move to the next image
		self.forward()

	def on_unhealthy_clicked(self):
		# Get image path
		image_path = os.path.join(self.image_dir, self.image_filenames[self.current_image_index])
		
		# Re-label image
		base_name = os.path.basename(image_path)
		new_filename = f"unhealthy_{base_name.split('_')[1]}"
		
		# Select output dir
		output_dir = os.path.join(os.path.dirname(__file__), "output_labelled")
		os.makedirs(output_dir, exist_ok=True)

		# Rename (move) the file to the new location with the new name
		new_path = os.path.join(output_dir, new_filename)

		shutil.copy(image_path, new_path)

		# Optional confirmation message
		print(f"Image '{base_name}' copied as '{new_filename}' into '{output_dir}'.")

		# Automatically move to the next image
		self.forward()

	def on_discard_clicked(self):
		# Delete the current image and move to the next one

		# Get the current image file's path
		image_path = os.path.join(self.image_dir, self.image_filenames[self.current_image_index])

		# Optional confirmation message
		print(f"Image '{self.image_filenames[self.current_image_index]}' has been discarded.")

		# Remove the image from the list of filenames
		del self.image_filenames[self.current_image_index]

		# Move to the next image, or show a popup if this is the last one
		if self.current_image_index < len(self.image_filenames):
			self.load_image(self.current_image_index)
		else:
			messagebox.showinfo("Info", "No more images left.")
			self.root.destroy()

def main():
	root = tk.Tk()
	image_dir = "/home/sunny/Documents/GitHub/MSc-Dissertation-2024-shared/3. Classification/sample_labelled_20"
	ImageLabeller(root, image_dir)
	root.mainloop()

if __name__ == "__main__":
	main()
