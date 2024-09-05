import os
import json
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

        # Load layout configuration from JSON file
        with open('layout_config.json') as config_file:
            config = json.load(config_file)
            self.create_buttons(config['buttons'])

        # Optional: Adjust grid configuration
        self.root.grid_rowconfigure(0, weight=1)
        for i in range(3):
            self.root.grid_columnconfigure(i, weight=1)

    def create_buttons(self, button_config):
        for i, button in enumerate(button_config):
            if button["command"] == "quit":
                cmd = self.root.quit  # Directly refer to the quit function
            else:
                cmd = getattr(self, button["command"])  # For other usual commands

            btn = tk.Button(self.root, text=button["text"], padx=10, pady=10, command=cmd)
            btn.grid(row=1 + (i // 3), column=i % 3, padx=(10, 10), pady=(10, 10), sticky="ew")

    def load_image(self, index):
        """Load an image for display in the UI."""
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
        image_path = os.path.join(self.image_dir, self.image_filenames[self.current_image_index])
        base_name = os.path.basename(image_path)
        new_filename = f"health_{base_name.split('_')[1]}"
        
        output_dir = os.path.join(os.path.dirname(__file__), "output_labelled")
        os.makedirs(output_dir, exist_ok=True)

        new_path = os.path.join(output_dir, new_filename)

        shutil.copy(image_path, new_path)

        # Optional confirmation message
        print(f"Image '{base_name}' copied as '{new_filename}' into '{output_dir}'.")

        self.forward()

    def on_unhealthy_clicked(self):
        image_path = os.path.join(self.image_dir, self.image_filenames[self.current_image_index])
        base_name = os.path.basename(image_path)
        new_filename = f"unhealthy_{base_name.split('_')[1]}"
        
        output_dir = os.path.join(os.path.dirname(__file__), "output_labelled")
        os.makedirs(output_dir, exist_ok=True)

        new_path = os.path.join(output_dir, new_filename)
        shutil.copy(image_path, new_path)

        # Optional confirmation message
        print(f"Image '{base_name}' copied as '{new_filename}' into '{output_dir}'.")
        
        self.forward()

    def on_discard_clicked(self):
        image_path = os.path.join(self.image_dir, self.image_filenames[self.current_image_index])
        print(f"Image '{self.image_filenames[self.current_image_index]}' has been discarded.")
        del self.image_filenames[self.current_image_index]

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
