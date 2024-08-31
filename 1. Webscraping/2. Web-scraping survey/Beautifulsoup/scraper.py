import requests
from bs4 import BeautifulSoup
import os

# URL of the webpage
url = 'https://www.mayoclinic.org/healthy-lifestyle/adult-health/in-depth/7-fingernail-problems-not-to-ignore/art-20546860'

# Make a directory to store the images
os.makedirs('downloaded_images', exist_ok=True)

# Send a HTTP request to the URL
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all image tags
images = soup.find_all('img')

# Make a directory to store the images
os.makedirs('output', exist_ok=True)  # Updated to point to 'output' subfolder

# Download each image
for img in images:
    src = img.get('src')
    if src:
        # Complete the src with the base URL if needed
        if not src.startswith('http'):
            src = 'https://www.mayoclinic.org' + src
        # Get the image content
        img_response = requests.get(src)
        # Save the image
        img_name = os.path.join('output', src.split('/')[-1])  # Updated to save in 'output' folder
        with open(img_name, 'wb') as f:
            f.write(img_response.content)
        print(f'Downloaded {img_name}')

print("All images have been downloaded successfully.")
