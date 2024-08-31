import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service  # Import Service
from selenium.webdriver.common.by import By

# Setup WebDriver
driver_path = 'G:/My Drive/M5. MSc Data Science and Analytics/15. Dissertation (CS5500)/Web-scraping/Selenium/chrome-webdriver/chromedriver.exe' 
print(driver_path)
service = Service(executable_path=driver_path)  # Create a Service object
driver = webdriver.Chrome(service=service)

# Open the target website
driver.get('https://www.mayoclinic.org/healthy-lifestyle/adult-health/in-depth/7-fingernail-problems-not-to-ignore/art-20546860')

# Create directory for images
os.makedirs('downloaded_images', exist_ok=True)  # Ensure folder name matches with the one used below

# Find all images
images = driver.find_elements(By.TAG_NAME, 'img')

# Download images
for idx, image in enumerate(images):
    src = image.get_attribute('src')
    if src:
        image_name = f'downloaded_images/image_{idx}.jpg'
        driver.get(src)
        driver.save_screenshot(image_name)
        time.sleep(1)  # Sleep to avoid rapid requests

# Close the browser
driver.quit()
