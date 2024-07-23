from playwright.sync_api import sync_playwright
import os

def download_images():
    base_url = "https://www.mayoclinic.org"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://www.mayoclinic.org/healthy-lifestyle/adult-health/in-depth/7-fingernail-problems-not-to-ignore/art-20546860")

        # Gather all image sources
        images = page.query_selector_all('img')
        image_urls = [image.get_attribute('src') for image in images if image.get_attribute('src')]

        # Ensure output directory exists
        os.makedirs('output', exist_ok=True)

        # Download images
        for index, url in enumerate(image_urls):
            try:
                # Create the full URL by concatenating the base URL with the relative path
                full_url = base_url + url if url.startswith('/') else url

                # Fetch and save image
                response = page.goto(full_url)
                if response.ok:
                    image_content = response.body()
                    with open(f'output/image_{index}.png', 'wb') as file:
                        file.write(image_content)
            except Exception as e:
                print(f"Failed to download {url}: {e}")

        browser.close()

if __name__ == '__main__':
    download_images()
