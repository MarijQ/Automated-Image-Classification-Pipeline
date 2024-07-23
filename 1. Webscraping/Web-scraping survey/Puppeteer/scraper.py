import asyncio
from pyppeteer import launch
import os

async def download_images():
    # Ensure the output directory exists
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    browser = await launch(headless=True, executablePath='C:/Program Files/Google/Chrome/Application/chrome.exe')
    page = await browser.newPage()
    await page.goto('https://www.mayoclinic.org/healthy-lifestyle/adult-health/in-depth/7-fingernail-problems-not-to-ignore/art-20546860', {'waitUntil': 'networkidle0'})
    images = await page.evaluate('''() => {
        return Array.from(document.images, img => img.src);
    }''')

    for index, image in enumerate(images):
        try:
            viewSource = await page.goto(image)
            buffer = await viewSource.buffer()
            file_name = os.path.join(output_dir, f'image{index}.png')  # Save images in the output folder
            with open(file_name, 'wb') as f:
                f.write(buffer)
            print(f'Downloaded {file_name}')
        except Exception as e:
            print(f'Failed to download {image}: {str(e)}')

    await browser.close()

asyncio.run(download_images())
