// imports
const fs = require('fs');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const axios = require('axios');
puppeteer.use(StealthPlugin());

// Function to download image
async function downloadImage(url, filepath) {
    const response = await axios({url, method: 'GET', responseType: 'stream'});
    return new Promise((resolve, reject) => {
        const writeStream = fs.createWriteStream(filepath);
        response.data.pipe(writeStream);
        writeStream.on('finish', () => resolve(`Successfully downloaded ${filepath}`));
        writeStream.on('error', reject);
    });
}

// Function to scrape images
async function scrapeImages(keyword, numImages) {
    // Create output directory if it doesn't exist
    const outputDir = './output';
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir);
    }

    // Load browser and navigate to google
    const browser = await puppeteer.launch({headless: false});
    const page = await browser.newPage();
    const first_page = 'https://www.google.com/search?q=watermelon&sca_esv=1994aae6371246d6&sca_upv=1&biw=1090&bih=911&udm=2&ei=d6poZo29L-rXhbIPyIeEoAg&ved=0ahUKEwjN1IiZqdSGAxXqa0EAHcgDAYQQ4dUDCBA&uact=5&oq=watermelon&gs_lp=Egxnd3Mtd2l6LXNlcnAiCndhdGVybWVsb24yCBAAGIAEGLEDMggQABiABBixAzIFEAAYgAQyCBAAGIAEGLEDMggQABiABBixAzIIEAAYgAQYsQMyCBAAGIAEGLEDMggQABiABBixAzIFEAAYgAQyBRAAGIAESKYRUKYGWN8PcAJ4AJABAJgBtQGgAYQHqgEDOC4yuAEDyAEA-AEBmAIMoALBB8ICDRAAGIAEGLEDGEMYigXCAgoQABiABBhDGIoFmAMAiAYBkgcEMTAuMqAH2jA&sclient=gws-wiz-serp'
    await page.goto(first_page, {waitUntil: 'networkidle2'});

    // Handle cookie consent popup
    const consentButtonSelector = '[id="L2AGLb"]';
    await page.waitForSelector(consentButtonSelector, {visible: true});
    await page.click(consentButtonSelector);

    // Type in keyword and press enter
    await page.waitForSelector('textarea[name="q"]'); // Ensure the textarea is loaded
    await page.focus('textarea[name="q"]'); // Focus on the textarea
    await page.keyboard.down('Shift');
    await page.keyboard.press('End');
    await page.keyboard.up('Shift');
    await page.keyboard.press('Backspace'); // Clear existing text
    await page.keyboard.type(keyword); // Type the keyword
    await page.keyboard.press('Enter'); // Press 'Enter'
    await page.waitForNavigation({waitUntil: 'networkidle2'});

    // Scroll to trigger images loading
    let loadedImages = 0;
    while (loadedImages < numImages) {
        await page.evaluate(() => {
            window.scrollBy(0, 1e5);
        });

        // Wait for at least 1 more image to load
        await page.waitForFunction(
            (currentLoadedImages) => {
                const imgs = document.querySelectorAll('img');
                return imgs.length >= currentLoadedImages;
            },
            {},
            loadedImages
        );
        loadedImages = await page.$$eval('img', imgs => imgs.length); // Get the number of images loaded
    }

    console.log(loadedImages, numImages)

    // Extract image URLs
    const imageElements = await page.$$eval('img', imgs => imgs.map(img => img.src));

    // Filter out non-image URLs and slice to the desired number of images
    const imageUrls = imageElements.filter(url => url.startsWith('http')).slice(0, numImages);

    // Download images
    for (let i = 0; i < imageUrls.length; i++) {
        const imageUrl = imageUrls[i];
        const filepath = `${outputDir}/image${i}.jpg`; // Update path to use 'output' folder
        try {
            const result = await downloadImage(imageUrl, filepath);
            console.log(result);
        } catch (error) {
            console.error(`Failed to download image ${imageUrl}: ${error}`);
        }
    }

    // await browser.close();
}

scrapeImages('fingernail', 1000);
