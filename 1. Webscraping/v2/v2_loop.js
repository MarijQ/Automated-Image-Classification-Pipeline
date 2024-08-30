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

async function processImageIdentifiers(page, numImages, outputDir) {
    let downloadedImages = 0;

    while (downloadedImages < numImages) {
        const { imageElements, debugInfo } = await page.evaluate(() => {
            const elements = Array.from(document.querySelectorAll('[jsname="dTDiAc"]'));
            if (elements.length === 0) return { imageElements: [], debugInfo: [] };

            const firstElement = elements[0];
            const docId = firstElement.getAttribute('data-docid');
            const debugInfo = [];

            // Correctly navigate to the anchor element within nested classes
            debugInfo.push(`Anchor inner HTML: ${firstElement.outerHTML}`);
            const imageUrl = firstElement.querySelector('div > h3 > a')?.outerHTML;


            return { imageElements: [{ docId, imageUrl }], debugInfo };
        });

        // Log the debug information
        debugInfo.forEach(info => console.log(info));

        console.log(imageElements);

        for (const { docId, imageUrl } of imageElements) {
            if (downloadedImages >= numImages) break;

            console.log(`Full URL: ${imageUrl} | Doc ID: ${docId}`);
            downloadedImages++;
        }

        await page.evaluate(() => window.scrollBy(0, 10000));
        await new Promise(r => setTimeout(r, 200));
    }
}


// Function to set up and scrape images
async function scrapeImages(keyword, numImages) {
    // Delete + recreate output directory
    const outputDir = './output';
    if (fs.existsSync(outputDir)) {
        fs.rmSync(outputDir, {recursive: true, force: true});
    }
    fs.mkdirSync(outputDir);

    // Load browser and navigate to google
    const browser = await puppeteer.launch({headless: false});
    const page = await browser.newPage();
    const first_page = 'https://www.google.com/search?q=watermelon&sca_esv=1994aae6371246d6&sca_upv=1&biw=1090&bih=911&udm=2&ei=d6poZo29L-rXhbIPyIeEoAg&ved=0ahUKEwjN1IiZqdSGAxXqa0EAHcgDAYQQ4dUDCBA&uact=5&oq=watermelon&gs_lp=Egxnd3Mtd2l6LXNlcnAiCndhdGVybWVsb24yCBAAGIAEGLEDMggQABiABBixAzIFEAAYgAQyCBAAGIAEGLEDMggQABiABBixAzIIEAAYgAQYsQMyCBAAGIAEGLEDMggQABiABBixAzIFEAAYgAQyBRAAGIAESKYRUKYGWN8PcAJ4AJABAJgBtQGgAYQHqgEDOC4yuAEDyAEA-AEBmAIMoALBB8ICDRAAGIAEGLEDGEMYigXCAgoQABiABBhDGIoFmAMAiAYBkgcEMTAuMqAH2jA&sclient=gws-wiz-serp';
    await page.goto(first_page, {waitUntil: 'networkidle2'});

    // Handle cookie consent popup
    const consentButtonSelector = '[id="L2AGLb"]';
    await page.waitForSelector(consentButtonSelector, {visible: true});
    await page.click(consentButtonSelector);

    // Type in keyword and press enter
    await page.waitForSelector('textarea[name="q"]');
    await page.focus('textarea[name="q"]');
    await page.keyboard.down('Shift');
    await page.keyboard.press('End');
    await page.keyboard.up('Shift');
    await page.keyboard.press('Backspace');
    await page.keyboard.type(keyword);
    await page.keyboard.press('Enter');
    await page.waitForNavigation({waitUntil: 'networkidle2'});

    // Call the function to process image identifiers
    await processImageIdentifiers(page, numImages, outputDir);

    // await browser.close();
}

scrapeImages('fingernail', 1);