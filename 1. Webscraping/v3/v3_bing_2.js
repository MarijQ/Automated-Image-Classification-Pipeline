// imports
const fs = require('fs');
const axios = require('axios');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const UserAgent = require('user-agents');
puppeteer.use(StealthPlugin());

// Function to generate a random delay
const randomDelay = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;

// Function to download image
async function downloadImage(url, filepath) {
    try {
        const response = await axios({
            url,
            method: 'GET',
            responseType: 'stream',
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        });
        return new Promise((resolve, reject) => {
            const writeStream = fs.createWriteStream(filepath);
            response.data.pipe(writeStream);
            writeStream.on('finish', () => resolve(`Successfully downloaded ${filepath}`));
            writeStream.on('error', reject);
        });
    } catch (error) {
        console.error(`Failed to download ${url}: ${error.message}`);
        throw error; // Rethrow the error for further handling
    }
}

// Function to process image identifiers
async function processImageIdentifiers(page, outputDir, n) {
    const mUrls = await page.evaluate(() => {
        const elements = Array.from(document.querySelectorAll('a.iusc'));
        return elements.map(element => {
            const mContent = element.getAttribute('m');
            const mData = JSON.parse(mContent); // Parse the JSON string
            return mData.murl; // Return the value of 'murl'
        });
    });

    // Limit to the first n URLs
    const limitedUrls = mUrls.slice(0, n);

    // Download images and save them as image1.jpg, image2.jpg, etc.
    for (let i = 0; i < limitedUrls.length; i++) {
        const url = limitedUrls[i];
        const filepath = `${outputDir}/image${i + 1}.jpg`;
        try {
            await downloadImage(url, filepath);
            await new Promise(resolve => setTimeout(resolve, randomDelay(1000, 3000))); // Random delay
        } catch (error) {
            console.error(`Error downloading image ${i + 1}: ${error.message}`);
        }
    }
}

// Function to set up and scrape images
async function scrapeImages(keyword, n) {
    const outputDir = './output';

    // Delete existing output directory if it exists
    if (fs.existsSync(outputDir)) {
        fs.rmSync(outputDir, { recursive: true, force: true });
    }
    fs.mkdirSync(outputDir); // Create a new output directory

    const browser = await puppeteer.launch({ headless: false });
    const page = await browser.newPage();
    const userAgent = new UserAgent();
    await page.setUserAgent(userAgent.toString()); // Random User-Agent
    const first_page = `https://www.bing.com/images/search?q=${encodeURIComponent(keyword)}`;
    await page.goto(first_page, { waitUntil: 'networkidle2' });

    // Call the function to process image identifiers
    await processImageIdentifiers(page, outputDir, n);

    await browser.close();
}

// Specify the keyword and number of images to scrape
scrapeImages('fingernail', 100); // Change 5 to any number of images you want to scrape
