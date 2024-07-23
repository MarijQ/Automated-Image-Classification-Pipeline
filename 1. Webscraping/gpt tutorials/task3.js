// Rename file to `.mjs` or add "type": "module" in package.json
import puppeteer from 'puppeteer';
import fs from 'fs';
import path from 'path';
import fetch from 'node-fetch'; // ES Module import

const downloadImage = async (url, outputPath) => {
    const response = await fetch(url);
    const buffer = await response.buffer();
    fs.writeFileSync(outputPath, buffer);
};

async function run() {
    const browser = await puppeteer.launch({ headless: false });
    const page = await browser.newPage();
    await page.goto('https://images.google.com', { waitUntil: 'domcontentloaded' });

    // Handle cookie consent popup
    const consentButtonSelector = '[id="L2AGLb"]'; // This selector might need adjustment based on your location and the specific site
    await page.waitForSelector(consentButtonSelector, { visible: true });
    await page.click(consentButtonSelector);

    const query = 'puppies'; // Change this to your desired search term
    await page.type('input[name="q"]', query);
    await page.keyboard.press('Enter');
    await page.waitForNavigation({ waitUntil: 'networkidle2' });

    const images = await page.evaluate(() => {
        const thumbnails = document.querySelectorAll('img.Q4LuWd');
        return Array.from(thumbnails).map(img => img.src);
    });

    const dir = './task3_output';
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }

    await Promise.all(images.slice(0, 5).map((url, index) => 
        downloadImage(url, path.join(dir, `image${index}.jpg`))
    ));

    await browser.close();
}

run();
