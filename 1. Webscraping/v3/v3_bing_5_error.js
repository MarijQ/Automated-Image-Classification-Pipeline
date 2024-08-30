// imports
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const UserAgent = require('user-agents');
puppeteer.use(StealthPlugin());

// Function to generate a random string of letters
const generateRandomString = (length) => {
    const characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
    let result = '';
    for (let i = 0; i < length; i++) {
        result += characters.charAt(Math.floor(Math.random() * characters.length));
    }
    return result;
};

// Function to generate a random delay
const randomDelay = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;

// Function to download image
async function downloadImage(url, filepath) {
    try {
        const response = await axios({
            url,
            method: 'GET',
            responseType: 'stream',
            timeout: 10000, // Set a timeout for the request
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        });

        // Check for successful response
        if (response.status !== 200) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return new Promise((resolve, reject) => {
            const writeStream = fs.createWriteStream(filepath);
            response.data.pipe(writeStream);
            writeStream.on('finish', () => {
                console.log(`Successfully downloaded ${filepath}`);
                resolve();
            });
            writeStream.on('error', (error) => {
                console.error(`Failed to write ${filepath}: ${error.message}`);
                reject(error);
            });
        });
    } catch (error) {
        // Handle specific HTTP errors
        if (error.response) {
            // Check for 403 or 404 errors
            if (error.response.status === 403 || error.response.status === 404 || error.response.status === 400) {
                console.error(`Skipping ${url} due to HTTP error: ${error.response.status}`);
                return null; // Return null to indicate this URL should not be retried
            }
        }
        console.error(`Failed to download ${url}: ${error.message}`);
        throw error; // Rethrow the error for further handling
    }
}

// Function to collect image URLs
async function collectImageUrls(page) {
    let imageUrls = [];

    for (let i = 0; i < 5; i++) {
        // Scroll the page by 10000 pixels
        await page.evaluate(() => {
            window.scrollBy(0, 10000);
        });
        console.log('Scrolled down 10000 pixels.');

        // Wait for a moment to allow new images to load
        await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for 1 second

        // Click the "See more images" button if it exists
        const buttonExists = await page.evaluate(() => {
            const button = document.querySelector('a.btn_seemore');
            if (button) {
                button.click();
                return true;
            }
            return false;
        });

        if (buttonExists) {
            console.log('Clicked "See more images" button.');
        } else {
            console.log('No "See more images" button found.');
        }

        // Scroll the page by 10000 pixels
        await page.evaluate(() => {
            window.scrollBy(0, 10000);
        });
        console.log('Scrolled down 10000 pixels.');

        // Wait for a moment to allow new images to load
        await new Promise(resolve => setTimeout(resolve, 5000)); // Wait for 5 seconds

        // Extract image URLs from the page
        const { newImageUrls, htmlContent } = await page.evaluate(() => {
            const elements = Array.from(document.querySelectorAll('a.iusc'));
            const newImageUrls = elements.map(element => {
                const mContent = element.getAttribute('m');
                const mData = JSON.parse(mContent); // Parse the JSON string
                return mData.murl; // Return the value of 'murl'
            });
            const htmlContent = document.documentElement.outerHTML; // Get the full HTML content
            return { newImageUrls, htmlContent };
        });

        // Save the HTML content to a text file
        // const filePath = path.join(__dirname, `page_content_${i + 1}.txt`);
        // fs.writeFileSync(filePath, htmlContent, 'utf8');
        // console.log(`Saved HTML content to ${filePath}`);

        imageUrls = [...new Set([...imageUrls, ...newImageUrls])]; // Merge and deduplicate URLs
        console.log(`Total number of image URLs extracted: ${imageUrls.length}`);
    }

    return imageUrls;
}

// Function to process image identifiers
async function processImageIdentifiers(outputDir, urlIdPairs, browserIndex, startIndex, failedUrls) {
    let successfulDownloads = 0; // Counter for successful downloads

    for (let i = 0; i < urlIdPairs.length; i++) {
        const [url, uniqueId] = urlIdPairs[i]; // Destructure the URL and unique identifier
        const filepath = `${outputDir}/${uniqueId}.jpg`; // Use uniqueId for naming
        try {
            const result = await downloadImage(url, filepath);
            if (result !== null) { // Only count if the result is not null
                successfulDownloads++; // Increment only for successful downloads
            }
            await new Promise(resolve => setTimeout(resolve, randomDelay(1000, 3000))); // Random delay
        } catch (error) {
            console.error(`Error downloading image ${startIndex + i} for browser ${browserIndex}: ${error.message}`);
            failedUrls.push(url); // Push the failed URL to the array
        }
    }

    return successfulDownloads; // Return the count of successful downloads
}

// Function to set up and scrape images
async function scrapeImages(keyword, totalImages, numBrowsers, headless_opt) {
    const outputDir = './output';

    // Delete existing output directory if it exists
    if (fs.existsSync(outputDir)) {
        fs.rmSync(outputDir, { recursive: true, force: true });
    }
    fs.mkdirSync(outputDir); // Create a new output directory

    const browser = await puppeteer.launch({ headless: headless_opt }); // Run in headless mode
    const page = await browser.newPage();
    const userAgent = new UserAgent();
    await page.setUserAgent(userAgent.toString()); // Random User-Agent
    const first_page = `https://www.bing.com/images/search?q=${encodeURIComponent(keyword)}`;
    await page.goto(first_page, { waitUntil: 'networkidle2' });

    // Collect all image URLs
    let allUrls = await collectImageUrls(page, totalImages);
    console.log(`Gathered ${allUrls.length} image URLs.`); // Print the number of URLs gathered
    if (headless_opt) { await browser.close(); }

    const failedUrls = []; // Array to store failed URLs
    let totalDownloaded = 0; // Initialize totalDownloaded here

    // Main loop to ensure we reach the target number of images
    while (totalDownloaded < totalImages) {
        // Split URLs among browsers
        const urlsPerBrowser = Math.ceil(allUrls.length / numBrowsers);
        const browserPromises = [];

        for (let i = 0; i < numBrowsers; i++) {
            const urlsForBrowser = allUrls.slice(i * urlsPerBrowser, (i + 1) * urlsPerBrowser);
            browserPromises.push(processImageIdentifiers(outputDir, urlsForBrowser, i + 1, (i * urlsPerBrowser) + 1, failedUrls));
        }

        const results = await Promise.all(browserPromises);
        totalDownloaded += results.reduce((acc, count) => acc + count, 0); // Sum the successful downloads from all browsers

        // Count the number of images in the output directory
        const files = fs.readdirSync(outputDir);
        const imageCount = files.filter(file => file.endsWith('.jpg')).length; // Count only .jpg files
        console.log(`Total images downloaded: ${imageCount}`); // Log the total count of downloaded images

        // If there are failed URLs, retry them
        if (failedUrls.length > 0) {
            console.log(`Retrying ${failedUrls.length} failed URLs...`);
            allUrls = failedUrls.slice(); // Prepare to retry failed URLs
            failedUrls.length = 0; // Clear the failed URLs for the next round
        } else {
            break; // Exit the loop if there are no failed URLs
        }
    }

    console.log(`Final total images downloaded: ${totalDownloaded}`); // Log the final count of downloaded images
}

// Specify the keyword, total number of images to scrape, and number of browsers
scrapeImages('fingernail', 1000, 100, false); // Change 400 to the total number of images you want to scrape and 200 to the number of browser instances
