// imports
const fs = require('fs'); // File system module for file operations
const axios = require('axios'); // HTTP client for making requests
const puppeteer = require('puppeteer-extra'); // Puppeteer for browser automation
const StealthPlugin = require('puppeteer-extra-plugin-stealth'); // Plugin to avoid detection
const UserAgent = require('user-agents'); // Module to generate random user agents
puppeteer.use(StealthPlugin()); // Use stealth plugin to avoid detection

// Predefined set of search terms for Beau's lines fingernail disease
const searchTerms = [
    "Beau's lines fingernail disease",
    "fingernail ridges Beau's lines",
    "Beau's lines symptoms nails",
    "nail disease Beau's lines",
    "Beau's lines causes fingernails"
];

// Set to store all unique URLs across all keywords
const uniqueUrls = new Set();

// Function to generate a random string of letters
const generateRandomString = (length) => {
    const characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'; // Characters to use
    let result = ''; // Initialize result string
    for (let i = 0; i < length; i++) {
        result += characters.charAt(Math.floor(Math.random() * characters.length)); // Append random character
    }
    return result; // Return the generated string
};

// Function to generate a random delay
const randomDelay = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min; // Generate random delay between min and max

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
            throw new Error(`HTTP error! status: ${response.status}`); // Throw error if status is not 200
        }

        return new Promise((resolve, reject) => {
            const writeStream = fs.createWriteStream(filepath); // Create write stream
            response.data.pipe(writeStream); // Pipe response data to file
            writeStream.on('finish', () => {
                console.log(`Successfully downloaded ${filepath}`); // Log success
                resolve(); // Resolve promise
            });
            writeStream.on('error', (error) => {
                console.error(`Failed to write ${filepath}: ${error.message}`); // Log error
                reject(error); // Reject promise
            });
        });
    } catch (error) {
        // Handle specific HTTP errors
        if (error.response) {
            // Check for 403 or 404 errors
            if (error.response.status === 403 || error.response.status === 404 || error.response.status === 400) {
                console.error(`Skipping ${url} due to HTTP error: ${error.response.status}`); // Log and skip specific errors
                return null; // Return null to indicate this URL should not be retried
            }
        }
        console.error(`Failed to download ${url}: ${error.message}`); // Log general error
        throw error; // Rethrow the error for further handling
    }
}

// Function to collect image URLs
async function collectImageUrls(page, totalImages) {
    const mUrls = new Map(); // Use a Map to store unique identifiers and URLs
    let previousUrlsCount = 0; // To track the number of URLs before the last scroll
    let sameCountIterations = 0; // Counter for same URL count iterations

    while (mUrls.size < totalImages) {
        const newUrls = await page.evaluate(() => {
            const elements = Array.from(document.querySelectorAll('a.iusc')); // Select image elements
            return elements.map(element => {
                const mContent = element.getAttribute('m'); // Get 'm' attribute
                const mData = JSON.parse(mContent); // Parse the JSON string
                return mData.murl; // Return the value of 'murl'
            });
        });

        newUrls.forEach(url => {
            if (!mUrls.has(url)) {
                const uniqueId = generateRandomString(10); // Generate a unique identifier
                mUrls.set(url, uniqueId); // Map the URL to its unique identifier
            }
        });

        console.log(`Currently collected ${mUrls.size} image URLs.`); // Log the number of URLs collected

        // Check if new images have been loaded
        if (mUrls.size === previousUrlsCount) {
            sameCountIterations++;
            if (sameCountIterations >= 5) {
                console.log('No new images loaded for 5 consecutive iterations. Exiting loop.'); // Log and break if no new images
                break; // Exit the loop if no new images are loaded for 5 consecutive iterations
            }
        } else {
            sameCountIterations = 0; // Reset the counter if new URLs are found
        }

        previousUrlsCount = mUrls.size; // Update the previous count

        // Scroll the page by 10000 pixels
        await page.evaluate(() => {
            window.scrollBy(0, 10000); // Scroll down
        });
        console.log('Scrolled down 10000 pixels.'); // Log the scroll

        // Wait for a moment to allow new images to load
        await new Promise(resolve => setTimeout(resolve, 1000)); // Adjust wait time as necessary

        // Wait for the "See more images" button to appear
        let buttonExists = false;
        const maxAttempts = 3; // Maximum number of attempts
        let attempts = 0;

        while (!buttonExists && attempts < maxAttempts) {
            buttonExists = await page.evaluate(() => {
                const button = document.querySelector('a.btn_seemore'); // Check for button
                return button !== null; // Return true if button exists
            });

            if (!buttonExists) {
                console.log('Waiting for "See more images" button to appear...'); // Log waiting
                await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for 1 second
                attempts++;

                // Scroll the page by 10000 pixels again
                await page.evaluate(() => {
                    window.scrollBy(0, 10000); // Scroll down again
                });
                console.log('Scrolled down 10000 pixels.'); // Log the scroll
            }
        }

        // If the button exists, click it
        if (buttonExists) {
            await page.evaluate(() => {
                const button = document.querySelector('a.btn_seemore'); // Select button
                button.click(); // Click the button
            });
            console.log('Clicked "See more images" button.'); // Log the click
        } else {
            console.log('Max attempts reached. No more "See more images" button found.'); // Log max attempts reached
        }
    }

    console.log(`Final count of collected image URLs: ${mUrls.size}`); // Log the final count
    return Array.from(mUrls.entries()).slice(0, totalImages); // Return the collected URLs and their unique identifiers
}

// Function to process image identifiers with duplicate check
async function processImageIdentifiersWithDupCheck(outputDir, urlIdPairs, browserIndex, startIndex, failedUrls, keywordStats) {
    let successfulDownloads = 0; // Counter for successful downloads
    let ignoredDuplicates = 0; // Counter for ignored duplicates

    for (let i = 0; i < urlIdPairs.length; i++) {
        const [url, uniqueId] = urlIdPairs[i]; // Destructure the URL and unique identifier

        // Check if the URL is already in the uniqueUrls set
        if (uniqueUrls.has(url)) {
            console.log(`Ignoring duplicate URL: ${url}`); // Log the duplicate
            ignoredDuplicates++; // Increment ignored duplicates counter
            continue; // Skip this URL
        }

        // Add the URL to the unique set to prevent future duplicates
        uniqueUrls.add(url);

        const filepath = `${outputDir}/${uniqueId}.jpg`; // Use uniqueId for naming
        try {
            const result = await downloadImage(url, filepath); // Attempt to download image
            if (result !== null) { // Only count if the result is not null
                successfulDownloads++; // Increment only for successful downloads
            }
            await new Promise(resolve => setTimeout(resolve, randomDelay(1000, 3000))); // Random delay
        } catch (error) {
            console.error(`Error downloading image ${startIndex + i} for browser ${browserIndex}: ${error.message}`); // Log error
            failedUrls.push(url); // Push the failed URL to the array
        }
    }

    // Update keyword statistics
    keywordStats.successfulDownloads += successfulDownloads;
    keywordStats.ignoredDuplicates += ignoredDuplicates;

    return successfulDownloads; // Return the count of successful downloads
}

// Function to scrape images for multiple search terms
async function scrapeMultipleKeywords(keywords, totalImagesPerKeyword, numBrowsers, headless_opt) {
    const outputDir = './output'; // Output directory

    // Create output directory if it doesn't exist
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir); // Create the directory if it doesn't exist
    }

    // Clear all files in the output directory
    const files = fs.readdirSync(outputDir);
    for (const file of files) {
        fs.unlinkSync(`${outputDir}/${file}`); // Delete each file
    }

    const keywordStatsMap = {}; // Object to store stats for each keyword

    for (const keyword of keywords) {
        console.log(`Starting scrape for keyword: ${keyword}`); // Log the current keyword

        // Initialize stats for the current keyword
        keywordStatsMap[keyword] = {
            successfulDownloads: 0,
            ignoredDuplicates: 0
        };

        // Scrape images for the current keyword
        await scrapeImagesWithDupCheck(keyword, totalImagesPerKeyword, numBrowsers, headless_opt, keywordStatsMap[keyword]);
    }

    console.log("Completed scraping for all keywords.");

    // Log the stats for each keyword
    console.log("Keyword Statistics:");
    for (const [keyword, stats] of Object.entries(keywordStatsMap)) {
        console.log(`Keyword: ${keyword}`);
        console.log(`  Successful Downloads: ${stats.successfulDownloads}`);
        console.log(`  Ignored Duplicates: ${stats.ignoredDuplicates}`);
    }
}

// Function to set up and scrape images with duplicate check
async function scrapeImagesWithDupCheck(keyword, totalImages, numBrowsers, headless_opt, keywordStats) {
    const browser = await puppeteer.launch({ headless: headless_opt }); // Launch browser
    const page = await browser.newPage(); // Open new page
    const userAgent = new UserAgent(); // Generate random user agent
    await page.setUserAgent(userAgent.toString()); // Set user agent
    const first_page = `https://www.bing.com/images/search?q=${encodeURIComponent(keyword)}`; // Construct search URL
    await page.goto(first_page, { waitUntil: 'networkidle2' }); // Navigate to search URL

    // Collect all image URLs
    let allUrls = await collectImageUrls(page, totalImages); // Collect URLs
    console.log(`Gathered ${allUrls.length} image URLs for keyword: ${keyword}.`); // Print the number of URLs gathered
    if (headless_opt) { await browser.close(); } // Close browser if headless

    const failedUrls = []; // Array to store failed URLs
    let totalDownloaded = 0; // Initialize totalDownloaded here
    let sameFailedCountIterations = 0; // Counter for same failed URL count iterations

    // Main loop to ensure we reach the target number of images
    while (totalDownloaded < totalImages) {
        // Split URLs among browsers
        const urlsPerBrowser = Math.ceil(allUrls.length / numBrowsers); // Calculate URLs per browser
        const browserPromises = []; // Array to store browser promises

        for (let i = 0; i < numBrowsers; i++) {
            const urlsForBrowser = allUrls.slice(i * urlsPerBrowser, (i + 1) * urlsPerBrowser); // Slice URLs for each browser
            browserPromises.push(processImageIdentifiersWithDupCheck('./output', urlsForBrowser, i + 1, (i * urlsPerBrowser) + 1, failedUrls, keywordStats)); // Process URLs
        }

        const results = await Promise.all(browserPromises); // Wait for all promises
        totalDownloaded += results.reduce((acc, count) => acc + count, 0); // Sum the successful downloads from all browsers

        // If there are failed URLs, retry them
        if (failedUrls.length > 0) {
            console.log(`Retrying ${failedUrls.length} failed URLs...`); // Log retry
            const currentFailedCount = failedUrls.length; // Store current failed count
            const urlsToRetry = failedUrls.slice(); // Prepare to retry failed URLs
            failedUrls.length = 0; // Clear the failed URLs for the next round

            // Retry the failed URLs
            const retryResults = await Promise.all(urlsToRetry.map(url => {
                return processImageIdentifiersWithDupCheck('./output', [[url, generateRandomString(10)]], 1, 1, failedUrls, keywordStats);
            }));

            // Count successful retries
            totalDownloaded += retryResults.reduce((acc, count) => acc + count, 0);

            // Check if the number of failed URLs has stayed the same
            if (currentFailedCount === failedUrls.length) {
                sameFailedCountIterations++;
                if (sameFailedCountIterations >= 3) {
                    console.log('No progress on failed URLs for 3 consecutive attempts. Exiting loop.'); // Log and break if no progress
                    break; // Exit the loop if no progress is made
                }
            } else {
                sameFailedCountIterations = 0; // Reset the counter if progress is made
            }
        } else {
            break; // Exit the loop if there are no failed URLs
        }

        // If we have downloaded enough images, break the loop
        if (totalDownloaded >= totalImages) {
            console.log(`Reached target of ${totalImages} images for keyword: ${keyword}. Exiting loop.`); // Log reaching target
            break; // Exit the loop if the target is reached
        }
    }

    console.log(`Final total images downloaded for keyword "${keyword}": ${totalDownloaded}`); // Log the final count of downloaded images
}

// Start the scraping process for multiple keywords
scrapeMultipleKeywords(searchTerms, 200, 10, true); // Parameters: keywords, total images per keyword, number of browsers, headless mode
