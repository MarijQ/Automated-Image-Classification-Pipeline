// Imports
const puppeteer = require('puppeteer');
const fs = require('fs');

async function run() {
    // Create directory for output
    const dir = './task2_output';
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }

    // Setup + load page
    const browser = await puppeteer.launch({headless: false});
    const page = await browser.newPage();
    await page.goto('https://google.com', {waitUntil: "domcontentloaded"});

    // Handle cookie consent popup
    const consentButtonSelector = '[id="L2AGLb"]'; // This selector might need adjustment based on your location and the specific site
    await page.waitForSelector(consentButtonSelector, { visible: true });
    await page.click(consentButtonSelector);

    // Search for keyword
    const searchQuery = "intel";
    await page.waitForSelector('input[id="jZ2SBf"]', {visible: true});
    await page.type('input[aria-label="Search"]', searchQuery);
    await page.keyboard.press('Enter'); // Simulate pressing the Enter key to submit the search
    await page.waitForNavigation({ waituntil: 'networkidle0' }); // Wait for the search results page to load completely

    // Output + exit
    await page.screenshot({path: `${dir}/search_results.png`}); // Save a screenshot of the search results
    await browser.close();
}

run();

//*[@id="APjFqb"]