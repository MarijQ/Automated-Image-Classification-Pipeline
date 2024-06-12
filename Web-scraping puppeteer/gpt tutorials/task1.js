const puppeteer = require('puppeteer'); // Import the Puppeteer library
const fs = require('fs'); // Import the file system module to interact with the file system

async function run() {
    const browser = await puppeteer.launch(); // Launch a new browser instance
    const page = await browser.newPage(); // Open a new page/tab in the browser
    await page.goto('https://example.com'); // Navigate to the specified URL
    
    const dir = './task1_output'; // Define the directory path for output
    
    // Check if the directory exists, if not, create it
    if (!fs.existsSync(dir)){
        fs.mkdirSync(dir, { recursive: true }); // The option { recursive: true } allows creating nested directories
    }

    await page.screenshot({path: `${dir}/example.png`}); // Take a screenshot and save it to the specified path
    await browser.close(); // Close the browser
}

run(); // Run the function
