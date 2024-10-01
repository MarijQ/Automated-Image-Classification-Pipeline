// ------------------- Initialisation -------------------

// imports
const {performance} = require('perf_hooks'); // Import performance for timing
const fs = require('fs'); // File system module for file operations
const axios = require('axios'); // HTTP client for making requests
const puppeteer = require('puppeteer-extra'); // Puppeteer for browser automation
const StealthPlugin = require('puppeteer-extra-plugin-stealth'); // Plugin to avoid detection
const UserAgent = require('user-agents'); // Module to generate random user agents
const xlsx = require('xlsx'); // Library for writing Excel files
const path = require('path'); // Add path module for handling file paths
puppeteer.use(StealthPlugin()); // Use stealth plugin to avoid detection

// User-defined search terms
const onychomycosis_terms = [
    "Onychomycosis fingernail fungus",
    "fungal infection onychomycosis nails",
    "onychomycosis symptoms nails",
    "onychomycosis treatment",
    "onychomycosis toenail infection",
    "nail fungus onychomycosis",
    "onychomycosis causes nails",
    "onychomycosis progression",
    "onychomycosis diagnosis nails",
    "onychomycosis images",
    "onychomycosis in nails",
    "severe onychomycosis nails",
    "onychomycosis appearance",
    "onychomycosis affected nails",
    "onychomycosis fungal nails",
    "onychomycosis necrosis nails",
    "onychomycosis discoloration",
    "onychomycosis thickened nails",
    "onychomycosis spread nails",
    "onychomycosis advanced nails"
];

const healthy_nails_terms = [
    "healthy fingernails",
    "normal fingernails",
    "strong fingernails",
    "clear fingernails",
    "well-maintained nails",
    "glossy fingernails",
    "natural healthy nails",
    "smooth healthy fingernails",
    "fingernail health",
    "undamaged fingernails",
    "fingernails without disease",
    "fingernails condition healthy",
    "fingernail growth healthy",
    "fingernail structure healthy",
    "fingernails appearance healthy",
    "fingernails care healthy",
    "fingernails smooth healthy",
    "fingernail shape healthy",
    "fingernails without blemishes",
    "fingernails vibrant health"
];

// Search engines to use
let search_engines = ['bing', 'duckduckgo'];

// List of Bing license options
const bing_license_options = {
    1: "", // "All" - No filter
    2: "+filterui:licenseType-Any", // "All Creative Commons"
    3: "+filterui:license-L1", // "Public domain"
    4: "+filterui:license-L2_L3_L4_L5_L6_L7", // "Free to share and use"
    5: "+filterui:license-L2_L3_L4", // "Free to share and use commercially"
    6: "+filterui:license-L2_L3_L5_L6", // "Free to modify, share, and use"
    7: "+filterui:license-L2_L3" // "Free to modify, share, and use commercially"
};

// List of DuckDuckGo license options
const duckduckgo_license_options = {
    1: "", // "All Licenses" - No filter
    2: "&license=Any", // "All Creative Commons"
    3: "&license=Public", // "Public Domain"
    4: "&license=Share", // "Free to Share and Use"
    5: "&license=ShareCommercially", // "Free to Share and Use Commercially"
    6: "&license=Modify", // "Free to Modify, Share, and Use"
    7: "&license=ModifyCommercially" // "Free to Modify, Share, and Use Commercially"
};

// Output directory
const output_dir = './output';

const MAX_FAILED_ATTEMPTS = 5; // Maximum allowed failed attempts per URL
const CONCURRENT_DOWNLOADS = 20; // Adjust this number based on desired concurrency

// ------------------- Main Function Call -------------------

// Execute the full script
(async () => {
    const license_bing = 1; // Change this number based on the desired Bing license option (1-7)
    const license_duckduckgo = 1; // Change this number based on the desired DuckDuckGo license option (1-7)
    const total_images = 10000; // Total target images
    const download_rate = 1; // Download success ratio (can be adjusted)
    const is_visible = false; // False for headless mode
    const is_quota = false; // Force splitting target up equally for different search combinations for debugging, false means greedy search

    // await full(onychomycosis_terms.slice(0), search_engines.slice(0, 2), total_images, download_rate, !is_visible, is_quota, license_bing, license_duckduckgo);
    // await benchmark_collection(onychomycosis_terms.slice(0), search_engines.slice(0, 2), 10, 400, 10, download_rate, !is_visible, false, license_bing, license_duckduckgo, 3);
    await benchmark_keywords(onychomycosis_terms.slice(0), search_engines.slice(0, 2), total_images, download_rate, !is_visible, false, license_bing, license_duckduckgo, 3);
    // await benchmark_concurrency(onychomycosis_terms.slice(0), search_engines.slice(0, 2), total_images, download_rate, !is_visible, false, license_bing, license_duckduckgo, 3);
})();

// Main function to run the entire process
async function full(search_terms, search_engines, total_images, download_rate, is_headless, is_quota, license_bing, license_duckduckgo) {
    // Stage 0: Initialisation
    setup_output_directory(output_dir);
    const master_data = []; // Initialize master_data to track all URLs

    // Stage 1: Image Collection
    console.log(`\n===== Image Collection =====`);
    await image_collecting(search_terms, search_engines, total_images, download_rate, is_headless, is_quota, license_bing, license_duckduckgo, master_data);

    // Save results to Excel
    save_to_excel(master_data); // Save results to Excel

    // Stage 2: Image Download
    console.log(`\n===== Image Download =====`);
    const keyword_stats = {successful_downloads: 0};
    await image_downloading(master_data, keyword_stats, total_images);

    // Save results to Excel
    save_to_excel(master_data); // Save results to Excel
}

// Function to set up output directory and clear files
function setup_output_directory(dir) {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir);
        console.log(`📂 Created output directory: ${dir}`);
    } else {
        // Clear existing files
        const files = fs.readdirSync(dir);
        for (const file of files) {
            fs.unlinkSync(`${dir}/${file}`);
        }
        console.log(`🧹 Cleared existing files in output directory: ${dir}`);
    }
}

// ------------------- Benchmark functions -------------------

async function benchmark_collection(search_terms, search_engines, min_images, max_images, increment, download_rate, is_headless, is_quota, license_bing, license_duckduckgo, repeatCount) {
    const stats_data = []; // To store benchmark results

    for (const search_engine of search_engines) {
        for (let total_images = min_images; total_images <= max_images; total_images += increment) {
            console.log(`\n🔍 Benchmarking: Search Engine = ${search_engine}, Total Images = ${total_images}`);

            const master_data = []; // To store collected URLs

            let total_collection_time = 0;
            let total_download_time = 0;
            let total_successful_downloads = 0;

            // Repeat the benchmarking process
            for (let repeat = 0; repeat < repeatCount; repeat++) {
                // Start timing the image collection process
                const collection_start_time = performance.now();

                // Run the image collection stage
                await image_collecting(search_terms, [search_engine], total_images, download_rate, is_headless, is_quota, license_bing, license_duckduckgo, master_data);

                const collection_end_time = performance.now();
                const collection_time = (collection_end_time - collection_start_time) / 1000; // Time in seconds

                // Initialize download statistics
                const keyword_stats = { successful_downloads: 0 };

                // Start timing the image downloading process
                const download_start_time = performance.now();

                // Run the image downloading stage
                await image_downloading(master_data, keyword_stats, total_images);

                const download_end_time = performance.now();
                const download_time = (download_end_time - download_start_time) / 1000; // Time in seconds

                // Accumulate times and successful downloads
                total_collection_time += collection_time;
                total_download_time += download_time;
                total_successful_downloads += keyword_stats.successful_downloads;

                // Log the result for this run
                console.log(`⏱️ Run ${repeat + 1}: Collection Time: ${collection_time.toFixed(2)}s, Download Time: ${download_time.toFixed(2)}s, Successful Downloads: ${keyword_stats.successful_downloads}`);
            }

            // Calculate averages
            const average_collection_time = (total_collection_time / repeatCount).toFixed(2);
            const average_download_time = (total_download_time / repeatCount).toFixed(2);
            const average_successful_downloads = (total_successful_downloads / repeatCount);

            // Log the average result for this configuration
            console.log(`📊 Average: Collection Time: ${average_collection_time}s, Download Time: ${average_download_time}s, Successful Downloads: ${average_successful_downloads}`);

            // Store the result in stats_data
            stats_data.push({
                total_images,
                search_engine,
                average_collection_time,
                average_download_time,
                average_successful_downloads
            });
        }
    }

    // Save benchmark results to Excel
    save_benchmark_collection_to_excel(stats_data);
}

function save_benchmark_collection_to_excel(stats_data) {
    const workbook = xlsx.utils.book_new();
    const worksheet_data = stats_data.map(entry => ({
        'Total Images': entry.total_images,
        'Search Engine': entry.search_engine,
        'Collection Time (s)': entry.collection_time,
        'Download Time (s)': entry.download_time,
        'Successful Downloads': entry.successful_downloads
    }));

    const worksheet = xlsx.utils.json_to_sheet(worksheet_data);
    xlsx.utils.book_append_sheet(workbook, worksheet, 'Benchmark Results');

    const filepath = `./stats_4-4.xlsx`;
    xlsx.writeFile(workbook, filepath);
    console.log(`📁 Benchmark results saved to ${filepath}`);
}

async function benchmark_keywords(search_terms, search_engines, total_images, download_rate, is_headless, is_quota, license_bing, license_duckduckgo, repetitions) {
    const aggregatedStats = []; // To accumulate stats across repetitions

    for (let rep = 1; rep <= repetitions; rep++) {
        console.log(`\n===== Benchmark2: Repetition ${rep} =====`);
        const startTime = performance.now();
        setup_output_directory(output_dir);
        const master_data = [];
        const stats = [];
        let total_unique_urls = 0;

        // Stage 1: Image Collection with Stats Recording
        console.log(`\n===== Benchmark2: Image Collection =====`);
        for (let i = 0; i < search_terms.length; i++) {
            const search_term = search_terms[i];
            for (let j = 0; j < search_engines.length; j++) {
                const search_engine = search_engines[j];
                let remaining = is_quota
                    ? Math.ceil(Math.ceil(total_images / download_rate) / (search_terms.length * search_engines.length))
                    : Math.ceil(total_images / download_rate) - total_unique_urls;

                console.log(`🔍 [${search_engine.toUpperCase()}][${search_term}] Starting collection with target: ${remaining} URLs`);

                let result;
                if (search_engine === "bing") {
                    result = await collection_bing(search_term, remaining, is_headless, master_data, license_bing);
                } else if (search_engine === "duckduckgo") {
                    result = await collection_duckduckgo(search_term, remaining, is_headless, master_data, license_duckduckgo);
                }

                const { new_unique, duplicates_in_search } = result;
                total_unique_urls += new_unique;
                const elapsedTime = ((performance.now() - startTime) / 1000).toFixed(2);

                // Record stats
                stats.push({
                    'Repetition': rep,
                    'Search Engine': search_engine,
                    'Search Term Number': i + 1,
                    'New Unique Images': new_unique,
                    'Duplicates': duplicates_in_search,
                    'Script Time (s)': parseFloat(elapsedTime)
                });

                console.log(`✅ [${search_engine.toUpperCase()}][${search_term}] yielded ${new_unique} new + ${duplicates_in_search} duplicate URLs (${elapsedTime}s elapsed)\n`);

                if (!is_quota && total_unique_urls >= Math.ceil(total_images / download_rate)) {
                    console.log(`🎯 Target reached. Ending collection.`);
                    break;
                }
            }

            if (!is_quota && total_unique_urls >= Math.ceil(total_images / download_rate)) {
                break;
            }
        }

        // Save collected URLs to Excel for this repetition
        save_to_excel(master_data);

        // Stage 2: Image Download
        console.log(`\n===== Benchmark2: Image Download =====`);
        const keyword_stats = {successful_downloads: 0};
        await image_downloading(master_data, keyword_stats, total_images);

        // Save downloaded results to Excel for this repetition
        save_to_excel(master_data);

        // Calculate and log total script time elapsed for this repetition
        const endTime = performance.now();
        const totalElapsedTime = ((endTime - startTime) / 1000).toFixed(2);
        console.log(`📊 Total script time elapsed for repetition ${rep}: ${totalElapsedTime} seconds`);

        // Accumulate stats
        aggregatedStats.push(...stats);
    }

    // Calculate averages
    const averagedStats = [];
    const grouped = aggregatedStats.reduce((acc, curr) => {
        const key = `${curr['Search Engine']}_${curr['Search Term Number']}`;
        if (!acc[key]) {
            acc[key] = { count: 0, NewUnique: 0, Duplicates: 0, ScriptTime: 0 };
        }
        acc[key].count += 1;
        acc[key].NewUnique += curr['New Unique Images'];
        acc[key].Duplicates += curr['Duplicates'];
        acc[key].ScriptTime += curr['Script Time (s)'];
        return acc;
    }, {});

    for (const key in grouped) {
        const [search_engine, search_term_number] = key.split('_');
        const data = grouped[key];
        averagedStats.push({
            'Search Engine': search_engine,
            'Search Term Number': parseInt(search_term_number),
            'Average New Unique Images': (data.NewUnique / data.count).toFixed(2),
            'Average Duplicates': (data.Duplicates / data.count).toFixed(2),
            'Average Script Time (s)': (data.ScriptTime / data.count).toFixed(2)
        });
    }

    // Save averaged benchmark2 stats to Excel
    save_benchmark_keywords_to_excel(averagedStats);
}

function save_benchmark_keywords_to_excel(stats_data) {
    const workbook = xlsx.utils.book_new();
    const worksheet_data = stats_data.map(entry => ({
        'Search Engine': entry['Search Engine'],
        'Search Term Number': entry['Search Term Number'],
        'Average New Unique Images': entry['Average New Unique Images'],
        'Average Duplicates': entry['Average Duplicates'],
        'Average Script Time (s)': entry['Average Script Time (s)']
    }));

    const worksheet = xlsx.utils.json_to_sheet(worksheet_data);
    xlsx.utils.book_append_sheet(workbook, worksheet, 'Benchmark2 Averages');

    const filepath = `./stats_4-5.xlsx`;
    xlsx.writeFile(workbook, filepath);
    console.log(`📁 Averaged Benchmark2 statistics saved to ${filepath}`);
}

async function benchmark_concurrency(search_terms, search_engines, total_images, download_rate, is_headless, is_quota, license_bing, license_duckduckgo, repetitions) {
    const concurrencyValues = [];
    for (let i = 1; i <= 256; i*=2) {
        concurrencyValues.push(i);
    }

    // Stage 1: Image Collection
    console.log(`\n===== Benchmark3: Image Collection =====`);
    const master_data = []; // Initialize master_data
    await image_collecting(search_terms, search_engines, total_images, download_rate, is_headless, is_quota, license_bing, license_duckduckgo, master_data);
    save_to_excel(master_data);

    const stats = [];

    for (const concurrency of concurrencyValues) {
        let totalTime = 0;

        for (let run = 1; run <= repetitions; run++) {
            // Reset outcomes
            master_data.forEach(entry => {
                if (entry.outcome !== "Duplicate") {
                    entry.outcome = "Untested";
                    entry.failed_attempts = 0;
                    entry.last_failure = null;
                }
            });

            console.log(`\n🔄 Benchmark3: Concurrent Downloads = ${concurrency}, Run = ${run}`);
            const startTime = performance.now();

            await image_downloading(master_data, { successful_downloads: 0 }, total_images, concurrency);

            const endTime = performance.now();
            const elapsedTime = (endTime - startTime) / 1000;
            totalTime += elapsedTime;
            console.log(`⏱️ Run ${run} completed in ${elapsedTime.toFixed(2)} seconds`);
        }

        const averageTime = (totalTime / 3).toFixed(2);
        stats.push({ concurrent_downloads: concurrency, average_time: averageTime });
        console.log(`📊 Average Time for ${concurrency} concurrent downloads: ${averageTime} seconds`);
    }

    // Save benchmark3 results to Excel
    const workbook = xlsx.utils.book_new();
    const worksheet_data = stats.map(entry => ({
        'Concurrent Downloads': entry.concurrent_downloads,
        'Average Script Time (s)': entry.average_time
    }));

    const worksheet = xlsx.utils.json_to_sheet(worksheet_data);
    xlsx.utils.book_append_sheet(workbook, worksheet, 'Benchmark3 Results');

    const filepath = `./stats_4-6.xlsx`;
    xlsx.writeFile(workbook, filepath);
    console.log(`📁 Benchmark3 results saved to ${filepath}`);
}

// ------------------- Collection Functions -------------------

async function image_collecting(search_terms, search_engines, total_images, download_rate, is_headless, is_quota, license_bing, license_duckduckgo, master_data) {
    let unique_count = 0;
    const adjusted_target = Math.ceil(total_images / download_rate);
    console.log(`🎯 Total target images: ${total_images}`);
    console.log(`🎯 Adjusted target (accounting for download success rate ${download_rate}): ${adjusted_target}`);

    // Calculate target_images_per_combination if is_quota is true
    let target_images_per_combination = adjusted_target;
    if (is_quota) {
        const total_combinations = search_terms.length * search_engines.length;
        target_images_per_combination = Math.ceil(
            adjusted_target / total_combinations
        );
        console.log(`🎯 Quota Enabled: Each combination will target ${target_images_per_combination} images.\n`);
    }

    // Run collection
    for (const search_term of search_terms) {
        for (const search_engine of search_engines) {
            if (!is_quota && unique_count >= adjusted_target) {
                console.log(`🎯 Target reached. Ending collection.`);
                break;
            }

            const remaining = is_quota
                ? target_images_per_combination -
                master_data.filter(
                    (entry) =>
                        entry.search_term === search_term &&
                        entry.search_engine === search_engine &&
                        entry.outcome !== "Duplicate"
                ).length
                : adjusted_target - unique_count;

            console.log(`🔍 [${search_engine.toUpperCase()}][${search_term}] Starting collection with target: ${remaining} URLs`);

            let new_unique = 0;
            let duplicates_in_search = 0;
            let new_urls_in_search = 0;

            if (search_engine === "bing") {
                const result = await collection_bing(search_term, remaining, is_headless, master_data, license_bing);
                new_unique = result.new_unique;
                duplicates_in_search = result.duplicates_in_search;
                new_urls_in_search = result.new_urls_in_search;
            } else if (search_engine === "duckduckgo") {
                const result = await collection_duckduckgo(search_term, remaining, is_headless, master_data, license_duckduckgo);
                new_unique = result.new_unique;
                duplicates_in_search = result.duplicates_in_search;
                new_urls_in_search = result.new_urls_in_search;
            }

            unique_count += new_unique;

            // Consolidated log statement with tick icon
            const success_rate = duplicates_in_search + new_urls_in_search > 0 ? Math.round((new_urls_in_search / (duplicates_in_search + new_urls_in_search)) * 100) : 0;
            console.log(`✅ [${search_engine.toUpperCase()}][${search_term}] yielded ${new_urls_in_search} new + ${duplicates_in_search} duplicate URLs (${success_rate}% collection rate)\n`);

            if (!is_quota && unique_count >= adjusted_target) {
                console.log(`🎯 Target reached. Ending collection.`);
                break;
            }
        }

        if (!is_quota && unique_count >= adjusted_target) {
            break;
        }
    }

    // Calculate and print collection success rate
    const collection_success_rate = ((unique_count / master_data.length) * 100).toFixed(0);
    console.log(`📊 All search combinations yielded ${unique_count} unique / ${master_data.length} total URLs considered (${collection_success_rate}% overall collection rate)\n`);
}

async function collection_bing(search_term, target_images, is_headless, master_data, license_bing) {
    const browser = await puppeteer.launch({headless: is_headless});
    const page = await browser.newPage();
    await page.setUserAgent(new UserAgent().toString());

    let new_unique = 0; // Count of new unique URLs
    let duplicates_in_search = 0; // Count of duplicates in this search
    let new_urls_in_search = 0; // Count of new URLs in this search

    const considered_urls = new Set(); // Track processed URLs

    // Construct the search URL with the license filter
    const base_url = 'https://www.bing.com/images/search';
    const query = encodeURIComponent(search_term);
    const license_filter = bing_license_options[license_bing] || "";
    const search_url = `${base_url}?q=${query}&qft=${license_filter}`;

    try {
        // Navigate to the filtered search URL
        await page.goto(search_url, {waitUntil: 'networkidle2'});
        console.log(`Conducted search`);

        // Reject cookies if the button appears
        try {
            await page.waitForSelector('#bnp_btn_reject', {timeout: 1000});
            await page.click('#bnp_btn_reject');
            console.log('Rejected cookies');
        } catch {
            // Cookie reject button not found; continue
        }

        let last_height = await page.evaluate('document.body.scrollHeight');
        let no_new_content_count = 0; // Counter for no new content

        // Collect images until target is reached
        while (new_unique < target_images) {
            // Scroll to the bottom to load more images
            await page.evaluate('window.scrollTo(0, document.body.scrollHeight)');

            // Wait until the new body height has increased or 1 second timeout
            try {
                await page.waitForFunction(
                    (prevHeight) => document.body.scrollHeight > prevHeight,
                    {timeout: 1000},
                    last_height
                );
                no_new_content_count = 0;
            } catch {
                no_new_content_count++;

                // Attempt to click "See more images" if available
                try {
                    await page.waitForSelector('a.btn_seemore.cbtn.mBtn', {visible: true, timeout: 100});
                    await page.click('a.btn_seemore.cbtn.mBtn');
                    console.log('🔘 Clicked the "See more images" button.');
                } catch {
                    console.log('⚠️ "See more images" button is not clickable or not found. Continuing...');
                }
            }

            // Extract image URLs
            const new_urls = await page.evaluate(() => {
                return Array.from(document.querySelectorAll('a.iusc'))
                    .map(el => {
                        try {
                            const data = JSON.parse(el.getAttribute('m'));
                            return data.murl || null;
                        } catch {
                            return null;
                        }
                    })
                    .filter(url => url);
            });

            // Iterate through each new URL and process it
            for (const url of new_urls) {
                // Ignore if already considered for this search combination, otherwise add to considered list
                if (considered_urls.has(url)) continue;
                considered_urls.add(url);

                // Check if the URL is a duplicate based on master_data
                const isDuplicate = master_data.some(entry =>
                    entry.URL === url &&
                    (entry.search_term !== search_term || entry.search_engine !== 'bing')
                );

                // Add new row in master_data
                master_data.push({
                    URL: url,
                    search_term,
                    image_ID: generate_random_string(10),
                    outcome: isDuplicate ? "Duplicate" : "Untested",
                    search_engine: 'bing',
                    failed_attempts: 0, // Initialize failed_attempts to 0
                    last_failure: null // Initialize last_failure to null
                });

                // Update counts
                if (isDuplicate) {
                    duplicates_in_search++;
                } else {
                    new_unique++;
                    new_urls_in_search++;
                    if (new_unique >= target_images) break;
                }
            }

            console.log(`Considered ${new_urls.length} URLs, found ${new_unique} new URLs.`);

            // Exit if no new content after multiple attempts
            if (no_new_content_count >= 3) {
                console.log(`⏹ [BING][${search_term}] No more content to load after multiple attempts.`);
                break;
            }

            let new_height = await page.evaluate('document.body.scrollHeight');
            if (new_height === last_height) {
                no_new_content_count++;
            } else {
                no_new_content_count = 0;
                last_height = new_height;
            }

            if (no_new_content_count >= 3) {
                console.log(`⏹ [BING][${search_term}] No more content to load after multiple attempts.`);
                break;
            }
        }

    } catch (error) {
        console.error(`❌ [BING][${search_term}] Error: ${error.message}`);
    } finally {
        if (is_headless) await browser.close();
    }

    return {new_unique, duplicates_in_search, new_urls_in_search};
}

async function collection_duckduckgo(search_term, target_images, is_headless, master_data, license_duckduckgo) {
    const browser = await puppeteer.launch({headless: is_headless});
    const page = await browser.newPage();
    await page.setUserAgent(new UserAgent().toString());

    let new_unique = 0; // Count of new unique URLs
    let duplicates_in_search = 0; // Count of duplicates in this search
    let new_urls_in_search = 0; // Count of new URLs in this search

    const considered_urls = new Set(); // Track processed URLs

    // Construct the DuckDuckGo image search URL with the license filter
    const base_url = 'https://duckduckgo.com/';
    const query = encodeURIComponent(search_term);
    const license_filter = duckduckgo_license_options[license_duckduckgo] || "";
    const search_url = `${base_url}?q=${query}&iax=images&ia=images${license_filter}`;

    try {
        // Navigate to the DuckDuckGo image search URL
        await page.goto(search_url, {waitUntil: 'networkidle2'});
        console.log(`Conducted search`);

        // Reject cookies if the button appears
        const rejectButton = await page.$('button#onetrust-reject-btn-handler');
        if (rejectButton) {
            await rejectButton.click();
            console.log('Rejected cookies');
        }

        let last_height = await page.evaluate('document.body.scrollHeight');
        let no_new_content_count = 0; // Counter for no new content

        // Collect images until target is reached
        while (new_unique < target_images) {
            // Scroll to the bottom to load more images
            await page.evaluate('window.scrollTo(0, document.body.scrollHeight)');

            // Wait until the new body height has increased or 1 second timeout
            try {
                await page.waitForFunction(
                    (prevHeight) => document.body.scrollHeight > prevHeight,
                    {timeout: 1000},
                    last_height
                );
                no_new_content_count = 0;
            } catch {
                no_new_content_count++;

                // Attempt to click "See more images" if available
                const seeMoreButton = await page.$('a.btn_seemore.cbtn.mBtn');
                if (seeMoreButton) {
                    try {
                        await seeMoreButton.click();
                        console.log('🔘 Clicked the "See more images" button.');
                    } catch (clickError) {
                        console.log('⚠️ "See more images" button is not clickable. Continuing...');
                    }
                }
            }

            // Extract and decode actual image URLs
            const new_urls = await page.evaluate(() => {
                return Array.from(document.querySelectorAll('div.tile--img__media img'))
                    .map(img => {
                        const src = img.src;
                        try {
                            const urlObj = new URL(src);
                            const actualURL = urlObj.searchParams.get('u');
                            return actualURL ? decodeURIComponent(actualURL) : null;
                        } catch {
                            return null;
                        }
                    })
                    .filter(url => url && url.startsWith('http'));
            });

            // Iterate through each new URL and process it
            for (const url of new_urls) {
                // Ignore if already considered for this search combination, otherwise add to considered list
                if (considered_urls.has(url)) continue;
                considered_urls.add(url);

                // Check if the URL is a duplicate based on master_data
                const isDuplicate = master_data.some(entry =>
                    entry.URL === url &&
                    (entry.search_term !== search_term || entry.search_engine !== 'duckduckgo')
                );

                // Add new row in master_data
                master_data.push({
                    URL: url,
                    search_term,
                    image_ID: generate_random_string(10),
                    outcome: isDuplicate ? "Duplicate" : "Untested",
                    search_engine: 'duckduckgo',
                    failed_attempts: 0, // Initialize failed_attempts to 0
                    last_failure: null // Initialize last_failure to null
                });

                // Update counts
                if (isDuplicate) {
                    duplicates_in_search++;
                } else {
                    new_unique++;
                    new_urls_in_search++;
                    if (new_unique >= target_images) break;
                }
            }

            console.log(`Considered ${new_urls.length} URLs, found ${new_unique} new URLs.`);

            // Exit if no new content after multiple attempts
            if (no_new_content_count >= 3) {
                console.log(`⏹ [DUCKDUCKGO][${search_term}] No more content to load after multiple attempts.`);
                break;
            }

            let new_height = await page.evaluate('document.body.scrollHeight');
            if (new_height === last_height) {
                no_new_content_count++;
            } else {
                no_new_content_count = 0;
                last_height = new_height;
            }

            if (no_new_content_count >= 3) {
                console.log(`⏹ [DUCKDUCKGO][${search_term}] No more content to load after multiple attempts.`);
                break;
            }
        }

    } catch (error) {
        console.error(`❌ [DUCKDUCKGO][${search_term}] Error: ${error.message}`);
    } finally {
        if (is_headless) await browser.close();
    }

    return {new_unique, duplicates_in_search, new_urls_in_search};
}

// ------------------- Downloading Functions -------------------

// Master function to download images with retries
async function image_downloading(master_data, keyword_stats, total_images, concurrency = CONCURRENT_DOWNLOADS) {
    console.log(`\n🚀 Starting image download with concurrency: ${concurrency}...`);

    let pending_urls = master_data.filter(entry =>
        (entry.outcome === "Untested" || entry.outcome === "Failed") &&
        entry.failed_attempts < MAX_FAILED_ATTEMPTS
    );

    while (pending_urls.length > 0 && keyword_stats.successful_downloads < total_images) {
        console.log(`🔄 Attempting to download ${pending_urls.length} images with up to ${concurrency} concurrent downloads.`);

        let index = 0;

        // Function to process next URL in the list
        async function next() {
            if (keyword_stats.successful_downloads >= total_images) return;

            if (index >= pending_urls.length) return;

            const entry = pending_urls[index];
            index++;

            await process_url(entry, keyword_stats, total_images);

            // Start next download
            await next();
        }

        // Start initial concurrent downloads
        const initialTasks = [];
        for (let i = 0; i < concurrency && i < pending_urls.length; i++) {
            initialTasks.push(next());
        }

        // Wait for all downloads to finish
        await Promise.all(initialTasks);

        // Update pending_urls for next iteration
        pending_urls = master_data.filter(entry =>
            (entry.outcome === "Untested" || entry.outcome === "Failed") &&
            entry.failed_attempts < MAX_FAILED_ATTEMPTS
        );
    }

    const overall_download_rate = ((keyword_stats.successful_downloads / master_data.length) * 100).toFixed(0);
    console.log(`📊 Successfully downloaded ${keyword_stats.successful_downloads} / ${master_data.length} unique URLs (${overall_download_rate}% overall download rate)`);
}

// Helper function to download a single image
async function process_url(entry, keyword_stats, total_images) {
    if (keyword_stats.successful_downloads >= total_images) return;

    const {URL, image_ID, failed_attempts, last_failure} = entry;

    // Check if failed_attempts is less than MAX_FAILED_ATTEMPTS
    if (failed_attempts >= MAX_FAILED_ATTEMPTS) {
        return;
    }

    // Adjust timeout based on last_failure
    let timeout;
    if (last_failure === 'timeout') {
        timeout = 1000 * Math.pow(2, failed_attempts);
    } else {
        timeout = 1000;
    }

    const filepath = `./output/${image_ID}.jpg`;

    try {
        const result = await download_image(URL, filepath, timeout);
        if (result === true) {
            entry.outcome = "Downloaded";
            keyword_stats.successful_downloads++;
            console.log(`✅ [downloaded ${keyword_stats.successful_downloads}] Downloaded: ${image_ID}.jpg`);
        } else if (typeof result === 'number' && result >= 400 && result < 500) {
            // Do not retry for 40X errors
            entry.outcome = `Failed (${result})`;
            console.log(`⚠️ [downloaded ${keyword_stats.successful_downloads}] Failed to download ${URL} - HTTP ${result}. Skipping further attempts.`);
        } else {
            // For other HTTP errors, increment failed_attempts
            entry.failed_attempts++;
            entry.outcome = "Failed";
            entry.last_failure = 'network';
            console.log(`⚠️ [downloaded ${keyword_stats.successful_downloads}] Failed to download ${URL} - HTTP ${result}. Attempt ${entry.failed_attempts}/${MAX_FAILED_ATTEMPTS}.`);
        }
    } catch (error) {
        entry.failed_attempts++;
        entry.outcome = "Failed";
        if (error.code === 'ECONNABORTED') {
            entry.last_failure = 'timeout';
            console.log(`⚠️ [downloaded ${keyword_stats.successful_downloads}] Timeout downloading ${URL}. Attempt ${entry.failed_attempts}/${MAX_FAILED_ATTEMPTS}.`);
        } else {
            entry.last_failure = 'network';
            console.log(`⚠️ [downloaded ${keyword_stats.successful_downloads}] Network error downloading ${URL}. Attempt ${entry.failed_attempts}/${MAX_FAILED_ATTEMPTS}.`);
        }
    }

    // Introduce a random delay between downloads
    await delay_random(100, 200);
}

// Function to download individual image from URL
async function download_image(url, filepath, timeout) {
    try {
        const response = await axios({
            url,
            method: 'GET',
            responseType: 'stream',
            timeout,
            headers: {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        });

        if (response.status !== 200) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return new Promise((resolve, reject) => {
            const write_stream = fs.createWriteStream(filepath);
            response.data.pipe(write_stream);
            write_stream.on('finish', () => resolve(true));
            write_stream.on('error', (error) => reject(error));
        });
    } catch (error) {
        if (error.response && [403, 404, 400].includes(error.response.status)) {
            return error.response.status;
        }
        throw error;
    }
}

// Function to introduce a random delay between min and max milliseconds
function delay_random(min, max) {
    return new Promise(resolve => setTimeout(resolve, Math.floor(Math.random() * (max - min + 1)) + min));
}

// ------------------- File operation functions -------------------

// Function to generate a random string
function generate_random_string(length) {
    const characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
        result += characters.charAt(Math.floor(Math.random() * characters.length));
    }
    return result;
}

function save_to_excel(master_data) {
    const workbook = xlsx.utils.book_new();
    const worksheet_data = master_data.map(entry => ({
        URL: entry.URL,
        Search_Term: entry.search_term,
        Search_Engine: entry.search_engine,
        Image_ID: entry.image_ID,
        Outcome: entry.outcome
    }));

    const worksheet = xlsx.utils.json_to_sheet(worksheet_data);
    xlsx.utils.book_append_sheet(workbook, worksheet, 'Results');

    const filepath = `./results.xlsx`;
    xlsx.writeFile(workbook, filepath);
    console.log(`📁 URL log saved to ${filepath}`);
}
