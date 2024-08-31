const puppeteer = require("puppeteer"); // ^19.6.3

let browser;
(async () => {
  const searchQuery = "stack overflow";

  browser = await puppeteer.launch({headless: false});
  const [page] = await browser.pages();
  await page.setRequestInterception(true);
  page.on("request", request => {
    request.resourceType() === "document" ? 
      request.continue() : request.abort();
  });
  await page.goto("https://www.google.com/", {waitUntil: "domcontentloaded"});

  // Handle cookie consent popup
  const consentButtonSelector = '[id="L2AGLb"]'; // This selector might need adjustment based on your location and the specific site
  await page.waitForSelector(consentButtonSelector, { visible: true });
  await page.click(consentButtonSelector);


  await page.waitForSelector('input[aria-label="Search"]', {visible: true});
  await page.type('input[aria-label="Search"]', searchQuery);
  await Promise.all([
    page.waitForNavigation({waitUntil: "domcontentloaded"}),
    page.keyboard.press("Enter"),
  ]);
  await page.waitForSelector(".LC20lb", {visible: true});
  const searchResults = await page.$$eval(".LC20lb", els => 
    els.map(e => ({title: e.innerText, link: e.parentNode.href}))
  );
  console.log(searchResults);
})()
  .catch(err => console.error(err))
  .finally(() => browser?.close());