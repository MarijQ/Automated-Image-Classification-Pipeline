const puppeteer = require("puppeteer-extra");
const StealthPlugin = require("puppeteer-extra-plugin-stealth");

puppeteer.use(StealthPlugin());

const searchQuery = "bugatti chiron";

async function getImagesData(page) {
  const imagesResults = [];
  let iterationsLength = 0;
  while (true) {
    const images = await page.$$(".OcgH4b .PNCib.MSM1fd");
    for (; iterationsLength < images.length; iterationsLength++) {
      images[iterationsLength].click();
      await page.waitForTimeout(2000);
      imagesResults.push(
        await page.evaluate(
          (iterationsLength) => ({
            thumbnail: document.querySelectorAll(".OcgH4b .PNCib.MSM1fd")[iterationsLength].querySelector(".Q4LuWd")?.getAttribute("src"),
            source: document.querySelectorAll(".OcgH4b .PNCib.MSM1fd")[iterationsLength].querySelector(".VFACy div")?.textContent.trim(),
            title: document.querySelectorAll(".OcgH4b .PNCib.MSM1fd")[iterationsLength].querySelector("h3")?.textContent.trim(),
            link: document.querySelectorAll(".OcgH4b .PNCib.MSM1fd")[iterationsAnnotationsLength].querySelector(".VFACy")?.getAttribute("href"),
            original: Array.from(document.querySelectorAll(".eHAdSb .n3VNCb"))
              .find((el) => !el.getAttribute("src").includes("data:image") && !el.getAttribute("src").includes("gstatic.com"))
              ?.getAttribute("src"),
          }),
          iterationsLength
        )
      );
    }
    await page.waitForTimeout(5000);
    const newImages = await page.$$(".OcgH4b .PNCib.MSM1fd");
    if (newImages.length === images.length || images.length > 100) break;
  }
  return imagesResults;
}

async function getGoogleImagesResults() {
  const browser = await puppeteer.launch({
    headless: false,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  const URL = `https://www.google.com/search?q=${encodeURI(searchQuery)}&tbm=isch&hl=en&gl=us`;

  const page = await browser.newPage();

  await page.setDefaultNavigationTimeout(60000);
  await page.goto(URL);

    // Handle cookie consent popup
    const consentButtonSelector = '[id="L2AGLb"]'; // This selector might need adjustment based on your location and the specific site
    await page.waitForSelector(consentButtonSelector, { visible: true });
    await page.click(consentButtonSelector);

  await page.waitForSelector(".PNCib");

  const imagesResults = await getImagesData(page);

  await browser.close();

  return imagesResults;
}

getGoogleImagesResults().then(imagesResults => {
  console.log(imagesResults);
}).catch(error => {
  console.error("Error fetching images:", error);
});
