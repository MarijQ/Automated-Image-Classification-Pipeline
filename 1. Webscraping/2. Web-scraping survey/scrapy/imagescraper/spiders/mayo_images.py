import scrapy

class MayoImagesSpider(scrapy.Spider):
    name = 'mayo_images'
    start_urls = ['https://www.mayoclinic.org/healthy-lifestyle/adult-health/in-depth/7-fingernail-problems-not-to-ignore/art-20546860']

    def parse(self, response):
        for img_url in response.css('img::attr(src)').extract():
            yield {
                'image_urls': [response.urljoin(img_url)]
            }
