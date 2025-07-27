import json
import importlib
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os

# Load website config
with open("C:\\Users\\Admin\\Desktop\\smart mirror\\data\\website.json", "r") as f:
    websites = json.load(f)

# Optional: Run Chrome headlessly
chrome_options = Options()
# chrome_options.add_argument("--headless")

driver = webdriver.Chrome(options=chrome_options)

all_results = []

for site in websites:
    try:
        print(f"Scraping {site['name']}...")
        scraper_module = importlib.import_module(f"scrapers.{site['scraper']}")
        results = scraper_module.scrape(driver, site["url"])
        print(f"✅ Scraped {len(results)} products from {site['name']}")
        all_results.extend(results)
    except Exception as e:
        print(f"❌ Failed to scrape {site['name']}: {e}")

driver.quit()

output_path = "scraped_recommendations.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n🎉 Final results saved to {output_path}")
