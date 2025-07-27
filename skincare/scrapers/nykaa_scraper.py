from selenium.webdriver.common.by import By
import time

def scrape(driver, url):
    driver.get(url)
    time.sleep(5)

    products = []

    for _ in range(2):
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight)")
        time.sleep(2)

    product_cards = driver.find_elements(By.CLASS_NAME, "css-1rd7vky")  # card container

    for card in product_cards[:20]:
        try:
            name = card.find_element(By.CLASS_NAME, "css-14srtjr").text
            price = card.find_element(By.CLASS_NAME, "css-17x46n5").text
            link = card.find_element(By.TAG_NAME, "a").get_attribute("href")

            products.append({
                "product_name": name,
                "brand": "Nykaa",
                "price": price,
                "link": link,
                "source": "nykaa.com"
            })
        except Exception:
            continue

    return products
