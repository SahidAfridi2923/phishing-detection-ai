import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def capture_screenshot(url, output_path="screenshot.png"):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(3)  # wait for page load
        driver.save_screenshot(output_path)
    except:
        return None
    finally:
        driver.quit()

    return output_path