import requests

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

import csv
import os
from bs4 import BeautifulSoup
import time

# loads page and waits for pg to load some stuff
def load_pg(driver, pg_num, pg_url_no_num):
    print("Loading page " + str(pg_num) + " at url: " + pg_url_no_num + str(pg_num))
    driver.get(pg_url_no_num + str(pg_num))

    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(1) 
    except Exception as e:
        print(f"Failed to load {pg_url_no_num + str(pg_num)}: {e}")
        time.sleep(1)
    
    soup = BeautifulSoup(driver.page_source, "html.parser")
    bp_container = soup.find("div", class_="css-19iex53")
    return soup, bp_container

# gets the last page number
def get_total_pgs(soup):
    pg_nums = []
    page_links = soup.find_all("a", attrs={'aria-label': True})
    for link in page_links:
        pg_nums.append(link.text.strip())
    return int(max(pg_nums))

# gets all the links to blueprints on one page
def get_page_links(bp_container, bp_links, base_url):
    container = bp_container.find_all("div", class_="css-kcpn77")
    for content in container:        
        a_tag = content.find('a')
        if a_tag and a_tag.get("href"):
            href = a_tag["href"]
            bp_links.append(base_url + href)

# Gets the (name, author, source, tags, url, date, & data) of a single blueprint
def get_info_from_pg(driver, bp_link, results):
    print("Getting data from: " + bp_link )
    driver.get(bp_link)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    try:
        # Wait until the button is clickable and click it
        buttons = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".css-kr92sn:not(.primary)"))  # show string button class
        )
        buttons.click()

        # Get blueprint title
        all_titles = soup.find_all("h2", class_="title css-9ih1lr")
        for title in all_titles:
            span = title.find("span", class_="text")
            if span:
                bp_title = title.text
                break

        # Get blueprint string
        textarea = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'textarea')))
        blueprint_string = textarea.get_attribute("value")

        # Get author and date created?
        descr = soup.find_all("dl")
        info_dict = {}
        for dl in descr:
            dt = dl.find('dt')
            dd = dl.find('dd')
            info_dict[dt.get_text(strip=True)] = dd.get_text(strip=True)

        # Get tags
        tags = soup.find_all("span", class_="tag")
        tag_texts = [tag.text.strip() for tag in tags]

        # add data to list of results
        results.append({'name': str(bp_title),  
                        'author': info_dict['User:'],
                        'source': 'factorioblueprints.tech',
                        'tags': tag_texts,
                        'url': bp_link,
                        'date': info_dict['Last updated:'],
                        'data': str(blueprint_string)})

    except Exception as e:
        print(f"Failed to extract data from {bp_link}: {e}")
        results.append({'name': bp_link, 'author': None, 'source': 'factorioblueprints.tech', 'tags': None, 'url': bp_link, 'date': None, 'data' : None})
    time.sleep(1)

def make_csv(results):
    filename = "factorio-tech_pg.csv"
    fieldnames = ['name', 'author', 'source', 'tags', 'url', 'date', 'data']
    write_header = not os.path.exists(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL  # ensures all values are quoted
        )
        if write_header:
            writer.writeheader()  # write header only once
        writer.writerows(results)


# -------- START OF CODE -------- #
# set up headless (no gui) chrome using selenium
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# UPDATE THESE (only for runtime purposes)
curr_pg = 31
last_pg = 35

base_url = "https://factorioblueprints.tech"
pg_url_no_num = base_url + "/?page="
all_bp_links = []

# Uncomment these if you want to do all pages at once (not recommended)
# soup, bp_container = load_pg(driver, curr_pg, pg_url_no_num)
# last_pg = get_total_pgs(soup)
# curr_pg += 1

# gets all links from the pages in the range curr_pg --> last_pg
last_iter = min(last_pg, last_pg) 
while curr_pg <= last_iter:
    soup, bp_container = load_pg(driver, curr_pg, pg_url_no_num)
    get_page_links(bp_container, all_bp_links, base_url)
    curr_pg += 1

# gets information from the list of pages
results = []
for link in all_bp_links:
    get_info_from_pg(driver, link, results)

#makes csv
make_csv(results)

driver.quit()
