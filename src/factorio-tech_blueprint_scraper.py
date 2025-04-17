import requests

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

import pandas as pd
import sys
import csv
import os
import ast
import json
from bs4 import BeautifulSoup
import time



# loads page and waits for pg to load some stuff
def load_pg(driver, pg_num, pg_url_no_num):
    print("LOADING PAGE " + str(pg_num) + " at url: " + pg_url_no_num + str(pg_num))
    driver.get(pg_url_no_num + str(pg_num))

    # wait 10 sec or until body load
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(1) 
    except Exception as e:
        print(f"Failed to load {pg_url_no_num + str(pg_num)}: {e}")
        time.sleep(1)
    
    soup = BeautifulSoup(driver.page_source, "html.parser")
    bp_container = soup.find("div", class_="css-19iex53")
    return soup, bp_container

def get_total_pgs(soup):
    pg_nums = []
    page_links = soup.find_all("a", attrs={'aria-label': True})
    for link in page_links:
        pg_nums.append(link.text.strip())
    return int(max(pg_nums))

def get_page_links(bp_container, bp_links, base_url):
    container = bp_container.find_all("div", class_="css-kcpn77")
    for content in container:
        # content_title = content.text.strip()[1:]
        # print("getting content for " + content_title)
        
        a_tag = content.find('a')
        ## getting all the links to the blueprints
        if a_tag and a_tag.get("href"):
            href = a_tag["href"]
            bp_links.append(base_url + href)


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

        # Get blueprint title (there are multiple of this class, but the first is always title)
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


        if blueprint_string and len(blueprint_string) > 32000:
            print("Blueprint string too long. Calling make_json")
            make_json(driver, soup, bp_link, str(bp_title))
            pass
        else: # if bp string <= 32000 char
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

def make_json(driver, soup, bp_link, bp_title):
    try:
        # Wait until the button is clickable and click it
        buttons = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".css-14h2mzn:not(.primary)"))  # show json button class
        )
        buttons.click()

        pre_tag = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".css-1ows65d")))

        if pre_tag:
            try:
                json_data = json.loads(pre_tag.text)
                # make safe file name for json
                safe_name = bp_title.replace(" ", "_").replace("/", "_")
                safe_name = safe_name[:45] + "_tag" + str(bp_link)[-3:]
                safe_name = "".join(c for c in safe_name if c.isalnum() or c in ('_', '-'))
                filename = os.path.join("factorio-tech_jsons", f"{safe_name}.json")
                # write to json
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4)
            except json.JSONDecodeError as e:
                print("Failed to decode JSON:", e)

    except Exception as e:
        print(f"Failed to extract data from {bp_link}: {e}")
    time.sleep(1)

def make_csv(results, pg_num):
    filename = "factorio-tech_" + str(pg_num) + ".csv"
    fieldnames = ['name', 'author', 'source', 'tags', 'url', 'date', 'data']
    write_header = not os.path.exists(filename)
    
    # print(results)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL  # ensures all values are quoted
        )
        if write_header:
            writer.writeheader()  # write header only once

        for result in enumerate(results):
            writer.writerow(result)

# --------------------------------------------------------------------------- #

# set up headless (no gui) chrome using selenium
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# set csv size limit
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
print(maxInt)

# UPDATE THESE
curr_pg = 61
last_pg = 87

base_url = "https://factorioblueprints.tech"
pg_url_no_num = base_url + "/?page="


soup, bp_container = load_pg(driver, curr_pg, pg_url_no_num)
last_pg_num = get_total_pgs(soup)
last_iter = min(last_pg, last_pg_num) 

while curr_pg <= last_iter:
    all_bp_links = []
    results = []
    soup, bp_container = load_pg(driver, curr_pg, pg_url_no_num)
    get_page_links(bp_container, all_bp_links, base_url)
    for link in all_bp_links:
        get_info_from_pg(driver, link, results)
    make_csv(results, curr_pg)
    curr_pg += 1

driver.quit()
