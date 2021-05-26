from urllib.request import urlopen
from bs4 import BeautifulSoup
import time
import pandas as pd


def get_articles_link(link):
    all_links = []
    try:
        html = urlopen(link)
    except Exception:
        pass
    try:
        bs = BeautifulSoup(html.read(), 'html.parser')
        links = bs.find('table', {'cellpadding': '0', 'cellspacing': '0', 'border': '0', 'width': '100%'}).find_all('a')
        for link in links:
            if 'href' in link.attrs:
                all_links.append(link.attrs['href'].replace('http:', 'https:'))
    except Exception:
        pass
    try:
        date = bs.find('b').get_text()
    except Exception:
        return None
    return all_links, date


def get_articles_data(all_links, date):
    count = 0
    for link in all_links:
        count += 1
        if count % 100 == 0:
            time.sleep(5)
        try:
            html = urlopen(link)
        except Exception:
            continue
        try:
            bs = BeautifulSoup(html, 'html.parser')
        except Exception:
            continue
        try:
            news_headline = bs.find('arttitle').get_text()
        except Exception:
            news_headline = 'NULL'
        try:
            news_description = bs.find('div', {'class': 'Normal'}).get_text()
        except Exception:
            news_description = 'NULL'
        try:
            news_type = bs.find('li', {'itemprop': 'itemListElement'}).findNext('li').get_text()
        except Exception:
            news_type = 'NULL'
        try:
            news_date = date
        except Exception:
            news_date = 'NULL'
        try:
            a = [news_headline, news_description, news_type, news_date]
            df2 = pd.DataFrame([a])
            df2.to_csv('TOI_data_2.csv', mode='a', header=False, index=False)
        except Exception:
            continue


def get_data(url):
    all_links, date = get_articles_link(url)
    get_articles_data(all_links, date)


final_link = pd.read_csv('dailyNewsLinks.csv')


for i in range(550, 6575):
    get_data(final_link['links'][i])
