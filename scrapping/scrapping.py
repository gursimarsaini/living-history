from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv

def all_days_url_fun(url):
    all_days_url=[]
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    for date in soup.find_all("td"): 
        for url in date.find_all("a"):
            try:
                all_days_url.append(url['href'])
            except:
                print('No URL')
    return all_days_url
# all_days_url=all_days_url_fun('https://www.thehindu.com/archive/web/2011/05/')
# all_days_url


def all_month_url_fun(url):
    all_month_url=[]
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    for web in soup.find_all(class_='archiveBorder'):
        for month in web.find_all("li"):
            for url in month.find_all("a"): 
                all_month_url.append(url['href'])
    return all_month_url

all_month_url=all_month_url_fun('https://www.thehindu.com/archive/')


# for i in range(0,len(all)):
#     a=all_days_url_fun(all_month_url[i])
#     all.append(a)
# all


# In[11]:


urls= []
article_id=[]
titles=[]
descriptions=[]
links=[]
dates=[]
news_types=[]
authers=[]



sr_no=0
# file=open('the_hindu_July_to_dec_2011.csv','w')
# writer=csv.writer(file)
for j in range(0,len(all)):
    for i in range(0,len(all[j])):
        date=all[j][i][37:-1]
        req = requests.get(all[j][i])
        soup = BeautifulSoup(req.text, 'html.parser')
        for a in soup.find_all('ul', class_='archive-list'): 
            for heading in a.find_all("a"):
                req1 = requests.get(heading['href'])
                soup1 = BeautifulSoup(req1.text, 'html.parser')
                url=heading['href']
                sr_no=sr_no+1
                article_id.append(sr_no)
                links.append(url)
    #         print(url)
    #         date=soup1.find('span', class_="blue-color ksl-time-stamp")

                heading=heading.text
                titles.append(heading)
                dates.append(date)


    #         print(heading['href'])

                try:
                    news_type=soup1.find('div', class_="article-exclusive").text.strip()
                    news_types.append(news_type)

                except:
                    news_type='Nan'
                    news_types.append(news_type)
                try:
                    auther=soup1.find('a', class_="auth-nm lnk").text.strip()
    #                 authers.append(auther)

                except:
                    auther='Nan'
    #                 authers.append(auther)
                try:
                    description=soup1.find('div', class_="paywall").text.strip()
                    descriptions.append(description)

                except:
                    description='Nan'
                    descriptions.append(description)

                print(url,'   ',date,'   ',heading,'   ',' ',news_type,'  ',auther,'   ' ,description)


# In[8]:



import pandas as pd 
df1 = pd.DataFrame(list(zip(article_id,dates,news_types,titles,descriptions,links)), 
               columns =['article_id','date','news_type', 'title','description','link']) 
df1.head(-10)
# df1.to_csv('the_hindu_april_2011.csv')


# In[9]:


df2 = pd.read_csv('the_hindu_april_2011.csv')
df2.head(-10)


# In[11]:


df = pd.concat([df2,df1], axis=0)
df.head(-10)
df.to_csv('the_hindu_april_2011.csv')

