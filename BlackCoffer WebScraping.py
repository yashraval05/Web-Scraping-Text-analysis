#!/usr/bin/env python
# coding: utf-8

# Importing required Libraries

# In[2]:


import pandas as pd
from bs4 import BeautifulSoup as bs
import requests


# In[3]:


df=pd.read_excel('Input.xlsx')


# In[4]:


df.head()


# In[5]:


df['URL_ID']=df['URL_ID'].astype(int)


# In[6]:


df.info()


# In[7]:


session = requests.Session()

session.headers['User-Agent']


# We use custom header since the default header tells that the request is coming from Python code, some websites block such requests

# In[8]:


headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
}


# In[9]:


contents = []

for url in df['URL']:
  result = session.get(url, headers = headers)
  doc = bs(result.content, "html.parser")
  
  body = ""

  for text in doc.find_all("p"):
    body = body + " " + str(text.text)
  title = doc.header.h1.text
  #contents.append(title)
  contents.append(title+'\n'+body) #fetching the title and body storing it into list

#creating separate text file for each URL_ID
for url_id in df['URL_ID']:
    file_name=url_id
    file_name=str(file_name)
    if(url_id>44):
        url_id = url_id-1 #since the id 44 is missing
    with open(file_name,'w',encoding='utf8') as f:
        f.write(contents[url_id-1])


# In[10]:


print(len(contents))
print(contents[0])


# In[11]:


df2 = pd.merge(df, pd.DataFrame(contents), left_index = True, right_index = True, how = "left")


# In[13]:


df2.columns = ['URL_ID', 'URL', 'Text_Contents']
df2


# We export another Output.csv file with column text content for text analysis in the later part of the project

# In[14]:


df2.to_csv('Output.csv')

