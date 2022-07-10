#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize # Can also use RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
import contractions
from textstat.textstat import textstatistics
import string


# In[2]:


df = pd.read_csv('Output.csv')


# In[3]:


df.head()


# In[4]:


df.drop(['Unnamed: 0'], axis = 1,inplace=True)
df.info()


# In[5]:


#lemmatizer
lemma = WordNetLemmatizer()


# In[6]:


#stopwords dictionary from nltk 

stop_words = stopwords.words('english')
print(len(stop_words))
print(stop_words)


# For custom stop words

# In[7]:


fileobj=open("StopWords_Generic.txt")
StopWords_Generic=[]
for line in fileobj:
    StopWords_Generic.append(line.strip())
print(len(StopWords_Generic))


# In[8]:


fileobj=open("StopWords_Auditor.txt")
StopWords_Auditor=[]
for line in fileobj:
    StopWords_Auditor.append(line.strip())
print(len(StopWords_Auditor))


# In[9]:


fileobj=open("StopWords_Currencies.txt")
StopWords_Currencies=[]
for line in fileobj:
    StopWords_Currencies.append(line.split()[0])
print(len(StopWords_Currencies))


# In[10]:


fileobj=open("StopWords_DatesandNumbers.txt")
StopWords_DatesandNumbers=[]
for line in fileobj:
    StopWords_DatesandNumbers.append(line.split()[0])
print(len(StopWords_DatesandNumbers))


# In[11]:


fileobj=open("StopWords_GenericLong.txt")
StopWords_GenericLong=[]
for line in fileobj:
    StopWords_GenericLong.append(line.strip())
print(len(StopWords_GenericLong))


# In[12]:


fileobj=open("StopWords_Geographic.txt")
StopWords_Geographic=[]
for line in fileobj:
    StopWords_Geographic.append(line.split()[0])
print(len(StopWords_Geographic))


# In[13]:


fileobj=open("StopWords_Names.txt")
StopWords_Names=[]
for line in fileobj:
    StopWords_Names.append(line.split()[0])
print(len(StopWords_Names))


# Combining all stop words together and exporting it into single text file

# In[14]:


all_StopWords=StopWords_Names+StopWords_Geographic+StopWords_GenericLong+StopWords_DatesandNumbers+StopWords_Currencies+StopWords_Auditor+StopWords_Generic
string1 = ' '.join([str(item) for item in all_StopWords])
print(len(all_StopWords))


# In[15]:


with open('all_StopWords.txt','w',encoding='utf8') as f:
    f.write(string1)


# In[16]:


with open('all_StopWords.txt','r') as f2:
    
   b=f2.read().split()
   stop_words = stop_words+b #extend the stop words from nltk and custom stop words
print(len(stop_words))
print(stop_words)


# In[17]:


#pre-processing text
def text_prep(x: str) -> list:
     corp = str(x).lower()
     corp = contractions.fix(corp) #replacing I'll with I will
     corp = re.sub('[^a-zA-Z]+',' ', corp).strip() 
     tokens = word_tokenize(corp)
     words = [t for t in tokens if t not in stop_words]
     lemmatize = [lemma.lemmatize(w) for w in words]
     return lemmatize


# In[18]:


df.head()


# In[19]:


# Applying the function
preprocess_tag = [text_prep(i) for i in df['Text_Contents']]
df["preprocess_txt"] = preprocess_tag


# Using custom negative and positive words

# In[20]:


file = open('negative-words.txt', 'r')
neg_words = file.read().split()
file = open('positive-words.txt', 'r')
pos_words = file.read().split()


# 1.Extracting Derived variables

# In[21]:


positive_count = df['preprocess_txt'].map(lambda x: len([i for i in x if i in pos_words]))
df['POSITIVE SCORE'] = positive_count
negative_count = df['preprocess_txt'].map(lambda x: len([i for i in x if i in neg_words]))
df['NEGATIVE SCORE'] = negative_count


# In[22]:


df['POLARITY SCORE'] = round((df['POSITIVE SCORE'] - df['NEGATIVE SCORE'])/(df['POSITIVE SCORE'] + df['NEGATIVE SCORE'] + 0.000001), 2)


# In[23]:


df['num_words'] = df['preprocess_txt'].map(lambda x: len(x))
df['SUBJECTIVITY SCORE'] = round((df['POSITIVE SCORE'] + df['NEGATIVE SCORE'])/(df['num_words'] + 0.000001), 2)


# In[24]:


df.head()


# 2.Analysis of Readability

# In[25]:


df['num_sentences'] = df['Text_Contents'].map(lambda x: len(sent_tokenize(x)))


# In[26]:


df['AVG SENTENCE LENGTH'] = round(df['num_words']/df['num_sentences'], 1)


# In[30]:


df.info()


# In[31]:


def syllables_count(text):
  return textstatistics().syllable_count(text)


# In[32]:


df['syl_count'] = df['preprocess_txt'].apply(lambda x: syllables_count(" ".join(x)))


# In[33]:


def complex_words(text):
  diff_words_set = set()
  words = text
  for word in words:
    syllable_count = syllables_count(word)
    if syllable_count > 2:
      diff_words_set.add(word)
  return len(diff_words_set)


# In[34]:


df['COMPLEX WORD COUNT'] = df['preprocess_txt'].apply(lambda x: complex_words(x))


# In[35]:


df['PERCENTAGE OF COMPLEX WORDS'] = round((df['COMPLEX WORD COUNT']/df['num_words']), 2)


# In[36]:


df['FOG INDEX'] = 0.4 * (df['AVG SENTENCE LENGTH'] + df['PERCENTAGE OF COMPLEX WORDS'])


# 3.Average Number of Words Per Sentence

# In[37]:


df['AVG NUMBER OF WORDS PER SENTENCE'] = round(df['num_words']/df['num_sentences'], 2)


# 4.Complex Word Count is done in COMPLEX WORD COUNT column

# 5.Word Count is done in num_words

# 6.Syllable Count Per Word

# In[38]:


df['SYLLABLE PER WORD'] = df['syl_count']/df['num_words']


# In[39]:


df.head()


# 7.Personal Pronouns

# In[40]:


def personal_pro(text):
  pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us))\b',re.I)
  pronouns = pronounRegex.findall(text)
  return len(pronouns)


# In[41]:


df['PERSONAL PRONOUNS'] = df['Text_Contents'].apply(lambda x: personal_pro(x))


# 8.Average Word Length

# In[42]:


def text_len(text):
  text = ''.join(text)
  filtered = ''.join(filter(lambda x: x not in string.punctuation, text))
  words = [word for word in filtered.split() if word]
  avg = sum(map(len, words))/len(words)
  return avg


# In[43]:


df['AVG WORD LENGTH'] = df['Text_Contents'].map(lambda x: text_len(x))


# In[44]:


df.head()


# Renaming and removing columns according to the requirement

# In[45]:


df.rename(columns = {'num_words':'WORD COUNT'}, inplace = True)


# In[47]:


df.drop(['Text_Contents','preprocess_txt','num_sentences','syl_count'], axis = 1,inplace=True)


# In[48]:


df.info()


# Exporting the final output

# In[49]:


df.to_csv('Final_Output.csv')

