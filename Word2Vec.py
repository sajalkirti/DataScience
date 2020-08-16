#!/usr/bin/env python
# coding: utf-8

# In[16]:


import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re


# In[17]:


paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India 
               stands up to the world, no one will respect us. Only strength respects strength. We must be 
               strong not only as a military power but also as an economic power. Both must go hand-in-hand. 
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of 
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. 
               I see four milestones in my career"""


# In[20]:


paragraph = """Chandrayaan-2 (candra      -yāna, transl. "mooncraft"; About this soundpronunciation (help·info)) is the second lunar exploration mission developed by the Indian Space Research Organisation (ISRO), after Chandrayaan-1. As of September 2019, it consists of a lunar orbiter, and also included the Vikram lander, and the Pragyan lunar rover, all of which were developed in India. The main scientific objective is to map and study the variations in lunar surface composition, as well as the location and abundance of lunar water.

The spacecraft was launched on its mission to the Moon from the second launch pad at the Satish Dhawan Space Centre in Andhra Pradesh on 22 July 2019 at 2.43 p.m. IST (09:13 UTC) by a GSLV Mark III M1. The craft reached the Moon's orbit on 20 August 2019 and began orbital positioning manoeuvres for the landing of the Vikram lander. The lander and the rover were scheduled to land on the near side of the Moon, in the south polar region at a latitude of about 70° south on 6 September 2019 and conduct scientific experiments for one lunar day, which approximates to two Earth weeks. A successful soft landing would have made India the fourth country after the Soviet Union, United States and China to do so.

However, the lander deviated from its intended trajectory while attempting to land on 6 September 2019 which caused a 'hard landing'. According to a failure analysis report submitted to ISRO, the crash was caused by a software glitch. ISRO may re-attempt a landing by the second quarter of 2021 with Chandrayaan-3."""


# In[21]:


# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)


# In[22]:


text


# In[28]:


# Preparing the dataset
sentences = nltk.sent_tokenize(text)
print(sentences)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]


# In[29]:


sentences


# In[30]:


# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)


words = model.wv.vocab


# In[33]:


# Finding Word Vectors
vector = model.wv['indian']


# In[39]:


vector


# In[36]:




# Most similar words
similar = model.wv.most_similar('indian')


# In[37]:


similar


# In[44]:


#words


# In[42]:




# Most similar words
similar = model.wv.most_similar('vikram')


# In[49]:


# Finding Word Vectors
vector = model.wv['moon']

# Most similar words
similar = model.wv.most_similar('moon')


# In[50]:


similar


# In[ ]:




