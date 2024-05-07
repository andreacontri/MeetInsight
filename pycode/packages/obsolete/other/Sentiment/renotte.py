# code from https://www.youtube.com/watch?v=szczpgOEdXs, uses the BERT model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

tokens = tokenizer.encode('It was good but couldve been better. Great', return_tensors='pt')
result = model(tokens)
# print(result)
result.logits
int(torch.argmax(result.logits))+1

r = requests.get('https://www.yelp.com/biz/social-brew-cafe-pyrmont')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class':regex})
reviews = [result.text for result in results]

import numpy as np
import pandas as pd
df = pd.DataFrame(np.array(reviews), columns=['review'])
df['review'].iloc[0]
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1

df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))
df
df['review'].iloc[3]