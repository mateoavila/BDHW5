import en_core_web_lg
from newsapi import NewsApiClient
import pickle
import pandas as pd
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plot
from nltk.corpus import stopwords
from wordcloud import WordCloud


nlp_eng = en_core_web_lg.load()
newsapi = NewsApiClient(api_key='aec1519c797f4d2a9d38eb866cb10b5d')

articles = newsapi.get_everything(q='coronavirus', language='en', from_param='2021-09-19', to='2020-10-17', sort_by='relevancy', page_size=100)

filename = 'articlesCOVID.pckl'
pickle.dump(articles, open(filename, 'wb'))
filename = 'articlesCOVID.pckl'
loaded_model = pickle.load(open(filename, 'rb'))
filepath = 'articlesCOVID.pckl'
pickle.dump(loaded_model, open(filepath, 'wb'))

dataframe = pd.DataFrame(articles['articles'])
tokenizer = RegexpTokenizer(r'\w+')

def get_keywords_eng(token):
    result = []
    stop_words = stopwords.words('english')

    for i in token:
        if (i in stop_words):
            continue
        else:
            result.append(i)
    return result


results = []
for content in dataframe.content.values:
    content = tokenizer.tokenize(content)
    results.append([val[0] for val in Counter(get_keywords_eng(content)).most_common(5)])
dataframe['keywords'] = results
print(results)


text = str(results)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plot.figure()
plot.imshow(wordcloud, interpolation="bilinear")
plot.axis("off")
plot.show()


