import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#import nltk
#nltk.download('stopwords')
#print(stopwords.words('english'))
print("Molimo sacekajte da se ucitaju podaci za trening!")
train_news_dataset = pd.read_csv('train_data.csv')
#print(train_news_dataset.shape)
#print(train_news_dataset.isnull().sum())

train_news_dataset = train_news_dataset.fillna('')

train_news_dataset['content'] = train_news_dataset['author'] + " " + train_news_dataset['title']
x_train_values = train_news_dataset.drop(columns = 'label', axis = 1)
y_train_values = train_news_dataset['label']

#print(x_train)
#print(y_train)

steamer = PorterStemmer()
def steamming_words(content):
    text = re.sub('[^a-zA-Z]', ' ', content)
    text = text.lower()
    text = text.split()
    text = [steamer.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text

train_news_dataset['content'] = train_news_dataset['content'].apply(steamming_words)
print("Podaci su uspesno uneti!\n")
x_train_values = train_news_dataset['content'].values
y_train_values = train_news_dataset['label'].values

#print(x_train_values)
#print(y_train_values)

vectorizer = TfidfVectorizer()
vectorizer.fit(x_train_values)
x_train_values = vectorizer.transform(x_train_values)

model = LogisticRegression()
model.fit(x_train_values, y_train_values)

print("Unesite vesti za testiranje: ")
while(True):
    title = input("Unesite naslov vesti: ")
    author = input("Unesite autora vesti: ")
    text = input("Unesite text vesti: ")
    content = dict()
    content['content'] = title + " " + author
    content['content'] = steamming_words(content['content'])
    x_test_value = []
    x_test_value.append(content['content'])
    x_test_value = vectorizer.transform(x_test_value)
    prediction = model.predict(x_test_value)
    if prediction == 1:
        print("Uneta vest je netacna!")
    else:
        print("Uneta vest je tacna!")
