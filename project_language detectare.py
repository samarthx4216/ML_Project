import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("session 24/ld.csv")
print(data['Language'].value_counts())
X=data['Text']
y=data['Language']
le=LabelEncoder()
y=le.fit_transform(y)
import re
data_list=[]
for text in X:
    text=re.sub('[!@#$%()*^:;,~`0-9]',' ',text)
    text=text.lower()
    data_list.append(text)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(data_list).toarray()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
print(model.fit(x_train, y_train))
y_pred = model.predict(x_test)
print(y_pred)
def predict(text):
    x=cv.transform([text]).toarray()
    lang=model.predict(x)
    print(lang)
    lang=le.inverse_transform(lang)
    print(lang)
    print("The langauge is in",lang[0])
predict("Hello and welcome dosto")
import pickle

pickle.dump(model, open('ldmodel.pkl','wb'))
pickle.dump(cv, open('ldvector.pkl','wb'))
import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = FastAPI()

model1 = pickle.load(open('ldmodel.pkl','rb'))
model2 = pickle.load(open('ldvector.pkl','rb'))


@app.get('/')
def index():
    return {'Deployment': 'Hello and Welcome to 5 Minutes Engineering'}

@app.post('/predict')
def nlp(text : str):
    x = model2.transform([text]).toarray()
    prediction = model1.predict(x)
    if(prediction == 0):
        output = 'Arabic'
    if(prediction == 1):
        output = 'Danish'
    if(prediction == 2):
        output = 'Dutch'
    if(prediction == 3):
        output = 'English'
    if(prediction == 4):
        output = 'French'
    if(prediction == 5):
        output = 'German'
    if(prediction == 6):
        output = 'Greek'
    if(prediction == 7):
        output = 'Hindi'
    if(prediction == 8):
        output = 'Italian'
    if(prediction == 9):
        output = 'Kannada'
    if(prediction == 10):
        output = 'Malayalam'
    if(prediction == 11):
        output = 'Portuguese'
    if(prediction == 12):
        output = 'Russian'
    if(prediction == 13):
        output = 'Spanish'
    if(prediction == 14):
        output = 'Swedish'
    if(prediction == 15):
        output = 'Tamil'
    if(prediction == 16):
        output = 'Turkish'

    return {"Prediction": output}


if __name__ == '__main__':
  uvicorn.run(app, host='127.0.0.1', port=10048)