import pandas as pd
import numpy as np
dataset = pd.read_csv('session 17/P2/spam_detection.csv')
dataset
dataset.label.value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset.label = le.fit_transform(dataset.label)
dataset
x = dataset[['num_links', 'num_words', 'num_special_chars', 'has_spammy_keywords']]
y = dataset[['label']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=1)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test,y_test)
import pickle
pickle.dump(model, open('mlclass.pkl','wb'))
import uvicorn
from fastapi import FastAPI
import pickle
app = FastAPI()
pickle_in = open("mlclass.pkl","rb")
classifier=pickle.load(pickle_in)

@app.get('/')
def index():
    return {'Deployment': 'Hello and Welcome to 5 Minutes Engineering'}

@app.post('/predict')
def predict(num_links:int, num_words:int, num_special_chars:int, has_spammy_keywords:int):

    prediction = classifier.predict([[num_links,num_words,num_special_chars,has_spammy_keywords]])
    if(prediction[0] == 0):
        prediction="NOT SPAM"
    else:
        prediction="SPAM"
    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)