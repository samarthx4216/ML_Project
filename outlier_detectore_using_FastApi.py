import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
x,y=make_blobs(n_samples=2000,centers=2,cluster_std=0.9,random_state=0)
plt.scatter(x[:,0],x[:,1])
plt.show()
df=pd.DataFrame(data=x,columns=['x1','x2'])
print(df)
from sklearn.covariance import EllipticEnvelope
ee=EllipticEnvelope(contamination=0.01)
ee.fit(df)
prediction=ee.predict(df)
print(df)
plt.scatter(df['x1'], df['x2'],c=prediction)
plt.show()
print(prediction)
outliers=np.where(prediction==-1)
print(outliers)
import pickle
pickle.dump(ee,open('outliers.pkl','wb'))
import uvicorn
from fastapi import FastAPI
app=FastAPI()
pickle_in=open('outliers.pkl','rb')
calssifier=pickle.load(pickle_in)
print(calssifier)
@app.get('/')
def index():
    return{'Deployment':"Hello and Welcome to 5 Minutes Engineering"}
@app.post('/prediction')
def predict(x1:int,x2:int):
    prediction=calssifier.predict([[x1,x2]])
    if(prediction[0]==1):
        prediction='not outliers'
    elif(prediction[0]==-1):
        prediction='outliers'
    return{
        "prediction" : prediction
    }
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)