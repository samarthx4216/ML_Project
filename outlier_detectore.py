# Importing required libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

# Generate synthetic dataset with 2 clusters
x, y = make_blobs(n_samples=2000, centers=2, cluster_std=0.9, random_state=0)

# Plot the generated data points
plt.scatter(x[:, 0], x[:, 1])
plt.show()

# Convert dataset into DataFrame
df = pd.DataFrame(data=x, columns=['x1', 'x2'])
print(df)

# Importing Elliptic Envelope for outlier detection
from sklearn.covariance import EllipticEnvelope

# Create model (1% contamination means ~1% points considered as outliers)
ee = EllipticEnvelope(contamination=0.01)

# Train the model on the dataset
ee.fit(df)

# Predict outliers (-1 for outlier, 1 for inlier)
prediction = ee.predict(df)
print(df)

# Visualize the data points with outliers highlighted
plt.scatter(df['x1'], df['x2'], c=prediction)
plt.show()

# Print the prediction array
print(prediction)

# Find the index positions of outliers
outliers = np.where(prediction == -1)
print(outliers)

# Save trained model into a pickle file
import pickle
pickle.dump(ee, open('outliers.pkl', 'wb'))

# Import FastAPI and Uvicorn for API deployment
import uvicorn
from fastapi import FastAPI

# Create FastAPI app
app = FastAPI()

# Load trained model from pickle file
pickle_in = open('outliers.pkl', 'rb')
classifier = pickle.load(pickle_in)
print(classifier)

# Root endpoint
@app.get('/')
def index():
    return {"Deployment": "Hello and Welcome to 5 Minutes Engineering"}

# Prediction endpoint
@app.post('/prediction')
def predict(x1: int, x2: int):
    """
    Predict whether a given point (x1, x2) is an outlier.
    Input: x1, x2 (integer values)
    Output: 'outliers' or 'not outliers'
    """
    prediction = classifier.predict([[x1, x2]])
    if prediction[0] == 1:
        prediction = 'not outliers'
    elif prediction[0] == -1:
        prediction = 'outliers'
    return {
        "prediction": prediction
    }

# Run the FastAPI app using Uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)
