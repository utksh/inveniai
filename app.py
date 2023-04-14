# import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle
from flask import Flask, request, jsonify

# load the dataset
df = pd.read_csv('medical_data.csv')

# split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(df['description'], df['class'], test_size=0.2, random_state=42)

# create a TF-IDF vectorizer
tfidf = TfidfVectorizer()

# fit and transform the training data using the vectorizer
train_data_transformed = tfidf.fit_transform(train_data)

# initialize a Linear Support Vector Classifier
svm = LinearSVC()

# train the model using the transformed training data and the corresponding labels
svm.fit(train_data_transformed, train_labels)

# save the trained model
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

# create a Flask app
app = Flask(__name__)

# define a route for the API
@app.route('/predict', methods=['POST'])
def predict():
    # load the saved model
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # get the request data
    data = request.get_json()
    description = data['description']
    
    # transform the description using the vectorizer
    description_transformed = tfidf.transform([description])
    
    # make a prediction using the loaded model
    prediction = model.predict(description_transformed)[0]
    
    # return the prediction as a JSON response
    response = {'prediction': prediction}
    return jsonify(str(response))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)