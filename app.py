# import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle
from flask import Flask, request, jsonify
import pika


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import spacy

# Load the dataset
df = pd.read_csv('medical_data.csv')

# Define a function to create new features
def create_features(df):
    # Create a feature for the length of the description
    df['desc_length'] = df['description'].apply(lambda x: len(x.split()))
    
    # Create a feature for the number of unique words in the description
    df['unique_words'] = df['description'].apply(lambda x: len(set(x.split())))
    
    # Create a feature for the number of adjectives in the description
    df['num_adjectives'] = df['description'].apply(count_adjectives)
    
    return df

# Define a function to count the number of adjectives in the description
def count_adjectives(text):
    # Load the spaCy English model
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    adjectives = [token for token in doc if token.pos_ == "ADJ"]
    return len(adjectives)

# Apply the feature engineering function to the dataset
# df = create_features(df)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['description']], df['class'], test_size=0.2, random_state=42)

# Define the TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit and transform the training data using TF-IDF
X_train_tfidf = tfidf.fit_transform(X_train['description'])

# Add the additional features to the training data
# X_train_final = pd.concat([pd.DataFrame(X_train_tfidf.toarray()), X_train[['desc_length', 'unique_words', 'num_adjectives']].reset_index(drop=True)], axis=1)
X_train_final =  pd.DataFrame(X_train_tfidf.toarray())
# Define the random forest classifier
rfc = RandomForestClassifier()

# Define the grid search parameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(rfc, param_grid, cv=5, n_jobs=-1)
rf = grid_search.fit(X_train_final, y_train)

# Print the best hyperparameters found by grid search
print("Best hyperparameters: ", grid_search.best_params_)

# Transform the testing data using TF-IDF
X_test_tfidf = tfidf.transform(X_test['description'])

# Add the additional features to the testing data
# X_test_final = pd.concat([pd.DataFrame(X_test_tfidf.toarray()), X_test[['desc_length', 'unique_words', 'num_adjectives']].reset_index(drop=True)], axis=1)
X_test_final = pd.DataFrame(X_test_tfidf.toarray())
# Predict the disease class using the trained model
y_pred = grid_search.predict(X_test_final)

# Evaluate the model using F1-score
print("F1-score: ", f1_score(y_test, y_pred, average='weighted'))

# save the trained model
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, World!"

# Define the route for the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.json
    
    # Load the trained model
    with open('trained_model.pkl', 'rb') as f:
        rfc_model = pickle.load(f)
    
    # Transform the input data using TF-IDF
    input_tfidf = tfidf.transform([input_data['description']])
    
    # Create the additional features for the input data
    # input_features = [[len(input_data['description'].split()), len(set(input_data['description'].split())), count_adjectives(input_data['description'])]]
    
    # Concatenate the TF-IDF and additional features for the input data
    # input_final = pd.concat([pd.DataFrame(input_tfidf.toarray()), pd.DataFrame(input_features, columns=['desc_length', 'unique_words', 'num_adjectives'])], axis=1)
    input_final = pd.DataFrame(input_tfidf.toarray())
    # Predict the disease class using the trained model
    prediction = rfc_model.predict(input_final)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

