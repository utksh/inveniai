# import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle
from flask import Flask, request, jsonify
import pika
import json


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import spacy

# Load the dataset
df = pd.read_csv('health_conditions.csv')

# Define a function to create new features
def create_features(df):
    # Create a feature for the length of the description
    df['desc_length'] = df['description'].apply(lambda x: len(x.split()))
    
    # Create a feature for the number of unique words in the description
    df['unique_words'] = df['description'].apply(lambda x: len(set(x.split())))
    
    # Load the spaCy English model
    nlp = spacy.load("en_core_web_sm")
    
    # Define a function to count the number of adjectives in the description
    def count_adjectives(text):
        doc = nlp(text)
        adjectives = [token for token in doc if token.pos_ == "ADJ"]
        return len(adjectives)
    
    # Create a feature for the number of adjectives in the description
    df['num_adjectives'] = df['description'].apply(count_adjectives)
    
    return df

# Apply the feature engineering function to the dataset
df = create_features(df)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['description', 'desc_length', 'unique_words', 'num_adjectives']], df['disease_class'], test_size=0.2, random_state=42)

# Define the TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit and transform the training data using TF-IDF
X_train_tfidf = tfidf.fit_transform(X_train['description'])

# Add the additional features to the training data
X_train_final = pd.concat([pd.DataFrame(X_train_tfidf.toarray()), X_train[['desc_length', 'unique_words', 'num_adjectives']].reset_index(drop=True)], axis=1)

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
grid_search.fit(X_train_final, y_train)

# Print the best hyperparameters found by grid search
print("Best hyperparameters: ", grid_search.best_params_)

# Transform the testing data using TF-IDF
X_test_tfidf = tfidf.transform(X_test['description'])

# Add the additional features to the testing data
X_test_final = pd.concat([pd.DataFrame(X_test_tfidf.toarray()), X_test[['desc_length', 'unique_words', 'num_adjectives']].reset_index(drop=True)], axis=1)

# Predict the disease class using the trained model
y_pred = grid_search.predict(X_test_final)

# Evaluate the model using F1-score
print("F1-score: ", f1_score(y_test, y_pred, average='weighted'))

# save the trained model
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, World!"

def on_request(ch, method, props, body):
    # load the saved model
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # parse the message body
    message = json.loads(body)
    description = message['description']

    # transform the description using the vectorizer
    description_transformed = tfidf.transform([description])

    # make a prediction using the loaded model
    prediction = model.predict(description_transformed)[0]

    # create a response message
    response = {'prediction': prediction}

    # send the response message back to the sender
    ch.basic_publish(
        exchange='',
        routing_key=props.reply_to,
        properties=pika.BasicProperties(
            correlation_id=props.correlation_id
        ),
        body=json.dumps(response)
    )

    # acknowledge the message
    ch.basic_ack(delivery_tag=method.delivery_tag)

@app.route("/predict", methods=["POST"])
def predict():
    # create a connection to the RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    # create a queue to receive messages
    channel.queue_declare(queue='prediction_queue')

    # generate a unique correlation ID
    correlation_id = str(uuid.uuid4())

    # create a message payload
    payload = {'description': request.json['description']}

    # send the message to the prediction queue
    channel.basic_publish(
        exchange='',
        routing_key='prediction_queue',
        properties=pika.BasicProperties(
            reply_to='prediction_response',
            correlation_id=correlation_id
        ),
        body=json.dumps(payload)
    )

    # set up a listener to receive the prediction response
    channel.basic_consume(
        queue='prediction_response',
        on_message_callback=on_request
    )

    # start consuming messages
    channel.start_consuming()

    # close the connection
    connection.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

