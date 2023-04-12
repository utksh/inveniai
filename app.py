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
    app.run()