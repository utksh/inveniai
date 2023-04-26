import psycopg2
import json
import pika
import requests
# Connect to the PostgreSQL database
conn = psycopg2.connect(
    host="0.0.0.0",
    database="predictions",
    user="postgres",
    password="postgres",
    port=5432,
)
print("consuming")
# Define a cursor object to execute SQL queries
cur = conn.cursor()

# Create a table for storing the inference results
cur.execute("""
    CREATE TABLE IF NOT EXISTS inference_results (
        id SERIAL PRIMARY KEY,
        patient_description TEXT,
        disease_class TEXT,
        predicted_class TEXT,
        predicted_probability FLOAT
    )
""")

# Define the RabbitMQ connection parameters
credentials = pika.PlainCredentials('guest', 'guest')
parameters = pika.ConnectionParameters('localhost', 5672, '/', credentials)

# Connect to the RabbitMQ server
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

# Declare the message queue
channel.queue_declare(queue='inference_queue')

# Define the callback function for consuming messages
def callback(ch, method, properties, body):
    # Convert the JSON string to a Python dictionary
    data = json.loads(body)

    # Make an API call for model inference
    url = 'http://localhost:8000/predict'
    headers = {'Content-type': 'application/json'}
    response = requests.post(url, headers=headers, json=data)

    # Parse the model prediction
    predicted_class = response.json()['predicted_class']
    predicted_probability = response.json()['predicted_probability']

    # Insert the inference results into the database
    cur.execute("""
        INSERT INTO inference_results (patient_description, disease_class, predicted_class, predicted_probability)
        VALUES (%s, %s, %s, %s)
    """, (data['patient_description'], data['class'], predicted_class, predicted_probability))
    conn.commit()

# Start consuming messages from the message queue
channel.basic_consume(queue='inference_queue', on_message_callback=callback, auto_ack=False)
print("Called callback")
channel.start_consuming()

# Close the database connection
cur.close()
conn.close()
