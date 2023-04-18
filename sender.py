import pika
import json

# Define the RabbitMQ connection parameters
credentials = pika.PlainCredentials('guest', 'guest')
parameters = pika.ConnectionParameters('localhost', 5672, '/', credentials)

# Connect to the RabbitMQ server
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

# Declare the message queue
channel.queue_declare(queue='inference_queue')

# Define the data to be sent for inference
data = {
    'description':'histologic abnormality of large and small coronary artery',
}

# Convert the data to a JSON string
message = json.dumps(data)

# Send the message to the message queue
channel.basic_publish(exchange='', routing_key='inference_queue', body=message)

# Close the RabbitMQ connection
connection.close()
