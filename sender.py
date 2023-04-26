import pika
import json
print("here")
# Define the RabbitMQ connection parameters
credentials = pika.PlainCredentials('guest', 'guest')
parameters = pika.ConnectionParameters('localhost', 5672, '/', credentials)
print("connection made")
# Connect to the RabbitMQ server
connection = pika.BlockingConnection(parameters)
print(connection.is_open, "Connection open")
channel = connection.channel()

# Declare the message queue
channel.queue_declare(queue='inference_queue')

# Define the data to be sent for inference
data = {
    'description':'histologic abnormality of large and small coronary artery',
}

# Convert the data to a JSON string
message = json.dumps(data)

print(message)
# Send the message to the message queue
channel.basic_publish(exchange='', routing_key='inference_queue', body=message, mandatory=True)
print(channel.basic_publish)

# Close the RabbitMQ connection
connection.close()
