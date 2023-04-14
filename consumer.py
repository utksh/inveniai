import pika
import requests
import json
import psycopg2


connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare(queue='prediction_queue')

conn = psycopg2.connect(host='localhost', dbname='health_conditions', user='postgres', password='password')


def callback(ch, method, properties, body):
    # make an API call to the prediction endpoint
    url = 'http://localhost:5000/predict'
    headers = {'Content-type': 'application/json'}
    data = {'description': json.loads(body)['description']}
    response = requests.post(url, json=data, headers=headers)

   # get the prediction result
    prediction = response.json()['prediction']
    confidence = response.json()['confidence']

    # insert the inference result into the database
    cur = conn.cursor()
    cur.execute("INSERT INTO inference_results (description, predicted_class, confidence) VALUES (%s, %s, %s)", (json.loads(body)['description'], prediction, confidence))
    conn.commit()

    # print the prediction result
    print(prediction)


    # acknowledge the message
    ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='prediction_queue', on_message_callback=callback)

channel.start_consuming()
