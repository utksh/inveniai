# Disease Classifier
This repository contains code for training a random forest classifier on a dataset of patient descriptions of different health conditions, and deploying the trained model as a Flask API. The repository also includes code for Dockerizing the API, setting up a RabbitMQ message queue, and consuming messages from the queue to make API calls for model inference. The inference results, along with the text on which model inference was made, are saved in a PostgreSQL database.

## Prerequisites  
Before running the code, you need to have the following software installed on your machine:

Python 3  
Docker  
RabbitMQ  
PostgreSQL  

You also need to install the following Python packages:  

pandas  
scikit-learn  
Flask  
requests  
psycopg2  
pika   
Usage    

To use the code, follow these steps:

Clone the repository to your local machine:

```bash

git clone https://github.com/utksh/inveniai.git cd disease-classifier

``` 

This will start two containers one for the app and a RabbitMQ consumer that consumes messages from the inference_queue message queue. 

```bash
docker-compose up
```


Send data for inference through the message queue by running the sender.py script:
```python
python sender.py
``` 
Check the inference results in the PostgreSQL database by connecting to the database and querying the inference_results table:

```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="mydatabase",
    user="myusername",
    password="mypassword"
)

cur = conn.cursor()

cur.execute("SELECT * FROM inference_results")

results = cur.fetchall()

for result in results:
    print(result)
