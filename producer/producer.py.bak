#!python3

import json
import time

from kafka import KafkaProducer

def json_serializer(data):
    return json.dumps(data).encode("utf-8")

if __name__ == "__main__":

    #read data
    with open('logFraud.json','rb') as file:
        file = json.load(file)

    #connect kafka server
    producer = KafkaProducer(bootstrap_servers=['localhost'], 
                             value_serializer=json_serializer)
    
    #push data to kafka server with topic "digitalskola"
    while True:
        for data in file:
            print(data)
            producer.send("digitalskola", data)
            #time.sleep(1)
