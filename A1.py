#Code for Appendix 1. Algorithms of the Intelligent Web 2nd Edition.

#Listing A.1.1
from kafka import KafkaClient, SimpleProducer

kafka = KafkaClient("localhost:9092")

producer = SimpleProducer(kafka)
producer.send_messages("test", "Hello World!")
producer.send_messages("test","This is my second message")
producer.send_messages("test","And this is my third!")

#Listing A.1.2
from kafka import KafkaClient, SimpleConsumer

kafka = KafkaClient("localhost:9092")				
consumer = SimpleConsumer(kafka,"mygroup","test")					

for message in consumer:
	print(message)

#Listing A.1.5
from kafka import KafkaClient, SimpleProducer

kafka = KafkaClient("localhost:9092")
producer = SimpleProducer(kafka,async=False,
			  req_acks=SimpleProducer.ACK_AFTER_CLUSTER_COMMIT,
			  ack_timeout=2000)

producer.send_messages("test-replicated-topic", "Hello Kafka Cluster!")
producer.send_messages("test-replicated-topic","Message to be replicated.")
producer.send_messages("test-replicated-topic","And so is this!")

#Listing A.1.8
from kafka import KafkaClient
from kafka.common import ProduceRequest
from kafka.protocol import KafkaProtocol,create_message

kafka = KafkaClient("localhost:9092")

f = open('A1.data','r')

for line in f:
	s = line.split("\t")[0]
	part = abs(hash(s)) % 3 
	req = ProduceRequest(topic="click-streams",partition=part,messages=[create_message(s)])
	resps = kafka.send_produce_request(payloads=[req], fail_on_error=True)
