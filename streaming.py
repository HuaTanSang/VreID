import findspark
findspark.init()

import pyspark
import cv2
import yaml
import numpy as np
import supervision as sv
from ultralytics import YOLO
from pyspark.sql import SparkSession
from pyspark.sql.types import BinaryType
from pyspark.sql.functions import udf

# Load config Kafka
def load_kafka_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

model = YOLO('./yolov8n.pt')

spark = SparkSession.builder \
    .appName("Use Yolo") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1") \
    .getOrCreate()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def process_frame(frame_bytes):
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    out = box_annotator.annotate(scene=frame, detections=detections)
    out = label_annotator.annotate(out, detections)

    _, buf = cv2.imencode('.jpg', out)
    return buf.tobytes()

process_udf = udf(process_frame, BinaryType())

spark_df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "read") \
    .option("startingOffsets", "latest") \
    .load()

spark_df = spark_df.withColumn("value", process_udf("value"))

query = spark_df.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "write") \
    .option("checkpointLocation", "./checkpoint/write") \
    .trigger(processingTime='1 seconds')\
    .start()

query.awaitTermination()
spark.stop()