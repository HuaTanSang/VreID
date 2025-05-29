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

# Khởi tạo YOLO model
model = YOLO('./yolov8n.pt')

# Khởi tạo Spark
spark = SparkSession.builder \
    .appName("Use Yolo") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1") \
    .getOrCreate()

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Hàm xử lý frame
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

# Đăng ký UDF
process_udf = udf(process_frame, BinaryType())

# Đọc stream từ Kafka topic 'read'
spark_df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "read") \
    .option("startingOffsets", "earliest") \
    .load()

# Áp dụng UDF lên cột 'value'
spark_df = spark_df.withColumn("value", process_udf("value"))

# Ghi xuống Kafka topic 'write'
query = spark_df.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "write") \
    .option("checkpointLocation", "./checkpoint/write") \
    .start()

query.awaitTermination()
spark.stop()