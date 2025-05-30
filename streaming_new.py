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
from pyspark.sql.functions import pandas_udf
import pandas as pd

# Bật Arrow cho pandas_udf
spark = SparkSession.builder \
    .appName("Use Yolo with Pandas UDF") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
model = YOLO("yolov8n.pt")
# Hàm xử lý batch of frames
@pandas_udf(BinaryType())
def process_frame_batch(frame_series: pd.Series) -> pd.Series:
    out_bytes = []
    for frame_bytes in frame_series:
        # decode
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            out_bytes.append(None)
            continue

        # detect & annotate
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        scene = box_annotator.annotate(scene=frame, detections=detections)
        scene = label_annotator.annotate(scene=scene, detections=detections)

        # encode lại
        _, buf = cv2.imencode('.jpg', scene)
        out_bytes.append(buf.tobytes())

    return pd.Series(out_bytes)

# Đọc stream từ Kafka topic 'read'
spark_df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "read") \
    .option("startingOffsets", "latest") \
    .load()

# Áp dụng pandas_udf lên cột 'value'
spark_df = spark_df.withColumn("value", process_frame_batch("value"))

# Ghi xuống Kafka topic 'write'
query = spark_df.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "write") \
    .option("checkpointLocation", "./checkpoint/write") \
    .trigger(processingTime='1 seconds') \
    .start()

query.awaitTermination()
spark.stop()
