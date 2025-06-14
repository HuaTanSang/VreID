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

spark = SparkSession.builder \
    .appName("Streaming") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()


box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
model = YOLO("yolov8n.pt")


@pandas_udf(BinaryType())
def process_frame_batch(frame_series: pd.Series) -> pd.Series:
    out_bytes = []
    for frame_bytes in frame_series:

        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            out_bytes.append(None)
            continue

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        scene = box_annotator.annotate(scene=frame, detections=detections)
        scene = label_annotator.annotate(scene=scene, detections=detections)

        _, buf = cv2.imencode('.jpg', scene)
        out_bytes.append(buf.tobytes())

    return pd.Series(out_bytes)

spark_df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "read") \
    .option("checkpointLocation", "./checkpoint/read") \
    .option("startingOffsets", "latest") \
    .load()

spark_df = spark_df.withColumn("value", process_frame_batch("value"))

query = spark_df.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "write") \
    .option("checkpointLocation", "./checkpoint/write") \
    .trigger(processingTime='1 seconds') \
    .start()

query.awaitTermination()
spark.stop()
