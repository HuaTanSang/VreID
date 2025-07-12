

import os 
import cv2 
import threading
import numpy as np 

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf 
from pyspark.sql.types import BinaryType

from ultralytics import YOLO 
from processingpipeline import ProcessingPipeline
import faiss


KAFKA_SERVERS = os.getenv('KAFKA_BROKERS', 'localhost:9092')
INPUT_TOPICS = "sending-cam1,sending-cam2"
CHECKPOINT_PATH = "/home/huatansang/Documents/Big-data/checkpointt"
MODEL_PATH = "/home/huatansang/Documents/Big-data/Re-Identification-Project/models/best.pt"


global_procedure = None 

def start_global_variables():
    """
    Initializing global variables for the pipeline.
    """
    
    global global_procedure
    if global_procedure is None:
        global_procedure = ProcessingPipeline(
                tracker = YOLO(MODEL_PATH), 
                detector = YOLO(MODEL_PATH), 
                index = faiss.IndexFlatIP(512),  
                index_to_pid = [],
                id2feature = {},
        )
        print("[INFO] Global variables initialized for the processing pipeline.")
    
    return global_procedure

def start_streaming(): 
    """
    Start the streaming process using Spark Structured Streaming."""
    import findspark 
    findspark.init() 
    
    spark = SparkSession.builder \
                .master('local') \
                .appName("vehicle-reid-chromadb") \
                .config("spark.jars.packages","org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.0") \
                .config("spark.sql.adaptive.enabled", "false") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
                .config("spark.serializer.objectStreamReset", "1") \
                .config("spark.rdd.compress", "false") \
                .config("spark.kryo.unsafe", "true") \
                .config("spark.kryoserializer.buffer.max", "512m") \
                .getOrCreate()

    spark_df = spark.readStream.format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_SERVERS) \
        .option("subscribe", INPUT_TOPICS) \
        .option("checkpointLocation", CHECKPOINT_PATH + "/read") \
        .option("startingOffsets", "latest") \
        .load() 
        
    spark_df = spark_df.withColumn("value", spark_df["value"].cast(BinaryType())) 
    

    @udf(BinaryType())
    def process_frame_udf(value, topic):
        try:
            print(f"[INFO] Start processing frame from topic: {topic}")
            global_procedure = start_global_variables()

            if not value:
                print(f"[ERROR] Empty buffer received from topic: {topic}")
                return value

            frame_buffer = np.frombuffer(value, dtype=np.uint8)
            frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)

            if frame is None:
                print(f"[ERROR] Cannot decode frame from topic: {topic}")
                return value

            out = global_procedure.process_frame(frame, topic)
            if out is None:
                print(f"[ERROR] Pipeline returned None for topic: {topic}")
                return value

            success, buffer = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not success:
                print(f"[ERROR] Failed to encode output frame for topic: {topic}")
                return value

            print(f"[SUCCESS] Processed and encoded frame from topic: {topic}")
            return buffer.tobytes()

        except Exception as e:
            print(f"[UDF-YOLO] Exception while processing frame from {topic}: {e}")
            import traceback
            traceback.print_exc()
            return value

    processed_spark = spark_df \
            .selectExpr("CAST(key AS STRING) as key",
                        "CAST(topic as STRING) as topic",
                        "value") \
            .withColumn("value", process_frame_udf("value", "topic"))


    query_cam1 = processed_spark \
        .filter("topic = 'sending-cam1'") \
        .select("key", "value") \
        .writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("topic", "receive-cam1") \
        .option("checkpointLocation", "./checkpoint/write-cam1") \
        .outputMode("append") \
        .start()


    query_cam2 = processed_spark \
        .filter("topic = 'sending-cam2'") \
        .select("key", "value") \
        .writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("topic", "receive-cam2") \
        .option("checkpointLocation", "./checkpoint/write-cam2") \
        .outputMode("append") \
        .start()

    query_cam1.awaitTermination()
    query_cam2.awaitTermination() 
    spark.stop() 


    thread = threading.Thread(target=start_global_variables).start()
    return thread 