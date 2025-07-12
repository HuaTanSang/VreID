"""
This file is to process frames for re-identification in a streaming context.
Depennds on the topic's name. It can perform to registering object or querying them. 

"""

import cv2 
import numpy as np 
import torch 
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType
from pipeline.utils import draw_bboxes
from pipeline.streaming_detection import tracker, detector
from pipeline.feature_extractor import extract_feature
from collections import Counter
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, BinaryType
from .processingpipeline import ProcessingPipeline
from ultralytics import YOLO 
import faiss
global_procedure = None 

def start_global_variables():
    """
    Initializing global variables for the pipeline."""
    global global_procedure
    if global_procedure is None:
        global_procedure = ProcessingPipeline(
                tracker = YOLO('/home/huatansang/Documents/Big-data/Re-Identification-Project/models/best.pt'), 
                detector = YOLO('/home/huatansang/Documents/Big-data/Re-Identification-Project/models/best.pt'),
                index = faiss.IndexFlatIP(512),  # Assuming 512 is the dimension of the feature vector
                index_to_pid = [],
                id2feature = {},
        )
        print("[INFO] Global variables initialized for the processing pipeline.")
    
    return global_procedure

    
def process_frame(value, topic):
    """
    Processing frame with UDF. 
    Using customized YOLO model to detect object. 
    """
    from pyspark.sql.functions import udf
    from pyspark.sql.types import BinaryType
    from pipeline.streaming_detection import tracker, detector, draw_bboxes

    try:
        print(f"[INFO] Start processing frame from topic: {topic}")

        frame_buffer = np.frombuffer(value, dtype=np.uint8)
        frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
        out = frame.copy()

        if frame is None:
            print(f"[ERROR] Cannot decode frame from topic: {topic}")
            return value

        # Routiing the topic
        if topic == "sending-cam1":
            res = tracker.track(
                source=frame, conf=0.5, imgsz=640,
                iou=0.45, persist=True, save=False, tracker="bytetrack.yaml"
            )
            ids = res[0].boxes.id

        elif topic == "sending-cam2":
            res = detector.predict(
                source=frame, conf=0.5, imgsz=640, iou=0.45, save=False
            )
            ids = None  # Because of predict does not have ids

        # Extract information for the boxes
        info = res[0].boxes
        bboxes = info.xyxy
        scores = info.conf

        # Draw bounding box
        if bboxes is not None:
            out = draw_bboxes(frame, bboxes, scores, ids)


        print(f"[SUCCESS] Processed and encoded frame from topic: {topic}")
        del res, ids

        return out, topic

    except Exception as e:
        print(f"[UDF-YOLO] Exception while processing frame from {topic}: {e}")
        import traceback
        traceback.print_exc()
        import gc
        gc.collect()
        return value