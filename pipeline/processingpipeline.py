import cv2 
import numpy as np 

from ultralytics import YOLO 
from register import * 
from query import * 
from utils import draw_bboxes
import faiss
import time 

class ProcessingPipeline:
    def __init__(self, 
                 detector: YOLO,
                 tracker: YOLO,
                 index: faiss.IndexFlatIP,
                 index_to_pid: list,
                 id2feature: dict,
                 draw_bboxes = draw_bboxes, 
                 register = register_features, 
                 query = query_features) -> None:
        
        self.draw_bboxes = draw_bboxes
        self.detector = detector
        self.tracker = tracker 
        self.register = register
        self.query = query
        self.index = index
        self.index_to_pid = index_to_pid
        self.id2feature = id2feature

    def process_frame(self, frame, topic): 
        """
        Processing frame with UDF. 
        Using customized YOLO model to track and detect object. 
        """
        try:
            print(f"[INFO] Start processing frame from topic: {topic}")

            out = frame.copy()

            # Routing the topic
            if topic == "sending-cam1":
                res = self.tracker.track(
                    source=frame, conf=0.5, imgsz=640,
                    iou=0.45, persist=True, save=False, tracker="bytetrack.yaml"
                )

                info = res[0].boxes
                bboxes = info.xyxy
                scores = info.conf
                ids = info.id
                self.register(out, bboxes, ids, self.index, self.index_to_pid, self.id2feature)

                out = self.draw_bboxes(frame, bboxes, scores, ids)
                print(f"[DEBUG] Processed topic {topic}, {len(bboxes)} boxes with ids: {ids}")

            else: #  topic == "sending-cam2"

                res = self.detector.predict(
                    source=frame, conf=0.5, imgsz=640, iou=0.45, save=False
                )
                info = res[0].boxes
                bboxes = info.xyxy if info is not None else []
                scores = info.conf if info is not None else []
                ids = None  # Because prediction does not have ids

                print(f"[DEBUG] Detected COMPLETE")
                # time.sleep(1)
                if len(bboxes) > 0:
                    new_bboxes, new_scores, new_ids = self.query(frame, bboxes, self.index_to_pid, self.index)
                    print("THIS IS QUERY FOR CAM2")
                    # time.sleep(1)
                    print(f"[DEBUG] Found {len(new_bboxes)} new boxes with ids: {new_ids}")
                    # time.sleep(10)  # Adding a small delay to simulate processing time
                    out = self.draw_bboxes(out, new_bboxes, new_scores, new_ids)
                else:
                    out = frame
                
                print(f"[DEBUG] Processed topic {topic}, {len(bboxes)} boxes with ids: {ids}")

            return out

        except Exception as e:
            print(f"[ERROR] {e} in processingpipeline.py at line 76")
            return frame         
            