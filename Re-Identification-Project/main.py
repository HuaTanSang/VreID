import sys
import os
sys.path.append(
    os.path.dirname(__file__)
)
import argparse
import cv2
import time
from utils import *
from ultralytics import YOLO
from register import *
import threading
from queue import Queue
import subprocess
from query import *

parser = argparse.ArgumentParser()
parser.add_argument('--input_video_path',  type=str, default='./video/Loc_1_0_inf.mp4')
parser.add_argument('--query_video_path',  type=str, default='./video/Loc_1_1_inf.mp4')
parser.add_argument('--track_model_path', type=str, default='./models/best.pt')
parser.add_argument('--show', type=int, default=0)
parser.add_argument('--time_delay', type=int, default=5)
args = parser.parse_args()

index = faiss.IndexFlatIP(512)
index_to_pid = []
id2feature = {}



if not os.path.exists(args.track_model_path):
    subprocess.run(['python3', 'train_yolo.py'], check=True)

detector_for_track = YOLO(args.track_model_path)
detector_for_detect = YOLO(args.track_model_path)


feature_queue = Queue(maxsize=100)
def feature_worker():
    while True:
        item = feature_queue.get()
        if item is None:
            break
        frame, bboxes, ids = item
        register_features(frame, bboxes, ids, index, index_to_pid, id2feature)
        feature_queue.task_done()

worker = threading.Thread(target=feature_worker, daemon=True)
worker.start()

rotation0 = (get_video_rotation(args.input_video_path)!=0)
rotation1 = (get_video_rotation(args.query_video_path)!=0)

cap0 = cv2.VideoCapture(args.input_video_path)
cap1 = cv2.VideoCapture(args.query_video_path)
if not cap0.isOpened() or not cap1.isOpened():
    print("Không thể mở video")
    sys.exit(1)

start_time = time.time()
skip = 2
frame_id0 = 0
frame_id1 = 0

width0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
height0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps0 = cap0.get(cv2.CAP_PROP_FPS) or 30

width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps1 = cap1.get(cv2.CAP_PROP_FPS) or 30
os.makedirs('./output_video',exist_ok=True)
write0 = cv2.VideoWriter('./output_video/input_video_with_bbox.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps0, (768,448))
write1 = cv2.VideoWriter('./output_video/input_video_with_reid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps1, (768,448))

while True:
    ok0 = cap0.grab()
    ok1 = cap1.grab()
    if not ok0 and not ok1:
        break
    out0 = None
    if ok0 and frame_id0 % 1 == 0:
        _, frame0 = cap0.retrieve()
        if rotation0:
            frame0 = cv2.rotate(frame0, cv2.ROTATE_180)

        res0 = detector_for_track.track(
            source=frame0,conf=0.5, imgsz=640, iou=0.45,
            persist=True, save=False, tracker="bytetrack.yaml"
        )
        info0 = res0[0].boxes
        bboxes0 = info0.xyxy
        ids0    = info0.id
        scores0 = info0.conf
        out0 = draw_bboxes(frame0, bboxes0, scores0, ids0)
        if bboxes0 is not None and ids0 is not None:
            if len(bboxes0) and len(ids0):
                feature_queue.put((frame0.copy(), bboxes0.clone(), ids0.clone()))
    out1 = None
    if ok1 and (time.time() - start_time) >= args.time_delay and frame_id1 % 1 == 0:
        _, frame1 = cap1.retrieve()
        if rotation1:
            frame1 = cv2.rotate(frame1, cv2.ROTATE_180)

        res1 = detector_for_detect.predict(
            source=frame1,conf=0.5, imgsz=640, iou=0.45, save=False
        )
        info1   = res1[0].boxes
        bboxes1 = info1.xyxy
        if bboxes1 is not None:
            if len(bboxes1):
                new_bboxes, new_scores, new_ids = query_features(
                    frame1, bboxes1, index_to_pid, index
                )
                out1 = draw_bboxes(frame1, new_bboxes, new_scores, new_ids)
            else:
                out1 = frame1
        else:
            out1 = frame1

    if out0 is not None:
        if args.show==1:
            cv2.imshow('Camera Input', cv2.resize(out0,(768,448),interpolation=cv2.INTER_AREA))
        write0.write(cv2.resize(out0,(768,448),interpolation=cv2.INTER_AREA))
    if out1 is not None:
        if args.show==1:
            cv2.imshow('Camera Query', cv2.resize(out1,(768,448),interpolation=cv2.INTER_AREA))
        write1.write(cv2.resize(out1,(768,448),interpolation=cv2.INTER_AREA))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id0 += 1
    frame_id1 += 1
    if args.show==1:
        time.sleep(0.005) 


cap0.release()
cap1.release()
write0.release()
write1.release()
cv2.destroyAllWindows()


feature_queue.put(None)
worker.join()
