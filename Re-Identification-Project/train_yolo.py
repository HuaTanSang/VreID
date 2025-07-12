from ultralytics import YOLO
import os
import sys
import argparse
import shutil
from utils import *
# Tạo parser
sys.path.append(
    os.path.dirname(__file__)
)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse argument")
    parser.add_argument('--config', type=str,default='./Train_Yolo_Dataset/config.yaml', help='config file')
    args = parser.parse_args()
    model = YOLO('yolo12s.pt')
    if os.path.exists(args.config):
        convert_datasets_to_yolo_format(vers=None)
    results = model.train(
        data=args.config,
        epochs=60,
        imgsz=640,
        batch=8,
    )
    os.makedirs('./models',exist_ok=True)
    shutil.copy2(os.path.join(results.save_dir,'weights/best.pt'),'./models')

