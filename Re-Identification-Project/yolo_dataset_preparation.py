import os
import sys
sys.path.append(os.path.dirname(__file__))
import argparse
from utils import convert_datasets_to_yolo_format
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_src',  type=str, default=None)
    args = parser.parse_args()
    convert_datasets_to_yolo_format(args.data_src,vers=None)