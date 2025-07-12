import os
import sys
sys.path.append(os.path.dirname(__file__))
import argparse
from utils import make_data_for_train_reid
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key',  type=str, default='itgcVjG83EOkfuwwV7HR')
    parser.add_argument('--data_src',  type=str, default=None)
    parser.add_argument('--model_path',  type=str, default=None)
    args = parser.parse_args()
    make_data_for_train_reid(data_src=args.data_src,save_path='Custom_ReID_Dataset',model_path=args.model_path)
