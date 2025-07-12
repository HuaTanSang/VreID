import os
import sys
sys.path.append(os.path.dirname(__file__))
import argparse
from utils import download_datasets
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key',  type=str, default='itgcVjG83EOkfuwwV7HR')
    parser.add_argument('--work_space',  type=str, default="final-bigdata-project")
    parser.add_argument('--project',  type=str, default="labeling-data-br0ae")
    args = parser.parse_args()
    download_datasets(api_key=args.api_key,work_space=args.work_space,project_name=args.project,vers=None)
