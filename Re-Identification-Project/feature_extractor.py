from torchreid.utils import FeatureExtractor
import os
import sys
import torch
sys.path.append(
    os.path.dirname(__file__)
)
extractor = FeatureExtractor(
    model_name='osnet_ain_x1_0',
    model_path=os.path.join(os.path.dirname(__file__),'model_reid/model', os.listdir(os.path.join(os.path.dirname(__file__),'model_reid/model'))[-1]),
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

def extract_feature(img):
    return extractor(img)
