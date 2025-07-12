from torchreid.utils import FeatureExtractor 
import torch 

OSNET_DIR = "/home/huatansang/Documents/Big-data/Re-Identification-Project/model_reid/model/model.pth.tar-100"

extractor = FeatureExtractor(
    model_name='osnet_x0_25',
    model_path=OSNET_DIR,
    device='cuda' if torch.cuda.is_available() else 'cpu') 

def extract_feature(image):
    """Extracting feature from image using OSNet model
    """

    features = extractor(image)
    return features