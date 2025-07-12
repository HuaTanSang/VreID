import torch

import os
import faiss
import cv2
import numpy as np
import os
import sys

sys.path.append(
    os.path.dirname(__file__)
)
from feature_extractor import extract_feature
if not os.path.exists(os.path.join(os.path.dirname(__file__),'model_reid/model', os.listdir(os.path.join(os.path.dirname(__file__),'model_reid/model'))[-1])):
    raise Exception('You must have model at /model_reid/model for find the best threshold')
def register_features(image, bboxes, pids,index,index_to_pid,id2feature):
    cp_image = image.copy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()
    if len(bboxes.shape) == 1:
        bboxes = bboxes.reshape(1, -1)
    bboxes = bboxes.tolist()
    pids = pids.tolist()

    for box, pid in zip(bboxes, pids):
        x_min, y_min, x_max, y_max = box
        person_img = cp_image[int(y_min):int(y_max), int(x_min):int(x_max)]
        if person_img.size == 0:
            continue
        cut_img = cv2.cvtColor(cv2.resize(person_img, (128, 256), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB)
        feat = extract_feature(cut_img)[0].detach().cpu().numpy().astype(np.float32)

        pid_str = str(pid)
        if pid_str not in id2feature:
            id2feature[pid_str] = []
        id2feature[pid_str].append(feat)

        if len(id2feature[pid_str]) % 5 == 0:
            feature_mean = np.sum(id2feature[pid_str], axis=0, keepdims=True)
            faiss.normalize_L2(feature_mean)
            index.add(feature_mean)
            index_to_pid.append(pid_str)
            id2feature[pid_str] = []
