import os, glob
import sys
sys.path.append(os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
import custom_dataset
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import extract_feature
from utils import *

if not os.path.exists(os.path.join(os.path.dirname(__file__),'model_reid/model', os.listdir(os.path.join(os.path.dirname(__file__),'model_reid/model'))[-1])):
    raise Exception('You must have model at /model_reid/model for find the best threshold')

def load_image_list(folder):
    data = []
    for path in glob.glob(os.path.join(folder, '*.jpg')):
        fname = os.path.basename(path)
        pid = int(fname.split('_')[0]) 
        data.append((path, pid))
    return data

def extract_features(data_list):
    feats, pids = [], []
    for img_path, pid in data_list:
        feat = extract_feature(img_path)[0].cpu().numpy()
        feats.append(feat)
        pids.append(pid)
    return np.vstack(feats), np.array(pids)

def extract_features(data_list):
    feats, pids = [], []
    for img_path, pid in data_list:
        feat = extract_feature(img_path)[0].cpu().numpy()
        feats.append(feat)
        pids.append(pid)
    return np.vstack(feats), np.array(pids)

if __name__ == '__main__':
    if not os.path.exists('./Custom_ReID_Dataset/dataset_0'):
        make_data_for_train_reid()
    query_list = load_image_list('./Custom_ReID_Dataset/dataset_0/query')
    gallery_list = load_image_list('./Custom_ReID_Dataset/dataset_0/gallery')


    query_feats, query_pids = extract_features(query_list)
    gallery_feats, gallery_pids = extract_features(gallery_list)

    scores, labels = [], []
    for qf, qpid in zip(query_feats, query_pids):
        sims = cosine_similarity(qf.reshape(1, -1), gallery_feats).flatten()
        for sim, gpid in zip(sims, gallery_pids):
            scores.append(sim)
            labels.append(int(qpid == gpid))

    scores = np.array(scores)
    labels = np.array(labels)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    youden = tpr - fpr
    best_ix = np.argmax(youden)
    best_thresh = thresholds[best_ix]

    # print(best_thresh)
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title(f"ROC Curve for ReID and best threshold {best_thresh}")
    plt.grid(True)
    plt.show()
