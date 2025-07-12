import os
import sys
sys.path.append(os.path.dirname(__file__))
from collections import Counter
import cv2
import torch
import faiss
import numpy as np
from feature_extractor import extract_feature

def vote_topk(scores, indices, index_to_pid, threshold=0.599):
    votes = []
    score_dict={}

    for score, idx in zip(scores, indices):
        if score >= threshold:
            pid = index_to_pid[idx]
            if pid not in score_dict:
                score_dict[pid]=0
            else:
                score_dict[pid]+=score
            votes.append(pid)
    if not votes:
        return -1,0
    
    counter = Counter(votes)
    voted_pid, cnt = counter.most_common(1)[0]

    if cnt <= max(len(votes)//2,2):
        return -1,0
    
    return voted_pid,score_dict[voted_pid]/cnt

def query_features(image, bboxes, index_to_pid, index):
    
    """Querying features from detected objects in the image.
    This function extracts features from the detected objects and searches them in the index.
    Args:
        image (np.ndarray): The input image containing detected objects.
        bboxes (list): List of bounding boxes for detected objects.
        index_to_pid (list): List mapping indices to person IDs.
        index (faiss.IndexFlatIP): FAISS index for searching features.
    Returns:
        new_bboxes (list): List of bounding boxes after querying.
        new_scores (list): List of scores for the queried bounding boxes.
        new_ids (list): List of person IDs for the queried bounding boxes."""
    

    cp_image = image.copy()
    
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()
    if len(bboxes.shape) == 1:
        bboxes = bboxes.reshape(1, -1)
        
    bboxes = bboxes.tolist()
    new_bboxes=[]
    new_scores=[]
    new_ids=[]
    
    for box in bboxes:
        x_min, y_min, x_max, y_max = box
        person_img = cp_image[int(y_min):int(y_max), int(x_min):int(x_max)]

        if person_img.size == 0:
            continue
        
        cut_img = cv2.cvtColor(cv2.resize(person_img, (128, 256), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB)
        query_feat = extract_feature(cut_img)[0].detach().cpu().numpy().astype(np.float32).reshape(1,-1)
        faiss.normalize_L2(query_feat)
        
        D, I = index.search(query_feat, 5)
        score = D[0]
        index_id = I[0]
        voted_pid,reid_score=vote_topk(score,index_id,index_to_pid)

        print(f"[DEBUG] {voted_pid}")
        
        if voted_pid != -1:
            new_bboxes.append(box)
            new_scores.append(reid_score)
            new_ids.append(str(voted_pid))
    
    return new_bboxes,new_scores,new_ids