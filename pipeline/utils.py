import cv2 
import torch 

def draw_bboxes(original_image, 
                bboxes,
                scores,
                ids=None): 
    """
    Drawing box for each object in the frame
    """
    def draw_box(image,
                 box,
                 conf,
                 id=None):
        """
        Drawing one box
        """

        x_min,y_min,x_max,y_max = box
        x_min,y_min,x_max,y_max=int(x_min),int(y_min),int(x_max),int(y_max)

        H,W = image.shape[:2]
        image = cv2.rectangle(image, (x_min,y_min), (x_max,y_max), (0,255,0), 2)
        image = cv2.putText(image, f'conf: {conf:.2f}',(x_min,max(y_min-30,0)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0,255,255), 2, cv2.LINE_AA)
        
        if id:
            image = cv2.putText(image, f'id: {id}',(x_min,max(y_min-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0,255,255), 2, cv2.LINE_AA)
            
        return image
    
    if ids is not None:
        if len(ids) != len(bboxes):
            raise Exception('List ids must have the same length with bboxes')
        
        if isinstance(ids,torch.Tensor):
            ids=ids.tolist()

    else:
        ids = [None]*len(bboxes)

    if isinstance(bboxes,torch.Tensor):
        bboxes = bboxes.tolist() if bboxes is not None else []
        scores = scores.tolist() if scores is not None else []

    if len(bboxes) ==0: 
        return original_image
    
    cp_image = original_image.copy()

    for box,conf,id in zip(bboxes,scores,ids):
        print(box,conf,id)
        cp_image=draw_box(cp_image,box,conf,id)
    
    print(f"[DEBUG] Drawed {len(bboxes)} boxes")
    return cp_image