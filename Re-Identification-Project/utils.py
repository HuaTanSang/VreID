import os
import sys
sys.path.append(os.path.dirname(__file__))
from roboflow import Roboflow
import numpy as np
import concurrent.futures
import glob
import shutil
import yaml
import torch
import cv2
from PIL import Image
from ultralytics import YOLO

def download_datasets(api_key='itgcVjG83EOkfuwwV7HR',work_space="final-bigdata-project",project_name="labeling-data-br0ae",vers=None):
    if vers is None:
        vers=[6,7,8,10]
    os.makedirs('Roboflow_Datasets',exist_ok=True)
    data_path= os.path.join(os.path.dirname(__file__),'Roboflow_Datasets')
    def download_one_dataset(ver,data_path):
        try:
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(work_space).project(project_name)
            version = project.version(ver)
            dataset = version.download("yolov5",location=data_path+f'/Reid_data_ver_{ver}')
        except:
            pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as Executer:
        Executer.map(lambda x :download_one_dataset(x,data_path),vers)
    print('SUCCESS!')
    print(f'Data Path: {data_path}')
    return data_path
    
def rewrite_label(label_paths,src_path):
    def write_label(label_paths,src_path):
        if os.path.getsize(label_paths) > 0:
            arr=np.loadtxt(
                label_paths
            )
            if len(arr.shape)==1:
                arr=arr.reshape(1,-1)
            arr[:,0]=0
            np.savetxt(src_path+'/'+label_paths.split('/')[-1],arr,"%.6f",delimiter=" ")
            return True
        return False
    all_txt_path = sorted(glob.glob(label_paths+'/*.txt'),key=lambda x:x.split('/')[-1])
    # print(all_txt_path)
    if len(all_txt_path)==0:
        raise Exception(f"Don't have any txt file in {all_txt_path}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as Executer:
        output=list(Executer.map(lambda x:write_label(x,src_path),all_txt_path))
    return output

def convert_datasets_to_yolo_format(data_path=None,vers=None):

    root_path = os.path.dirname(__file__)
    yolo_data_path = os.path.join(root_path,'Train_Yolo_Dataset')
    img_path = os.path.join(yolo_data_path,'images')
    label_path =  os.path.join(yolo_data_path,'labels')
    os.makedirs(img_path,exist_ok=True)
    os.makedirs(label_path,exist_ok=True)
    if data_path is None :
        download_datasets(vers=vers)
        data_path  =root_path+'/Roboflow_Datasets'
        if not os.path.exists(
            data_path
        ): raise Exception(
            "Don't have root data path to convert"
        )
    for type_ in ['train','valid']:
        # try:
        original_img_path=data_path+f'/*/{type_}/images'
        original_label_path=data_path+f'/*/{type_}/labels'
        os.makedirs(img_path+f'/{type_}',exist_ok=True)
        os.makedirs(label_path+f'/{type_}',exist_ok=True)
        all_image_paths = np.array(sorted(glob.glob(original_img_path+'/*.jpg'),key=lambda x:x.split('/')[-1]))
        conditions = rewrite_label(original_label_path,label_path+f'/{type_}')
        # print(conditions)
        all_image_paths=all_image_paths[conditions].tolist()
        # print(all_image_paths)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as Executer:
            Executer.map(
                lambda x:shutil.copy2(x,img_path+f'/{type_}/'),all_image_paths
                )
        # except:pass
    data = {
        'path': yolo_data_path,
        'train': 'images/train',
        'val': 'images/valid' if os.path.exists(yolo_data_path+'/images/valid') else '',
        'test': '',
        'names': {
            0: 'person',
        }
    }
    with open(f"{yolo_data_path}/config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)
    print(f'Path to config file: {yolo_data_path}/config.yaml')
    return f"{yolo_data_path}/config.yaml"

def draw_bboxes(original_image,bboxes,scores,ids=None): 
    def draw_box(image,box,conf,id=None):
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
        bboxes=bboxes.tolist()
        scores=scores.tolist()
    if len(bboxes)==0:return original_image
    cp_image = original_image.copy()
    for box,conf,id in zip(bboxes,scores,ids):
        print(box,conf,id)
        cp_image=draw_box(cp_image,box,conf,id)
    return cp_image

def conver_format_bbox(box,H=900,W=1600):
    x,y,w,h= box
    x=x*W
    y=y*H
    w = w * W
    h = h * H

    x_min = x- w / 2
    y_min = y - h / 2
    x_max = x + w / 2
    y_max = y + h / 2
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def cut_image(image, bbox):
    h,w = image.shape[:2]
    cp_image = image.copy()
    x_min, y_min, x_max, y_max = conver_format_bbox(bbox,h,w)
    print(f"Cutting image with bbox: {bbox} -> ({x_min}, {y_min}), ({x_max}, {y_max})")
    return cv2.cvtColor(cv2.resize(cp_image[y_min:y_max, x_min:x_max], (128, 256),interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB).astype(np.uint8)


def save_image_for_reid(image_paths,label_paths,config_yaml,save_path='Custom_ReID_Dataset',model_path=None):
    root_path = os.path.dirname(__file__)
    save_path = os.path.join(root_path,save_path)
    os.makedirs(save_path,exist_ok=True)
    with open(config_yaml,"r") as f:
        config = yaml.safe_load(f)
        mapping_id = config['names']
    def save_img(image_path,label_path,mapping_id,model_path=None):
        if model_path is None:
            if os.path.exists(os.path.join(os.path.dirname(__file__),'models/best.pt')):
                model_path = os.path.join(os.path.dirname(__file__),'models/best.pt')
        model = YOLO(model_path) if model_path is not None else None
        path_inf = image_path.split('/')
        frame_id = path_inf[-1].split('-')[1].split('_')[0]
        cam_id = int(path_inf[-1].split('_')[2])+1
        name_dataset = path_inf[-4] 
        if os.path.getsize(label_path) > 0: 
            arr=np.loadtxt(
                label_path
            )
            if len(arr.shape)==1:
                arr=arr.reshape(1,-1)
            img = cv2.imread(image_path)
            if model is not None:
                info_box = model.predict(img,iou=0.45,save=False,show=False)[0].boxes
                bboxes = info_box.xywh.detach().cpu().numpy()
                if len(bboxes)==0:return
                conf = info_box.conf.detach().cpu().numpy()
                if len(bboxes.shape)!=2:
                    bboxes=bboxes.reshape(1,-1)
            for info in  arr:
                id, box =mapping_id[int(info[0])],info[1:]
                if  model is not None:
                    box=np.array(box)
                    min_id = 0
                    min_val = 1e3
                    for i in range(len(bboxes)):
                        pt1=bboxes[i][:2]
                        pt2=box[:2]*np.array([1600,900])
                        d =  np.linalg.norm(pt1 - pt2)
                        if d < min_val:
                            min_val=d
                            min_id=i
                    if conf[min_id]<0.66: continue
                    elif min_val > 7.5: continue
                    else:
                        if len(arr)!=len(bboxes):
                            with open("temp.log", 'a') as f:
                                f.write(f"d = {min_val} id = {min_id} frame {frame_id} cam_id = {cam_id}\n")
                cut_img = cut_image(img,box)
                img_pil = Image.fromarray(cut_img)
                save_name = str(id)+f'_{name_dataset}_c{cam_id}_{frame_id}.jpg'
                img_pil.save(os.path.join(save_path,save_name), format='JPEG')
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as Executer:
        Executer.map(
            lambda x:save_img(x[0],x[1],mapping_id,model_path),zip(image_paths,label_paths)
        )
def make_data_for_train_reid(data_src=None,save_path='Custom_ReID_Dataset',model_path=None):
    if os.path.exists(os.path.join(os.path.dirname(__file__),'temp.log')):
        open('temp.log', 'w').close()
    root_path = os.path.dirname(__file__)
    for i in range(100):
        if os.path.exists(os.path.join(root_path,save_path)):
            save_path=save_path+f'({i})'
        else:break
    if data_src is None:
        if not os.path.exists(os.path.join(root_path,'Roboflow_Datasets')):
            download_datasets()
            data_src= 'Roboflow_Datasets'
        else:
            data_src= 'Roboflow_Datasets'
    for name_dataset in os.listdir(os.path.join(root_path,data_src)):
        config_path = os.path.join(root_path,data_src,name_dataset,'data.yaml')
        image_paths = sorted(glob.glob(os.path.join(root_path,data_src,name_dataset,'*/images/*.jpg')))
        label_paths = sorted(glob.glob(os.path.join(root_path,data_src,name_dataset,'*/labels/*.txt')))
        save_image_for_reid(image_paths,label_paths,config_path,save_path,model_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as Executer:
        output = list(set(Executer.map(lambda x:'_'.join(x.split('_')[:-2]),glob.glob(os.path.join(root_path,save_path,'*.jpg')))))
    re_mapping={
        output[i]:i for i in range(len(output))
    }
    os.makedirs(os.path.join(root_path,save_path,'dataset_0','bounding_box_train'),exist_ok=True)
    os.makedirs(os.path.join(root_path,save_path,'dataset_0','query'),exist_ok=True)
    os.makedirs(os.path.join(root_path,save_path,'dataset_0','gallery'),exist_ok=True)

    def rename(old_path,re_mapping):
        key='_'.join(old_path.split('_')[:-2])
        new_path = f'{re_mapping[key]:04d}_'+'_'.join(old_path.split('_')[-2:])
        if 'c1' in new_path:
            shutil.move(old_path, os.path.join(root_path,save_path,'dataset_0','bounding_box_train',new_path))
        else:
            shutil.move(old_path, os.path.join(root_path,save_path,'dataset_0','query',new_path))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as Executer:
        Executer.map(lambda x: rename(x,re_mapping),glob.glob(os.path.join(root_path,save_path,'*.jpg')))
    query = sorted(glob.glob(os.path.join(root_path,save_path,'dataset_0','query','*.jpg')))
    for q in query:
        if np.random.rand()<0.4:
            new_cam_for_gallery=int(q.split('/')[-1].split('_')[1][1])+1
            name_gallery =q.split('/')[-1].split('_')
            name_gallery[1]=f'c{new_cam_for_gallery}'
            shutil.move(q,os.path.join(root_path,save_path,'dataset_0','gallery','_'.join(name_gallery)))
import ffmpeg
def get_video_rotation(video_path):
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    
    for stream in video_streams:
        tags = stream.get('tags', {})
        rotation = int(tags.get('rotate', 0))
        return rotation
    return 0