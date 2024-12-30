import os
import re
import json
import torch
import math
import cv2
import numpy as np
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Compose, Resize




def load_scap_bbox(json_path):

    with open(json_path, 'r') as f:
        data = json.load(f)
        bbox = data[0]['bbox']  # scaphoid bbox: [x1, y1, x2, y2], fracture bbox: [[x1, y1],[x2, y2],[x3, y3],[x4, y4]]
        bbox = [int(x) for x in bbox] if bbox is not None else [0.,0.,0.,0.]
        return bbox


def load_frac_bbox(json_path):

    with open(json_path, 'r') as f:
        data = json.load(f)
        bbox = data[0]['bbox']  # scaphoid bbox: [x1, y1, x2, y2], fracture bbox: [[x1, y1],[x2, y2],[x3, y3],[x4, y4]]
        bbox = [[cor for cor in cors] for cors in bbox ] if bbox is not None else [[0.,0.],[0.,0.],[0.,0.],[0.,0.]]
        has_frac = 1. if data[0]['name'] is not None else 0.
        return bbox, has_frac



class ImageDataset(Dataset):
    def __init__(self, args):    
        self.datas = {'scaphoid':[], 'fracture':[]}
        for img_name in os.listdir(args.img_dir):            
            img_path = os.path.join(args.img_dir, img_name)

            scap_path = os.path.join(args.scap_dir, img_name.replace('.jpg', '.json'))
            frac_path = os.path.join(args.frac_dir, img_name.replace('.jpg', '.json'))

            if os.path.exists(img_path) and os.path.exists(scap_path) and os.path.exists(frac_path):
                scap_cor = load_scap_bbox(scap_path)   
                img = cv2.imread(img_path)
                scap_img = img[scap_cor[1]:scap_cor[3], scap_cor[0]:scap_cor[2]]  
                
                if scap_img.size != 0:
                    # Scaphoid
                    scap_data = {
                        'name': img_name,
                        'img': img_path,
                        'cor': scap_cor,
                    }
                    self.datas['scaphoid'].append(scap_data)

                    # Fracture     
                    scap_img_dir = args.frac_dir.replace('annotations','images')
                    os.makedirs(scap_img_dir, exist_ok=True)
                    scap_img_path = os.path.join(scap_img_dir, img_name)
                    cv2.imwrite(scap_img_path, scap_img)
                    frac_cor, has_frac = load_frac_bbox(frac_path)
                    frac_data = {
                        'name': img_name,
                        'img': scap_img_path,
                        'cor': frac_cor,
                        'cls': has_frac
                    }
                    self.datas['fracture'].append(frac_data)      

            self.save_to_json('all_datas.json')
                
    def save_to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.datas, f, indent=4)   

    def __len__(self):
        return len(self.datas['scaphoid'])    

    def __getitem__(self, idx):     
        return self.datas['scaphoid'][idx], self.datas['fracture'][idx]



class ScaphoidImageDataset(ImageDataset):
    def __init__(self, args):
        super().__init__(args)
        
        self.transforms = Compose([ToTensorV2(p=1.0)], 
                                    bbox_params={
                                      'format': 'pascal_voc', 
                                      'label_fields': ['labels']
                                    })

    def __getitem__(self, idx):
        data, _ =  super().__getitem__(idx)

        img_path, cor = data['img'], data['cor']
        img = cv2.imread(img_path)
        img_arr = img.astype(np.float32) / 255.0 # convert to float instead of original double to be computed with weight on cuda (type: TensorFloat)

        target = {
            'boxes': torch.tensor([cor], dtype=torch.float32),
            'labels': torch.ones((1,), dtype=torch.int64),
            'img_id': torch.tensor(idx),
            'area': torch.as_tensor((cor[3]-cor[1])*(cor[2]-cor[0]), dtype=torch.float32),
            'iscrowd': torch.zeros((1,), dtype=torch.int64)
        }
 
        sample = {
            'image': img_arr,
            'bboxes':[cor],
            'labels':[1]
        }
        sample = self.transforms(**sample)  

        return sample['image'], target


       

class FractureImageDataset(ImageDataset):
    def __init__(self, args, scap_dir=None):
        super().__init__(args)
        if scap_dir is not None:
            for img_name in os.listdir(scap_dir):
                matched_info = next((info for info in self.datas['fracture'] if info['name'] == img_name), None)
                if matched_info:
                    matched_info['img'] = os.path.join(scap_dir, img_name)
        self.transform = Compose([
            Resize(height=244, width=244), 
            ToTensorV2()
        ])
    
    def __getitem__(self, idx):
        _, data = super().__getitem__(idx)

        img_path, cor, cls = data['img'], data['cor'], data['cls']
        img = cv2.imread(img_path)
        img_arr = img.astype(np.float32) / 255.0

        img_tensor = self.transform(image=img_arr)['image']

        cor_tensor = torch.tensor([cor], dtype=torch.float32)
        cls_tensor = torch.tensor(cls, dtype=torch.long)
        
        return img_tensor, cls_tensor, cor_tensor