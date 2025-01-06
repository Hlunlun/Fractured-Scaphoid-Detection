import os
import cv2
import shutil
import numpy as np
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from yolo_anno import load_json, create_fracture_data, collate_hand_data, create_hand_data
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as AreaPolygon


class HandDetector:
    def __init__(self, train=True, args=None):
        if train:
            self.args = args        
            self.hand_config = "yolo_config/data/hand.yaml"
            self.hand_project = "yolo_config/runs/train_hand"
            self.frac_config = "yolo_config/data/fracture.yaml"
            self.frac_project = "yolo_config/runs/train_frac"        
            self.data_prep()

    def data_prep(self):        
        fracture_datas = load_json('fracture')
        train_dataset, val_dataset = train_test_split(fracture_datas , test_size=0.2, random_state=42)
        # dst_dir = '/home/lun/projects/yolov11-obb/datasets/hand'
        dst_dir = 'yolo_config/datasets/fracture'
        create_fracture_data(train_dataset, dst_dir, 'train')
        create_fracture_data(val_dataset, dst_dir, 'val')



        datas = collate_hand_data(load_json())
        train_dataset, val_dataset = train_test_split(datas, test_size=0.2, random_state=42)
        # dst_dir = '/home/lun/projects/yolov11-obb/datasets/hand'
        dst_dir = 'yolo_config/datasets/hand'
        create_hand_data(train_dataset, dst_dir, 'train')
        create_hand_data(val_dataset, dst_dir, 'val')


    def train(self):        
        self.hand_model = YOLO("yolo11x-obb.pt") 
        self.hand_model.train(data=self.hand_config, 
                            epochs=100, 
                            imgsz=1024,
                            device='cuda:1',
                            optimizer='SGD',
                            close_mosaic=10,
                            project=self.hand_project,
                            name='exp')

        self.frac_model = YOLO("yolo11x-obb.pt")
        self.frac_model.train(data=self.frac_config, 
                            epochs=100, 
                            imgsz=1024,
                            device='cuda:0',
                            optimizer='SGD',
                            close_mosaic=10,
                            project=self.frac_project,
                            name='exp')


    def latest_exp(self, dir_path):
        exp_folders = [folder for folder in os.listdir(dir_path) if folder.startswith("exp")]
        sorted_folders = sorted(exp_folders, key=lambda x: int(x[3:]))
        return sorted_folders[-1] if sorted_folders else None

    def _detect_hand(self, img_name, img_path):
        self.hand_datas = collate_hand_data(load_json()) 
        self.hand_model = YOLO('yolo_config/weights/hand_best.pt')
        results = self.hand_model(img_path, conf=0.)

        # Choose best confidence            
        scaphoid_results = [x for x in results[0].obb if x.cls == 1]
        fracture_results = [x for x in results[0].obb if x.cls == 0]
        scaphoid_result = max(scaphoid_results, key=lambda x: x.conf)
        fracture_result = max(fracture_results, key=lambda x: x.conf)

        # Plot prediction
        output_path = f'prediction/hand/{img_name}'
        scap_xy4 = scaphoid_result.xyxyxyxy.detach().cpu().squeeze().tolist()
        self.plot_xyxyxyxy(scap_xy4, img_path, output_path, (0, 255, 0))
        if(fracture_result.conf >= 0.3):
            frac_xy4 = fracture_result.xyxyxyxy.detach().cpu().squeeze().tolist()
            self.plot_xyxyxyxy(frac_xy4, output_path, output_path, (0, 255, 0))

        # Plot target            
        tgt = list(filter(lambda x: x['name'] == img_name, self.hand_datas))
        if len(tgt) > 0:
            tgt = tgt[0]
            self.plot_xyxyxyxy(tgt['scap_cor'], output_path, output_path, (0, 0, 255))
            if(tgt['frac_cls']==1):
                self.plot_xyxyxyxy(tgt['frac_cor'], output_path, output_path, (0, 0, 255))

            return {
                'img_name': img_name,
                'scap_iou': self.cal_iou(scap_xy4, tgt['scap_cor']),
                'frac_iou': self.cal_iou(frac_xy4, tgt['frac_cor']) if tgt['frac_cls']==1 and fracture_result.conf >= 0.3 else 0,
                'cls':tgt['frac_cls']
            }
        else:
            return {
                'img_name': img_name,
                'scap_iou': 0.,
                'frac_iou': 0.,
                'cls':tgt['frac_cls']
            }

    def _detect_fracture(self, img_name, img_path):
        self.hand_datas = collate_hand_data(load_json())  
        self.frac_model = YOLO('yolo_config/weights/fracture_best.pt')
        results = self.frac_model(img_path, conf=0.)            
        max_conf_result = max(results[0].obb, key=lambda x: x.conf)

        # Plot prediction
        output_path = f'prediction/fracture/{img_name}'
        frac_xy4 = max_conf_result.xyxyxyxy.detach().cpu().squeeze().tolist()
        frac_xy4n = max_conf_result.xyxyxyxyn.detach().cpu().squeeze().tolist()
        # if max_conf_result.conf > 0.5:
        self.plot_xyxyxyxy(frac_xy4, img_path,output_path, (0, 255, 0))
        self.plot_xyxyxyxyn(frac_xy4n, img_path, output_path, (0, 255, 0))

        # Plot target
        tgt = list(filter(lambda x: x['name'] == img_name, self.fracture_datas))
        if len(tgt) > 0:
            if max_conf_result.conf > 0.5:
                self.plot_xyxyxyxy(tgt[0]['cor'], output_path, output_path, (0, 0, 255))
            else:
                self.plot_xyxyxyxy(tgt[0]['cor'], img_path, output_path, (0, 0, 255))

        return {
            'img_name': img_name,
            'iou': self.cal_iou(frac_xy4, tgt[0]['cor']) if len(tgt)>0 and max_conf_result.conf > 0.5 else 0,
            'cls': tgt[0]['cls']
        }

    def detect_hand(self, dir_path='ip_data'):     
        self.hand_datas = collate_hand_data(load_json())   
        dir_path = os.path.join(dir_path, 'scaphoid_detection/images')
        # hand_weights = self.latest_exp(self.hand_project)

        ious = []
        for img_name in os.listdir(dir_path):            
            img_path = os.path.join(dir_path, img_name)
            iou = self._detect_hand(img_name, img_path)
            ious.append(iou)

        return ious


    def detect_fracture(self, dir_path='prediction/classifier'):
        self.fracture_datas = load_json('fracture')
        # frac_weights = self.latest_exp(self.frac_project)

        ious = []
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            iou = self._detect_fracture(img_name, img_path)
            ious.append(iou)

        return ious


    def cal_iou(self, cor1, cor2):
        poly1 = AreaPolygon(cor1)
        poly2 = AreaPolygon(cor2)
        
        intersection_area = poly1.intersection(poly2).area        
        union_area = poly1.area + poly2.area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0

    def plot_xywhr(self, xywhr, image_path, save_path, color=(0, 255, 0)):      
        x, y, w, h, r = xywhr

        cos_r = np.cos(r)
        sin_r = np.sin(r)
        dx = w / 2
        dy = h / 2
        
        offsets = np.array([
            [-dx, -dy],
            [dx, -dy],
            [dx, dy],
            [-dx, dy]
        ])
        
        rotation_matrix = np.array([
            [cos_r, -sin_r],
            [sin_r, cos_r]
        ])    

        rotated_offsets = offsets @ rotation_matrix.T
        vertices = (rotated_offsets + np.array([x, y])).astype(int)

        image = cv2.imread(image_path)
        pts = vertices.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        cv2.imwrite(save_path, image)
        
    def plot_xyxyxyxy(self, xyxyxyxy, image_path, save_path, color=(0, 255, 0)):
        absolute_vertices = np.array(xyxyxyxy, dtype=np.int32)

        image = cv2.imread(image_path)
        
        pts = absolute_vertices.reshape((-1, 1, 2))  # 調整格式為 cv2.polylines 接受的格式
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        cv2.imwrite(save_path, image)

    def plot_xyxyxyxyn(self, xyxyxyxyn, image_path, save_path, color=(0, 255, 0)):
        image = cv2.imread(image_path)
        image_width = image.shape[0]
        image_height = image.shape[1]
        
        absolute_vertices = np.array(
            [[int(x * image_width), int(y * image_height)] for x, y in xyxyxyxyn],
            dtype=np.int32
        )
        
        pts = absolute_vertices.reshape((-1, 1, 2)) 
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        cv2.imwrite(save_path, image)





