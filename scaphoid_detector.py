import os
import cv2
import torch
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataset import ScaphoidImageDataset
from torch.optim import Adam
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ScaphoidDetector():
    def __init__(self, args):
        # 1. Prepare dataset
        self.scap_dataset = ScaphoidImageDataset(args)
        self.test_ratio = args.test_ratio
        self.random_state = args.random_state
        self.lr = args.lr

        
        # 2. Set parameters  
        self.device = args.device
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.num_epochs = args.num_epochs
        self.model_dir = args.model_dir
        self.save_epoch = args.save_epoch

        # 3. Initialize model   
        self.num_classes = args.num_classes
        self.scap_model = self.create_model().to(self.device)
        self.frac_model = self.create_model().to(self.device)

        # 4. Define optimizer
        params = [p for p in self.scap_model.parameters() if p.requires_grad]
        self.scap_optim = torch.optim.SGD(params, lr=self.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.frac_optim = Adam(self.frac_model.parameters(), lr=self.lr)

        # 5. Results folder
        self.re_dir = os.path.join(f'performance/detector/{self._get_time()}')
        os.makedirs(self.re_dir, exist_ok=True)

    def train(self):
        # Train scaphoid model
        train_dataset, test_dataset = train_test_split(self.scap_dataset, test_size=self.test_ratio, random_state=self.random_state)
        self.train_scap_metrics =  [] 
        for epoch in range(self.num_epochs):
            self.train_scap_metrics += self._run(self.scap_model, train_dataset, epoch, 'scaphoid', self.scap_optim) 
        self.test_scap_metrics = self._run(self.scap_model, test_dataset, 0)


    def create_model(self):
        # load a model pre-trained on COCO        
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model      
    
    def load_model(self, name):
        model = self.create_model()
        model.load_state_dict(torch.load(f'{self.model_dir}/{name}/model_99.pth'))
        return model.to(self.device)


    def _run(self, model, dataset, epoch=0, model_name='scaphoid', optim=None):        
        metrics = []

        if optim is not None:
            # Train and update parameters for model
            print(f'Epoch {epoch}')
            mode = "Training"
            train_data, val_data = train_test_split(dataset, test_size=self.test_ratio, random_state=self.random_state)
            train_loader = DataLoader(train_data, batch_size=self.train_batch_size, shuffle=True, collate_fn=self.collate_fn)
            val_loader = DataLoader(val_data, batch_size=self.train_batch_size, shuffle=True, collate_fn=self.collate_fn)
        else:
            # Test model without updating parameters
            print()
            mode = "Testing"
            train_loader = DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False, collate_fn=self.collate_fn)

        # Train or Test the model
        progress_bar = tqdm(train_loader, desc=mode)
        for imgs, tgts in progress_bar:
            if optim is not None:
                model.train()
                loss_dict = model(imgs, tgts)
                losses = sum(loss for loss in loss_dict.values())
                optim.zero_grad()
                losses.backward()
                optim.step()
            else:
                model.eval()
                preds = model(imgs)
                metrics += self.cal_metric(preds, tgts)

        # Validation during training
        if optim is not None:
            progress_bar = tqdm(val_loader, desc="Evaluating")
            for imgs, tgts in progress_bar:   
                model.eval()             
                preds = model(imgs)
                metrics += self.cal_metric(preds, tgts)

            # Save model checkpoints
            if epoch == 0 or (epoch + 1) % self.save_epoch == 0:
                save_path = os.path.join(self.model_dir, f'{model_name}/model_{epoch}.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

        return metrics
    

    
    def collate_fn(self, batch):
        imgs = [sample[0].to(self.device) for sample in batch]
        tgts = [{k:v.to(self.device) for k, v in sample[1].items()} for sample in batch]        
        return imgs, tgts

    def cal_metric(self, preds, tgts):
        metric_functions = {
            'ious': lambda pred, tgt: self.cal_iou(pred,tgt)['best_iou'],
            'boxes': lambda pred, tgt: self.cal_iou(pred,tgt)['best_box'],
        }        
        # Calculate metrics using the mapping
        results = [{name: func(pred, tgt) for name, func in metric_functions.items()} for pred, tgt in zip(preds, tgts) ]
                
        return results
    

    def cal_iou(self, pred, tgt): 
        pred_boxes = pred['boxes'].detach().cpu().numpy()
        tgt_boxes = tgt['boxes'].detach().cpu().numpy()

        best_iou = 0.
        best_box = [0.,0.,0.,0.]
        
        results = {
            'best_iou': best_iou, 
            'best_box': best_box
        }

        if pred_boxes.shape[0] > 0: 
            tgt_box = tgt_boxes[0]
            for pred_box in pred_boxes: 
                # Calculate intersection coordinates
                inter_x1 = max(pred_box[0], tgt_box[0])
                inter_y1 = max(pred_box[1], tgt_box[1])
                inter_x2 = min(pred_box[2], tgt_box[2])
                inter_y2 = min(pred_box[3], tgt_box[3])

                if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
                    continue  # Skip this prediction if there's no intersection
                
                intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                # Area of prediction and target boxes
                pred_area = abs(pred_box[2] - pred_box[0]) * abs(pred_box[3] - pred_box[1])
                tgt_area = abs(tgt_box[2] - tgt_box[0]) * abs(tgt_box[3] - tgt_box[1])

                # Calculate IoU
                union_area = pred_area + tgt_area - intersection
                iou = intersection / union_area if union_area > 0 else 0    

                # Update best box and iou
                if iou > best_iou:
                    best_iou = iou
                    best_box = pred_box
                    results = {
                        'best_iou': best_iou, 
                        'best_box': best_box
                    }

        return results
    
    def _get_time(self):        
        now = datetime.now()
        time = now.strftime("%Y%m%d-%H%M%S")
        return time

    
    def plot(self):
        """
        Plot Loss, Accuracy, IoU, Accuracy, Recall, Precision, F1
        """          

        self._plt(self.train_scap_metrics, 'train_scap_metrics', 'Scaphoid model: Evaluating During Training')
        self._plt(self.test_scap_metrics, 'test_scap_metrics', 'Scaphoid model: Testing')


    def _plt(self, metrics, img_name, title):
        plt.figure()
        df = pd.DataFrame(metrics)
        plt.title(title)
        # 為每個欄位畫折線圖
        df.plot(kind='line', subplots=True, layout=(len(df.columns), 1), figsize=(25, len(df.columns) * 3))
        # 顯示圖表
        plt.tight_layout()
        plt.savefig(os.path.join(self.re_dir, f'{img_name}.jpg')) 


    def detect(self, dir_path='ip_data'):

        # Parameters
        parser = argparse.ArgumentParser()
        parser.add_argument('--img-dir', type=str, default=f'{dir_path}/scaphoid_detection/images')
        parser.add_argument('--scap-dir', type=str, default=f'{dir_path}/scaphoid_detection/annotations')
        parser.add_argument('--frac-dir', type=str, default=f'{dir_path}/fracture_detection/annotations')
        args = parser.parse_args()

        # Load dataset and model
        scap_dataset = ScaphoidImageDataset(args)
        scap_model = self.load_model('scaphoid')
        scap_metrics = self._run(scap_model, scap_dataset)

        # Data list
        infos = [[item['name'], item['img']] for item in scap_dataset.datas['scaphoid']]
        pred_boxes = [metric['boxes'] for metric in scap_metrics]
        tgt_boxes = [scap_dataset.__getitem__(i)[1]['boxes'].squeeze().detach().numpy() for i in range(len(scap_dataset))]

        # Mark Scaphoid in images
        for idx, (info, pred_box, tgt_box) in enumerate(
            zip(infos, pred_boxes, tgt_boxes)
        ):
            
            img = cv2.imread(info[1])
            img_arr = img.astype(np.uint8)
            img_arr_2 = img.astype(np.uint8)

            cv2.rectangle(img_arr, (int(pred_box[0]), int(pred_box[1])), (int(pred_box[2]), int(pred_box[3])), (0, 255, 0), thickness=8)
            cv2.rectangle(img_arr, (int(tgt_box[0]), int(tgt_box[1])), (int(tgt_box[2]), int(tgt_box[3])), (0, 0, 255), thickness=5)

            output_path= f"prediction/scaphoid_rec/{info[0]}"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img_arr)            
            output_path = os.path.join(self.re_dir, f'scaphoid_rec/{info[0]}')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img_arr)

            # Crop Scaphoid            
            if pred_box[1]<pred_box[3] and pred_box[0]<pred_box[2]:
                img_arr_scap = img_arr_2[int(pred_box[1]):int(pred_box[3]), int(pred_box[0]):int(pred_box[2])]                
            else:
                img_arr_scap = img_arr_2[int(tgt_box[1]):int(tgt_box[3]), int(tgt_box[0]):int(tgt_box[2])]
                      
            output_path = os.path.join(self.re_dir, f'scaphoid/{info[0]}')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img_arr_scap) 
            output_path = os.path.join(f"prediction/scaphoid/{info[0]}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img_arr_scap)   


    

