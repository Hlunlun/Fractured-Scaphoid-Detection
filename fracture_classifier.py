import os
import cv2
import torch
import argparse
import torch.nn as nn
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from dataset import FractureImageDataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd


class FractureClassifier:
    def __init__(self, args):
        # Dataset
        self.dataset = FractureImageDataset(args)

        # Parameters
        self.num_classes = args.num_classes
        self.device = args.device        
        self.lr = args.lr
        self.test_ratio = args.test_ratio
        self.random_state = args.random_state
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.model_dir = args.model_dir
        self.num_epochs = args.num_epochs
        self.save_epoch = args.save_epoch
        
        # Model
        self.model = self.create_model().to(self.device)
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.optim = Adam(self.model.classifier.parameters(), lr=self.lr, weight_decay=args.weight_decay)

        # Result        
        self.re_dir = os.path.join(f'performance/classifier/{self._get_time()}')
        os.makedirs(self.re_dir, exist_ok=True)

    def create_model(self):
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)        
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)
        return model
    
    def load_model(self):
        model = self.create_model()
        model.load_state_dict(torch.load(f'{self.model_dir}/classifier/model_99.pth'))
        return model.to(self.device)
    
    def train(self):        
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=self.test_ratio, random_state=self.random_state)
        self.train_metrics =  [] 
        for epoch in range(self.num_epochs):
            self.train_metrics += self._run(self.model, train_dataset, epoch, self.optim) 
        self.test_metrics = self._run(self.model, test_dataset, 0)
        print(self.test_metrics)

        self.plot()

    def _run(self, model, dataset, epoch=0, optim=None):
        metrics = []

        if optim is not None:
            # Train and update parameters for model
            print(f'Epoch {epoch}')
            mode = "Training"
            model.train()
            train_data, val_data = train_test_split(dataset, test_size=self.test_ratio, random_state=self.random_state)
            train_loader = DataLoader(train_data, batch_size=self.train_batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.train_batch_size, shuffle=True)
        else:
            # Test model without updating parameters
            print()
            mode = "Testing"
            model.eval()
            train_loader = DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False)

        # Train or Test the model
        progress_bar = tqdm(train_loader, desc=mode)
        for imgs, clss, _ in progress_bar:
            imgs, clss = imgs.to(self.device), clss.to(self.device)
            preds = model(imgs)
            losses = self.criterion(preds, clss)
            if optim is not None:
                optim.zero_grad()
                losses.backward()
                progress_bar.set_postfix(loss=losses.item())
                optim.step()
            else:
                metrics += self.cal_metric(preds, clss)



        # Validation during training
        if optim is not None:
            model.eval()    
            progress_bar = tqdm(val_loader, desc="Evaluating")
            for imgs, clss, _ in progress_bar:    
                imgs, clss = imgs.to(self.device), clss.to(self.device)        
                preds = model(imgs)
                metrics += self.cal_metric(preds, clss)

            if epoch == 0 or (epoch + 1) % self.save_epoch == 0:
                save_path = os.path.join(self.model_dir, f'classifier/model_{epoch}.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

        return metrics
    
    def cal_metric(self, preds, tgts):
        preds_np = torch.argmax(preds, dim=1).cpu().numpy()
        tgts_np = tgts.cpu().numpy()

        metric_functions = {
            'accus': accuracy_score,
            'recalls': recall_score,
            'precisions': precision_score,
            'f1s': f1_score, 
            'cls': lambda pred,_: pred,            
        }                
        results = [{name: func(preds_np, tgts_np) for name, func in metric_functions.items()}] 
        results[0]['losses'] = self.criterion(preds, tgts).item()
                
        return results
    
    def _get_time(self):        
        now = datetime.now()
        time = now.strftime("%Y%m%d-%H%M%S")
        return time

    def plot(self):
        """
        Plot Loss, Accuracy, IoU, Accuracy, Recall, Precision, F1
        """          
        self._plt(self.train_metrics, 'train_metrics', 'Classifier model: Evaluating During Training')
        self._plt(self.test_metrics, 'test_metrics', 'Classifier model: Testing')


    def _plt(self, metrics, img_name, title):
        plt.figure()
        df = pd.DataFrame(metrics)
        plt.title(title)
        # 為每個欄位畫折線圖
        df.plot(kind='line', subplots=True, layout=(len(df.columns), 1), figsize=(25, len(df.columns) * 3))
        # 顯示圖表
        plt.tight_layout()
        plt.savefig(os.path.join(self.re_dir, f'{img_name}.jpg')) 

    def classify(self, dir_path='ip_data'):
        # Parameters
        parser = argparse.ArgumentParser()
        parser.add_argument('--img-dir', type=str, default=f'{dir_path}/scaphoid_detection/images')
        parser.add_argument('--scap-dir', type=str, default=f'{dir_path}/scaphoid_detection/annotations')
        parser.add_argument('--frac-dir', type=str, default=f'{dir_path}/fracture_detection/annotations')
        args = parser.parse_args()

        scap_img_dir = 'prediction/scaphoid'

        dataset = FractureImageDataset(args, scap_img_dir)
        model = self.load_model()
        metrics = self._run(model, dataset)
        
        infos_loader = DataLoader(dataset.datas['fracture'], batch_size=self.test_batch_size, shuffle=False)
        preds = [metric['cls'] for metric in metrics]
        
        for infos, preds in zip(infos_loader, preds):
            for name, path, pred in zip(infos['name'], infos['img'], preds):                             
                if pred == 1.0:
                    output_path= f"prediction/classifier/{name}"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, cv2.imread(path))            
                    output_path = os.path.join(self.re_dir, f'fclassifier/{name}')
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, cv2.imread(path))

        print(len(os.listdir("prediction/classifier")))
        print(len(os.listdir("prediction/scaphoid")))
        




