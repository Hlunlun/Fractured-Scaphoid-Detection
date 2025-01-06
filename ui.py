import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QGroupBox, QLabel, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QPolygonF



class MainWindow(QMainWindow):
    def __init__(self, scap_detector, frac_classifier, hand_detector):
        super().__init__()
        self.setWindowTitle("MainWindow")
        self.setGeometry(100, 100, 1200, 700)
        self.current_image_index = -1
        self.image_paths = []

        # 主佈局
        self.main_layout = QHBoxLayout()

        # 圖片選擇區域
        self.image_group = QGroupBox("Image")
        self.image_layout = QVBoxLayout()
        self.load_button = QPushButton("Load Folder")
        self.load_button.clicked.connect(self.load_folder)
        self.image_layout.addWidget(self.load_button)

        self.image_nav_layout = QHBoxLayout()
        self.pre_button = QPushButton("Pre")
        self.pre_button.clicked.connect(self.show_previous_image)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_image)
        self.image_nav_layout.addWidget(self.pre_button)
        self.image_nav_layout.addWidget(self.next_button)
        self.image_layout.addLayout(self.image_nav_layout)

        self.current_image_label = QLabel("Current Image:")
        self.image_display = QLabel()
        self.image_layout.addWidget(self.current_image_label)
        self.image_layout.addWidget(self.image_display)
        self.image_group.setLayout(self.image_layout)
        self.main_layout.addWidget(self.image_group)

        # metrhod 1.
        self.detect_group = QGroupBox("FasterRCNN + VGG16 + YOLOv11-OBB")
        self.detect_layout = QVBoxLayout()
        self.detect_button = QPushButton("Detection")
        self.detect_button.clicked.connect(self.detect_image)
        self.detect_layout.addWidget(self.detect_button)

        self.scap_iou_label = QLabel("Scaphoid IoU:")
        self.frac_iou_label = QLabel("Fracture IoU:")
        self.accuracy_label = QLabel("Accuracy:")
        self.precision_label = QLabel("Precision:")
        self.recall_label = QLabel("Recall:")        
        self.f1_label = QLabel("F1:")        
        self.detect_display = QLabel()

        self.detect_layout.addWidget(self.scap_iou_label)
        self.detect_layout.addWidget(self.frac_iou_label)
        self.detect_layout.addWidget(self.accuracy_label)
        self.detect_layout.addWidget(self.precision_label)
        self.detect_layout.addWidget(self.recall_label)
        self.detect_layout.addWidget(self.f1_label)
        self.detect_layout.addWidget(self.detect_display)

        self.detect_group.setLayout(self.detect_layout)
        self.main_layout.addWidget(self.detect_group)

        # metrhod 2.
        self.detect_group2 = QGroupBox("YOLOv11-OBB")
        self.detect_layout2 = QVBoxLayout()
        self.detect_button2 = QPushButton("Detection")
        self.detect_button2.clicked.connect(self.detect_image2)
        self.detect_layout2.addWidget(self.detect_button2)

        self.scap_iou_label2 = QLabel("Scaphoid IoU:")
        self.frac_iou_label2 = QLabel("Fracture IoU:")
        self.detect_display2 = QLabel()

        self.detect_layout2.addWidget(self.scap_iou_label2)
        self.detect_layout2.addWidget(self.frac_iou_label2)
        self.detect_layout2.addWidget(self.detect_display2)

        self.detect_group2.setLayout(self.detect_layout2)
        self.main_layout.addWidget(self.detect_group2)


        # 設定主視窗
        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        # detector
        self.scap_detector = scap_detector
        self.frac_classifier = frac_classifier
        self.hand_detector = hand_detector

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")   
        
        self.image_paths = [
            os.path.join(folder_path, 'scaphoid_detection/images', file)
            for file in os.listdir(os.path.join(folder_path, 'scaphoid_detection/images'))
            if file.endswith(".jpg")
        ]
        self.image_names = [
            file
            for file in os.listdir(os.path.join(folder_path, 'scaphoid_detection/images'))
            if file.endswith(".jpg")
        ]
        self.current_image_index = 0
        self.show_image()    
        
        QMessageBox.information(self, "Info", "Start predicting")
        self.scap_ious = self.scap_detector.detect(folder_path)
        self.frac_met = self.frac_classifier.classify(folder_path)        
        self.frac_ious = self.hand_detector.detect_fracture()
        self.hand_ious = self.hand_detector.detect_hand(folder_path)
        QMessageBox.information(self, "Info", "Successfully predicting")        

        self.accuracy_label.setText(f"Accuracy: {self.frac_met['accu']}")
        self.precision_label.setText(f"Precision: {self.frac_met['precision']}")
        self.recall_label.setText(f"Recall: {self.frac_met['recall']}")
        self.f1_label.setText(f"F1:{self.frac_met['f1']}")
        

    def show_image(self):
        image_path = self.image_paths[self.current_image_index]
        img_name = self.image_names[self.current_image_index]
        self.current_image_label.setText(f"Current Image: {img_name}")

        pixmap = QPixmap(image_path)
        self.image_display.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))

    def show_previous_image(self):
        self.current_image_index -= 1
        if self.current_image_index < 0:
            self.current_image_index = len(self.image_paths) - 1
        self.show_image()

    def show_next_image(self):  
        self.current_image_index += 1
        if self.current_image_index >= len(self.image_paths):
            self.current_image_index = 0 
        self.show_image()

    def detect_image(self):
        img_name = self.image_names[self.current_image_index]   
        scap_info = list(filter(lambda x: x['img_name'] == img_name, self.scap_ious)) 
        frac_info = list(filter(lambda x: x['img_name'] == img_name, self.frac_ious)) 
        self.scap_iou_label.setText(f"Scaphoid IoU: {scap_info[0]['iou']}")
        if len(frac_info)>0:            
            self.frac_iou_label.setText(f"Fracture IoU: {frac_info[0]['iou']}")
            self.show_detection(f'prediction/fracture/{img_name}')
        else:            
            self.frac_iou_label.setText(f"Fracture : 0")
            self.show_detection(f'prediction/scaphoid_rec/{img_name}')

    
    def show_detection(self, img):
        pixmap = QPixmap(img)
        self.detect_display.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))


    def detect_image2(self):
        img_name = self.image_names[self.current_image_index]
        info = list(filter(lambda x: x['img_name'] == img_name, self.hand_ious)) 

        self.scap_iou_label2.setText(f"Scaphoid IoU: {info[0]['scap_iou']}")
        self.frac_iou_label2.setText(f"Fracture IoU: {info[0]['frac_iou']}")
        self.show_detection2(f'prediction/hand/{img_name}')

    
    def show_detection2(self, img):
        pixmap = QPixmap(img)
        self.detect_display2.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))
        
