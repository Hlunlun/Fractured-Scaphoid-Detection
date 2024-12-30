import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QGroupBox, QLabel
)
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QPolygonF


class MainWindow(QMainWindow):
    def __init__(self, detector):
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

        # 偵測區域
        self.detect_group = QGroupBox("Detect")
        self.detect_layout = QVBoxLayout()
        self.detect_button = QPushButton("Detection")
        self.detect_button.clicked.connect(self.detect_image)
        self.detect_layout.addWidget(self.detect_button)

        self.iou_label = QLabel("IoU:")
        self.accuracy_label = QLabel("Accuracy:")
        self.precision_label = QLabel("Precision:")
        self.recall_label = QLabel("Recall:")
        
        self.detect_display = QLabel()

        self.detect_layout.addWidget(self.iou_label)
        self.detect_layout.addWidget(self.accuracy_label)
        self.detect_layout.addWidget(self.precision_label)
        self.detect_layout.addWidget(self.recall_label)
        self.detect_layout.addWidget(self.detect_display)

        self.detect_group.setLayout(self.detect_layout)
        self.main_layout.addWidget(self.detect_group)

        # intialize detector
        self.detector = detector

        # 設定主視窗
        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")        
        if not folder_path:
            return
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        self.images = [ ]
        if self.image_paths:
            # 預設顯示第一張圖片
            self.current_image_index = 0
            self.show_image()
        else:
            # 如果資料夾內沒有符合條件的檔案
            print("No valid image files found in the selected folder.")
        
    def show_image(self):

        image_path = self.image_paths[self.current_image_index]
        self.current_image_label.setText(f"Current Image: {image_path}")

        # load image
        pixmap = QPixmap(image_path)
        # Resize and show image
        self.image_display.setPixmap(pixmap.scaled(500, 500))

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
        image = self.image_paths[self.current_image_index]
        label, scap_img, pred_cor, tgt_cor, iou = self.detector.predict(image)


        # Show
        self.iou_label.setText(f"IoU: {iou}")
        self.accuracy_label.setText("Accuracy: 0.90")
        self.precision_label.setText("Precision: 0.88")
        self.recall_label.setText("Recall: 0.87")

        self.show_detection(scap_img, tgt_cor, pred_cor)

    
    def show_detection(self, scap_img, tgt_cor, pred_cor):
        pixmap = QPixmap(scap_img)
        self.detect_display.setPixmap(pixmap.scaled(500, 500))
