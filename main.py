import os
import sys
import shutil
import argparse
from ui import MainWindow
from torchvision import transforms
from hand_detector import HandDetector
from scaphoid_detector import ScaphoidDetector
from fracture_classifier import FractureClassifier
from PyQt5.QtWidgets import QApplication

"""Run in command line
# train model
python main.py --train 1

# run app
python main.py
"""

def argparse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=0, help='To train the model')
    parser.add_argument('--img-dir', type=str, default="ip_data/scaphoid_detection/images")
    parser.add_argument('--scap-dir', type=str, default="ip_data/scaphoid_detection/annotations")
    parser.add_argument('--frac-dir', type=str, default="ip_data/fracture_detection/annotations")
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--test-batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--random-state', type=float, default=42)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save-epoch', type=int, default=10)
    parser.add_argument('--model-dir', type=str, default="saved_model")
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)   
    parser.add_argument('--momentum', type=float, default=0.9)        
    parser.add_argument('--weight-decay', type=float, default=0.001)
    args = parser.parse_args() 
    return args

def reset_dirs(args):    
    if args.train == 1:
        shutil.rmtree('saved_model')
        os.makedirs(os.path.join(args.model_dir,'scaphoid'), exist_ok=True)
        os.makedirs(os.path.join(args.model_dir,'classifier'), exist_ok=True)
    else:        
        shutil.rmtree('prediction')
        os.makedirs('prediction/fracture', exist_ok=True)
        os.makedirs('prediction/scaphoid', exist_ok=True)
        os.makedirs('prediction/scaphoid_rec', exist_ok=True)
        os.makedirs('prediction/classifier', exist_ok=True)
        os.makedirs('prediction/hand', exist_ok=True)

def system_init(args):  
    scap_detector = ScaphoidDetector(args)
    frac_classifier = FractureClassifier(args)
    hand_detector = HandDetector(args)
    
    if args.train == 1:
        scap_detector.train()
        frac_classifier.train()
        hand_detector.train()
    else:
        scap_detector.detect()
        frac_classifier.classify()
        hand_detector.detect_fracture()
        hand_detector.detect_hand()
        # app = QApplication(sys.argv)
        # window = MainWindow(detector)
        # window.show()
        # sys.exit(app.exec_())

def main():
    args = argparse_args()
    reset_dirs(args)
    system_init(args)

if __name__ == "__main__":              
    main()




    
