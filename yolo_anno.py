import os
import cv2
import json
import shutil
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

def load_json(split = None):
    with open('all_datas.json', 'r') as f:
        datas = json.load(f)

    return datas[split] if split is not None else datas



def write_labels(labels_with_coords, label_path, width=None, height=None):
    with open(label_path, 'w') as f:
        for cls, cors in labels_with_coords:
            # Create label array: [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
            label = [cor for sublist in cors for cor in sublist]
            
            # Normalize using numpy array operations
            label = np.array(label, dtype=np.float32)
            if width is not None and height is not None:
                label[0::2] = label[0::2] / width   # x coordinates
                label[1::2] = label[1::2] / height  # y coordinates
                
            # Write to file
            label_str = ' '.join(f'{x}' for x in label)
            f.write(f'{cls} {label_str}\n')


    # with open(label_path, 'w') as f:
    #     for cls, cors in labels_with_coords:
    #         # Normalize
    #         label = [[cor[0]/width, cor[1]/height] for cor in cors]

    #         # Create label array: [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
    #         faltten_label = [cor for sublist in label for cor in sublist]

    #         # Write to file
    #         label_str = ' '.join(f'{x:.6f}' for x in faltten_label)
    #         f.write(f'{cls} {label_str}\n')

def create_fracture_data(dataset, dst_dir, split='train'):
    paths = []
    for data in dataset:
        if data['cls'] == 1:
            out_dir = os.path.join(dst_dir, 'images', split)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy(data['img'], out_dir)

            img_path = f'../fracture/images/{split}/{data["name"]}'
            paths.append(img_path)


            # Get image dimensions for normalization
            with Image.open(data['img']) as img:
                img_width, img_height = img.size

            label_path = os.path.join(dst_dir, 'labels', split, f"{data['name'].replace('.jpg', '')}.txt")
            os.makedirs(os.path.dirname(label_path), exist_ok=True)

            write_labels([(0, data['cor'])], label_path, img_width, img_height)

    out_path = os.path.join(dst_dir, f'{split}.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(paths))


def collate_hand_data(datas):
    collate_data = []
    for scap, frac in zip(datas['scaphoid'], datas['fracture']):
        scap_bbox = scap['cor']

        scap_cor = [[scap_bbox[0], scap_bbox[1]],
                    [scap_bbox[2], scap_bbox[1]],
                    [scap_bbox[2], scap_bbox[3]],
                    [scap_bbox[0], scap_bbox[3]]]

        frac_cor = [[cor[0]+scap_bbox[0], cor[1]+scap_bbox[1]] for cor in frac['cor']]
        
        data = {
            'name': scap['name'],
            'path': scap['img'],
            'scap_cls': 1,
            'frac_cls': frac['cls'],
            'scap_cor': scap_cor,
            'frac_cor': frac_cor,
        }
        collate_data.append(data)
    
    
    with open('hand_datas.json', 'w') as f:
        json.dump(collate_data, f, indent=4) 
    return collate_data



def plot_xyxyxyxy(vertices, image_path, name, save_dir):
    absolute_vertices = np.array(vertices, dtype=np.int32)
    image = cv2.imread(image_path)

    image_width = image.shape[0]
    image_height = image.shape[1]

    # 繪製多邊形框
    pts = absolute_vertices.reshape((-1, 1, 2))  # 調整格式為 cv2.polylines 接受的格式
    cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.imwrite(f'{save_dir}/{name}', image)




def create_hand_data(dataset, dst_dir, split='train'):
    paths = []
    for data in dataset:
        out_dir = os.path.join(dst_dir, 'images', split)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(data['path'], out_dir)

        img_path = f"../hand/images/{split}/{data['name']}"
        paths.append(img_path)


        # Get image dimensions for normalization
        with Image.open(data['path']) as img:
            img_width, img_height = img.size

        label_path = os.path.join(dst_dir, 'labels', split, f"{data['name'].replace('.jpg', '')}.txt")
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        labels_with_coords = []
        if data['frac_cls'] == 1:
            labels_with_coords.append((0, data['frac_cor']))
        # plot_xyxyxyxy(data['scap_cor'], data['path'], data['name'], 'ip_data/scaphoid_detection/images_rec')
            # plot_xyxyxyxy(data['frac_cor'], f'ip_data/scaphoid_detection/images_rec/{data['name']}', data['name'], 'ip_data/fracture_detection/images_rec')
        labels_with_coords.append((1, data['scap_cor']))
        write_labels(labels_with_coords, label_path, img_width, img_height)
        
    out_path = os.path.join(dst_dir, f'{split}.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(paths))











