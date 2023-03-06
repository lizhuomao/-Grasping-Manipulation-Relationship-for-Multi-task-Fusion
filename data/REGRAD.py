import cv2
import os
import json
import random

from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


class REGRAD(object):
    def __init__(self):
        self.num_class_dict = {}
        self.class_num_dict = {}
        self.CLASS_NAME = []
        self.dataset_dicts = []
        self.dataset_dicts = self.get_regrad_dicts()

    def get_regrad_dicts(self, data_dir="/root/autodl-nas/REGRAD/REGRAD_v1/", is_train=True, height=960, width=1280):
        if len(self.dataset_dicts) != 0:
            return self.dataset_dicts
        id_cnt = 0
        class_num = 0
        data_dir = os.path.join(data_dir, 'Relation_Part')
        if is_train:
            dataset_dir = os.path.join(data_dir, 'train')
        else:
            dataset_dir = os.path.join(data_dir, 'test')

        scene_id = os.listdir(dataset_dir)
        scene_id.sort()
        for s_id in scene_id:
            data_scene_dir = os.path.join(dataset_dir, s_id)
            camera_angles = os.listdir(data_scene_dir)
            for c_a in camera_angles:
                if not c_a.isdigit():
                    continue
                json_dir = os.path.join(data_scene_dir, c_a)
                json_file = os.path.join(json_dir, 'info.json')
                with open(json_file) as f:
                    imgs_annos = json.load(f)
                record = {}
                filename = os.path.join(json_dir, 'rgb.jpg')

                record["file_name"] = filename
                record["image_id"] = id_cnt
                record["height"] = height
                record["width"] = width
                id_cnt += 1

                objs = []
                for anno in imgs_annos:
                    bbox = anno["bbox"]
                    if bbox is None:
                        continue
                    bbox_mode = BoxMode.XYXY_ABS
                    category_id = anno["category"]
                    category = anno["model_name"]
                    if category not in self.class_num_dict:
                        self.num_class_dict[str(class_num)] = category
                        category_id = class_num
                        self.class_num_dict[category] = class_num
                        class_num += 1
                    else:
                        category_id = self.class_num_dict[category]
                    

                    segmentation = anno["segmentation"]
                    if len(segmentation[0]) <= 2:
                        continue
                    py = segmentation[0]
                    px = segmentation[1]
                    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                    poly = [p for x in poly for p in x]
                    obj = {
                        "bbox": bbox,
                        "bbox_mode": bbox_mode,
                        "segmentation": [poly],
                        "category_id": category_id
                    }
                    objs.append(obj)
                record["annotations"] = objs
                self.dataset_dicts.append(record)
        
        self.CLASS_NAME = [self.num_class_dict[str(i)] for i in range(len(self.num_class_dict))]
                
        return self.dataset_dicts


# Test:
# get_regrad_dicts("/root/autodl-nas/REGRAD/REGRAD_v1/")


# data = REGRAD()
# DatasetCatalog.register("REGRAD", data.get_regrad_dicts)
# MetadataCatalog.get("REGRAD").thing_classes = data.CLASS_NAME
# metadata = MetadataCatalog.get("REGRAD")

# for d in random.sample(data.dataset_dicts, 3):
#     img = cv2.imread(d['file_name'])
#     cv2.imwrite("origin.jpg", img)
#     visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imwrite("test.jpg", out.get_image()[:, :, ::-1])
