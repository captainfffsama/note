[原文](https://www.guyuehome.com/37048)

这篇博客作为下篇博客的准备工作，[下篇博客](实例分割2_Mask2Former数据集制作和训练.md)介绍 **Mask2Former** 的使用，2022 年了，当然要用 Transformer 来代替卷积。  
Github：[https://github.com/facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former)

# 1. 数据集制作
## 1.1 数据标注
### 1.1.1 自己标注

标注软件 Labelme：[https://github.com/wkentaro/labelme/releases](https://github.com/wkentaro/labelme/releases)  
建议标签文件保存在图片所在的目录，因为标签文件中会自动保存对应图片的路径，否则后续需要修改。

### 1.1.2 从 mask 图片转 JSON

可能事先已经获得语义分割的 Mask 图片，但是没有 JSON 格式的标签文件，这里我选择使用轮廓检测 + 凸包拟合来获得实例的凸多边形，然后将信息写入 JSON 文件。

代码见 [5.2 Mask 转 JSON](实例分割1_Detectron2数据集制作并注册数据集训练.md#5.2%20Mask%20转%20JSON)

## 1.2 Labelme 标签转 COCO 格式

代码见 [5.1 Labelme 转 COCO](#5.1%20Labelme%20转%20COCO)

由所有 JSON 文件生成三个 JSON 文件，包含所有图片 JSON 文件中的信息，如下图所示：  
![](https://www.guyuehome.com/Uploads/Editor/202203/780113933cf94bfebc27e33070c2d532.png)

## 1.3 代码运行提示

如果不需要划分测试集，可以在代码中自行修改训练集和验证集的占比。  
我这里设置的数据集比例为：train:val:test=6:2:2。  
下面是目录的结构：  
![](https://www.guyuehome.com/Uploads/Editor/202203/fd3a86d2d16446efb3c505dd24a97960.png)

# 2. 环境准备

根据自己的 CUDA 版本自行选择适合自己系统的版本安装，这里我的是 CUDA10.2。

```bash
conda create -n mask2former python=3.8
conda activate mask2former
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
conda install --channel https://conda.anaconda.org/Zimmf cudatoolkit=10.2
pip install opencv-python, labelme

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
```

# 3. 训练
## 3.1 注册数据集

在服务器上训练时，Home 目录的内存一般有限，数据集会放置在其他目录，为了实现自定义目录下的数据集均可以训练，选择使用注册数据集的方式。

相关解读可参考 [4. 参考](#4.%20参考)。

代码见 [5.3 注册数据集](#5.3%20注册数据集)

## 3.2 参数配置

设置学习率，迭代次数等内容来覆盖掉默认的配置参数。

代码见 [5.4 配置文件](#5.4%20配置文件)

## 3.3 训练与预测

```bash

# 单卡训练
python tools/train_net1.py \
    --config-file configs/my_config.yaml \
    --num-gpus 1 \
    SOLVER.IMS_PER_BATCH 2 \
    SOLVER.BASE_LR 0.0025

# 指定某张卡训练，这里我指定 3 号卡
python tools/train_net1.py \
    --config-file configs/my_config.yaml \
    --num-gpus 1 \
    MODEL.DEVICE "cuda:3" \
    SOLVER.IMS_PER_BATCH 2 \
    SOLVER.BASE_LR 0.0025

# 多卡训练，每张卡上两个batchsize
python tools/train_net1.py \
    --config-file configs/my_config.yaml \
    --num-gpus 4 \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.0025

# 断点续训
python train_net1.py \
    --config-file configs/my_config.yaml \
    --num-gpus 1 \
    --resume

# 评估
python train_net1.py \
  --config-file configs/my_config.yaml \
  --eval-only \
  MODEL.WEIGHTS output/model_final.pth
```

```bash
# 预测一张图片，权重选择最后输出的，我这里没有完全训练完，只有中间结果
python ./demo/demo.py --config-file ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --input "xxx.jpg" --confidence-threshold 0.5 --output "./output1/xxx.jpg" --opts MODEL.WEIGHTS ./output/model_0059999.pth

# 批量预测图片，输出目录需要自己创建，否则会找不到目录而报错
python ./demo/demo.py --config-file ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --input "图片目录/*.jpg" --confidence-threshold 0.5 --output "./output1/" --opts MODEL.WEIGHTS ./output/model_0059999.pth
```

# 4. 参考

1. [Detectron2 01-注册机制 Registry 实现](https://zhuanlan.zhihu.com/p/93835858)

# 5. 源码
## 5.1 Labelme 转 COCO

```python
# -*- coding:utf-8 -*-

import json
import glob
import random
import PIL.Image
import numpy as np
from labelme import utils

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path="./tran.json"):
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            print(f"{num} : {json_file}")
            with open(json_file, "r", encoding="utf8", errors="ignore") as fp:
                data = json.load(fp)  # 加载json文件
                self.images.append(self.image(data, num))
                for shapes in data["shapes"]:
                    label = shapes["label"]
                    if label not in self.label:
                        self.categories.append(self.categorie(label))
                        self.label.append(label)
                    points = shapes[
                        "points"
                    ]  # 这里的point是用rectangle标注得到的，只有两个点，需要转成四个点
                    points.append([points[0][0], points[1][1]])
                    points.append([points[1][0], points[0][1]])
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

    def image(self, data, num):
        image = {}
        img = utils.img_b64_to_arr(data["imageData"])  # 解析原图片数据
        # img=io.imread(data['imagePath']) # 通过图片路径打开图片
        # img = cv2.imread(data['imagePath'], 0)
        height, width = img.shape[:2]
        img = None
        image["height"] = height
        image["width"] = width
        image["id"] = num + 1
        image["file_name"] = data["imagePath"].split("/")[-1]

        self.height = height
        self.width = width

        return image

    def categorie(self, label):
        categorie = {}
        categorie["supercategory"] = "Cancer"
        categorie["id"] = len(self.label) + 1  # 0 默认为背景
        categorie["name"] = label
        return categorie

    def annotation(self, points, label, num):
        annotation = {}
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        annotation["image_id"] = num + 1
        # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
        annotation["bbox"] = list(map(float, self.getbbox(points)))
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        # annotation['category_id'] = self.getcatid(label)
        annotation["category_id"] = self.getcatid(label)  # 注意，源代码默认为1
        annotation["id"] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie["name"]:
                return categorie["id"]
        return 1

    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA) # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1) # 画多边形 内部像素值为1
        polygons = points

        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        """从mask反算出其边框 mask：[h,w] 0、1组成的图片 1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）"""
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r] # [x1,y1,x2,y2]
        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco["images"] = self.images
        data_coco["categories"] = self.categories
        data_coco["annotations"] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(
            self.data_coco,
            open(self.save_json_path, "w"),
            ensure_ascii=False,
            indent=4,
            cls=MyEncoder,
        )  # indent=4 更加美观显示
        print(f"Json file save in {self.save_json_path}.")

def main():
    # 读取目录
    labelme_json = glob.glob(r"/mnt/lijun/fcg_data_word/joint_word_json/*.json")
    json_size = len(labelme_json)

    # 划分训练集和验证集
    random.seed(10)
    train_dataset = random.sample(labelme_json, int(float(json_size * 0.6)))
    labelme_json_remain = list(set(labelme_json) - set(train_dataset))  # 取补集
    val_dataset = random.sample(labelme_json_remain, int(float(json_size * 0.2)))
    test_dataset = list(set(labelme_json_remain) - set(val_dataset))
    print(f"训练集数量：{len(train_dataset)}")
    print(f"验证集数量：{len(val_dataset)}")
    print(f"测试集数量：{len(test_dataset)}")

    # 格式转换
    labelme2coco(val_dataset, "./coco/annotations/instances_val2017.json")
    labelme2coco(test_dataset, "./coco/annotations/instances_test2017.json")
    labelme2coco(train_dataset, "./coco/annotations/instances_train2017.json")

if __name__ == "__main__":
    main()
```

## 5.2 Mask 转 JSON

代码中调用了 Labelme 转 COCO 中的类，算是数据处理的主函数。

```python
import os
import json
import glob
import base64
import random
import cv2 as cv
import numpy as np
import labelme2coco_server as l2c

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print(f'-- new folder "{path}" --')
    else:
        print(f'-- the folder "{path}" is already here --')

def find_hulls(img):
    # 检测二进制图像的轮廓
    contours, hierarchy = cv.findContours(
        img[:, :, 0], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    # 获取所有轮廓的凸包
    hulls_list = []

    for i in range(len(contours)):
        l0 = []
        cnt = contours[i]
        hull = cv.convexHull(cnt)
        for j in range(len(hull)):
            l1 = [float(hull[j][0][0]), float(hull[j][0][1])]
            l0.append(l1)
        hulls_list.append(l0)

    return hulls_list

def mask2json(img, img_path, points):
    shapes = []
    # for i in range(len(points)):
    for point in points:
        if len(point) > 2:  # 防止有的没有形成闭合的轮廓
            shape = {
                "label": "1",
                "points": point,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
            shapes.append(shape)

    img_height = np.shape(img)[0]
    img_width = np.shape(img)[1]

    data = {
        "version": "4.6.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": "." + img_path,
        "imageData": str(base64.b64encode(open(img_path, "rb").read())).split("b'")[1],
        "imageHeight": img_height,
        "imageWidth": img_width,
    }

    return data

def main():
    img_folder_path = "图片所在目录"
    mask_folder_path = "图片掩码所在目录"
    json_folder_path = "JOSN文件将要保存的目录"
    mkdir(json_folder_path)
    mkdir("coco/annotations/")
    mkdir("coco/train2017/")
    mkdir("coco/val2017/")

    img_names = os.listdir(img_folder_path)
    for img_name in img_names:
        # Get path
        name = img_name.split(".jpg")[0]
        print(img_name)
        mask_name = name + ".png"
        img_path = img_folder_path + img_name
        mask_path = mask_folder_path + mask_name
        print(f"图片路径：{img_path}")

        # Read img
        img = cv.imread(img_path)
        mask = cv.imread(mask_path)

        # Processing
        hulls = find_hulls(mask)
        data = mask2json(img, img_path, hulls)
        with open(json_folder_path + name + ".json", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"已写入 {name}.json")

    # 读取目录
    labelme_json = glob.glob(r"json文件所在目录/*.json")
    json_size = len(labelme_json)
    print(f"数据集数量：{json_size}")

    # 划分训练集和验证集
    random.seed(10)
    train_dataset = random.sample(labelme_json, int(float(json_size * 0.6)))
    labelme_json_remain = list(set(labelme_json) - set(train_dataset))  # 取补集
    val_dataset = random.sample(labelme_json_remain, int(float(json_size * 0.2)))
    test_dataset = list(set(labelme_json_remain) - set(val_dataset))
    print(f"训练集数量：{len(train_dataset)}")
    print(f"验证集数量：{len(val_dataset)}")
    print(f"测试集数量：{len(test_dataset)}")

    # 格式转换
    l2c.labelme2coco(train_dataset, "./coco/annotations/instances_train2017.json")
    l2c.labelme2coco(val_dataset, "./coco/annotations/instances_val2017.json")
    l2c.labelme2coco(test_dataset, "./coco/annotations/instances_test2017.json")

if __name__ == "__main__":
    main()
```

## 5.3 注册数据集

 tools/train_net1.py 代码如下：

```python
#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
from collections import OrderedDict

import cv2
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.visualizer import Visualizer

class Trainer(DefaultTrainer):
    """We use the "DefaultTrainer" which contains pre-defined default logic for standard training workflow. They may not work for you, especially if you are working on a new research project. In that case you can write your own training loop. You can use "tools/plain_train_net.py" as an example."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Create evaluator(s) for a given datasets. This uses the special metadata "evaluator_type" associated with each builtin datasets. For your own datasets, you can simply create an evaluator manually in your script and do not have to worry about the hacky if-else logic here."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the datasets {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation …")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

class Register:
    """用于注册自己的数据集"""

    CLASS_NAMES = ["__background__", "1"]
    ROOT = "图片所在目录"

    def __init__(self):
        self.CLASS_NAMES = Register.CLASS_NAMES
        # 数据集路径
        self.ANN_ROOT = "/home/user/yourdir/detectron2/datasets/coco/annotations"

        self.TRAIN_PATH = Register.ROOT
        self.VAL_PATH = Register.ROOT

        self.TRAIN_JSON = os.path.join(self.ANN_ROOT, "instances_train2017.json")
        self.VAL_JSON = os.path.join(self.ANN_ROOT, "instances_val2017.json")

        # 声明数据集的子集
        self.PREDEFINED_SPLITS_DATASET = {
            "coco_my_train": (self.TRAIN_PATH, self.TRAIN_JSON),
            "coco_my_val": (self.VAL_PATH, self.VAL_JSON),
        }

    def register_dataset(self):
        """purpose: register all splits of datasets with PREDEFINED_SPLITS_DATASET 注册数据集（这一步就是将自定义数据集注册进Detectron2）"""
        for key, (image_root, json_file) in self.PREDEFINED_SPLITS_DATASET.items():
            self.register_dataset_instances(
                name=key, json_file=json_file, image_root=image_root
            )

    @staticmethod
    def register_dataset_instances(name, json_file, image_root):
        """purpose: register datasets to DatasetCatalog, register metadata to MetadataCatalog and set attribute 注册数据集实例，加载数据集中的对象实例"""
        DatasetCatalog.register(
            name, lambda: load_coco_json(json_file, image_root, name)
        )
        MetadataCatalog.get(name).set(
            json_file=json_file, image_root=image_root, evaluator_type="coco"
        )

    def plain_register_dataset(self):
        """注册数据集和元数据"""
        # 训练集
        DatasetCatalog.register(
            "coco_my_train", lambda: load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH)
        )
        MetadataCatalog.get("coco_my_train").set(
            thing_classes=self.CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
            evaluator_type="coco",  # 指定评估方式
            json_file=self.TRAIN_JSON,
            image_root=self.TRAIN_PATH,
        )

        # DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
        # 验证/测试集
        DatasetCatalog.register(
            "coco_my_val", lambda: load_coco_json(self.VAL_JSON, self.VAL_PATH)
        )
        MetadataCatalog.get("coco_my_val").set(
            thing_classes=self.CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
            evaluator_type="coco",  # 指定评估方式
            json_file=self.VAL_JSON,
            image_root=self.VAL_PATH,
        )

    def checkout_dataset_annotation(self, name="coco_my_val"):
        """查看数据集标注，可视化检查数据集标注是否正确， 这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界 可选择使用此方法"""
        # dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
        dataset_dicts = load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH)
        print(len(dataset_dicts))
        for i, d in enumerate(dataset_dicts, 0):
            # print(d)
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(
                img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5
            )
            vis = visualizer.draw_dataset_dict(d)
            # cv2.imshow('show', vis.get_image()[:, :, ::-1])
            cv2.imwrite("out/" + str(i) + ".jpg", vis.get_image()[:, :, ::-1])
            # cv2.waitKey(0)
            if i == 200:
                break

def setup(args):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    Register().register_dataset()  # register my dataset

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """ If you'd like to do anything fancier than the standard training logic, consider writing your own training loop (see plain_train_net.py) or subclassing the trainer. """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
```

## 5.4 配置文件

configs/my_config.yaml

```yaml
_BASE_: "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
DATASETS:
  TRAIN: ("coco_my_train",)
  TEST: ("coco_my_val",)
MODEL:
  RETINANET:
    NUM_CLASSES: 2 # 类别数+1, 因为有background
  # WEIGHTS: "../tools/output/model_final.pth"
SOLVER:
  # IMS_PER_BATCH: 16
  # 初始学习率
  BASE_LR: 0.0025
  # 迭代到指定次数，学习率进行衰减
  # STEPS: (210000, 250000)
  # MAX_ITER: 270000
  # CHECKPOINT_PERIOD: 5000
# TEST:
# EVAL_PERIOD: 3000
OUTPUT_DIR: "./output1"
```