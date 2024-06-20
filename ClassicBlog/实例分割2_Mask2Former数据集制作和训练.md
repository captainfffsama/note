[原文](https://www.guyuehome.com/37167)

[toc]

截至博客发布，Mask2Former rank 7 在 COCO test-dev 实例分割排名中。

Github 链接：[https://github.com/facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former)

![](https://www.guyuehome.com/Uploads/Editor/202203/bf022c2626c84f2eb7731020c48da245.png)

# 1. 环境配置

```bash
conda create -n mask2former python=3.8
conda activate mask2former
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
conda install --channel https://conda.anaconda.org/Zimmf cudatoolkit=10.2
pip install opencv-python

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
git clone git@github.com:facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh

# 数据集预处理需要的库
pip install opencv-python labelme
```

# 2. 数据集
## 2.1 数据集制作

参考上一篇博客：[实例分割1_Detectron2数据集制作并注册数据集训练](实例分割1_Detectron2数据集制作并注册数据集训练.md)

## 2 .2 注册数据集

主要是修改 `train_net.py` 文件，这里的使用方法与 Dectectron2 是相同的。

主要是编写了 Register 类（在其中指定了数据集所在路径，可直接从 Dectectron2 那边做过的复制），同时引入了 Dectectron2 的一些注册数据集用的 module 。

代码见附录\[1\]。

# 3. 训练
## 3.1 配置文件

在 `configs` 文件夹下新建 `my_config.yaml` 文件，我这里基于实例分割的配置文件 `coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml` ，也可以选用其他的。

不同配置文件对应于不同的数据集格式，即使同样是 coco 数据集，实例分割和全景分割时，实例分割的配置文件会利用 `xxx_mapper.py` 来生成额外的字典参数，如 `instances` 参数。

`my_config.json` 文件的内容如下：

```yaml
_BASE_: "coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
DATASETS:
  TRAIN: ("coco_my_train",)
  TEST: ("coco_my_val",)
MODEL:
  RETINANET:
    NUM_CLASSES: 1
  # WEIGHTS: "../tools/output/model_final.pth"
  ROI_HEADS:
    NUM_CLASSES: 1
    NMS_THRESH_TEST: 0.5
  # DEVICE: "cuda:3"
SOLVER:
  # IMS_PER_BATCH: 2
  # 初始学习率
  BASE_LR: 0.0025
  # 迭代到指定次数，学习率进行衰减
  # STEPS: (210000, 250000)
  # MAX_ITER: 270000
  # CHECKPOINT_PERIOD: 5000
# TEST:
#   EVAL_PERIOD: 3000
OUTPUT_DIR: "./output1"
```

## 3.2 训练命令

```bash
# 训练
python train_net1.py \
    --config-file configs/my_config.yaml \
    --num-gpus 1 \
    SOLVER.IMS_PER_BATCH 2 \
    SOLVER.BASE_LR 0.0025

# 多 GPU 训练
python train_net1.py \
    --config-file configs/my_config.yaml \
    --num-gpus 4 \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.0025

# 断点续训
python train_net1.py \
    --config-file configs/my_config.yaml \
    --num-gpus 4 \
    --resume
```

## 3.3 预测脚本

```bash
# 预测单张
python ./demo/demo.py \
    --config-file ./configs/my_config.yaml \
    --input "1.jpg" \
    --confidence-threshold 0.5 --output "./output1/1.jpg" \
    --opts MODEL.WEIGHTS ./output/model_0059999.pth

# 预测多张（输出目录需要新建）
python ./demo/demo.py \
    --config-file ./configs/my_config.yaml \
    --input "路径/*.jpg" \
    --confidence-threshold 0.5 --output "output1/" \
    --opts MODEL.WEIGHTS ./output/model_0059999.pth
```

## 3 .4 评估模型

```bash
# 评估
python train_net1.py \
    --config-file configs/my_config.yaml \
    --eval-only \
    MODEL.WEIGHTS output/model_final.pth
```

# 附录：代码
## [1] 注册数据集

`train_net1.py` 文件内容如下：

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import cv2
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_train_loader,
    DatasetCatalog,
)
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import load_coco_json

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(
                    COCOPanopticEvaluator(dataset_name, output_folder)
                )
        # COCO
        if (
            evaluator_type == "coco_panoptic_seg"
            and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
        ):
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if (
            evaluator_type == "coco_panoptic_seg"
            and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON
        ):
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder
                )
            )
        # Mapillary Vistas
        if (
            evaluator_type == "mapillary_vistas_panoptic_seg"
            and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
        ):
            evaluator_list.append(
                InstanceSegEvaluator(dataset_name, output_dir=output_folder)
            )
        if (
            evaluator_type == "mapillary_vistas_panoptic_seg"
            and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON
        ):
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder
                )
            )
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if (
            evaluator_type == "ade20k_panoptic_seg"
            and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
        ):
            evaluator_list.append(
                InstanceSegEvaluator(dataset_name, output_dir=output_folder)
            )
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = (
                        hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation …")
        model = SemanticSegmentorWithTTA(cfg, model)
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
    ROOT = "填自己的"

    def __init__(self):
        self.CLASS_NAMES = Register.CLASS_NAMES
        # 数据集路径
        self.ANN_ROOT = "填自己的"

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
        """
        purpose: register all splits of datasets with PREDEFINED_SPLITS_DATASET
        注册数据集（这一步就是将自定义数据集注册进Detectron2）
        """
        for key, (image_root, json_file) in self.PREDEFINED_SPLITS_DATASET.items():
            self.register_dataset_instances(
                name=key, json_file=json_file, image_root=image_root
            )

    @staticmethod
    def register_dataset_instances(name, json_file, image_root):
        """
        purpose: register datasets to DatasetCatalog,
                 register metadata to MetadataCatalog and set attribute
        注册数据集实例，加载数据集中的对象实例
        """
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
        """
        查看数据集标注，可视化检查数据集标注是否正确，
        这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界
        可选择使用此方法
        """
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
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former"
    )
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

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
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