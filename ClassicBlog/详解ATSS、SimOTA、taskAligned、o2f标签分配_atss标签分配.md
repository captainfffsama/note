[toc]

[原文](https://blog.csdn.net/qq_16952303/article/details/132321125)

标签分配也是影响目标检测 AP 的一个重要因素，仅是训练技巧，不会增加推理耗时。例如 ATSS 就采用自适应正负样本分配的方式带来 2% 的无痛涨点。最近也是好奇这些标签分配具体是怎么做的，故记录目前学习到的 4 种算法。代码目前基本上没啥详细注释（因为是直接抄来的），后续会精读加注释。

# 论文链接

[ATSS](https://arxiv.org/pdf/1912.02424v4.pdf)  
[SimOTA(YOLOX)](https://arxiv.org/pdf/2107.08430v2.pdf)  
[taskAligned(TOOD)](https://arxiv.org/pdf/2108.07755v3.pdf)  
[o2f](https://arxiv.org/pdf/2303.11567v1.pdf)

# ATSS


## 原理介绍

![](https://img-blog.csdnimg.cn/facccd33c9e14ae9b31b70e78973d8fd.png)

  
**直接上 ATSS 标签分配算法步骤，用人话说就是**：

1.  对于每一个 GT，根据中心距离最近的原则，每个 FPN 层选择 k 个 anchor，假如是 4 个 FPN 层，那一共会到 4 k 个
2.  计算 GT 和 k 个 anchor 之间的 IOUs
3.  求 IOUs 的均值和均方差, mean、std
4.  将均值和均方差相加的和作为 IOU 阈值
5.  将 IOU 高于阈值的挑出来作为正样本  
    算法就是这么简单，却很有效果。以 IOU 均值和方差之和作为均值的好处在于：若 IOU 都很高，此时均值也较高，说明匹配的都挺好的，所以 IOU 阈值应该较高；若 IOU 差异大，此时 IOU 的方差就会大，就会出现一些匹配的比较好，还有一些匹配的比较差，此时加上方差就会把匹配的差的去掉。IOU 方差有利于确定哪几个 FPN 层来预测目标，如下图所示：  
    ![](https://img-blog.csdnimg.cn/d45f86eb72be48f3845d76019be13689.png)
      

    (a) 的方差比较高，均值加方差，此时只有 level 3 的被保留了，(b) 的方差就比较低，1、2 都可适配。

## 代码

抄自 PaddleDetection

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ..bbox_utils import iou_similarity, batch_iou_similarity
from ..bbox_utils import bbox_center
from .utils import (check_points_inside_bboxes, compute_max_iou_anchor,
                    compute_max_iou_gt)

__all__ = ['ATSSAssigner']

@register
class ATSSAssigner(nn.Layer):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection
     via Adaptive Training Sample Selection
    """
    __shared__ = ['num_classes']

    def __init__(self,
                 topk=9,
                 num_classes=80,
                 force_gt_matching=False,
                 eps=1e-9):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching
        self.eps = eps

    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list,
                             pad_gt_mask):
        gt2anchor_distances_list = paddle.split(
            gt2anchor_distances, num_anchors_list, axis=-1)
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0, ] + num_anchors_index[:-1]
        is_in_topk_list = []
        topk_idxs_list = []
        for distances, anchors_index in zip(gt2anchor_distances_list,
                                            num_anchors_index):
            num_anchors = distances.shape[-1]
            _, topk_idxs = paddle.topk(
                distances, self.topk, axis=-1, largest=False)
            topk_idxs_list.append(topk_idxs + anchors_index)
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(
                axis=-2).astype(gt2anchor_distances.dtype)
            is_in_topk_list.append(is_in_topk * pad_gt_mask)
        is_in_topk_list = paddle.concat(is_in_topk_list, axis=-1)
        topk_idxs_list = paddle.concat(topk_idxs_list, axis=-1)
        return is_in_topk_list, topk_idxs_list

    @paddle.no_grad()
    def forward(self,
                anchor_bboxes,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None,
                pred_bboxes=None):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt
        7. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            anchor_bboxes (Tensor, float32): pre-defined anchors, shape(L, 4),
                    "xmin, xmax, ymin, ymax" format
            num_anchors_list (List): num of anchors in each level
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes,
                    shape(B, n, 1), if None, then it will initialize with one_hot label
            pred_bboxes (Tensor, float32, optional): predicted bounding boxes, shape(B, L, 4)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C), if pred_bboxes is not None, then output ious
        """
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3

        num_anchors, _ = anchor_bboxes.shape
        batch_size, num_max_boxes, _ = gt_bboxes.shape

        
        if num_max_boxes == 0:
            assigned_labels = paddle.full(
                [batch_size, num_anchors], bg_index, dtype=gt_labels.dtype)
            assigned_bboxes = paddle.zeros([batch_size, num_anchors, 4])
            assigned_scores = paddle.zeros(
                [batch_size, num_anchors, self.num_classes])
            return assigned_labels, assigned_bboxes, assigned_scores

        
        ious = iou_similarity(gt_bboxes.reshape([-1, 4]), anchor_bboxes)
        ious = ious.reshape([batch_size, -1, num_anchors])

        
        gt_centers = bbox_center(gt_bboxes.reshape([-1, 4])).unsqueeze(1)
        anchor_centers = bbox_center(anchor_bboxes)
        gt2anchor_distances = (gt_centers - anchor_centers.unsqueeze(0)) \
            .norm(2, axis=-1).reshape([batch_size, -1, num_anchors])

        
        
        is_in_topk, topk_idxs = self._gather_topk_pyramid(
            gt2anchor_distances, num_anchors_list, pad_gt_mask)

        
        
        iou_candidates = ious * is_in_topk
        iou_threshold = paddle.index_sample(
            iou_candidates.flatten(stop_axis=-2),
            topk_idxs.flatten(stop_axis=-2))
        iou_threshold = iou_threshold.reshape([batch_size, num_max_boxes, -1])
        iou_threshold = iou_threshold.mean(axis=-1, keepdim=True) + \
                        iou_threshold.std(axis=-1, keepdim=True)
        is_in_topk = paddle.where(iou_candidates > iou_threshold, is_in_topk,
                                  paddle.zeros_like(is_in_topk))

        
        is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes)

        
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        
        
        mask_positive_sum = mask_positive.sum(axis=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile(
                [1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = paddle.where(mask_multiple_gts, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        
        if self.force_gt_matching:
            is_max_iou = compute_max_iou_gt(ious) * pad_gt_mask
            mask_max_iou = (is_max_iou.sum(-2, keepdim=True) == 1).tile(
                [1, num_max_boxes, 1])
            mask_positive = paddle.where(mask_max_iou, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        assigned_gt_index = mask_positive.argmax(axis=-2)

        
        batch_ind = paddle.arange(
            end=batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = paddle.gather(
            gt_labels.flatten(), assigned_gt_index.flatten(), axis=0)
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = paddle.where(
            mask_positive_sum > 0, assigned_labels,
            paddle.full_like(assigned_labels, bg_index))

        assigned_bboxes = paddle.gather(
            gt_bboxes.reshape([-1, 4]), assigned_gt_index.flatten(), axis=0)
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

        assigned_scores = F.one_hot(assigned_labels, self.num_classes + 1)
        ind = list(range(self.num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = paddle.index_select(
            assigned_scores, paddle.to_tensor(ind), axis=-1)
        if pred_bboxes is not None:
            
            ious = batch_iou_similarity(gt_bboxes, pred_bboxes) * mask_positive
            ious = ious.max(axis=-2).unsqueeze(-1)
            assigned_scores *= ious
        elif gt_scores is not None:
            gather_scores = paddle.gather(
                gt_scores.flatten(), assigned_gt_index.flatten(), axis=0)
            gather_scores = gather_scores.reshape([batch_size, num_anchors])
            gather_scores = paddle.where(mask_positive_sum > 0, gather_scores,
                                         paddle.zeros_like(gather_scores))
            assigned_scores *= gather_scores.unsqueeze(-1)

        return assigned_labels, assigned_bboxes, assigned_scores

```

# SimOTA

### 原理介绍

![](https://img-blog.csdnimg.cn/01d5f706066b4297b99ae524eabc555d.png)

  

盗个图，SimOTA 的算法流程也极其简单，不多说了看图把。需要说明的是，前期由于模型预测不准，dynamic_k 基本就为 1。

### 代码

再次抄自 PaddleDetection

```python
import paddle
import numpy as np
import paddle.nn.functional as F

from ppdet.modeling.losses.varifocal_loss import varifocal_loss
from ppdet.modeling.bbox_utils import batch_bbox_overlaps
from ppdet.core.workspace import register

@register
class SimOTAAssigner(object):
    """Computes matching between predictions and ground truth.
    Args:
        center_radius (int | float, optional): Ground truth center size
            to judge whether a prior is in center. Default 2.5.
        candidate_topk (int, optional): The candidate top-k which used to
            get top-k ious to calculate dynamic-k. Default 10.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 3.0.
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        num_classes (int): The num_classes of dataset.
        use_vfl (int): Whether to use varifocal_loss when calculating the cost matrix.
    """
    __shared__ = ['num_classes']

    def __init__(self,
                 center_radius=2.5,
                 candidate_topk=10,
                 iou_weight=3.0,
                 cls_weight=1.0,
                 num_classes=80,
                 use_vfl=True):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.num_classes = num_classes
        self.use_vfl = use_vfl

    def get_in_gt_and_in_center_info(self, flatten_center_and_stride,
                                     gt_bboxes):
        num_gt = gt_bboxes.shape[0]

        flatten_x = flatten_center_and_stride[:, 0].unsqueeze(1).tile(
            [1, num_gt])
        flatten_y = flatten_center_and_stride[:, 1].unsqueeze(1).tile(
            [1, num_gt])
        flatten_stride_x = flatten_center_and_stride[:, 2].unsqueeze(1).tile(
            [1, num_gt])
        flatten_stride_y = flatten_center_and_stride[:, 3].unsqueeze(1).tile(
            [1, num_gt])

        
        l_ = flatten_x - gt_bboxes[:, 0]
        t_ = flatten_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - flatten_x
        b_ = gt_bboxes[:, 3] - flatten_y

        deltas = paddle.stack([l_, t_, r_, b_], axis=1)
        is_in_gts = deltas.min(axis=1) > 0
        is_in_gts_all = is_in_gts.sum(axis=1) > 0

        
        gt_center_xs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_center_ys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        ct_bound_l = gt_center_xs - self.center_radius * flatten_stride_x
        ct_bound_t = gt_center_ys - self.center_radius * flatten_stride_y
        ct_bound_r = gt_center_xs + self.center_radius * flatten_stride_x
        ct_bound_b = gt_center_ys + self.center_radius * flatten_stride_y

        cl_ = flatten_x - ct_bound_l
        ct_ = flatten_y - ct_bound_t
        cr_ = ct_bound_r - flatten_x
        cb_ = ct_bound_b - flatten_y

        ct_deltas = paddle.stack([cl_, ct_, cr_, cb_], axis=1)
        is_in_cts = ct_deltas.min(axis=1) > 0
        is_in_cts_all = is_in_cts.sum(axis=1) > 0

        
        is_in_gts_or_centers_all = paddle.logical_or(is_in_gts_all,
                                                     is_in_cts_all)

        is_in_gts_or_centers_all_inds = paddle.nonzero(
            is_in_gts_or_centers_all).squeeze(1)

        
        is_in_gts_and_centers = paddle.logical_and(
            paddle.gather(
                is_in_gts.cast('int'), is_in_gts_or_centers_all_inds,
                axis=0).cast('bool'),
            paddle.gather(
                is_in_cts.cast('int'), is_in_gts_or_centers_all_inds,
                axis=0).cast('bool'))
        return is_in_gts_or_centers_all, is_in_gts_or_centers_all_inds, is_in_gts_and_centers

    def dynamic_k_matching(self, cost_matrix, pairwise_ious, num_gt):
        match_matrix = np.zeros_like(cost_matrix.numpy())
        
        topk_ious, _ = paddle.topk(pairwise_ious, self.candidate_topk, axis=0)
        
        dynamic_ks = paddle.clip(topk_ious.sum(0).cast('int'), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = paddle.topk(
                cost_matrix[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            match_matrix[:, gt_idx][pos_idx.numpy()] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        
        extra_match_gts_mask = match_matrix.sum(1) > 1
        if extra_match_gts_mask.sum() > 0:
            cost_matrix = cost_matrix.numpy()
            cost_argmin = np.argmin(
                cost_matrix[extra_match_gts_mask, :], axis=1)
            match_matrix[extra_match_gts_mask, :] *= 0.0
            match_matrix[extra_match_gts_mask, cost_argmin] = 1.0
        
        match_fg_mask_inmatrix = match_matrix.sum(1) > 0
        match_gt_inds_to_fg = match_matrix[match_fg_mask_inmatrix, :].argmax(1)

        return match_gt_inds_to_fg, match_fg_mask_inmatrix

    def get_sample(self, assign_gt_inds, gt_bboxes):
        Pos_inds = np.Unique (np.Nonzero (assign_gt_inds > 0)[0])
        Neg_inds = np.Unique (np.Nonzero (assign_gt_inds == 0)[0])
        Pos_assigned_gt_inds = assign_gt_inds[pos_inds] - 1

        If gt_bboxes. Size == 0:
            
            Assert pos_assigned_gt_inds. Size == 0
            Pos_gt_bboxes = np. Empty_like (gt_bboxes). Reshape (-1, 4)
        Else:
            If len (gt_bboxes. Shape) < 2:
                Gt_bboxes = gt_bboxes.Resize (-1, 4)
            Pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        Return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    Def __call__(self,
                 Flatten_cls_pred_scores,
                 Flatten_center_and_stride,
                 Flatten_bboxes,
                 Gt_bboxes,
                 Gt_labels,
                 Eps=1 e-7):
        """Assign gt to priors using SimOTA.
        TODO: add comment.
        Returns:
            Assign_result: The assigned result.
        """
        Num_gt = gt_bboxes. Shape[0]
        Num_bboxes = flatten_bboxes. Shape[0]

        If num_gt == 0 or num_bboxes == 0:
            
            Label = np.Ones ([num_bboxes], dtype=np. Int 64) * self. Num_classes
            Label_weight = np.Ones ([num_bboxes], dtype=np. Float 32)
            Bbox_target = np. Zeros_like (flatten_center_and_stride)
            Return 0, label, label_weight, bbox_target

        Is_in_gts_or_centers_all, is_in_gts_or_centers_all_inds, is_in_boxes_and_center = self. Get_in_gt_and_in_center_info (
            Flatten_center_and_stride, gt_bboxes)

        
        Valid_flatten_bboxes = flatten_bboxes[is_in_gts_or_centers_all_inds]
        Valid_cls_pred_scores = flatten_cls_pred_scores[
            Is_in_gts_or_centers_all_inds]
        Num_valid_bboxes = valid_flatten_bboxes. Shape[0]

        Pairwise_ious = batch_bbox_overlaps (valid_flatten_bboxes,
                                            Gt_bboxes)  
        If self. Use_vfl:
            Gt_vfl_labels = gt_labels.Squeeze (-1). Unsqueeze (0). Tile (
                [num_valid_bboxes, 1]). Reshape ([-1])
            Valid_pred_scores = valid_cls_pred_scores.Unsqueeze (1). Tile (
                [1, num_gt, 1]). Reshape ([-1, self. Num_classes])
            Vfl_score = np.Zeros (valid_pred_scores. Shape)
            Vfl_score[np.Arange (0, vfl_score. Shape[0]), gt_vfl_labels.Numpy (
            )] = pairwise_ious.Reshape ([-1])
            Vfl_score = paddle. To_tensor (vfl_score)
            Losses_vfl = varifocal_loss (
                Valid_pred_scores, vfl_score,
                Use_sigmoid=False). Reshape ([num_valid_bboxes, num_gt])
            Losses_giou = batch_bbox_overlaps (
                Valid_flatten_bboxes, gt_bboxes, mode='giou')
            Cost_matrix = (
                Losses_vfl * self. Cls_weight + losses_giou * self. Iou_weight +
                Paddle. Logical_not (is_in_boxes_and_center). Cast ('float 32') *
                100000000)
        Else:
            Iou_cost = -paddle.Log (pairwise_ious + eps)
            gt_onehot_label = (F.one_hot (
                Gt_labels.Squeeze (-1). Cast (paddle. Int 64),
                Flatten_cls_pred_scores. Shape[-1]). Cast ('float 32'). Unsqueeze (0)
                               .tile ([num_valid_bboxes, 1, 1]))

            Valid_pred_scores = valid_cls_pred_scores.Unsqueeze (1). Tile (
                [1, num_gt, 1])
            cls_cost = F.binary_cross_entropy (
                Valid_pred_scores, gt_onehot_label, reduction='none'). Sum (-1)

            Cost_matrix = (
                Cls_cost * self. Cls_weight + iou_cost * self. Iou_weight +
                Paddle. Logical_not (is_in_boxes_and_center). Cast ('float 32') *
                100000000)

        Match_gt_inds_to_fg, match_fg_mask_inmatrix = \
            Self. Dynamic_k_matching (
                Cost_matrix, pairwise_ious, num_gt)

        
        Assigned_gt_inds = np.Zeros ([num_bboxes], dtype=np. Int 64)
        Match_fg_mask_inall = np. Zeros_like (assigned_gt_inds)
        Match_fg_mask_inall[is_in_gts_or_centers_all.Numpy (
        )] = match_fg_mask_inmatrix

        Assigned_gt_inds[match_fg_mask_inall.Astype (
            Np. Bool)] = match_gt_inds_to_fg + 1

        Pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds \
            = self. Get_sample (assigned_gt_inds, gt_bboxes.Numpy ())

        Bbox_target = np. Zeros_like (flatten_bboxes)
        Bbox_weight = np. Zeros_like (flatten_bboxes)
        Label = np.Ones ([num_bboxes], dtype=np. Int 64) * self. Num_classes
        Label_weight = np.Zeros ([num_bboxes], dtype=np. Float 32)

        If len (pos_inds) > 0:
            Gt_labels = gt_labels.Numpy ()
            Pos_bbox_targets = pos_gt_bboxes
            Bbox_target[pos_inds, :] = pos_bbox_targets
            Bbox_weight[pos_inds, :] = 1.0
            If not np.Any (gt_labels):
                Label[pos_inds] = 0
            Else:
                Label[pos_inds] = gt_labels.Squeeze (-1)[pos_assigned_gt_inds]

            Label_weight[pos_inds] = 1.0
        If len (neg_inds) > 0:
            Label_weight[neg_inds] = 1.0

        Pos_num = max (pos_inds. Size, 1)

        Return pos_num, label, label_weight, bbox_target
```

# TaskAligned assigner


## 原理介绍

这是 TOOD 里面的一种方法，被用在了百度的 pp-picodet 中，非常强，感觉也很精妙，配合 VFL loss 效果很好。下面介绍一下算法步骤：

1.  将 iou 和分类分数一起算一个综合分数 t = s α ∗ u β t=s^\\alpha*u^\\beta t=sα∗uβ
2.  根据 t 选 topK 个预测框作为正样本。
3.  计算新的软标签 t ′ = t / m a x t ∗ m a x u t'=t/max\_t * max\_u t′=t/maxt​∗maxu​这里对 t 进行了一个归一化操作，让最大值对应最大的 IOU。感觉这个非常强，分类分数和 IOU 都并合并进了软标签。

## 代码

```python
from __future__ import absolute_import
From __future__ import division
From __future__ import print_function

Import paddle
Import paddle. Nn as nn
Import paddle. Nn. Functional as F

From ppdet. Core. Workspace import register
From .. Bbox_utils import batch_iou_similarity
From .utils import (gather_topk_anchors, check_points_inside_bboxes,
                    Compute_max_iou_anchor)

__all__ = ['TaskAlignedAssigner']

@register
Class TaskAlignedAssigner (nn. Layer):
    """TOOD: Task-aligned One-stage Object Detection
    """

    Def __init__(self, topk=13, alpha=1.0, beta=6.0, eps=1 e-9):
        Super (TaskAlignedAssigner, self).__init__()
        Self. Topk = topk
        Self. Alpha = alpha
        Self. Beta = beta
        Self. Eps = eps

    @paddle. No_grad ()
    Def forward (self,
                Pred_scores,
                Pred_bboxes,
                Anchor_points,
                Num_anchors_list,
                Gt_labels,
                Gt_bboxes,
                Pad_gt_mask,
                Bg_index,
                Gt_scores=None):
        R"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py

        The assignment is done in following steps
        1. Compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. Select top-k bbox as candidates for each gt
        3. Limit the positive sample's center in gt (because the anchor-free detector
           Only can predict positive distance)
        4. If an anchor box is assigned to multiple gts, the one with the
           Highest iou will be selected.
        Args:
            Pred_scores (Tensor, float 32): predicted class probability, shape (B, L, C)
            Pred_bboxes (Tensor, float 32): predicted bounding boxes, shape (B, L, 4)
            Anchor_points (Tensor, float 32): pre-defined anchors, shape (L, 2), "cxcy" format
            Num_anchors_list (List): num of anchors in each level, shape (L)
            Gt_labels (Tensor, int 64|int 32): Label of gt_bboxes, shape (B, n, 1)
            Gt_bboxes (Tensor, float 32): Ground truth bboxes, shape (B, n, 4)
            Pad_gt_mask (Tensor, float 32): 1 means bbox, 0 means no bbox, shape (B, n, 1)
            Bg_index (int): background index
            Gt_scores (Tensor|None, float 32) Score of gt_bboxes, shape (B, n, 1)
        Returns:
            Assigned_labels (Tensor): (B, L)
            Assigned_bboxes (Tensor): (B, L, 4)
            Assigned_scores (Tensor): (B, L, C)
        """
        Assert pred_scores. Ndim == pred_bboxes. Ndim
        Assert gt_labels. Ndim == gt_bboxes. Ndim and \
               Gt_bboxes. Ndim == 3

        Batch_size, num_anchors, num_classes = pred_scores. Shape
        _, num_max_boxes, _ = gt_bboxes. Shape

        
        If num_max_boxes == 0:
            Assigned_labels = paddle.Full (
                [batch_size, num_anchors], bg_index, dtype=gt_labels. Dtype)
            Assigned_bboxes = paddle.Zeros ([batch_size, num_anchors, 4])
            Assigned_scores = paddle.Zeros (
                [batch_size, num_anchors, num_classes])
            Return assigned_labels, assigned_bboxes, assigned_scores

        
        Ious = batch_iou_similarity (gt_bboxes, pred_bboxes)
        
        Pred_scores = pred_scores.Transpose ([0, 2, 1])
        Batch_ind = paddle.Arange (
            End=batch_size, dtype=gt_labels. Dtype). Unsqueeze (-1)
        Gt_labels_ind = paddle.Stack (
            [batch_ind.Tile ([1, num_max_boxes]), gt_labels.Squeeze (-1)],
            Axis=-1)
        Bbox_cls_scores = paddle. Gather_nd (pred_scores, gt_labels_ind)
        
        Alignment_metrics = bbox_cls_scores.Pow (self. Alpha) * ious.Pow (
            Self. Beta)

        
        Is_in_gts = check_points_inside_bboxes (anchor_points, gt_bboxes)

        
        
        Is_in_topk = gather_topk_anchors (
            Alignment_metrics * is_in_gts, self. Topk, topk_mask=pad_gt_mask)

        
        Mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        
        
        Mask_positive_sum = mask_positive.Sum (axis=-2)
        If mask_positive_sum.Max () > 1:
            Mask_multiple_gts = (mask_positive_sum.Unsqueeze (1) > 1). Tile (
                [1, num_max_boxes, 1])
            Is_max_iou = compute_max_iou_anchor (ious)
            Mask_positive = paddle.Where (mask_multiple_gts, is_max_iou,
                                         Mask_positive)
            Mask_positive_sum = mask_positive.Sum (axis=-2)
        Assigned_gt_index = mask_positive.Argmax (axis=-2)

        
        Assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        Assigned_labels = paddle.Gather (
            Gt_labels.Flatten (), assigned_gt_index.Flatten (), axis=0)
        Assigned_labels = assigned_labels.Reshape ([batch_size, num_anchors])
        Assigned_labels = paddle.Where (
            Mask_positive_sum > 0, assigned_labels,
            Paddle. Full_like (assigned_labels, bg_index))

        Assigned_bboxes = paddle.Gather (
            Gt_bboxes.Reshape ([-1, 4]), assigned_gt_index.Flatten (), axis=0)
        Assigned_bboxes = assigned_bboxes.Reshape ([batch_size, num_anchors, 4])

        assigned_scores = F.one_hot (assigned_labels, num_classes + 1)
        Ind = list (range (num_classes + 1))
        Ind.Remove (bg_index)
        Assigned_scores = paddle. Index_select (
            Assigned_scores, paddle. To_tensor (ind), axis=-1)
        
        Alignment_metrics *= mask_positive
        Max_metrics_per_instance = alignment_metrics.Max (axis=-1, keepdim=True)
        Max_ious_per_instance = (ious * mask_positive). Max (axis=-1,
                                                           Keepdim=True)
        Alignment_metrics = alignment_metrics / (
            Max_metrics_per_instance + self. Eps) * max_ious_per_instance
        Alignment_metrics = alignment_metrics.Max (-2). Unsqueeze (-1)
        Assigned_scores = assigned_scores * alignment_metrics

        Return assigned_labels, assigned_bboxes, assigned_scores
```

# O 2 f 用于端对端密集检测的 1 对少量标签分配策略


## 原理介绍

这是 2023 年 CVPR（虽然我读完很怀疑。。。可能是我没体会到精妙之处），它的动机是这样的：它想去除 NMS 这个步骤，之所以要 NMS 是因为训练的时候一个 GT 会匹配到多个正样本，除了最正的那个正样本之外，其它的正样本都叫 ambiguous sample，这些 ambiguous sample 作者认为一开始可以对分类 loss 的贡献应该随着训练的推移减少，它对应的分类标签值要逐渐降低，如下图所示：  
![](https://img-blog.csdnimg.cn/61cde77e1f5e475ea4125fbc30039635.png)

  
**然后它这个标签分配算法跟 TOOD 贼像，流程是一样的，就是计算分数的时候有差别：** 

![](https://img-blog.csdnimg.cn/32b57c8369ab4eafbe423362aa683fcd.png)

  
**然后它这个归一化的时候改了下，让 ambiguous sample 的标签值逐渐降低，核心就是这一点。我在 TaskAligned 的基础上改成了这种标签计算方式，初步效果变差，继续探索中。。。总觉得这文章说的不详细，也没多少实验。**   
![](https://img-blog.csdnimg.cn/2c9f8cf08cd2465c8f5849d6285857f4.png)