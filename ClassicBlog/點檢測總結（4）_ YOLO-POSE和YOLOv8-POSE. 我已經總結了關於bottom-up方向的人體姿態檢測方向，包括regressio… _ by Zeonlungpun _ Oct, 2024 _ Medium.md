[

![](https://miro.medium.com/v2/da:true/resize:fill:88:88/0*GA26dCHZTpPXzAgD)






](https://medium.com/@zeonlungpun?source=post_page---byline--739a75310ef4--------------------------------)

我已經總結了關於bottom-up方向的人體姿態檢測方向，包括regression based和heatmap base的方法。可以看出，heatmap base的方法後處理會相對複雜，且對於多人的姿態檢測時，更顯得複雜。接下來這個blog我要研究、總結一下top-down方法，即先用detector檢測出人，再檢測出關鍵點。這類方法也可以看作直接回歸出點。

主要研究兩個模型：YOLO-POSE（主要基於YOLOv5）和YOLOv8-POSE。

這兩種方法本質上就是在YOLOv5或者YOLOv8的基礎上加一個pose的預測頭，在做目標檢測的同時對人體關鍵點進行檢測。主要思路和好處如下所示：

> _Existing heatmap based two-stage approaches are sub-optimal as they are not end-to-end trainable and training relies on a surrogate L1 loss that is not equivalent to maximizing the evaluation metric, i.e. Object Keypoint Similarity (OKS). Our framework allows us to train the model end-to-end and optimize the OKS metric itself. The proposed model learns to jointly detect bounding boxes for multiple persons and their corresponding 2D poses in a single forward pass and thus bringing in the best of both top-down and bottom-up approaches. Proposed approach doesn’t require the postprocessing of bottom-up approaches to group detected keypoints into a skeleton as each bounding box has an associated pose, resulting in an inherent grouping of the keypoints._

![](https://miro.medium.com/v2/resize:fit:875/0*AGMylKSlM4SdzJRn)

添加图片注释，不超过 140 字（可选）

網絡的輸出現在變成這樣：bounding box的4個座標，框的置信度，人類別的置信度；以及每個人預測出17個keypoints，每個keypoint的座標和置信度：

![](https://miro.medium.com/v2/resize:fit:875/0*6bqtLD5-bCt8JcmQ)

添加图片注释，不超过 140 字（可选）

優點總結就是：

*   多人關鍵點檢測更快速、方便

> _solving multi-person pose estimation in line with object detection since major challenges like scale variation and occlusion are common to both. Thus, taking the first step toward unifying these two fields. Our approach will directly benefit from any advancement in the field of Object detection._

*   避免繁瑣的後處理步驟

> _heatmap-free approach uses standard OD postprocessing instead of complex post-processing involving Pixel level NMS, adjustment, refinement, line-integral, and various grouping algorithms. The approach is robust because of end-to-end training without independent post-processing._

*   可以直接利用網絡優化評價指標OKS

> _Extended the idea of IoU loss from box detection to keypoints. Object keypoint similarity (OKS) is not just used for evaluation but as a loss for training. OKS loss is scale-invariant and inherently gives different weighting to different keypoints_

類似於IOU LOSS，這裡提出了OKS loss。該方法的好處就是直接利用網絡優化metrics指標，而不是間接優化：

> _Conventionally, heat-map based bottom-up approaches use L1 loss to detect keypoints. However, L1 loss may not necessarily be suitable to obtain optimal OKS. Again, L1 loss is naïve and doesn’t take into consideration scale of an object or the type of a keypoint._

![](https://miro.medium.com/v2/resize:fit:875/0*u1pW1rvcpBqKJZPp)

添加图片注释，不超过 140 字（可选）

關鍵點置信度損失：

![](https://miro.medium.com/v2/resize:fit:875/0*2_ZhNhZfENoD8Bd2)

添加图片注释，不超过 140 字（可选）

yolov8-obb的主要思路就是在檢測頭的基礎上加一個obb模塊，在進行框預測的同時順便進行關鍵點檢測。其中需要注意的是，正負樣本分配、NMS都是以目標檢測爲基準；即先目標檢測，再關鍵點檢測。

配置文件

```
\# Ultralytics YOLO 🚀, AGPL-3.0 license  
\# YOLOv8-pose keypoints/pose estimation model. For Usage examples see https://docs.ultralytics.com/tasks/pose
```

```
\# Parameters  
nc: 1  # number of classes  
kpt\_shape: \[17, 3\]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)  
scales: # model compound scaling constants, i.e. 'model=yolov8n-pose.yaml' will call yolov8-pose.yaml with scale 'n'  
  # \[depth, width, max\_channels\]  
  n: \[0.33, 0.25, 1024\]  
  s: \[0.33, 0.50, 1024\]  
  m: \[0.67, 0.75, 768\]  
  l: \[1.00, 1.00, 512\]  
  x: \[1.00, 1.25, 512\]\# YOLOv8.0n backbone  
backbone:  
  # \[from, repeats, module, args\]  
  - \[-1, 1, Conv, \[64, 3, 2\]\]  # 0-P1/2  
  - \[-1, 1, Conv, \[128, 3, 2\]\]  # 1-P2/4  
  - \[-1, 3, C2f, \[128, True\]\]  
  - \[-1, 1, Conv, \[256, 3, 2\]\]  # 3-P3/8  
  - \[-1, 6, C2f, \[256, True\]\]  
  - \[-1, 1, Conv, \[512, 3, 2\]\]  # 5-P4/16  
  - \[-1, 6, C2f, \[512, True\]\]  
  - \[-1, 1, Conv, \[1024, 3, 2\]\]  # 7-P5/32  
  - \[-1, 3, C2f, \[1024, True\]\]  
  - \[-1, 1, SPPF, \[1024, 5\]\]  # 9\# YOLOv8.0n head  
head:  
  - \[-1, 1, nn.Upsample, \[None, 2, 'nearest'\]\]  
  - \[\[-1, 6\], 1, Concat, \[1\]\]  # cat backbone P4  
  - \[-1, 3, C2f, \[512\]\]  # 12 - \[-1, 1, nn.Upsample, \[None, 2, 'nearest'\]\]  
  - \[\[-1, 4\], 1, Concat, \[1\]\]  # cat backbone P3  
  - \[-1, 3, C2f, \[256\]\]  # 15 (P3/8-small) - \[-1, 1, Conv, \[256, 3, 2\]\]  
  - \[\[-1, 12\], 1, Concat, \[1\]\]  # cat head P4  
  - \[-1, 3, C2f, \[512\]\]  # 18 (P4/16-medium) - \[-1, 1, Conv, \[512, 3, 2\]\]  
  - \[\[-1, 9\], 1, Concat, \[1\]\]  # cat head P5  
  - \[-1, 3, C2f, \[1024\]\]  # 21 (P5/32-large)  
#kpt\_shape 一般爲 (17,3)  
  - \[\[15, 18, 21\], 1, Pose, \[nc, kpt\_shape\]\]  # Pose(P3, P4, P5)
```

注意，多了個Pose結構。

pose結構

```
class Pose(Detect):  
    """YOLOv8 Pose head for keypoints models."""
```

```
 def \_\_init\_\_(self, nc=80, kpt\_shape=(17, 3), ch=()):  
        """Initialize YOLO network with default parameters and Convolutional Layers."""  
        super().\_\_init\_\_(nc, ch)  
        self.kpt\_shape = kpt\_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)  
        self.nk = kpt\_shape\[0\] \* kpt\_shape\[1\]  # number of keypoints total  
        self.detect = Detect.forward  
        #中間過渡層的channel數量  
        c4 = max(ch\[0\] // 4, self.nk)  
       #三個輸出頭  
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch) def forward(self, x):  
        """Perform forward pass through YOLO model and return predictions."""  
        bs = x\[0\].shape\[0\]  # batch size  
#同時輸出檢測框和關鍵點  
        kpt = torch.cat(\[self.cv4\[i\](x\[i\]).view(bs, self.nk, -1) for i in range(self.nl)\], -1)  # (bs, 17\*3, h\*w)  
        x = self.detect(self, x)  
        if self.training:  
            return x, kpt  
        pred\_kpt = self.kpts\_decode(bs, kpt)  
        return torch.cat(\[x, pred\_kpt\], 1) if self.export else (torch.cat(\[x\[0\], pred\_kpt\], 1), (x\[1\], kpt)) def kpts\_decode(self, bs, kpts):  
        """Decodes keypoints."""  
        ndim = self.kpt\_shape\[1\]  
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER\_FOR\_GREATER\_OP\_CODES' bug  
            y = kpts.view(bs, \*self.kpt\_shape, -1)  
            a = (y\[:, :, :2\] \* 2.0 + (self.anchors - 0.5)) \* self.strides  
         #如果 ndim == 3，對第三個維度（置信度）應用 sigmoid 函數。  
            if ndim == 3:  
                a = torch.cat((a, y\[:, :, 2:3\].sigmoid()), 2)  
            return a.view(bs, self.nk, -1)  
        else:  
            y = kpts.clone()  
            if ndim == 3:  
                y\[:, 2::3\] = y\[:, 2::3\].sigmoid()  # sigmoid (WARNING: inplace .sigmoid\_() Apple MPS bug)  
            y\[:, 0::ndim\] = (y\[:, 0::ndim\] \* 2.0 + (self.anchors\[0\] - 0.5)) \* self.strides  
            y\[:, 1::ndim\] = (y\[:, 1::ndim\] \* 2.0 + (self.anchors\[1\] - 0.5)) \* self.strides  
            return y
```

模型預測關鍵點相對於某個**anchor point** 的偏移量。這樣的設計使模型更容易學習並提高準確性。模型輸出的 y\[:, 0::ndim\] 和 y\[:, 1::ndim\] 分別代表 x 和 y 坐標的預測值。

*   這些預測值通常是相對於 anchor 的偏移量，範圍一般在 \[-1, 1\] 之間。
*   y\[:, 0::ndim\] \* 2.0 + (self.anchors\[0\] — 0.5) 中的 \* 2.0 將預測的偏移量放大，使其範圍變為 \[0, 2\]，然後加上 (self.anchors\[0\] — 0.5) 將其轉換為相對於 anchor 的實際偏移量。
*   self.anchors\[0\] 和 self.anchors\[1\] 是對應於 x 和 y 坐標的 anchor 點，-0.5 是為了調整偏移，使得預測值可以精確對應到 anchor 的周圍。
*   最後，結果乘以 self.strides。strides 表示在圖像中的網格步幅，確保偏移量轉換為圖像中的實際像素坐標。這意味著 strides 决定了 anchor 在原图中的大小比例，每一个 stride 相当于在图像中的实际像素数量。

```
from ultralytics.engine.results import Results  
from ultralytics.models.yolo.detect.predict import DetectionPredictor  
from ultralytics.utils import DEFAULT\_CFG, LOGGER, ops  

```

```
class PosePredictor(DetectionPredictor):  
    """  
    A class extending the DetectionPredictor class for prediction based on a pose model. Example:  
        \`\`\`python  
        from ultralytics.utils import ASSETS  
        from ultralytics.models.yolo.pose import PosePredictor args = dict(model='yolov8n-pose.pt', source=ASSETS)  
        predictor = PosePredictor(overrides=args)  
        predictor.predict\_cli()  
        \`\`\`  
    """ def \_\_init\_\_(self, cfg=DEFAULT\_CFG, overrides=None, \_callbacks=None):  
        """Initializes PosePredictor, sets task to 'pose' and logs a warning for using 'mps' as device."""  
        super().\_\_init\_\_(cfg, overrides, \_callbacks)  
        self.args.task = 'pose'  
        if isinstance(self.args.device, str) and self.args.device.lower() == 'mps':  
            LOGGER.warning("WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "  
                           'See [https://github.com/ultralytics/ultralytics/issues/4031.')](https://github.com/ultralytics/ultralytics/issues/4031.')) def postprocess(self, preds, img, orig\_imgs):  
        """Return detection results for a given input image or list of images."""  
 #先對框作NMS  
        preds = ops.non\_max\_suppression(preds,  
                                        self.args.conf,  
                                        self.args.iou,  
                                        agnostic=self.args.agnostic\_nms,  
                                        max\_det=self.args.max\_det,  
                                        classes=self.args.classes,  
                                        nc=len(self.model.names)) if not isinstance(orig\_imgs, list):  # input images are a torch.Tensor, not a list  
            orig\_imgs = ops.convert\_torch2numpy\_batch(orig\_imgs)  
 #NMS後的結果  
        results = \[\]  
        for i, pred in enumerate(preds):  
            orig\_img = orig\_imgs\[i\]  
           #letterbox的逆過程，主要針對框的預測  
            pred\[:, :4\] = ops.scale\_boxes(img.shape\[2:\], pred\[:, :4\], orig\_img.shape).round()  
            pred\_kpts = pred\[:, 6:\].view(len(pred), \*self.model.kpt\_shape) if len(pred) else pred\[:, 6:\]  
           #與ops.scale\_boxes類似，主要針對點  
            pred\_kpts = ops.scale\_coords(img.shape\[2:\], pred\_kpts, orig\_img.shape)  
            img\_path = self.batch\[0\]\[i\]  
            results.append(  
                Results(orig\_img, path=img\_path, names=self.model.names, boxes=pred\[:, :6\], keypoints=pred\_kpts))  
        return results
```

本質就是上文提及的OKS LOSS：

```
class KeypointLoss(nn.Module):  
    """Criterion class for computing training losses."""
```

```
 def \_\_init\_\_(self, sigmas) -> None:  
        """Initialize the KeypointLoss class."""  
        super().\_\_init\_\_()  
        self.sigmas = sigmas def forward(self, pred\_kpts, gt\_kpts, kpt\_mask, area):  
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""  
        d = (pred\_kpts\[..., 0\] - gt\_kpts\[..., 0\]) \*\* 2 + (pred\_kpts\[..., 1\] - gt\_kpts\[..., 1\]) \*\* 2  
        kpt\_loss\_factor = kpt\_mask.shape\[1\] / (torch.sum(kpt\_mask != 0, dim=1) + 1e-9)  
        e = d / (2 \* self.sigmas) \*\* 2 / (area + 1e-9) / 2  # from cocoeval  
        return (kpt\_loss\_factor.view(-1, 1) \* ((1 - torch.exp(-e)) \* kpt\_mask)).mean()
```

全部損失：

主要是框損失（dfl loss+ ciou loss）、類別損失、oks loss（點損失）、點的置信度損失。

```
class v8PoseLoss(v8DetectionLoss):  
    """Criterion class for computing training losses."""
```

```
 def \_\_init\_\_(self, model):  # model must be de-paralleled  
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""  
        super().\_\_init\_\_(model)  
        self.kpt\_shape = model.model\[-1\].kpt\_shape  
        #點的置信度損失  
        self.bce\_pose = nn.BCEWithLogitsLoss()  
        is\_pose = self.kpt\_shape == \[17, 3\]  
        nkpt = self.kpt\_shape\[0\]  # number of keypoints  
        sigmas = torch.from\_numpy(OKS\_SIGMA).to(self.device) if is\_pose else torch.ones(nkpt, device=self.device) / nkpt  
        self.keypoint\_loss = KeypointLoss(sigmas=sigmas) def \_\_call\_\_(self, preds, batch):  
        """Calculate the total loss and detach it."""  
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt\_location, kpt\_visibility  
        feats, pred\_kpts = preds if isinstance(preds\[0\], list) else preds\[1\]  
##把三個尺度的檢測頭進行拼接，再拆分成box預測值和類別概率預測值  
        pred\_distri, pred\_scores = torch.cat(\[xi.view(feats\[0\].shape\[0\], self.no, -1) for xi in feats\], 2).split(  
            (self.reg\_max \* 4, self.nc), 1) # B, grids, ..  
#(bs,total\_anchor\_num,num\_class)-->(bs,num\_class,total\_anchor\_num)  
        pred\_scores = pred\_scores.permute(0, 2, 1).contiguous()  
 # (bs,total\_anchor\_num,reg\_max\*4)-->(bs,reg\_max\*4,total\_anchor\_num)  
        pred\_distri = pred\_distri.permute(0, 2, 1).contiguous()  
#(bs,total\_anchor\_num,2)-->(bs,2,total\_anchor\_num)  
        pred\_kpts = pred\_kpts.permute(0, 2, 1).contiguous() dtype = pred\_scores.dtype  
        imgsz = torch.tensor(feats\[0\].shape\[2:\], device=self.device, dtype=dtype) \* self.stride\[0\]  # image size (h,w)  
        anchor\_points, stride\_tensor = make\_anchors(feats, self.stride, 0.5) # Targets  
        batch\_size = pred\_scores.shape\[0\]  
        batch\_idx = batch\['batch\_idx'\].view(-1, 1)  
        targets = torch.cat((batch\_idx, batch\['cls'\].view(-1, 1), batch\['bboxes'\]), 1)  
\# 將box信息由歸一化尺度轉換到輸入圖像尺度，並對bath內每張圖像的gt個數進行對齊(目標個數都設定一個統一的值M，  
#方便進行矩陣運算)  
    # M值的設定規則為，選取batch內最大的gt\_num作為M  
    # targets: (bs,M(n\_max\_boxes) ,5) ,其中5 = cls,cx,cy,width,height  
        targets = self.preprocess(targets.to(self.device), batch\_size, scale\_tensor=imgsz\[\[1, 0, 1, 0\]\])  
        gt\_labels, gt\_bboxes = targets.split((1, 4), 2)  # cls, xyxy  
\# # 通過對四個坐標值相加，如果為0，則說明該gt信息為填充信息，在mask中為False，後期計算過程中會進行過濾  
        mask\_gt = gt\_bboxes.sum(2, keepdim=True).gt\_(0) #  # Pboxes:得到預測的框座標四個值的偏移量  
        pred\_bboxes = self.bbox\_decode(anchor\_points, pred\_distri)  # xyxy, (b, h\*w, 4)  
     #得到預測的點座標偏移量  
        pred\_kpts = self.kpts\_decode(anchor\_points, pred\_kpts.view(batch\_size, -1, \*self.kpt\_shape))  # (b, h\*w, 17, 3)  
正樣本篩選過程  
    """  
target\_labels(Tensor): shape(bs, num\_total\_anchors)  
target\_bboxes(Tensor): shape(bs, num\_total\_anchors, 4)  
target\_scores(Tensor): shape(bs, num\_total\_anchors, num\_classes)  
fg\_mask(Tensor): shape(bs, num\_total\_anchors):mask of each anchor boxes  
(distinguish the positive and negative samples)  
    """  
        \_, target\_bboxes, target\_scores, fg\_mask, target\_gt\_idx = self.assigner(  
            pred\_scores.detach().sigmoid(), (pred\_bboxes.detach() \* stride\_tensor).type(gt\_bboxes.dtype),  
            anchor\_points \* stride\_tensor, gt\_labels, gt\_bboxes, mask\_gt) target\_scores\_sum = max(target\_scores.sum(), 1) # # Cls loss:每個都類都使用二元交叉熵損失  
        loss\[3\] = self.bce(pred\_scores, target\_scores.to(dtype)).sum() / target\_scores\_sum  # BCE # Bbox loss  
        if fg\_mask.sum():  
            target\_bboxes /= stride\_tensor  
            loss\[0\], loss\[4\] = self.bbox\_loss(pred\_distri, pred\_bboxes, anchor\_points, target\_bboxes, target\_scores,  
                                              target\_scores\_sum, fg\_mask)  
            keypoints = batch\['keypoints'\].to(self.device).float().clone()  
            keypoints\[..., 0\] \*= imgsz\[1\]  
            keypoints\[..., 1\] \*= imgsz\[0\] loss\[1\], loss\[2\] = self.calculate\_keypoints\_loss(fg\_mask, target\_gt\_idx, keypoints, batch\_idx,  
                                                             stride\_tensor, target\_bboxes, pred\_kpts)  
#各個損失加權求和  
        loss\[0\] \*= self.hyp.box  # box gain  
        loss\[1\] \*= self.hyp.pose  # pose gain  
        loss\[2\] \*= self.hyp.kobj  # kobj gain  
        loss\[3\] \*= self.hyp.cls  # cls gain  
        loss\[4\] \*= self.hyp.dfl  # dfl gain return loss.sum() \* batch\_size, loss.detach()  # loss(box, cls, dfl) @staticmethod  
    def kpts\_decode(anchor\_points, pred\_kpts):  
        """Decodes predicted keypoints to image coordinates."""  
        y = pred\_kpts.clone()  
        y\[..., :2\] \*= 2.0  
        y\[..., 0\] += anchor\_points\[:, \[0\]\] - 0.5  
        y\[..., 1\] += anchor\_points\[:, \[1\]\] - 0.5  
        return y
```

重點來了，calculate keypoint loss:

這裡有個參數masks，代表由目標檢測進行分配的正樣本。所以實際上是在目標檢測基礎上進行關鍵點檢測。

```
def calculate\_keypoints\_loss(self, masks, target\_gt\_idx, keypoints, batch\_idx, stride\_tensor, target\_bboxes,  
                                 pred\_kpts):  
        """  
        Calculate the keypoints loss for the model.
```

```
 This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is  
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is  
        a binary classification loss that classifies whether a keypoint is present or not. Args:  
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N\_anchors).  
            target\_gt\_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N\_anchors).  
            keypoints (torch.Tensor): Ground truth keypoints, shape (N\_kpts\_in\_batch, N\_kpts\_per\_object, kpts\_dim).  
            batch\_idx (torch.Tensor): Batch index tensor for keypoints, shape (N\_kpts\_in\_batch, 1).  
            stride\_tensor (torch.Tensor): Stride tensor for anchors, shape (N\_anchors, 1).  
            target\_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N\_anchors, 4).  
            pred\_kpts (torch.Tensor): Predicted keypoints, shape (BS, N\_anchors, N\_kpts\_per\_object, kpts\_dim). Returns:  
            (tuple): Returns a tuple containing:  
                - kpts\_loss (torch.Tensor): The keypoints loss.  
                - kpts\_obj\_loss (torch.Tensor): The keypoints object loss.  
        """  
        batch\_idx = batch\_idx.flatten()  
        batch\_size = len(masks) # Find the maximum number of keypoints in a single image  
        max\_kpts = torch.unique(batch\_idx, return\_counts=True)\[1\].max() # Create a tensor to hold batched keypoints  
        # （bs, max\_kpts,N\_kpts\_per\_object, kpts\_dim ）  
        batched\_keypoints = torch.zeros((batch\_size, max\_kpts, keypoints.shape\[1\], keypoints.shape\[2\]),  
                                        device=keypoints.device) # Fill batched\_keypoints with keypoints based on batch\_idx  
        for i in range(batch\_size):  
            keypoints\_i = keypoints\[batch\_idx == i\]  
            batched\_keypoints\[i, :keypoints\_i.shape\[0\]\] = keypoints\_i # Expand dimensions of target\_gt\_idx to match the shape of batched\_keypoints  
       #(BS, N\_anchors,1,1)  
        target\_gt\_idx\_expanded = target\_gt\_idx.unsqueeze(-1).unsqueeze(-1) # Use target\_gt\_idx\_expanded to select keypoints from batched\_keypoints  
        selected\_keypoints = batched\_keypoints.gather(  
            1, target\_gt\_idx\_expanded.expand(-1, -1, keypoints.shape\[1\], keypoints.shape\[2\])) # Divide coordinates by stride  
        selected\_keypoints /= stride\_tensor.view(1, -1, 1, 1) kpts\_loss = 0  
        kpts\_obj\_loss = 0 if masks.any():  
       #只計算正樣本的oks損失  
      #正樣本由目標檢測進行分配，實際上是在目標檢測基礎上進行關鍵點檢測  
            gt\_kpt = selected\_keypoints\[masks\]  
            area = xyxy2xywh(target\_bboxes\[masks\])\[:, 2:\].prod(1, keepdim=True)  
            pred\_kpt = pred\_kpts\[masks\]  
            #把vis=0的keypoint去掉  
            kpt\_mask = gt\_kpt\[..., 2\] != 0 if gt\_kpt.shape\[-1\] == 3 else torch.full\_like(gt\_kpt\[..., 0\], True)  
            kpts\_loss = self.keypoint\_loss(pred\_kpt, gt\_kpt, kpt\_mask, area)  # pose loss if pred\_kpt.shape\[-1\] == 3:  
                kpts\_obj\_loss = self.bce\_pose(pred\_kpt\[..., 2\], kpt\_mask.float())  # keypoint obj loss return kpts\_loss, kpts\_obj\_loss
```