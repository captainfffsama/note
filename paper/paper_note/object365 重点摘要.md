#目标检测 
#数据集

[toc]

# Abstract
没啥好说

# 1.引言
注意第三段quality提到的质量控制.
Quality. In addition to the size, annotation quality is of great importance when building a dataset. To ensure quality, we divide the annotation pipeline into three steps, which can significantly reduce the job requirement for the annotators. Besides the annotators, we also include inspectors and examiners to review the quality of the annotations. To reduce ambiguities during the annotation process, we apply two consistency rules. This annotation pipeline ensures that we obtain high-quality annotation with high efficiency.

将标注步骤分解成三步来减少标注人员的工作难度.添加巡检和考官来进一步控制标注质量.同时设定来两个一致性原则来减少标注歧义.

# 2.相关工作
没啥好说

# 3.数据集
## 3.1 数据收集
### 3.1.1 数据来源
主要来源Flicker

### 3.1.2 目标类别
分11个父类:human and related accessories, living room, clothes, kitchen, instrument, transportation, bathroom, electronics, food (vegetables), office supplies, and animal.

又将11个父类细分成442个子类,然后标来100K张图之后,结合VOC和COCO,选出来365类 作为最终标准类.

### 3.1.3 非标志型图片
学COCO,去掉了一些适合图片分类而非检测的图片,比如仅仅包含一个目标在图片中心的.

## 3.2 标注
### 3.2.1 标注步骤
由于很难记住365类且一部分不包含目标和标志性图片应该去掉.设计了3个步骤:
1. 二分类.仅保留非标志性,且至少包含一个11父类目标的图片.
2. 给1中筛出的图片打上对应的父类标签(不用框,仅分类),可能不止一个标签.
3. 对2中分好的图片,标注bbox,每个标注员仅仅标注一个父类标签包含的子类实例.

![](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200908115200.png)

### 3.2.2标注人员
将人分成标注员,巡检员,考官.
- 标注员:以上三步都要做.第三步时,一个标注员一次仅标一子类,开标之前还要培训和考试.
- 巡检员:检查所有标注员的标注,发现错就退回错的图片给标注员
- 考官:抽看看100张巡检检查过的图片,发现一个错就返回整个样本 batch.

### 3.2.3 一致性原则
- **分类原则:** 功能优先原则.

![](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200908120941.png)
左图既可以视为水龙头也可以视为是茶壶,考虑到实际功能,标水龙头.同理右图中的玩具熊,标玩具而不是熊.

```
自我思考:针对自己数据集,也应订立一些指导思想
```

- **画框原则:** 无歧义最大框原则.
![](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200908130426.png)
右图框整个塔会导致歧义成 tower 类,所以仅仅框钟表的部分.

## 3.3 统计
略