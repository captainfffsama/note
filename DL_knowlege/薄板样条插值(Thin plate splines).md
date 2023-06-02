#图像处理 #代码片段

[toc]

# 自用代码

```python
import json
import base64
from collections import defaultdict
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

import debug_tools as D

COLOR_DICT=mcolors.CSS4_COLORS


def generate_point_pair(points_json):
    all_points=defaultdict(list)
    for point_data in points_json:
        all_points[point_data['label']].append(np.array(point_data['points']).squeeze())

    h=np.sqrt(np.sum((all_points['1'][0]-all_points['2'][0])**2))
    w=np.sqrt(np.sum((all_points['4'][0]-all_points['1'][0])**2))
        # +np.sqrt(np.sum((all_points['3'][0]-all_points['2'][0])**2))//2

    dst_pts=[[50,h+50],[50,50],[w+50,50],[w+50,h+50]]

    inner_pt_num=len(all_points['5'])

    inner_pts=[[(i+1)*(w/(inner_pt_num+1))+50,h+50] for i in range(inner_pt_num)]
    
    out_pt_num=len(all_points['6'])
        
    out_pts=[[(i+1)*(w/(out_pt_num+1))+50,50] for i in range(out_pt_num)]
    

    dst_pts =dst_pts+inner_pts+out_pts

    all_points['5'].sort(key=lambda x: np.sum((x-all_points['1'][0])**2))
    all_points['6'].sort(key=lambda x: np.sum((x-all_points['2'][0])**2))

    ori_all_points = all_points['1']+all_points['2']+all_points['3']+all_points['4']+all_points['5']+all_points['6']
    return np.stack(ori_all_points,axis=0),np.array(dst_pts),h,w

def get_color():
    return random.sample(COLOR_DICT.keys(),1)[0]

def show_img(img,dst_img,src_pts,dst_pts):
    fig, axs = plt.subplots(2,1, figsize=(15 * 1, 15 *1))
    axs[0].imshow(img[:,:,::-1])
    axs[1].imshow(dst_img[:,:,::-1])
    axs[0].plot(src_pts[:,0],src_pts[:,1],'o','r')
    axs[1].plot(dst_pts[:,0],dst_pts[:,1],'o','r')
    for idx,(kp_a,kp_b) in enumerate(zip(src_pts,dst_pts)):
        if idx in (0,1,2,3,4,(src_pts.shape[0]//2+2)):
            color=get_color()

        con =  patches.ConnectionPatch(
            xyA=kp_a, coordsA=axs[0].transData,
            xyB=kp_b, coordsB=axs[1].transData,
            arrowstyle="-", shrinkB=5,color=color)
        axs[1].add_artist(con)

    plt.show()

def main(json_path):
    with open(json_path,'r') as fr:
        anno_content=json.load(fr)
    imgdata=anno_content['imageData']
    imgdata=base64.b64decode(imgdata)
    img_array=np.fromstring(imgdata,np.uint8)
    img=cv2.imdecode(img_array,1)

    src_point,dst_point,dst_h,dst_w=generate_point_pair(anno_content['shapes'])
    src_point=np.around(src_point)
    dst_point=np.around(dst_point)
    np.save("/data/indoor_meter/test1/pt_src",src_point)
    np.save("/data/indoor_meter/test1/pt_dst",dst_point)

    matches=[cv2.DMatch(i,i,0) for i in range(src_point.shape[0])]
    tps = cv2.createThinPlateSplineShapeTransformer(0)

    src_point=src_point.reshape(1,-1,2)
    dst_point=dst_point.reshape(1,-1,2)
    tps.estimateTransformation(dst_point.astype(np.int32),src_point.astype(np.int32),matches)
    print(tps.getRegularizationParameter())
    print(tps.applyTransformation(src_point,None))

    new_img=tps.warpImage(img)

    show_img(img,new_img,src_point.squeeze(),dst_point.squeeze())

if __name__=="__main__":
    json_path='/data/indoor_meter/test1/1.json'
    main(json_path)

```

# 几个注意点
1. 点的对应关系一定要对
2. 注意 `estimateTransformation` 中是 dst 在前,src 在后
# 相关资料
- <https://blog.csdn.net/weixin_42028608/article/details/106128409>
- <https://github.com/cheind/py-thin-plate-spline/blob/master/thinplate/pytorch.py>
- <https://blog.csdn.net/VictoriaW/article/details/70161180>
- <https://blog.csdn.net/zhaominyiz/article/details/104114538>