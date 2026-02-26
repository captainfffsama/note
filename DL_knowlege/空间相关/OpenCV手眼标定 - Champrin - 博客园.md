使用 OpenCV 进行手眼标定，版本需要 4.1.0 及以上。

为了确定手眼变换，至少需要两个具有非平行旋转轴的运动。因此，至少需要 3 个不同的姿势，但强烈建议使用更多的姿势。—— OpenCV 官方文档提示

`cv::calibrateHandEye()`
------------------------

> [OpenCV有关手眼标定官方文档](https://docs.opencv.org/4.1.0/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b)

```
void cv::calibrateHandEye(
    InputArrayOfArrays R\_gripper2base,
    InputArrayOfArrays t\_gripper2base,
    InputArrayOfArrays R\_target2cam,
    InputArrayOfArrays t\_target2cam,
    OutputArray R\_cam2gripper,
    OutputArray t\_cam2gripper,
    HandEyeCalibrationMethod method = CALIB\_HAND\_EYE\_TSAI
) 
``` 

*   矩阵输入参数类型都为 `std::vector<cv::Mat>`，即 张标定图片，需要有 组相关的矩阵依序放入 `std::vector` 以传入参数
*   `method`：求解 个 方程组的方法
    *   [`method`类型官方文档](https://docs.opencv.org/4.1.0/d9/d0c/group__calib3d.html#gad10a5ef12ee3499a0774c7904a801b99)
    *   `CALIB_HAND_EYE_TSAI`：一种完全自主高效的3D机器人手眼标定新技术
    *   `CALIB_HAND_EYE_PARK`：机器人传感器标定，求解欧几里德群上的
    *   `CALIB_HAND_EYE_HORAUD`：手眼标定
    *   `CALIB_HAND_EYE_ANDREFF`：在线手眼标定
    *   `CALIB_HAND_EYE_DANIILIDIS`：使用双四元数的手眼标定

坐标系：

*   `base`：机械臂基地坐标系
*   `gripper`：机械臂末端坐标系（夹爪坐标系）
*   `cam`：相机坐标系
*   `target`：标定板坐标系（世界坐标系）

方法变量名跟眼在手上的情况关联，代表：

对于眼在手上
------

眼在手上，标定求解的是 相机坐标系 相对于 机械臂末端坐标系 的变换矩阵 。

输入：

*   `R_gripper2base`：机械臂末端坐标系 相对于 机械臂基底坐标系的 旋转矩阵
*   `t_gripper2base`：机械臂末端坐标系 相对于 机械臂基底坐标系的 平移矩阵
*   `R_target2cam`：世界坐标系 相对于 相机坐标系的 旋转矩阵
*   `t_target2cam`：世界坐标系 相对于 相机坐标系的 平移矩阵

输出：

*   `R_cam2gripper`：相机坐标系 相对于 机械臂末端坐标系的 旋转矩阵
*   `t_cam2gripper`：相机坐标系 相对于 机械臂末端坐标系的 平移矩阵

对于眼在手外
------

眼在手外，标定求解的是 相机坐标系 相对于 机械臂基底坐标系 的转换矩阵 。

输入：

*   `R_gripper2base`为 机械臂基底坐标系 到 机械臂末端坐标系的 旋转矩阵
*   `t_gripper2base`为 机械臂基底坐标系 到 机械臂末端坐标系的 平移矩阵
*   `R_target2cam`为 标定板坐标系 到 相机坐标系的 旋转矩阵
*   `t_target2cam`为 标定板坐标系 到 相机坐标系的 平移矩阵

输出：

*   `R_cam2gripper`相机坐标系 到 机械臂基底坐标系的 旋转矩阵
*   `t_cam2gripper`相机坐标系 到 机械臂基底坐标系的 平移矩阵

由机械臂正运动学解出 机械臂基底坐标系 与 机械臂末端坐标系的 之间的变换矩阵
---------------------------------------

和，可通过解算机械臂的位姿求得  
对于每个位姿，通常会返回六个参数：  
这六个参数是机械臂末端在基底坐标系下的位姿的表示  
即：  
为欧拉角，欧拉角(旋转角)到旋转矩阵的转换，见`旋转矩阵.md`

solvePnP
--------

使用OpenCV中的`solvePnP`，可以求解相机到标定板坐标系的旋转矩阵和平移矩阵，即和

> solvePnP是用来求解2D-3D的位姿对应关系，在这里，图片是(2D)，而标定板坐标系是(3D)  
> 利用solvePnP函数，就可以得到图片（相机坐标系）与标定板坐标系的变换关系

```
bool cv::solvePnP(
    InputArray objectPoints,
    InputArray imagePoints,
    InputArray cameraMatrix,
    InputArray distCoeffs,
    OutputArray rvec,
    OutputArray tvec,
    bool useExtrinsicGuess = false,
    int flags = SOLVEPNP\_ITERATIVE
) 
``` 

输入：  
`objectPoints`为棋盘格坐标系，由棋盘格真实坐标生成，一般以棋盘格左上角顶点建立，轴坐标都为0  
斗  
`imagePoints`为图片识别到的角点坐标，与objectPoints中的值一一对应  
`cameraMatrix`为相机内参矩阵  
`distCoeffs`为相机畸变矩阵

输出：  
`rvec`为标定板坐标系到相机坐标系的旋转向量，可用`cv::Rodrigues()`变为旋转矩阵  
`tvec`为标定板坐标系到相机坐标系的平移向量

即得：  

**注：**   
输入时，可以使用`vector of vector`，即`vector<vector<>>`，输出是`vector`，意味着可以一次性输入所有位姿图片，然后计算得到每张图的变换矩阵，从而进行手眼标定

注意事项
----

1.  直接利用`cv::calibrateCamera`得到的，来作为标定板坐标系到相机坐标系的输入  
    利用`cv::calibrateCamera`得到的，它的内参是一个`新相机内参`，而之后进行手眼标定得到的转换矩阵也是对于`新相机内参`而言的  
    而在识别待抓取物体时，进行的PnP坐标结算，输入的相机内参是`老内参`，不是`新相机内参`，如果再使用对于`新相机内参`而言的手眼标定得到的转换矩阵，进行到机械臂基底的转换，这样就会有错误了，导致每次抓出总会有偏差  
    因此，无论是手眼标定，还是PnP结算，都要使用同一个相机内参
2.  识别标定板的角点标反了  
    由于在建立棋盘格上的三维坐标系的时候，我们默认是从棋盘格左上角到右下角建立的，如果识别反了，则会有个别图片棋盘格识别的角点和输入的棋盘格三维坐标对应不上，导致标定错误  
    要注意拍摄标定图片时的规范操作，不要让原本标定板最右下角的角点在原本标定板最左上角的角点的左边
3.  手眼标定时，标定板面积过小且只在图片中心处移动  
    这样会使得标定的结果更偏向于中心处准确，而图片边缘处会略微的不准
4.  手眼标定时，机械臂末端位姿只上下前后移动，旋转角几乎不变  
    这样时没法得到准确的旋转矩阵的，6个分量都需要移动，才能计算准确
5.  手眼标定位姿太少(图片太少)  
    图片太少计算结果的稳定性比较差，建议十张以上
6.  抓取时点云坐标系不是手眼标定的相机坐标系，常见于使用RGB-D相机  
    使用厂商提供的RGB相机到点云坐标系的变换矩阵，变换相机坐标系到实际点云坐标系用于抓取

\_\_EOF\_\_

[![](https://pic.cnblogs.com/avatar/956615/20230107154037.png)
](https://pic.cnblogs.com/avatar/956615/20230107154037.png)

*   **本文作者：**  [Champrin](https://www.cnblogs.com/champrin)
*   **本文链接：**  [https://www.cnblogs.com/champrin/p/17643041.html](https://www.cnblogs.com/champrin/p/17643041.html)
*   **关于博主：**  评论和私信会在第一时间回复。或者[直接私信](https://msg.cnblogs.com/msg/send/champrin)我。
*   **版权声明：**  本博客所有文章除特别声明外，均采用 [BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/ "BY-NC-SA") 许可协议。转载请注明出处！
*   **声援博主：**  如果您觉得文章对您有帮助，可以点击文章右下角**【推荐】** 一下。