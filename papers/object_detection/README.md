# Object Detection

##### 2014

### RCNN 

<b><details><summary>**"Rich feature hierarchies for accurate object detection and semantic segmentation"**</summary></b>

![architecture](images/RCNN/architecture.jpg)

#### Questions

1. 简要介绍RCNN

   RCNN是最早使用CNN进行目标检测的方法，具体来说，RCNN首先用selective search选择候选框，resize成固定大小然后输入预训练好的CNN中，保存feature map，然后通过SVM进行分类，最后进行边界框的回归。

2. RCNN的缺点

   1. 多阶段训练，训练复杂
   2. 训练时间和空间消耗大，RCNN将每个候选框输入CNN进行运算，造成了大量的重复运算，且SVM分类效果一般，参数量大，速度慢。
   3. 预测速度慢

3. 简述NMS：

   主要目的是剔除过于接近的候选框。具体来说，对于IoU超过阈值的框，只保留可信度较高的。

4. 损失函数：

   预训练CNN时对于和ground truth的IoU大于0.5的框视为正类，训练SVM时阈值为0.3

</details>

### SPPNet 

<b><details><summary>**"Spatial pyramid pooling in deep convolutional networks for visual recognition"**</summary></b>

![architecture](images/SPPNet/architecture.jpg)

![feature](images/SPPNet/feature.jpg)

![pyramid](images/SPPNet/pyramid.jpg)

#### Questions

1. SPPNet主要思想

   SPPNet主要时为了解决CNN必须输入特定尺寸图像的问题。SPPNet采用了spatial pyramid pooling的方法，对神经网络的输出特征图进行max-pooling，将其pooling到不同尺寸，这样便可以使用不同尺寸的全连接层，检测不同尺度的特征

2. 相比RCNN的改进

   SPPNet只将输入图像通过CNN一次，在最后的输出特征图上，根据不同尺度的classifier来进行分类与检测，节约了大量的计算

3. 为什么在特征图上也可以实现检测

   因为文章发现输入图像中的目标会在特征图的对应位置上产生强响应，因此不必像RCNN一样在输入之前进行特征位置选择。

4. SPPNet的缺点

   多阶段训练，需要硬盘存储

</details>

##### 2015

### Fast R-CNN 

<b><details><summary>**"Fast R-CNN"**</summary></b>

​	![Architecture](images\FastRCNN\Architecture.JPG)

A Fast R-CNN network takes as input an entire image and a set of object proposals. The network first processes the whole image with several convolutional (*conv*) and max pooling layers to produce a conv feature map. Then, for each object proposal a region of interest (*RoI*) pooling layer extracts a fixed-length feature vector from the feature map. Each feature vector is fed into a sequence of fully connected (*fc*) layers that finally branch into two sibling output layers: one that produces softmax probability estimates over K* object classes plus a catch-all “background” class and another layer that outputs four real-valued numbers for each of the *K* object classes. Each set of 4 values encodes refined bounding-box positions for one of the *K* classes.

#### Questions

1. Fast-RCNN的改进

   - RCNN每次提取RoI都要通过一次卷积，FastRCNN借鉴SPPNet的结论，每幅图像只进行一次卷积，而在特征图上通过映射区域选取RoI，大大减少了卷积的运算量。

   - 通过multi-task loss实现single-stage训练，训练可直接更新所有网络层。


2. loss函数：
   $$
L(p,u,t^u,v) = L_{cls}(p,u) + \lambda[u \ge 1]L_{loc}(t^u,v)
   $$
   其中p为预测值，k个类别有k+1个值，u为对应的真值，$L_{cls}$为log loss。v代表真实的边界框的四个坐标值，t为预测值，对每个类别输出对应的坐标偏移值，$L_{loc}$为平滑L1.
   
3. 如何实现多尺度的识别：

   
   1. 暴力搜索：所有图像在输入和输出时处理成固定像素
   2. 图像金字塔：训练时随机采样金字塔尺度

</details>

### Faster R-CNN 

<b><details><summary>**"Faster R-CNN: Towards real-time object detection with region proposal networks"**</summary></b>

![multi-scale](images/FasterRCNN/multi-scale.jpg)

![architecture](images/FasterRCNN/architecture.jpg)

![RPN](images/FasterRCNN/RPN.jpg)

#### Questions

1. 最重要的改进

   之前框架最大的局限在于候选区域的选定，本文提出RPN来自动生成region proposals，只在FastRCNN的基础上增加了几层全连接网络。

2. RPN的原理与loss

   RPN建立在FastRCNN的卷积层的基础上，输出为候选框以及候选框是否包含物体。通过在VGG或ZF网络输出的最后一层特征图上，进行3x3卷积，在每一个滑动窗的中心设置k（k=9）个anchor boxes，对应回原图九种不同尺寸不同长宽比的框，当卷积滑动就可以覆盖原图几乎所有可能的候选窗；卷积之后有两个相邻的1x1卷积，分别为分类层（输出2k个值）和回归层（输出4k个坐标偏移值）。损失函数如下：
   $$
   L(\{p_i\},\{t_i\})=\frac{1}{N_{cls}}\sum_i L_{cls}(p_i, p_i^*)+\lambda\frac{1}{N_{reg}}\sum_i p_i^* L_{reg}(t_i, t_i^*)
   $$
   其中，分类损失为log loss，回归损失为smooth-L1.

   用anchor的好处是可以实现平移不变性。

   在训练时，正样本为IOU最大的anchor以及IOU大于0.7的anchor，负样本为IOU小于0.3的anchor，其他忽略。

3. 如何训练

   通过4步训练来实现RPN与RCNN的参数共享：

   1. 用ImageNet的预训练网络训练RPN
   2. 用上一步RPN得到的候选框训练FastRCNN，RCNN同样适用ImageNet预训练网络，此时二者不共享参数
   3. 用上一步得到的RCNN的网络作为RPN的基网络，固定该网络，只训练RPN的顶上几层，此时二者已共享基网络的参数
   4. 最后一步，训练RCNN的顶上几层

#### Reference

[1] [pytorch implementation](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)

</details>

### YOLO 

<b><details><summary>**"You only look once: Unified, real-time object detection"**</summary></b>

![model](images/YOLO/model.jpg)

#### Questions

1. YOLO与之前目标识别框架的不同

   YOLO是单阶段检测模型，将分类与检测任务放在一起，训练简单，速度更快。此外YOLO在预测时分类器可以看到整幅图像的信息，因此YOLO对北京的分类错误比FastRCNN少一半。此外，YOLO的泛化性能更强。

2. YOLO 的原理

   YOLO在整张图的输入上，预测所有物体框与所有类别，从而实现单阶段的快速检测。具体来说，YOLO将输入图像分为S x S 个网格，如果物体中心落入某个网格中，该网格用来检测那个物体。每个网格会预测出B个bounding boxes和对应的confidence值。confidence代表该box内有物体的概率，即$ Pr(Object) * IOU$，若无物体则为0，这样confidence为一个介于0与IOU之间的值。每个bounding box包括五个预测值x，y，w，h和confidence，前四个分别为预测框的中心与宽高，而confidence则代表IOU。

   每个网格还预测C个条件概率$Pr(Class_i | Object)$，即在网格中有物体的情况下物体为类别i的概率。每个网格预测C个，与框的数量B无关。在测试时我们将二者相乘
   $$
   Pr(Class_i | Object) * Pr(Object) * IOU = Pr(Class_i) * IOU
   $$
   这样就得到了每一类的confidence。

3. Loss设计

   $$
   \lambda_{coord}\sum^{S^2}_{i=0}\sum^B_{j=0}1^{obj}_{ij}[(x_i-\hat{x_i})^2 + (y_i-\hat{y_i})^2]  + \lambda_{coord}\sum^{S^2}_{i=0}\sum^B_{j=0}1^{obj}_{ij}[(\sqrt{w_i}-\sqrt{\hat{w_i}})^2 + (\sqrt{h_i}-\sqrt{\hat{h_i}})^2] + \sum^{S^2}_{i=0}\sum^B_{j=0}1_{ij}^{obj}(C_i-\hat{C_i})^2+\lambda_{noobj}\sum^{S^2}_{i=0}\sum^B_{j=0}1_{ij}^{noobj}(C_i-\hat{C_i})^2 + \sum_{i=0}^{S^2}1_i^{obj}\sum_{c\in classes}(p_i(c)-\hat{p_i}(c))^2
   $$
   前两项代表每一个网格以及其预测出的每一个有物体的框的位置的回归loss，其中$1_{ij}^{obj}$代表第i个网格的第j个框是否有物体。注意这里对每个网格，只会选出IOU最大的那个框，也就是B个框中只有一个框会对loss产生影响。

   接下来的两项代表每一个有物体的框的预测confidence即IOU的损失以及没有物体的框的confidence损失。这里注意到有很多很多框是没有物体的，因此第四项loss容易更大，因此将这一项的权重$\lambda{noobj}$设为0.5，而将回归损失的权重设为5.

   最后一项为每个有物体的网格对应的物体的种类的预测损失。

   应该注意到这个Loss函数的设计，只有有物体的网格会对loss造成影响，且每个网格中只有一个框（IOU最大的框）会对loss造成影响。

4. 缺点

   - 对于小物体或成群出现的物体会有识别困难。
   - 对于不同尺度的物体识别能力欠佳。
   - 小框的小错误比大框的小错误在IOU上产生的影响更大，这是设计中不合理的一个点。

#### Reference

**[1] [You Only Look Once: Unified, Real-Time Object Detection](http://arxiv.org/pdf/1506.02640)**



</details>

### SSD 

<b><details><summary>**"SSD: Single Shot MultiBox Detector"**</summary></b>
	
</details>

##### 2016

### R-FCN 

<b><details><summary>**"R-FCN: Object Detection via Region-based Fully Convolutional Networks"**</summary></b>
	
</details>

### YOLOv2 

<b><details><summary>**"YOLO9000: Better, Faster, Stronger"**</summary></b>
	
</details>

### FPN 

<b><details><summary>**"Feature Pyramid Networks for Object Detection"**</summary></b>
	
</details>

##### 2017

### Mask R-CNN

<b><details><summary>**"Mask R-CNN"**</summary></b>
	
</details>

### RetinaNet 

<b><details><summary>**"Focal Loss for Dense Object Detection"**</summary></b>
	
</details>

### DCN 

<b><details><summary>**"Deformable Convolutional Networks"**</summary></b>
	
</details>

##### 2018

### YOLOv3 

<b><details><summary>**"YOLOv3: An Incremental Improvement"**</summary></b>
	
</details>

## Image Segmentation

FCN

DeconvNet

U-Net

SegNet

DeepLab

PSPNet

Mask-RCNN