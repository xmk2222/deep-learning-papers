# What should we do to find an ideal job as a deep learning engineer

# Content

[1. Books](#1books)

[2. Papers](#2papers)

[3. Concept Explanation](#3concept-explanation)

# 1.Books
- [ ] 李航 **"统计学习方法"**
- [ ] 周志华 **"机器学习"**
- [ ] 何海涛 **"剑指offer"**
- [ ] 诸葛越 **"百面机器学习"**
- [ ] scutan90 **"DeepLearning-500-questions"** https://github.com/scutan90/DeepLearning-500-questions
- [ ] Stephen Prata **"C++ Primer Plus"**
- [ ] **"数据结构"**
- [ ] Ian Goodfellow [**"深度学习"**](https://exacity.github.io/deeplearningbook-chinese/)

# 2.Papers

## Classical Network Structures

##### 2012
- [x] AlexNet [**"Imagenet classification with deep convolutional neural networks"**](#AlexNet)
##### 2014
- [x] VGGNet [**"Very deep convolutional networks for large-scale image recognition"**](#VGGNet)
- [x] Network in Network [**"Network In Network"**](#NIN)
##### 2015
- [x] GoogLeNet [**"Going deeper with convolutions"**](#GoogLeNet)
- [ ] ResNet **"Deep residual learning for image recognition"**
- [ ] Inception-v3 [**"Rethinking the Inception Architecture for Computer Vision"**](#InceptionV3)
##### 2016
- [ ] Inception-v4 [**"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"**](#Inception-v4)
- [ ] Attention **"Show, Attend and Tell Neural Image Caption Generation with Visual Attention"**
- [ ] SqueezeNet **"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"**
##### 2017
- [ ] Xception **"Xception: Deep Learning with Depthwise Separable Convolutions"**
- [x] MobileNet [**"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"**](#MobileNet)
- [ ] ResNeXt **"Aggregated Residual Transformations for Deep Neural Networks"**
- [ ] ShuffleNet **"ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices"**
##### 2018
- [ ] DenseNet **"Densely Connected Convolutional Networks"**
- [ ] MobileNetV2 [**"MobileNetV2: Inverted Residuals and Linear Bottlenecks"**](#MobileNetV2)
- [ ] NASNet **"Learning Transferable Architectures for Scalable Image Recognition"**
- [ ] ShuffleNetV2 **"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"**

##### 2019

- [ ] MobileNetV3 **"Searching for MobileNetV3"**
- [ ] MnasNet **"MnasNet: Platform-Aware Neural Architecture Search for Mobile"**

## Models

##### 2014
- [ ] DropOut **"Dropout: a simple way to prevent neural networks from overfitting"**
##### 2015
- [x] BatchNorm [**"Batch normalization: Accelerating deep network training by reducing internal covariate shift"**](#BatchNorm)
- [ ] Net2Net **"Net2net: Accelerating learning via knowledge transfer"**

## Efficient Computation
##### 2015
- [ ] Deep Compression **"Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding"**
##### 2017
- [ ] Survey **"A Survey of Model Compression and Acceleration for Deep Neural Networks"**
##### 2018

- [ ] Survey **"Recent Advances in Efficient Computation of Deep Convolutional Neural Networks"**

## Optimization

##### 2013
- [ ] Momentum **"On the importance of initialization and momentum in deep learning"**
##### 2014
- [ ] Adam **"Adam: A method for stochastic optimization"**
##### 2016
- [ ] Neural Optimizer **"Learning to learn by gradient descent by gradient descent"**

## Object Detection

##### 2014
- [ ] RCNN **"Rich feature hierarchies for accurate object detection and semantic segmentation"**
- [ ] SPPNet **"Spatial pyramid pooling in deep convolutional networks for visual recognition"**
##### 2015
- [ ] Faster R-CNN **"Faster R-CNN: Towards real-time object detection with region proposal networks"**
- [ ] YOLO **"You only look once: Unified, real-time object detection"**
##### 2016
- [ ] R-FCN **"R-FCN: Object Detection via Region-based Fully Convolutional Networks"**
##### 2017
- [ ] **"Mask R-CNN"**

## Deep Generative Model

##### 2013
- [ ] VAE **"Auto-encoding variational bayes"**
##### 2014
- [ ] GAN **"Generative adversarial nets"**
##### 2015
- [ ] VAE with attention **"DRAW: A recurrent neural network for image generation"**


## Deep Reinforcement Learning

##### 2013

- [ ] **"Playing atari with deep reinforcement learning"**

##### 2015

- [ ] **"Human-level control through deep reinforcement learning"**
- [ ] **"Continuous control with deep reinforcement learning"**
##### 2016
- [ ] **"Asynchronous methods for deep reinforcement learning"**
- [ ] AlphaGo **"Mastering the game of Go with deep neural networks and tree search"**
- [ ] **"Deep reinforcement learning with double q-learning"**

##### 2017

- [ ] **"Deep reinforcement learning: An overview"**
- [ ] **"Target-driven visual navigation in indoor scenes using deep reinforcement learning"**
- [ ] **"Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates"**
- [ ] **"Playing FPS games with deep reinforcement learning"**

##### 2018

- [ ] **"Rainbow: Combining improvements in deep reinforcement learning"**
- [ ] **"Deep reinforcement learning that matters"**

---------------------------
# Papers Summaries

## AlexNet 

#### Reference

[1][**"Imagenet classification with deep convolutional neural networks"**](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

[**back to top**](#content)


## VGGNet 

1. The use of stack 3×3 filters is effient than of 5×5 or 7×7 filters

2. A deep net with small filters outperforms a shallow net with larger filters

3. Combining the outputs of several models by averaging their soft-max class posteriors improves the performance due to complementarity of the models

#### Reference

[1][**"Very deep convolutional networks for large-scale image recognition"**](https://arxiv.org/pdf/1409.1556.pdf)

[2][Review: VGGNet — 1st Runner-Up (Image Classification), Winner (Localization) in ILSVRC 2014](https://medium.com/coinmonks/paper-review-of-vggnet-1st-runner-up-of-ilsvlc-2014-image-classification-d02355543a11)

[3][Keras implement vgg-16](https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py)

[4][Keras implement vgg-19](https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py)

[back to top](#content)


## NIN

1. Proposed a new network structure--mlpconv layer

  ![mlpconv](./images/NIN/mlpconv.png)

2. Usually, fully connected layers are used at the end of network, however, they are prone to overfitting. This article used global average pooling layer as the last layer of the network, it is more native to the convolution structure by enforcing correspondences between feature maps and categories, and could prevent over-fitting.

#### Reference

[1][**"Network In Network"**](https://arxiv.org/abs/1312.4400)

[2][Review: NIN — Network In Network (Image Classification)](https://towardsdatascience.com/review-nin-network-in-network-image-classification-69e271e499ee)

[**back to top**](#content)


## GoogLeNet 

![inception_naive](./images/GoogleNet/inception_module_naive.png)

![inception_module](./images/GoogleNet/Inception_module.png)

1. 1×1 convolution is used as a dimension reduction module to reduce the computation. By reducing the computation bottleneck, depth and width can be increased
2. When image’s coming in, different sizes of convolutions as well as max pooling are tried. Then different kinds of features are extracted.
3. Global average pooling is used nearly at the end of network by averaging each feature map from 7×7 to 1×1, and authors found that a move from FC layers to average pooling improved the top-1 accuracy by about 0.6%.
4. Auxiliary classifiers for combating gradient vanishing problem, also providing regularization.
5. besides the network design, the other stuffs like ensemble methods, multi-scale and multi-crop approaches are also essential to reduce the error rate

![googlenet](./images/GoogleNet/googlenet.png)

#### Questions

- inception 结构的优点

  文章认为，通过1x1, 3x3, 5x5并列最后拼接的结构，可以让模型同时感受到多尺度的特征。

- 1x1卷积的作用

  作用是对feature的通道数进行降维，可以大幅度减少模型的运算量，实际上这种分解方式可以通过low-rank来解释，用两个卷积层来等效一个卷积，但大大降低了运算量

- 预测为什么使用global average pooling

  相比全连接层，先通过global average pooling再连接Dense层或直接激活可以大大降低运算量，因为一个模型的最后一层FC往往参数量十分巨大，而且论文证明使用global average pooling的效果也略微更好一点

- 辅助分类器的作用

  对抗梯度消失问题，同时有一定的正则化作用

#### Reference

[1][**"Going deeper with convolutions"**](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)

[2][Pytorch implement](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py)

[3][Review: GoogLeNet (Inception v1)— Winner of ILSVRC 2014 (Image Classification)](https://medium.com/coinmonks/paper-review-of-googlenet-inception-v1-winner-of-ilsvlc-2014-image-classification-c2b3565a64e7)

[**back to top**](#content)

## InceptionV3

1. Factorizing Convolutions with Large Filter Size

In theory, we can **replace any n x n convolution by a 1 x n convolution followed by a n x 1 convolution** and the computational cost saving increases dramatically as n grows.

In practice, **it is found that employing this factorization does not work well on early layers, but it gives very good results on medium grid-size.**

![moduleA](./images/InceptionV2/moduleA.png)

![moduleB](./images/InceptionV2/moduleB.png)

![moduleC](./images/InceptionV2/moduleC.png)

2. Utility of Auxiliary Classifiers

The auxiliary classifiers act as **relularizer**.

![auxiliary](./images/InceptionV2/auxiliary.png)

3. Efficient Grid Size Reduction

**Conventionally**, such as AlexNet and VGGNet, the feature map downsizing is done by max pooling. But the drawback is either **too greedy by max pooling followed by conv layer**, or **too expensive by conv layer followed by max pooling**. Here, an efficient grid size reduction is proposed as follows:

![grid](./images/InceptionV2/grid.png)

With the efficient grid size reduction, **320 feature maps** are done by **conv with stride 2**. **320 feature maps** are obtained by **max pooling**. And these 2 sets of feature maps are **concatenated as 640 feature maps** and go to the next level of inception module.

**Less expensive and still efficient network** is achieved by this efficient grid size reduction.

4. Overall Architecture

![architecture](./images/InceptionV2/architecture.png)

5. General Design Principles
   1. **Avoid representational bottlenecks, especially early in the network.** One should avoid bottlenecks with extreme compression. In general, the representation size should gently decrease. Theoretically, information content can not be assessed merely by the dimensionality of the representation as it discards important factors like correlation structure, the dimensional merely provides a rough estimate of information content.
   2. **Higher dimensional representations are easier to process locally within a network.** Increasing the activation per tile in a network allows for more disentangled features. The resulting networks will train faster.
   3. **Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power.** The strong correlation between adjacent units results in much less loss of information during dimension reduction.
   4. **Balance the width and depth of the network.** Increasing both the width and depth of the network can contribute to higher quality network.

#### Questions

- 相比googleNet有哪些改进

  - 用两个相连的3x3卷积代替5x5卷积，降低了运算复杂度。
  - 通过low-rank分解，将nxn的卷积分解为两个1xn和nx1的卷积，降低了模型的运算复杂度。
  - 通过stride=2的卷积和avg_pool的filter concat实现feature降维，是一种折中的解决方案。
  - 证明了辅助分类器的正则化作用

- 有什么缺点

  过于宽且深度不同的Inception block会大大降低模型的训练以及预测速度

#### Reference 

[1] [Rethinking the Inception Architecture for Computer Vision](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)

[2] [Keras implement](https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py)

[3] [Pytorch implement](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py)

[4] [Review: Inception-v3 — 1st Runner Up (Image Classification) in ILSVRC 2015](https://medium.com/@sh.tsang/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c)

[**back to top**](#content)

## InceptionV4

1. It is studied that whether the Inception itself can be made more efficient by making it deeper and wider.

   In order to optimize the training speed, **the layer sizes was tuned carefully to balance the computation between the various model sub-networks**.

2. Since the residual connections are of inherent importance for  training very deep architectures. **it is natural to replace the filter concatenation stage of the Inception architecture with residual connections.**

   **Inception-Resnet-v1 was training much faster, but reached slightly worse final accuracy than Inception-v3.** 

   If the number of filters exceeded 1000, the residual variants started to exhibit instabilities, and the network just “died” early during training. This could be prevented, neither by lowering the learning rate, nor by adding an extra BN to this layer. However, scaling down the residuals before adding them to the previous layer activation seemed to stabilize the training.

   Two-phase training is also suggested, where the first "warm-up" phase is done with very low learning rate, followed by a second phase with high learning rate.

#### Questions

- 有什么改进
  - 微调inception结构来弥补其在运算速度上的劣势
  - 改用residual结构来提高模型训练的速度

#### Reference

[1] [[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14806/14311)

[2] [Keras Implementation](https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_resnet_v2.py)

[**back to top**](#content)

## MobileNet

1. Depthwise separable convolution

![DepthWiseConv](./images/MobileNet/DepthWiseConv.png)

(This figure is a little confusing that the last 1x1 conv actually is a ordinary 1x1 conv layer that has depth of M, and there are N such filters)

Standard convolution has the computatianal cost of:
$$
D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F
$$

where $D_K$ is the size of the kernel, $D_F$is the size of the input feature map, $M$ and $N$ is the number of input and out put channels.

Depthwise saparable convolutions cost:

$$
D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F
$$

2. Width Multiplier α is introduced to **control the input width of a layer**, for a given layer and width multiplier α, tαhe number of input channels M becomes αM and the number of output channels N bocomes αN

3. Resolution Multiplier ρ is introduced to **control the input image resolution**of the network

4. Overall architecture

![architecture](./images/MobileNet/architecture.png)

#### Questions

- 请简要介绍MobileNet

  MobileNet是一种轻量级神经网络模型，它通过Depthwise separable convolution的方式大大降低了运算量。DW卷积的原理是对输入的每一通道分离，分别用一个filter进行卷积操作，得到一个feature map，之后所有的feature map再通过一个1x1PW卷积进行通道变换，解决DW卷积导致的通道之间信息交流不畅的问题。

- 与传统卷积的运算量对比

  假定输入为$D_k \times D_k \times M$的tensor，输出通道数为N，传统卷积的运算量为$D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$, DW卷积的运算量为$D_K \cdot D_K \cdot M  \cdot D_F \cdot D_F$, PW卷积的运算量为$M \cdot N \cdot D_F \cdot D_F$.

#### Reference

[1][**"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"**](https://arxiv.org/abs/1704.04861)

[2][Keras implementation](https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py)

[3][Review: MobileNetV1 — Depthwise Separable Convolution (Light Weight Model)](https://towardsdatascience.com/review-mobilenetv1-depthwise-separable-convolution-light-weight-model-a382df364b69)

[**back to top**](#content)


## MobileNetV2

1. Linear Bottlenecks

Deep networks only have the power of a linear classifier on the non-zero volume part of the output domain, on the other hand, when ReLU collapses the channel, it inevitably loses information in that channel.

	- if the manifold of interest remains non-zero volume after ReLU transformation, it corresponds to a linear transformation.
	- ReLU is capable of preserving complete information about the input manifold, but only if the input manifold lies in a low-dimensional subspace of the input space.

Assuming the manifold of interest is low-dimensional we can capture this by inserting linear bottleneck layers into the convolutional blocks. Experimental evidence suggests that **using linear layers is crucial as it prevents non-linearities from destroying too much information**.

2. Inverted residuals

![Inverted](./images/MobileNetV2/inverted.png)

$h \cdot w \cdot k \cdot t (k + d^2 + k')$

3. Convolutional Blocks

![conv block](./images/MobileNetV2/ConvBlocks.png)

The first 1x1 Conv in MobileNetV2 is used for expanding input depth (by 6 default).

4. Overall Architecture

![architecture](./images/MobileNetV2/architecture.png)

#### Questions

- V2相比V1有哪些改进

  1. 采用inverted residual结构，在block中第一个1x1conv和DW conv之间进行通道扩大以提取更多特征，并在stride=1的block最后采用与ResNet类似的相加结构。
  2. 为避免ReLU对特征的破坏，在residual block的相加之前的1x1conv采用线性激活

- 为什么使用线性激活

  文章认为网络的激活函数会产生一份信息的副本，且此副本存在某种低维子空间表示，因此可以对通道进行降维，但是当维度较低时非线性激活会产生较大的信息损失。因此文章先将通道数扩大，再进行激活，而最后通道缩小时则采用线性激活以避免信息损失。

#### Reference

[1][**"MobileNetV2: Inverted Residuals and Linear Bottlenecks"**](https://arxiv.org/abs/1801.04381)

[2][Keras implementation](https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py)

[3][Review: MobileNetV2 — Light Weight Model (Image Classification)](https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c)

[**back to top**](#content)


## BatchNorm

1. Why we use batch normalization?

The distribution of each layer's input changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rate and careful parameter initialization, and makes it notoriously hard to train with saturating nonlinearities. That is **internal covariate shift**!

2. How  does batch norm work?

batch normalization normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation.

batch normalization adds two trainable parameters to each layer, so the normalized output is multiplied by a “standard deviation” parameter (gamma) and add a “mean” parameter (beta). In other words, batch normalization lets SGD do the denormalization by changing only these two weights for each activation, instead of losing the stability of the network by changing all the weights

![algorithm1](./images/BatchNorm/algorithm.png)

![algorithm2](./images/BatchNorm/algorithm2.png)

> There is a subtle difference between training and inferencing, During training, it is normalized by 

> $\sigma_B^2 \gets \frac{1}{m}\sum_i^m(x_i-\mu_B)^2$

> Then when testing, it is normalized by **unbiased variance estimate**:

> $Var[x] \gets \frac{m}{m-1}E_B[\sigma_B^2]$

3. Advantages 
   1. Batch normalization reduces the amount by what hidden unit values shift around (covariance shift )
   2. Batch normalization has a beneficial effect on the gradient flow through the network, by reducing the dependence of gradients on the scale of the parameters or for their initial values. This allow us to use much higher learning rates without the risk of divergence.
   3. Batch normalization regularizes the model and reduces the need for dropout.
   4. Batch normalized makes it possible to use saturating nonlinearities by preventing the network form getting stuck in the saturated modes.

#### Questions

- BN解决什么问题

  模型训练时，层与层之间存在高度耦合性，当前一层的参数更新后，当前层的输入的分布便发生改变，这增加了训练的难度，需要更小的学习率和初始化策略。对于类似sigmoid之类存在饱和区的激活函数来说这个问题更加严重。

  BN的作用就是在模型的层与层之间进行归一化，具体的操作是对每一mini-batch，在每一特征维度上进行归一化， 使其满足均值为0，方差为1的分布。

  然而这样做会导致数据失去部分其原本的表达能力，落入sigmoid或tanh的线性激活区。因此增加了两个参数$\gamma, \beta$进行线性变换$Z_j = \gamma_j\hat{Z_j}+\beta_j$, 当$\gamma^2=\sigma^2, \beta=\mu$时是一个原始分布的等价变换。同时对于sgd来说BN层的训练参数限定为$\gamma$和$\beta$，增加了模型的稳定性。

- 预测阶段BN怎么产生作用

  保留训练时每个batch的均值与方差，在预测时计算整个数据均值与方差的**无偏估计**，作为预测数据的归一化参数。

- BN是无偏估计吗

  是。

- BN的作用

  - 使网络每层的输入分布相对稳定，加速训练
  - 降低了梯度传播对参数的的敏感性，可以使用更大的学习率
  - 因为相当于为每一层的数据增加了噪声，具有一定的正则化作用，提高了模型的泛化性能。使模型可以不使用dropout。（有论文证明同时使用dropout和BN效果不如只使用他们中的一个）
  - 使深度模型可以使用sigmoid或tanh等具有饱和区的激活函数

- BN中batch size的影响

  当batch size过小时不建议使用BN，均值和方差具有较大的随机性。

- BN的参数数量

  BN层计算参数时将整个输入的feature map当做一个特征进行处理，因此参数数量只与输入的通道数相关（4C）

#### Reference

[1][**"Batch normalization: Accelerating deep network training by reducing internal covariate shift"**](https://arxiv.org/pdf/1502.03167v3.pdf)

[2][Review: Batch normalization in Neural Networks](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c)

[3][Review: Batch Normalization — What the hey?](https://gab41.lab41.org/batch-normalization-what-the-hey-d480039a9e3b)

[**back to top**](#content)

# 3.Concept Explanation

## Content
[Loss Function](#loss-function)

[Optimizers](#optimizers)

[Gradient Explode and Vanish](#Gradient-Explode-and-Vanish)

## Loss Function

#### Mean Squared Error

$$
MSE = \frac{1}{n} \sum_{i=1}^n (y_i-\hat{y}_i)^2
$$

#### Mean Absolute Error

$$
MSE = \frac{1}{n} \sum_{i=1}^n |y_i-\hat{y}_i|
$$

*L1 loss is more robust to outliers, but its derivatives are not continuous, making it inefficient to find the solution. L2 loss is sensitive to outliers, but gives a more stable and closed form solution (by setting its derivative to 0.)*

#### Mean Absolute Percentage Error 

#### Mean Squared Logarithmic Error

#### Squared Hinge 

#### Hinge

#### Huber

$$
L_\delta = \left\{\begin{array}{lr}
	\frac{1}{2}(y-\hat{y})^2\quad &if|(y-\hat{y}| < \delta)\\
	\delta(y-\hat{y}) - \frac{1}{2}\delta\quad&otherwise
	\end{array}
\right.
$$

*Typically used for regression. It’s less sensitive to outliers than the MSE as it treats error as square only inside an interval.*

#### Categorical Hinge 

#### Logcosh

#### Categorical Crossentropy 

In binary classification
$$
-(ylog(p) + (1-y)log(1-p))
$$
if M > 2(i.e.  multi-class classification)
$$
-\sum_{c=1}^M y_clog(p_c)
$$


#### Sparse Categorical Crossentropy

#### Binary Crossentropy 

#### Kullback-Leibler Divergence

#### Poisson

#### Cosine Proximity

[**back to contents**](#contents)

### Optimizers

#### Adadelta

#### Adagrad

#### Adam

#### Conjugate Gradients

#### BFGS

#### Momentum

#### Nesterov Momentum

#### Newton's Method

#### RMSProp

#### SGD



## Gradient Explode and Vanish

1. What is gradient explosion or vanishing

These problems arise during training of a deep network when the gradients are being propagated back in time all the way to the initial layer. The gradients coming from the deeper layers have to go through continuous matrix multiplications because of the the chain rule, and as they approach the earlier layers, if they have small values (<1), they shrink exponentially until they vanish and make it impossible for the model to learn , this is the vanishing gradient problem. While on the other hand if they have large values (>1) they get larger and eventually blow up and crash the model, this is the exploding gradient problem.

The difficulty that arises is that when the parameter gradient is very large, a gradient descent parameter update could throw the parameters very far, into a region where the objective function is larger, undoing much of the work that had been done to reach the current solution. And when the parameter tradient is very small, the back propagation won't work at all.

2. Why does gradient explosion happen

- Poor choice of learning rate that results in large weight updates.
- Poor choice of data preparation, allowing large differences in the target variable.
- Poor choice of loss function, allowing the calculation of large error values.

3. Why does gradient vanishing happen

Certain activation functions, like the sigmoid function, squishes a large input space into a small input space between 0 and 1. Therefore, a large change in the input of the sigmoid function will cause a small change in the output. Hence, the derivative becomes small.

When n hidden layers use an activation like the sigmoid function, n small derivatives are multiplied together. Thus, the gradient decreases exponentially as we propagate down to the initial layers.

4. Dealing with exploding gradient

Exploding gradients can be avoided in general by careful configuration of the network model, such as choice of small learning rate, scaled target variables, and a standard loss function. Nevertheless, exploding gradients may still be an issue with recurrent networks with a large number of input time steps.

There is a good method to prevent gradient explosion--gradient clipping, which place a predefined threshold on the gradients to prevent it from getting too large.

Keras supports gradient clipping on each optimization algorithm, with the same scheme applied to all layers in the model.

Gradient norm scaling involves changing the derivatives of the loss function to have a given vector norm when the L2 vector norm (sum of the squared values) of the gradient vector exceeds a threshold value.

```
# configure sgd with gradient norm clipping
opt = SGD(lr=0.01, momentum=0.9, clipnorm=1.0)
```

Gradient value clipping involves clipping the derivatives of the loss function to have a given value if a gradient value is less than a negative threshold or more than the positive threshold.

```
# configure sgd with gradient value clipping
opt = SGD(lr=0.01, momentum=0.9, clipvalue=0.5)
```

5. Dealing with Vanishing Gradients

The simplest solution is to use other activation functions, such as ReLU, which doesn’t cause a small derivative.

Residual networks are another solution, as they provide residual connections straight to earlier layers

Finally, batch normalization layers can also resolve the issue. As stated before, the problem arises when a large input space is mapped to a small one, causing the derivatives to disappear. Batch normalization reduces this problem by simply normalizing the input so |x| doesn’t reach the outer edges of the sigmoid function. 

#### Reference:

[1] [How to Avoid Exploding Gradients With Gradient Clipping](https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/)

[2] [The curious case of the vanishing & exploding gradient](https://medium.com/learn-love-ai/the-curious-case-of-the-vanishing-exploding-gradient-bf58ec6822eb)

[3] [The Vanishing Gradient Problem](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)

[**back to contents**](#content)



