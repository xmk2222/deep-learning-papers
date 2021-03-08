# Deep Learning Papers

# Content

[1. Papers](#1papers)

[2. Concept Explanation](#2concept-explanation)

# 1.Papers

## Content

[Image Classification](papers/image_classification/README.md)

[Models](papers/modules/README.md)

[Compact Network Design](papers/compact_network/README.md)

[Neural Architecture Search](papers/neural_architecture_search/README.md)

[Efficient Computation](papers/efficient_computation/README.md)

[Optimization](papers/optimization/README.md)

[Object Detection](papers/object_detection/README.md)

[Deep Generative Model](papers/generative_model/README.md)

[Deep Reinforcement Learning](papers/reinforcement_learning/README.md)


# 2.Concept Explanation

## Contents
[Loss Function](#loss-function)

[Optimizers](#optimizers)

[Gradient Explode and Vanish](#Gradient-Explode-and-Vanish)

[k-means](#k-means)

[PCA](#pca)

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

## k-means

#### Questions

- 简要介绍一下k-means

  k-means是一种无监督的聚类方法，通过不断迭代将数据聚为k类。具体操作方法为先选择k个数据作为初始中心，之后计算每个数据与每个类中心的距离，将数据归为距离最近的类。之后更新每一类的中心为类内所有点的均值，继续迭代计算距离，重新分类，重新计算均值。不断迭代直到分类不再发生改变。
  
- k-means初始值的选取对最终结果有影响吗？
  
  有
  
- k-means计算距离应该计算什么距离

  应该根据具体情况选择距离度量，不同的距离度量会产生不同的结果。可以选取的常用度量有欧式距离、曼哈顿距离、马氏距离等
  

[**back to contents**](#content)

## PCA

#### Questions

- 简要介绍一下PCA

  PCA即主成成分分析，作用是通过线性映射将数据从高维空间映射到低位空间，并尽量满足信息量不丢失，是数据降维到主要方法之一。PCA寻找数据方差最大的方向，作为第一主成分，之后寻找与之前找到的主成分方向正交的方向中，方差最大的方向，最为次一级主成分，依次将数据映射到最能反映数据信息量的方向上，达到降维的效果。
  
  PCA的具体操作流程为，首先对数据去均值，之后求数据的协方差矩阵，协方差矩阵的对角线代表每一特征的方差，非对角线上为协方差。优化目标为将此协方差矩阵对角化，寻找变换矩阵。具体方法为求协方差矩阵的特征值与特征向量，其中特征向量对应投影方向，特征值对应原始特征投影到该方向之后到方差，这样既可以找到相互正交的向量，也可以找到特征值即方差较大的方向。
  
- PCA的优缺点

  优点：降维的同时，可以消除数据维度之间的相关性
  
  缺点：失去数据原有含义

- 为什么要进行去均值操作

  去均值之后计算数据的协方差变得非常简单，只要计算$x \cdot x^T$即可。
  

[**back to contents**](#content)

