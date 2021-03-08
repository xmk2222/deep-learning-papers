# Modules

##### 2014

### DropOut 

<b><details><summary>**"Dropout: a simple way to prevent neural networks from overfitting"**</summary></b>
	
</details>

##### 2015

### BatchNorm 

<b><details><summary>**"Batch normalization: Accelerating deep network training by reducing internal covariate shift"**</summary></b>
	
1. Why we use batch normalization?

The distribution of each layer's input changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rate and careful parameter initialization, and makes it notoriously hard to train with saturating nonlinearities. That is **internal covariate shift**!

2. How  does batch norm work?

batch normalization normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation.

batch normalization adds two trainable parameters to each layer, so the normalized output is multiplied by a “standard deviation” parameter (gamma) and add a “mean” parameter (beta). In other words, batch normalization lets SGD do the denormalization by changing only these two weights for each activation, instead of losing the stability of the network by changing all the weights

![algorithm1](images/BatchNorm/algorithm.png)

![algorithm2](images/BatchNorm/algorithm2.png)

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
  
- BN的缺点

  - 当batch size太小的时候，归一化没有什么意义，数据分布依然会剧烈震荡
  - 在RNN中，每个时间点的数据分布不同，需要为每个time-step进行归一化，大大增加了模型的复杂度

#### Reference

[1][**"Batch normalization: Accelerating deep network training by reducing internal covariate shift"**](https://arxiv.org/pdf/1502.03167v3.pdf)

[2][Review: Batch normalization in Neural Networks](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c)

[3][Review: Batch Normalization — What the hey?](https://gab41.lab41.org/batch-normalization-what-the-hey-d480039a9e3b)
	
</details>

##### 2016

### Layer Normalization

<b><details><summary>**"Layer Normalization"**</summary></b>

#### Reference

[1] [Layer Normalization](https://arxiv.org/abs/1607.06450)

</details>

### Weight Normalization

<b><details><summary>**"Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks"**</summary></b>

#### Reference

[1] [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868)

</details>

### Rectifier

<b><details><summary>**"Delving deep into rectifiers: Surpassing human-level performance on imagenet classification"**</summary></b>
	
</details>

### Net2Net 

<b><details><summary>**"Net2net: Accelerating learning via knowledge transfer"**</summary></b>
	
</details>