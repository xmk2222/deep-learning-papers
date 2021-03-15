# Optimization
## Content
- [x] [SGD](#sgd)
- [x] [Momentum](#momentum)
- [x] [NAG](#nag)
- [x] [AdaGrad](#adagrad)
- [ ] [Adam](#adam)

-------------------
### SGD

**"Stochastic Gradient Descent"**
$$
w_{t+1} = w_t -\alpha\frac{\partial L}{\partial w_t}
$$
#### Questions

1. SGD的缺点

   缺点：对每一样本进行更新，收敛速度慢，同时loss函数常常会剧烈震荡，需要对输入数据进行洗牌。

2. mini-batch sgd and batch sgd

   - mini-batch sgd：每次用一小批样本更新梯度。优点是收敛过程更稳定，也更快，也可以利用矩阵运算加速。缺点是容易被困在鞍点和局部最优
   - batch sgd：对整个数据集计算梯度然后更新。优点是每次更新方向一定是最优；缺点是会被困在鞍点或局部最优，且数据集大时更新速度非常慢

--------------------
### Momentum 

**"On the importance of initialization and momentum in deep learning"**
$$
w_{t+1} = w_t - \alpha V_t\\
V_t = \beta V_{t-1} + (1 - \beta)\frac{\part L}{\part w_t}\\
\beta = 0.9
$$

#### Questions

1. Momentum的优点

   每次梯度更新时考虑之前的梯度大小和方向，模拟物理中的动量，这样可以让梯度的收敛更快且不易落入局部最优或鞍点

2. Momentum的缺点

   动量过大容易从坡的一端冲过坡底冲上坡的另一端，来回震荡

---------------------
### NAG

**"Nesterov Accelerated Gradient"**
$$
w_{t+1} = w_t - \alpha V_t\\
V_t = \beta V_{t-1} + (1-\beta)\frac{\part L}{\part w^*}\\
w^* = w_t - \alpha V_{t-1} \\
\beta = 0.9
$$

#### Questions

1. NAG的思想是什么

   为了避免动量法走的太快的问题，我们不是用当前的w的梯度进行更新，而是用预估的下一步的w的梯度进行更新。具体方法是用上一次的梯度，估计下一次更新后w的位置，然后计算该位置的梯度，用于叠加到动量上更新这一步的梯度。这样做的好处是当动量法到坡底后，可以预知下一步的梯度变化从而减速。


--------------------
### AdaGrad

**"Adam: A method for stochastic optimization"**
$$
w_{t+1} = w_t - \frac{\alpha}{\sqrt{S_t+\epsilon}} \frac{\part L}{\part w_t}\\
S_t = S_{t-1} + (\frac{\part L}{\part w_t})^2\\
\alpha = 0.001\\
\epsilon = 10^{-7}
$$

----------------
### Adam 

**"Adam: A method for stochastic optimization"**


---------------------------------------
### Reference

[1] [10 Gradient Descent Optimisation Algorithms + Cheat Sheet](https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9)

[2] [优化器详解](https://www.cnblogs.com/guoyaohua/p/8542554.html)

[3] [一文看懂各种神经网络优化算法：从梯度下降到Adam方法](https://zhuanlan.zhihu.com/p/27449596)
