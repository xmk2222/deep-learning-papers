# Neural Architecture Search
## Content
### 2016
- [ ] [Nas](#nas)

### 2018
- [x] [NASNet](#nasnet)

### 2019
- [ ] [MNASNet](#mnasnet)

-----------------
### Nas 

**Neural Architecture Search with Reinforcement Learning**
	

-------------------------
### NASNet 
**"Learning Transferable Architectures for Scalable Image Recognition"**
	
1. **Search for an architectural building block on a small dataset and the transfer the block to a larger dataset.**

   Design a search space so that the complexity of the architecture is independent of the depth of the network and the size of the input images. Searching for the best network architecture is therefore reduced to searching for the best cell structure.

2. **Predict a generic convolutional cell expressed in terms of "motifs"**

   By doing this, it is enough to only predict a **Normal Cell** and a **Reduction Cell**, and simply stack them in series.

   ![cell](../../images/NasNet/cell.png)

3. RNN Controller

   ![rnn](../../images/NasNet/rnn.png)

   ![steps](../../images/NasNet/steps.png)

   - The algorithm appends the newly-created hidden state to the set of existing hidden states as a potential input in subsequent blocks. The controller RNN repeats the above 5 prediction steps B times corresponding to the B blocks in a convolutional cell.(B=5)

   - In steps 3 and 4, the controller RNN selects an operation to apply to the hidden states:

   ![operations](../../images/NasNet/operations.png)

   - In step 5 the controller RNN selects a method to combine the two hidden states, either (1) element-wise addition between two hidden states or (2) concatenation between two hidden states along the ﬁlter dimension
- Specifically, the controller RNN is a one-layer LSTM with 100 hidden units at each layer and 2×5*B* softmax predictions for the two convolutional cells (where *B* is typically 5) associated with each architecture decision.
   - Each of the 10*B* predictions of the controller RNN is associated with a probability. The joint probability of a child network is the product of all probabilities at these 10*B* softmaxes. This joint probability is used to compute the gradient for the controller RNN.
   - The gradient is scaled by the validation accuracy of the child network to update the controller RNN such that the controller assigns low probabilities for bad child networks and high probabilities for good child networks.
   
   ![schema](../../images/NasNet/schema.png)
   
   ![a](../../images/NasNet/a.png)
   
   ![b](../../images/NasNet/b.png)
   
   ![c](../../images/NasNet/c.png)

#### Reference

[1] [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/pdf/1707.07012.pdf)

[2] [Review: NASNet — Neural Architecture Search Network (Image Classification)](https://medium.com/@sh.tsang/review-nasnet-neural-architecture-search-network-image-classification-23139ea0425d)

[3] [Keras Implementation](https://github.com/keras-team/keras-applications/blob/master/keras_applications/nasnet.py)
	

--------------------
### MnasNet 

<b><details><summary>**"MnasNet: Platform-Aware Neural Architecture Search for Mobile"**</summary></b>
	
</details>

### MobileNetV3

<b><details><summary>**Searching for MobileNetV3**</summary></b>



#### Questions

#### Reference

[1] [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

</details>
