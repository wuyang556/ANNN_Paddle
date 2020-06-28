## Asymmetric Non-local Neural Networks for Semantic Segmentation

论文链接：https://arxiv.org/pdf/1908.07678.pdf

**Non-local Block 和 Asysmmetric Non-local Block结构对比：**

![](src\images\Non-local Block.png)

传统CNN不能高效的捕捉 long range dependency，而Non-local 和 spatial pyramid pooling 可以有效解决这个问题。Non-local是计算每个点和所有点的相关性（即每个像素点都对应一张attention map）；而本文认为不需要计算每个点的attention map，而是计算每个像素点与某几个局部区域的 attention map。

### Asymmetric Non-local Neural Network

<img src="src\images\ANNN.png" style="zoom:67%;" />

网络结构组成：

1. 使用Dilated ResNet提取语义特征，Stage 4和Stage 5的feature map大小一样，均为8x
2. AFNB融合Stage 4和Stage 5 的feature maps
3. AFNB后接APNB，使用残差连接