## 神经网络层及其功能：
### 全连接层（Fully Connected Layer / Dense Layer）
* 功能：将输入数据从一个维度变换到另一个维度，通过线性变换和激活函数实现。
* 常用模块：torch.nn.Linear
* 公式：[ y = Wx + b ]
* 应用场景：广泛应用于多层感知机（MLP）、分类任务等。

### 卷积层（Convolutional Layer）
* 功能：提取输入数据的局部特征，常用于图像处理任务。
* 常用模块：torch.nn.Conv2d（二维卷积）、torch.nn.Conv1d（一维卷积）
* 公式：[ y = \text{conv}(x, W) + b ]
* 应用场景：图像识别、目标检测、自然语言处理等。

### 池化层（Pooling Layer）
* 功能：降低特征图的尺寸，减少参数数量，提高计算效率。
* 常用模块：torch.nn.MaxPool2d（最大池化）、torch.nn.AvgPool2d（平均池化）
* 应用场景：图像处理、特征降维等。

### 激活函数层（Activation Layer）
* 功能：引入非线性，使神经网络能够学习复杂的函数。
* 常用模块：torch.nn.ReLU、torch.nn.Sigmoid、torch.nn.Tanh、torch.nn.LeakyReLU、torch.nn.Softmax
* 应用场景：几乎所有的神经网络中都会使用激活函数。

### 归一化层（Normalization Layer）
* 功能：对输入数据进行归一化处理，加速训练过程，提高模型的稳定性。
* 常用模块：torch.nn.BatchNorm2d（批归一化）、torch.nn.LayerNorm（层归一化）
* 应用场景：图像处理、自然语言处理等。

### 循环层（Recurrent Layer）
* 功能：处理序列数据，具有记忆功能，能够捕捉时间上的依赖关系。
* 常用模块：torch.nn.RNN、torch.nn.LSTM、torch.nn.GRU
* 应用场景：自然语言处理、时间序列预测等。

### 残差层（Residual Layer）
* 功能：解决深层网络中的梯度消失问题，通过残差连接（skip connection）实现。
* 常用模块：自定义实现
* 应用场景：深度卷积神经网络（如 ResNet）

### 注意力层（Attention Layer）
* 功能：使模型能够关注输入数据的某些部分，提高模型的表征能力。
* 常用模块：自定义实现
* 应用场景：自然语言处理、图像识别等。

### 损失层（Loss Layer）
* 功能：计算模型的损失，用于优化模型参数。
* 常用模块：torch.nn.CrossEntropyLoss、torch.nn.MSELoss、torch.nn.BCELoss
* 应用场景：分类任务、回归任务等。
 
### 常见的激活函数
* ReLU (Rectified Linear Unit)：
  * 公式：( f(x) = \max(0, x) )
  * 特点：简单高效，计算速度快；能够有效缓解梯度消失问题；但可能会导致“死区”问题（即某些神经元的输出始终为零）。
  * 适用场景：广泛应用于卷积神经网络和深度神经网络。

* Sigmoid：
  * 公式：( f(x) = \frac{1}{1 + e^{-x}} )
  * 特点：将输入压缩到 (0, 1) 之间，适用于二分类问题；但容易导致梯度消失。
  * 适用场景：逻辑回归、二分类问题。

* Tanh (Hyperbolic Tangent)：
  * 公式：( f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} )
  * 特点：将输入压缩到 (-1, 1) 之间，输出均值接近零；但同样容易导致梯度消失。
  * 适用场景：隐藏层、RNN。

* Leaky ReLU：
  * 公式：( f(x) = \max(\alpha x, x) )，其中 (\alpha) 是一个小的正数（如 0.01）
  * 特点：解决了 ReLU 的“死区”问题，允许负值通过；但选择合适的 (\alpha) 可能需要调参。
  * 适用场景：深度神经网络。

* Softmax：
  * 公式：( f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} )
  * 特点：将输入向量转换为概率分布，适用于多分类问题。
  * 适用场景：输出层、多分类问题。
  
* Swish：
  * 公式：( f(x) = x \cdot \sigma(x) )，其中 (\sigma(x)) 是 Sigmoid 函数
  * 特点：非单调函数，性能优于 ReLU；但计算复杂度较高。
  * 适用场景：深度神经网络

* 梯度消失：在深度神经网络中，如果所有层都是线性的，梯度在反向传播过程中可能会逐渐变小，导致训练困难。激活函数可以帮助缓解这一问题。
* 梯度爆炸：某些激活函数（如ReLU）具有恒定的梯度，有助于防止梯度爆炸

* 损失函数：nn.MSELoss(), nn.CrossEntropyLoss(), nn.BCELoss()

* 优化器：optim.SGD(), optim.Adam(), optim.RMSprop()
 
权重矩阵
偏置向量
