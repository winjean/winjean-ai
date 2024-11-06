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
 
 
拟合、过拟合、欠拟合

### 权重、偏置、权重矩阵、偏置向量
* 权重（Weights）：
    权重是连接两个神经元之间的边上的数值，表示前一层神经元对后一层神经元的影响程度。  
    在前向传播过程中，输入数据与权重相乘，然后传递给下一层。  
* 权重矩阵（Weight Matrix）：
    当神经网络的一层有多个神经元时，权重可以组织成一个矩阵。  
    权重矩阵的每一行对应前一层的一个神经元，每一列对应后一层的一个神经元。  
    例如，如果前一层有 ( n ) 个神经元，后一层有 ( m ) 个神经元，则权重矩阵的形状为 ( m \times n )。
* 偏置（Biases）：
    偏置是每个神经元的附加参数，用于调整激活函数的位置。  
    偏置使得模型更加灵活，能够更好地拟合数据。  
    在前向传播过程中，偏置被加到加权输入的总和上。
* 偏置向量（Bias Vector）：
    当神经网络的一层有多个神经元时，偏置可以组织成一个向量。  
    偏置向量的长度等于该层神经元的数量。  
    例如，如果某一层有 ( m ) 个神经元，则偏置向量的形状为 ( m \times 1 )
