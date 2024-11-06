### 损失函数（Loss Function）
损失函数在机器学习和深度学习中扮演着至关重要的角色，它用于衡量模型预测值与真实值之间的差异。  
选择合适的损失函数对于模型的性能至关重要。以下是一些常见的损失函数及其适用场景：
* 均方误差（Mean Squared Error, MSE）
    公式：[ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 ]  
    适用场景：回归任务，当目标变量是连续值时。  
    特点：对异常值敏感，因为误差的平方会放大较大的误差。  
* 均方对数误差（Mean Squared Logarithmic Error, MSLE）  
    公式：[ \text{MSLE} = \frac{1}{n} \sum_{i=1}^{n} (\log(1 + y_i) - \log(1 + \hat{y}_i))^2 ]  
    适用场景：回归任务，特别是当目标变量的范围很大且对数变换后更符合正态分布时。  
    特点：对较小的误差更为敏感，适用于目标变量呈指数增长的情况。  
* 绝对误差（Mean Absolute Error, MAE）  
    公式：[ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| ]  
    适用场景：回归任务，对异常值不敏感。  
    特点：计算简单，对所有误差一视同仁。  
* 交叉熵损失（Cross-Entropy Loss）  
  * 二分类交叉熵：  
  公式：[ \text{Binary Cross-Entropy} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]]  
  适用场景：二分类任务。  
  * 多分类交叉熵：  
  公式：[ \text{Categorical Cross-Entropy} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{m} y_{ij} \log(\hat{y}_{ij}) ]
  适用场景：多分类任务。  
  特点：适用于概率输出，能够有效处理类别不平衡问题。   
* 合页损失（Hinge Loss）  
    公式：[ \text{Hinge Loss} = \max(0, 1 - y_i \cdot \hat{y}_i) ]  
    适用场景：支持向量机（SVM）等分类任务。  
    特点：鼓励模型将正例和负例尽可能分开，适用于线性分类器。  
* Huber损失  
    公式： [ \text{Huber Loss} = \begin{cases} \frac{1}{2}(y_i - \hat{y}_i)^2 & \text{if } |y_i - \hat{y}_i| \leq \delta \ \delta (|y_i - \hat{y}_i| - \frac{1}{2}\delta) & \text{otherwise} \end{cases} ]  
    适用场景：回归任务，对异常值不敏感。  

* 特点：结合了MSE和MAE的优点，对小误差采用平方损失，对大误差采用线性损失。
选择合适的损失函数需要根据具体的任务和数据特性来决定。不同的损失函数会对模型的训练过程和最终性能产生显著影响。