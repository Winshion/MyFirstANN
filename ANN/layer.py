"""构建神经网络层"""
import numpy as np


class Layer:
    """
    神经网络的‘层’，使用sigmoid激活函数。
    """

    def __init__(self, nrnSize=None, inputVect=None):
        #                         神经元数     输入向量
        self.nenSize = nrnSize
        if(inputVect):  # 判断是否为输入层
            self.a = np.array(inputVect).flatten()
            self.delta = np.ndarray(nrnSize, 0)
            self.partial = np.ndarray(nrnSize, 0)
            self.z = None
            self.theta = None  # theta只有在建完网络之后才能确定下来，先待定
        else:
            self.z = np.ndarray(nrnSize, 0)
            self.a = np.ndarray(nrnSize, 0)
            self.partial = np.ndarray(nrnSize, 0)
            self.delta = np.ndarray(nrnSize, 0)
            self.theta = None


class ANN:
    def __init__(self, layerLst):
        self.layerLst = layerLst  # layerLst表示层的列表
        self.alpha = 0  # 学习率
        self.length = len(layerLst)

    def optimize(self, alpha):
        """初始化函数，进行层与层之间的连接"""
        self.alpha = alpha
        for i in range(self.length - 1):
            self.layerLst[i].theta = np.ndarray(
                (self.layerLst[i + 1].nrnSize, self.layerLst[i].nrnSize))

    def fwrdProp(self):
        """前向传播方法"""
        for i in range(1, self.length):
            for j in range(self.layerLst[i].nrnSize):
                self.layerLst[i].z[j] = self.layerLst[i].theta[j].dot(
                    self.layerLst[i - 1].a
                )
            self.layerLst[i].a = 1 / (1 + np.exp(-self.layerLst[i].z))

    def backProp(self, epsilon):
        """反向传播算法，并更新每个theta"""
        pass
