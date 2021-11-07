import numpy as np


class Layer:
    def __init__(self, nrnSize=0):
        self.a = np.ndarray((nrnSize, 1))
        self.z = np.ndarray((nrnSize, 1))
        self.delta = np.ndarray((nrnSize, 1))
        self.bias = 1
        self.theta = None
        # partial和θ形状相同，储存暂时的偏导数
        self.partial = None
        # nrnSize存储该层神经元的长度
        self.nrnSize = nrnSize

    def setTheta(self, this, next):
        self.theta = np.random.randn(this, next)

    def setPartial(self, this, next):
        self.partial = np.random.randn(this, next)


class ANN:
    def __init__(self, lyrlst):
        self.lyrlst = lyrlst
        self.alpha = 0.5  # 学习率
        self.l = len(lyrlst)
        self.isOptimized = False

    def optimize(self, alpha=0.5):
        """初始化（优化）:做层与层之间θ的连接处理"""
        self.alpha = alpha
        for i in range(self.l-1):
            self.lyrlst[i].setTheta(self.lyrlst[i].nrnSize,
                                    self.lyrlst[i+1].nrnSize)
            self.lyrlst[i].setPartial(self.lyrlst[i].nrnSize,
                                      self.lyrlst[i+1].nrnSize)
        self.isOptimized = True

    def fwrdprop(self, X):
        """一次前向传播"""
        self.lyrlst[0].a = X.reshape((1, len(X)))
        for i in range(1, self.l):
            self.lyrlst[i].z = ((self.lyrlst[i-1].a @ self.lyrlst[i-1].theta))
            # print("z: ", self.lyrlst[i].z)
            self.lyrlst[i].a = (1 / (1 + np.exp(-self.lyrlst[i].z)))
            # print("a: ", self.lyrlst[i].a)

    def backprop(self, y):
        """一次反向传播"""
        self.lyrlst[-1].delta = self.lyrlst[-1].a - y  # 最后一层的反向传播
        for i in range(self.l - 2, -1, -1):
            # 更新delta
            self.lyrlst[i].delta = (self.lyrlst[i+1].delta * (
                self.lyrlst[i+1].a * (1 - self.lyrlst[i+1].a)
            )) @ self.lyrlst[i].theta.T
            # 更新partial
            #for j in range(len(self.lyrlst[i].a[0])):
            #   self.lyrlst[i].partial[j, :] = self.lyrlst[i].a[0][j]
            # print("PARTIAL: ",self.lyrlst[i].partial)
            self.lyrlst[i].partial = ((self.lyrlst[i+1].delta * (
                self.lyrlst[i+1].a * (1 - self.lyrlst[i+1].a)
            )).T @ self.lyrlst[i].a).T
        #     print('--------------------------')
        #     print("Layer Num:", i+1)
        #     print("a:", self.lyrlst[i].a)
        #     print("delta:", self.lyrlst[i].delta)
        #     print("theta:", self.lyrlst[i].theta)
        #     print('--------------------------')
        
        print("output:", self.lyrlst[-1].a)

    def adapt(self):
        """梯度下降中更新参数的函数"""
        for i in range(self.l - 1):
            self.lyrlst[i].theta -= (self.alpha * self.lyrlst[i].partial)
            # print("Layer: ", i + 1)
            # print("Theta:", self.lyrlst[i].theta)

    def fit(self, Xli, yli, epoch=5):
        if not self.isOptimized:
            print("请先初始化！")
            exit()
        Xli = np.array(Xli)
        yli = np.array(yli)
        if len(Xli) != len(yli):
            print("训练样本与标签长度不一致！")
            exit()
        for i in range(epoch):
            for j in range(len(Xli)):
                self.fwrdprop(Xli[j])
                self.backprop(yli[j])
                self.adapt()
            print("Epoch ", i, "over!")

    def predict(self, Xpred):
        Xpred = np.array(Xpred)
        if len(Xpred) != self.lyrlst[0].nrnSize:
            print("测试样本与模型输入预计长度不一致！")
            exit()
        ypred = np.ndarray((len(Xpred), 1))
        for i in range(Xpred):
            self.fwrdprop(Xpred[i])
        return self.lyrlst[-1].a.argmax() + 1
