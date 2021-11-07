import numpy as np
import NN

x = [[1.286, 5.638]]
y = [[0,0,0,1,0]]
model = NN.ANN([
    NN.Layer(nrnSize=2),
    NN.Layer(nrnSize=5),
    NN.Layer(nrnSize=5)
])
model.optimize(alpha = 0.5)
model.fit(x, y, epoch=100)
print(model.predict([[1.280, 5.638]]))