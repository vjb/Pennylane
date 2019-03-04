import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer, AdagradOptimizer

dev = qml.device('default.qubit', wires=1)

@qml.qnode(dev)
def circuit(var):

    qml.RX(var[0], wires=0)
    qml.RY(var[1], wires=0)
    
    return qml.expval.PauliZ(0)


def objective(var):
    return circuit(var)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
#matplotlib inline

fig = plt.figure(figsize = (6, 4))
ax = fig.gca(projection='3d')

X = np.arange(-3.1, 3.1, 0.1)
Y = np.arange(-3.1, 3.1, 0.1)
length = len(X)
xx, yy = np.meshgrid(X, Y)
Z = np.array([[objective([x, y]) for x in X] for y in Y]).reshape(length, length)
surf = ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm, antialiased=False)

ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.zaxis.set_major_locator(MaxNLocator(nbins = 5, prune = 'lower'))

#plt.show()

var_init = np.array([-0.011, -0.012])
objective(var_init)

gd = GradientDescentOptimizer(0.4)


var = var_init
var_gd = [var]

for it in range(100):
    var = gd.step(objective, var)

    if (it + 1) % 5 == 0:
        var_gd.append(var)
        print('Objective after step {:5d}: {: .7f} | Angles: {}'.format(it + 1, objective(var), var) )

ada = AdagradOptimizer(0.4)

var = var_init
var_ada = [var]

for it in range(100):
    var = ada.step(objective, var)
    
    if (it + 1) % 5 == 0:
        var_ada.append(var)
        print('Objective after step {:5d}: {: .7f} | Angles: {}'.format(it + 1, objective(var), var) )


