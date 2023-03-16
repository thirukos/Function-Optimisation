import numpy as np
import matplotlib.pyplot as plt


colors = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 
    'red', 'blue', 'lime'
]

def contourPlot(func, T, xs, fs, rngs=None, is_3d=True, legend=None):
    # %matplotlib widget
    if rngs is None:
        X = np.linspace(-20, 10, 100)
        Y = np.linspace(-20, 10, 100)
    else:
        X = np.linspace(*rngs[0], 100)
        Y = np.linspace(*rngs[1], 100)
    Z = []
    for x in X:
        z = []
        for y in Y: z.append(func([x, y], T))
        Z.append(z)
    Z = np.array(Z)
    X, Y = np.meshgrid(X, Y)
    if is_3d:
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, Z, 80)
        ax.set_xlabel('$x_0$')
        ax.set_ylabel('$x_1$')
        ax.set_zlabel('$f(x, T)$')
    else:
        plt.contour(X, Y, Z, 80)
        plt.xlabel('$x_0$')
        plt.ylabel('$x_1$')
    for i in range(len(xs)):
        x0 = [x[1] for x in xs[i]]
        x1 = [x[0] for x in xs[i]]
        if is_3d:
            ax.plot(x0, x1, fs[i], color='dimgrey',
                marker='x', markeredgecolor=colors[i], markersize=3)
        else:
            plt.plot(x0, x1, color='dimgrey',
                marker='x', markeredgecolor=colors[i], markersize=3)
            plt.xlim([-10, 10])
            plt.ylim([-20, 10])
    if legend is not None: plt.legend(legend)
    plt.show()