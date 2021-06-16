import numpy as np
import matplotlib
import itertools
class Grapher:

    def __init__(self, lower_limit, upper_limit):
        self.x_array = np.array([])
        self.y_array = np.array([])
        self.z_array = np.array([])
        self.t_array = np.array([])
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def _mesh(self, x_min, x_max, y_min, y_max):
        dx = (x_max - x_min) / 100
        dy = (y_max - y_min) / 100
        x, y = np.mgrid[x_min:x_max+dx:dx, y_min:y_max+dx:dy]
        return x, y


    def plot(self, f,  title=""):
        """Generic method for plotting a function on some mesh grid. Intended
    to be used only internally.

        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        from mpl_toolkits.mplot3d import Axes3D
        #self.tostring()
        X, Y = self._mesh(self.lower_limit, self.upper_limit, self.lower_limit, self.upper_limit)
        Z = np.zeros(X.shape)
        for i, j in itertools.product(range(X.shape[0]), range(X.shape[1])):
            Z[i, j] = f(np.array([X[i,j], Y[i,j]]))

        # From https://matplotlib.org/examples/mplot3d/surface3d_demo.html
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.text2D(0.05, 0.95, title, transform=ax.transAxes)

        t_array = self.t_array / np.amax(self.t_array)
        for i in range(1, len(self.x_array)):
            if(i < len(self.x_array) - 1):
                ax.plot(self.x_array[i - 1:i + 1], self.y_array[i - 1:i + 1], self.z_array[i - 1:i + 1],
                       c=(t_array[i - 1], 0, 0), zorder=10)
            else:
                ax.plot(self.x_array[i - 1:i + 1], self.y_array[i - 1:i + 1], self.z_array[i - 1:i + 1],
                        c=(t_array[i - 1], 1, 0), zorder=10)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, zorder=0)
        ax.set_zlim(np.min(self.z_array), np.max(self.z_array))
        #ax.set_zlim(np.min(Z), np.max(Z))
        #ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        plt.draw()
        plt.show()
        #input("Press <ENTER> to continue...")
        plt.close(fig)

    def addValue(self, float_array: np.array, fitness, time):
        self.x_array = np.append(self.x_array, float_array[0])
        self.y_array = np.append(self.y_array, float_array[1])
        self.z_array = np.append(self.z_array, fitness)
        self.t_array = np.append(self.t_array, time)

    def tostring(self):
        print("len(x_array): " + str(len(self.x_array)))
        print("len(y_array): " + str(len(self.y_array)))
        print("len(z_array): " + str(len(self.z_array)))
        print("len(t_array): " + str(len(self.t_array)))
