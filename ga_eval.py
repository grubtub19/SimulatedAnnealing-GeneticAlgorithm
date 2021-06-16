#!/usr/bin/env python3

"""Various functions from the First and Second International Contest
on Evolutionary Optimization. These functions accept multi-dimensional
vectors and output a single real value. The optimization goal is to
find a vector x* that minimizes each f.

Note that several of these functions can be parameterized; see the
function descriptions for details.

Each function has the following definitions:

    f(x) - The actual function definition
    f_c(x) - A boolean function that returns true if x fits all constraints

For many of these functions, the constraints can be encoded into the
specification of your chromosomes. However, the bump function will
need another method to deal with constraint violations.

To include this code in your project, you will need to install numpy
and matplotlib. If you have pip, this can be done with:

    sudo pip3 install numpy matplotlib

If you run this file with

    python3 ga_eval.py

you will see a visualization of each function on its approximate
domain. This information can be useful in deciding what modifications
will be necessary.

"""

import numpy as np
import matplotlib
import itertools
import shekel_params
import langermann_params

_norm = lambda x: np.linalg.norm(x, 1)**2
_inf_norm = lambda x: np.max(x**2)

################################################################

def sphere(x):
    """
    The sphere model. A basic function, where the minimum is at (1,1).

    Range: x_i in [-5, 5]

    :param numpy.ndarray x: The input vector
    """
    return _norm(x - 1.0)

def sphere_c(x):
    return np.all(np.logical_and(
        x >= -5,
        x <= 5
    ))

################################################################

def griew(x, D=4000.0):
    """
    Griewank's function. Similar to the sphere function, but much
    noisier around each point. The obvious minimum is at (100,100).

    Range: x_i in [0, 200]

    :param numpy.ndarray x: The input vector
    """
    n = len(x)
    term1 = _norm(x - 100) / D
    term2 = np.prod(np.cos((x-100) / np.sqrt(np.arange(n)+1)))
    return term1 - term2 + 1.0

def griew_c(x):
    return np.all(np.logical_and(
        x >= 0,
        x <= 200
    ))
    return True

################################################################

def shekel(x, holes=shekel_params.sample_holes, weights=shekel_params.sample_weights):
    """
    Modified Shekel's Foxholes. The function is mostly uniform, except
    for points very close to each "foxhole," in which the function
    dramatically decreases.

    Range: x_i in [0, 10]

    :param numpy.ndarray x: The (n x 1) input vector
    :param numpy.ndarray holes: An (n x m) matrix, consisting of m (n x 1) holes
    :param numpy.ndarray weights: A (m x 1) vector specifying the depth of each hole
    """
    n = len(x)
    m = holes.shape[0]
    if n > holes.shape[1]:
        raise ValueError(
            "Dimension of x is greater than dimension of holes. ({} > {})"
            .format(n, holes.shape[1])
        )
    return sum([-1.0 / (_norm(x - holes[j,:n] + weights[j])) for j in range(m)])

def shekel_c(x):
    return np.all(np.logical_and(
        x >= 0,
        x <= 10
    ))

################################################################

def micha(x, m=10):
    """
    Michalewitz's function. A noisy function with many local minima.

    Range: x_i in [-100, 100]

    :param numpy.ndarray x: The (n x 1) input vector
    :param float m: Parameter that affects the level of variation in the curves
    """
    n = len(x)
    cos_pi_6 = np.cos(np.pi / 6)
    sin_pi_6 = np.sin(np.pi / 6)
    y = np.zeros(n)
   # print("PRINTING")
    #print(y[::2])
    y[::2] = [x1 * cos_pi_6 - x2 * sin_pi_6 for (x1, x2) in zip(x[:-1], x[1:])]
    y[1::2] = [x1 * sin_pi_6 + x2 * cos_pi_6 for (x2, x1) in zip(x[1:], x[:-1])]
    y[n-1] = x[n-1]
    return sum(np.sin(y) * np.sin((np.arange(n) + 1) * y**2 / np.pi)**(2*m))

def micha_c(x):
    ## This works anywhere and couldn't find any constraint
    ## specification. A good choice of range might be (-100, 100) for
    ## any dimension.
    return np.all(np.logical_and(
        x >= -100,
        x <= 100
    ))

################################################################

def langermann(x, a=langermann_params.sample_a, c=langermann_params.sample_c):
    """Langermann's function. Another noisy function, although local maxima/minima are located near points given in the a matrix.

    Range: x_i in [0, 10]

    :param numpy.ndarray x: The (n x 1) input vector
    :param numpy.ndarray a: An (n x m) matrix of m (n x 1) vectors; each specifying a "region of more instability"
    :param numpy.ndarray c: An (m x 1) vector of weights associated with each vector in a.
    """
    n = len(x)
    m = a.shape[0]
    if n > a.shape[1]:
        raise ValueError(
            "Dimension of x is greater than dimension of a. ({} > {})"
            .format(n, a.shape[1])
        )
    term1 = [np.exp(-_norm(x - a[i,:n]) / np.pi) for i in range(m)]
    term2 = [np.cos(np.pi * _norm(x - a[i,:n])) for i in range(m)]
    return sum([c_i * term1_i * term2_i for (c_i, term1_i, term2_i) in zip(c, term1, term2)])

def langermann_c(x):
    return np.all(np.logical_and(
        x >= 0,
        x <= 10
    ))

################################################################

default_center_point = np.array([1., 1.3, .8, -.4, -1.3, 1.6, -.2, -.6, .5, 1.4,
                                 1., 1.3, .8, -.4, -1.3, 1.6, -.2, -.6, .5, 1.4])

def odd_square(x, center_point=default_center_point, c=0.2):
    """The Odd Square function. As the function approaches the center
point, it begins to oscillate more and more.

    Range: x_i is in (-5 pi, 5 pi)

    :param numpy.ndarray x: The (n x 1) input vector
    """
    n = len(x)
    dist = _inf_norm(x - center_point[:n])
    term1 = np.exp(-dist / (2 * np.pi))
    term2 = np.cos(np.pi * dist)
    term3 = 1 + c * dist / (dist + 0.01)
    return term1 * term2 * term3

def odd_square_c(x):
    return np.all(np.logical_and(
        x >= -5 * np.pi,
        x <= 5 * np.pi
    ))

################################################################

def bump(x):
    """The Bump function. Very smooth, but note that the constraints on
    this function cannot be easily encoded into the chromosonal design
    of the GA.

    Range: x_i is in (0,100)

    :param numpy.ndarray x: The (n x 1) input vector
    """
    n = len(x)
    cos_x = np.cos(x)
    term1 = np.sum(cos_x**4) - 2 * np.prod(cos_x**2)
    term2 = np.sqrt(np.sum(np.multiply(1.0 + np.arange(n), x**2)))
    if term2 != 0:
        return abs(term1 / term2)
    else:
        return 0

def bump_c(x):
    ## \prod_{i=1}^n x_i \geq 0.75
    #return np.prod(x) >= 0.75 and np.sum(x) <= 7.5 * len(x)
    return np.all(np.logical_and(
        x >= 0,
        x <= 5
    ))

################################################################

def _mesh(x_min, x_max, y_min, y_max):
    dx = (x_max - x_min) / 100
    dy = (y_max - y_min) / 100
    x, y = np.mgrid[x_min:x_max+dx:dx, y_min:y_max+dx:dy]
    return x, y


def _plot_f(f, X, Y, title=""):
    """Generic method for plotting a function on some mesh grid. Intended
to be used only internally.

    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from mpl_toolkits.mplot3d import Axes3D
    Z = np.zeros(X.shape)
    for i, j in itertools.product(range(X.shape[0]), range(X.shape[1])):
        Z[i, j] = f(np.array([X[i,j], Y[i,j]]))

    # From https://matplotlib.org/examples/mplot3d/surface3d_demo.html
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_zlim(np.min(Z), np.max(Z))
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.draw()
    plt.pause(1)
    input("Press <ENTER> to continue...")
    plt.close(fig)


if __name__ == '__main__':
    _plot_f(bump, *_mesh(0.1, 5, 0.1, 5), title="The Bump Function")
    _plot_f(sphere, *_mesh(-5, 5, -5, 5), title="The Sphere Function")
    _plot_f(griew, *_mesh(0, 200, 0, 200), title="Griewank's function")
    _plot_f(shekel, *_mesh(0, 10, 0, 10), title="Modified Shekel's Foxholes")
    _plot_f(micha, *_mesh(-10, 10, -10, 10), title="Michalewitz's function")
    _plot_f(langermann, *_mesh(0, 10, 0, 10), title="Langermann's function")
    _plot_f(odd_square, *_mesh(-5 * np.pi, 5 * np.pi, -5 * np.pi, 5 * np.pi), title="Odd Square Function")

