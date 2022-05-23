#EE2703 APL ASS 5
#NAME SHEKHAR DIWAKAR
#EE20B123

import numpy as np
from sys import argv
import scipy.linalg as s
import matplotlib.pyplot as pylab
import mpl_toolkits.mplot3d.axes3d as p3

Nx, Ny, radius, Niter = 25, 25, 8, 1700  # Variables

argc = len(argv)
# get user inputs
if len(argv) == 5:
    Nx = argv[1]
    Ny = argv[2]
    radius = argv[3]
    Niter = argv[4]

# initialize potential
phi = np.zeros((Nx, Ny))
x = np.linspace(-0.5, 0.5, num=Nx)
y = np.linspace(-0.5, 0.5, num=Ny)

Y, X = np.meshgrid(y, x, sparse=False)
ii = np.where(X ** 2 + Y ** 2 < (0.35) ** 2)
phi[ii] = 1.0

# plot potential
pylab.figure(1)
pylab.title("Potential", fontsize=16)
pylab.xlabel("X")
pylab.ylabel("Y")
pylab.contourf(X, Y, phi)

# helper functions for the iterations
def update_phi(phi, phiold):
    mask = np.where(X ** 2 + Y ** 2 < (0.35) ** 2)
    phi[1:-1, 1:-1] = 0.25 * (
        phiold[1:-1, 0:-2] + phiold[1:-1, 2:] + phiold[0:-2, 1:-1] + phiold[2:, 1:-1]
    )
    phi[:, 0] = phi[:, 1]  # Left Boundary
    phi[:, Nx - 1] = phi[:, Nx - 2]  # Right Boundary
    phi[0, :] = phi[1, :]  # Top Boundary
    phi[Ny - 1, :] = 0
    phi[mask] = 1.0
    return phi


err = np.zeros(Niter)
# the iterations
for k in range(Niter):
    phiold = phi.copy()
    phi = update_phi(phi, phiold)
    err[k] = np.max(np.abs(phi - phiold))
    if err[k] == 0:
        print("Reached steady state at ", k, " Iterations")
        break

# plotting Error on semilog
pylab.figure(2)
pylab.title("Error on a semilog plot", fontsize=16)
pylab.xlabel("No of iterations")
pylab.ylabel("Error")
pylab.semilogy(range(Niter), err)

# plotting Error on loglog
pylab.figure(9)
pylab.title("Error on a loglog plot", fontsize=16)
pylab.xlabel("No of iterations")
pylab.ylabel("Error")
pylab.loglog((np.asarray(range(Niter)) + 1), err)
pylab.loglog((np.asarray(range(Niter)) + 1)[::50], err[::50], "ro")
pylab.legend(["real", "every 50th value"])

# helper function for getting best fit
def get_fit(y, Niter, lastn=0):
    log_err = np.log(err)[-lastn:]
    X = np.vstack([(np.arange(Niter) + 1)[-lastn:], np.ones(log_err.shape)]).T
    log_err = np.reshape(log_err, (1, log_err.shape[0])).T
    return s.lstsq(X, log_err)[0]


# Helper function to plot errors
def plot_error(err, Niter, a, a_, b, b_):
    pylab.figure(3)
    pylab.title("Best fit for error on a loglog scale", fontsize=16)
    pylab.xlabel("No of iterations")
    pylab.ylabel("Error")
    x = np.asarray(range(Niter)) + 1
    pylab.loglog(x, err)
    pylab.loglog(x[::100], np.exp(a + b * np.asarray(range(Niter)))[::100], "ro")
    pylab.loglog(x[::100], np.exp(a_ + b_ * np.asarray(range(Niter)))[::100], "go")
    pylab.legend(["errors", "fit1", "fit2"])

    # now semilog
    pylab.figure(4)
    pylab.title("Best fit for error on a semilog scale", fontsize=16)
    pylab.xlabel("No of iterations")
    pylab.ylabel("Error")
    pylab.semilogy(x, err)
    pylab.semilogy(x[::100], np.exp(a + b * np.asarray(range(Niter)))[::100], "ro")
    pylab.semilogy(x[::100], np.exp(a_ + b_ * np.asarray(range(Niter)))[::100], "go")
    pylab.legend(["errors", "fit1", "fit2"])


def find_net_error(a, b, Niter):
    return -a / b * np.exp(b * (Niter + 0.5))


b, a = get_fit(err, Niter)
b_, a_ = get_fit(err, Niter, 500)
plot_error(err, Niter, a, a_, b, b_)
# plotting cumulative error
iter = np.arange(100, 1501, 100)
pylab.figure(5)
pylab.grid(True)
pylab.title("Plot of Cumulative Error values On a loglog scale", fontsize=16)
pylab.loglog(iter, np.abs(find_net_error(a_, b_, iter)), "ro")
pylab.xlabel("iterations")
pylab.ylabel("Net  maximum error")
pylab.legend(loc="best")


# plotting 2d contour of final potential
pylab.figure(6)
pylab.title("2D Contour plot of potential", fontsize=16)
pylab.xlabel("X")
pylab.ylabel("Y")
pylab.plot((ii[0] - Nx / 2) / Nx, (ii[1] - Ny / 2) / Ny, "ro")
pylab.contourf(Y, X[::-1], phi)
pylab.colorbar()
pylab.legend(loc="best")

# plotting 3d contour of final potential
fig1 = pylab.figure(7)  # open a new figure
ax = p3.Axes3D(fig1)  # Axes3D is the means to do a surface plot
pylab.title("The 3-D surface plot of the potential", fontsize=16)
surf = ax.plot_surface(Y, X, phi.T, rstride=1, cstride=1, cmap=pylab.cm.jet)
pylab.legend(loc="best")

Jx = 1 / 2 * (phi[1:-1, 0:-2] - phi[1:-1, 2:])
Jy = 1 / 2 * (phi[:-2, 1:-1] - phi[2:, 1:-1])

# plotting current density
pylab.figure(8)
pylab.title("Vector plot of current flow", fontsize=16)
pylab.quiver(Y[1:-1, 1:-1], -X[1:-1, 1:-1], -Jx[:, ::-1], -Jy)
pylab.plot((ii[0] - Nx / 2) / Nx, (ii[1] - Ny / 2) / Ny, "ro")
pylab.legend(loc="best")
pylab.show()

