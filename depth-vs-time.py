""" Plot the crevasse depth vs. time using FEniCS output. """


from __future__ import division
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import warnings


# suppress FEniCS output to terminal
set_log_active(False)
warnings.filterwarnings("ignore")


def load_data(i=1):
    """Load data and extract the damage variable (at integration points)."""
    try:
        path = "output/data/%d.h5" % int(i)
        mesh = Mesh()
        hdf5 = HDF5File(mpi_comm_world(), path, "r")
        hdf5.read(mesh, "mesh", False)
        nd = mesh.geometry().dim()
        scalar_quad = FiniteElement("Quadrature", cell=mesh.ufl_cell(),
                                    degree=2, quad_scheme="default")
        SQ = FunctionSpace(mesh, scalar_quad)
        xQ1, xQ2 = SQ.tabulate_dof_coordinates().reshape((-1, nd)).T
        dmg = Function(SQ)
        hdf5.read(dmg, "damage")
        m = hdf5.attributes("mesh")
        time = m["current time"]
        hdf5.close()
        dmg = dmg.vector().array()
        return time, xQ2, dmg
    except:
        return None, None, None


"""Construct arrays for plotting."""

Dcr = 0.6
ds_H = []
time = []
k = 1
inc = 10
end = int(input("\nNumber of last data file >> "))
loop = True

while loop:
    if k > end:
        k = end
    if k == end:
        loop = False
    timek, x2, D = load_data(k)
    if p:
        H = max(x2)
        ds = H-min(x2[D > Dcr])
        ds_H.append(ds/H)
        time.append(timek)
        print("%.1f %%" % (k/end*100))
    k += inc


"""Plot the depth vs. time curve."""

plt.figure()
plt.plot(np.array(time)/24, ds_H, "c-", lw=2)
plt.grid()
plt.ylim(0, 1)
plt.xlabel("Time (days)")
plt.ylabel("$\mathregular{d_s/H}$")
plt.savefig("output/figs/dvt.png")

print("Saved: output/figs/dvt.pdf")
