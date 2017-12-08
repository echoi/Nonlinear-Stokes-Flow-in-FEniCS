""" Generate a mesh using the FEniCS meshing tools. """


from __future__ import division
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np


# suppress FEniCS output to terminal
set_log_active(False)


"""Domain specifications."""

L, H = 500, 125  # length and height
nx, ny = 10, 10  # notch width and height
name = "notch"  # mesh name


"""Generate mesh."""

slab = Rectangle(Point(0, 0), Point(L, H))
notch = Rectangle(Point(L/2 - nx/2, H - ny), Point(L/2 + nx/2, H))
domain = slab

kx, ky = 4, int(H/5)
kn = 1
for i in range(kx):
    for j in range(ky):
        x = L/2 - 10 + 5*i
        y = j*5
        p1 = Point(x, y)
        p2 = Point(x+5, y+5)
        domain.set_subdomain(kn, Rectangle(p1, p2))
        kn += 1
mesh = generate_mesh(domain, 40)


"""Refine mesh."""

cc = CellFunction("bool", mesh, False)
rw = 3*nx
for cell in cells(mesh):
    m = cell.midpoint()
    if (0.8 < m[0]/L < 1) and (0.8 < m[1]/H < 1):
        cc[cell] = True
    if (0.9 < m[1]/H) and not (-rw/2 < m[0]-L/2 < rw/2):
        cc[cell] = True
mesh = refine(mesh, cc)

for i in range(1):
    cc = CellFunction("bool", mesh, False)
    for cell in cells(mesh):
        h = cell.h()
        m = cell.midpoint()
        if L/2-10 <= m[0] <= L/2+10:
            cc[cell] = True
    mesh = refine(mesh, cc)

x1, x2 = mesh.coordinates().T
print("\n  %d cells\n  %d vertices" % (len(mesh.cells()), len(x1)))
hmax = 0.0
for cell in cells(mesh):
    h = cell.h()
    m = cell.midpoint()
    if (L-nx)/2 <= m[0] <= (L+nx)/2:
        hmax = max(h, hmax)
print("  Max cell size in damage zone: %f" % hmax)


"""Output the mesh."""

hdf5 = HDF5File(mpi_comm_world(), "mesh/hdf5/" + name + ".h5", "w")
hdf5.write(mesh, "mesh")
hdf5.close()
File("mesh/xml/%s.xml" % name) << mesh


"""Generate a plot of the mesh."""

tri = np.asarray([cell.entities(0) for cell in cells(mesh)])

fig = plt.figure(figsize=(20, 20))
plt.title("Mesh: %s.h5" % name, fontsize=20, y=1.04)
ax = fig.add_subplot(111, label='main')
plt.ylim(-10, H + 10)
plt.xlim(-10, L + 10)
plt.plot(x1, x2, "k.", ms=3)
plot = plt.triplot(x1, x2, tri, "c-", lw=0.75)
ax.set_aspect("equal")
plt.tight_layout()
path = "output/figs/mesh-%s.pdf" % name
plt.savefig(path, bbox_inches="tight")
print("  Saved %s\n" % path)
