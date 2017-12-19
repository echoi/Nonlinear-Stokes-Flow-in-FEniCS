"""Run an updated-Lagrangian Wave simulation with damping for quasi-static problems. 
Important features include:
    - nonlinear viscosity
    - incompressibility
    - poro-mechanics formulation for hydraulic fracture
"""

# Copyright (C) 2010 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg 2008-2011
#
# First added:  2010-04-30
# Last changed: 2012-11-12

from __future__ import print_function
from fenics import *
# from dolfin import *
from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt

# suppress FEniCS output to terminal
set_log_active(False)

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# output directory
output_dir = ""
name = "notch_finer"
mesh_name = "notch_finer"

comm = MPI.COMM_WORLD  # MPI communications
rank = comm.Get_rank()  # number of current process
size = comm.Get_size()  # total number of processes


if rank == 0:
    start = time.clock()
    time_log = open(output_dir + "output/details/" + name + "_time_log.txt", "w")
    time_log.close()
    with open(output_dir
              + "output/details/" + name + "_simulation_log.txt", "w") as sim_log:
        if size == 1:
            sim_log.write("Stokes flow in FEniCS.\nRunning on "
                          "1 processor.\n" + "-"*64 + "\n")
        else:
            sim_log.write("Stokes flow in FEniCS.\nRunning on "
                          "%d processors.\n" % size + "-"*64 + "\n")

def write_hdf5(timestep, mesh, data, time=0):
    """Write output from the simulation in .h5 file format."""
    output_path = output_dir + "output/data/" + str(timestep) + ".h5"
    hdf5 = HDF5File(mpi_comm_world(), output_path, "w")
    hdf5.write(mesh, "mesh")
    m = hdf5.attributes("mesh")
    m["current time"] = float(time)
    m["current step"] = int(timestep)
    for value in sorted(data):
        hdf5.write(data[value], value)
    hdf5.close()


def load_mesh(path):
    """Load a mesh in .h5 file format."""
    mesh = Mesh()
    hdf5 = HDF5File(mpi_comm_world(), path, "r")
    hdf5.read(mesh, "mesh", False)
    hdf5.close()
    return mesh


def update(dv, u0, v0, dt):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    dv_vec = dv.vector()
    u0_vec, v0_vec = u0.vector(), v0.vector()

    # Update velocity and nodal coordinates
    # v = dv + v0
    v_vec = dv_vec + v0_vec
    u_vec = dt * v_vec

    # Update (u0 <- u0)
    v0.vector()[:] = v_vec
    u0.vector()[:] = u_vec

# External load
class Traction(UserExpression):

    def __init__(self, dt, t, old, **kwargs):
        self.t   = t
        self.dt  = dt
        self.old = old
        if has_pybind11():
            super().__init__(**kwargs)

    def eval(self, values, x):

        # 'Shift' time for n-1 values
        t_tmp = self.t
        if self.old and t > 0.0:
            t_tmp -= self.dt

        cutoff_t = 10.0*1.0/32.0;
        weight = t_tmp/cutoff_t if t_tmp < cutoff_t else 1.0

        values[0] = 1.0*weight
        values[1] = 0.0

    def value_shape(self):
        return (2,)

"""Material parameters."""

rho_ice = 917  # density of ice (kg/m^3)
rho_H2O = 1020  # density of seawater (kg/m^3)
grav = 9.81  # gravity acceleration (m/s**2)
temp = -10 + 273  # temperature (K)
B0 = 2.207e-3  # viscosity coefficient (kPa * yr**(1/3))
B0 *= 1e3  # convert to (Pa * yr**(1/3))
B0 *= (365*24)**(1/3)  # convert to (Pa * hour**(1/3))
BT = B0*np.exp(3155/temp - 0.16612/(273.39 - temp)**1.17)


"""Damage parameters."""

alpha = 0.21  # weight of max principal stress  in Hayhurst criterion
beta = 0.63  # weight of von Mises stress in Hayhurst criterion
r = 0.43  # damage exponent
B = 5.232e-7  # damage coefficient
k1, k2 = -2.63, 7.24  # damage rate dependency parameters
Dcr = 0.6  # critical damage
Dmax = 0.99  # maximum damage
lc = 10  # nonlocal length scale

"""Set simulation time and timestepping options."""

t_total = 13  # total time (hours)
t_elapsed = 0  # current elapsed time (hours)
t_delay_dmg = 0  # delay damage (hours)
max_Delta_t = 0.5  # max time increment (hours)
max_Delta_D = 0.1  # max damage increment
output_increment = 10  # number of steps between output
time_counter = 0  # current time step


"""Mesh details."""
hs = 0  # water level in crevasse (normalized with crevasse height)
hw = 0  # water level at terminus (absolute height)
mesh = load_mesh(output_dir + "mesh/hdf5/" + mesh_name + ".h5")
nd = mesh.geometry().dim()  # mesh dimensions (2D or 3D)
if nd == 3:
    L, H, W = 500, 125, 300  # domain dimensions: Length (x1 dimension), height (x2 dim.) and width (x3 dim.)
elif nd == 2:
    L, H = 500, 125  # domain dimensions: Length (x1 dimension), height (x2 dim.) and width (x3 dim.)

Rxx = 1/2*rho_ice*grav*H - 1/2*rho_H2O*grav*(hw**2)/H  # restrictive stress

# Sub domain for clamp at left end
def left(x, on_boundary):
    return x[0] < 0.001 and on_boundary

# Sub domain for rotation at right end
def right(x, on_boundary):
    return x[0] > L - 1e-3 and on_boundary

# Load mesh and define function space
# mesh = Mesh("./dolfin_fine.xml.gz")

# Define function space
V = VectorFunctionSpace(mesh, "CG", 2)

# Test and trial functions
dv = TrialFunction(V)
r = TestFunction(V)

E  = 1.0
nu = 0.0
mu    = E / (2.0*(1.0 + nu))
lmbda = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))

# Mass density andviscous damping coefficient
rho = 1.0

# Time stepping parameters
dt      = 1.0/32.0
t       = 0.0
T       = 10*dt

# Fields from previous time step (displacement, velocity, acceleration)
u0 = Function(V)
v0 = Function(V)

# External forces (body and applied tractions
f  = Constant((0.0, 0.0))
p  = Traction(dt, t, False, degree=1)
p0 = Traction(dt, t, True, degree=1)

# Create mesh function over the cell facets
boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
force_boundary = AutoSubDomain(right)
force_boundary.mark(boundary_subdomains, 3)

# Define measure for boundary condition integral
dss = ds(subdomain_data=boundary_subdomains)

# Stress tensor
def sigma(r):
    return 2.0*mu*sym(grad(r)) + lmbda*tr(sym(grad(r)))*Identity(len(r))

# Forms
# a = factor_m1*inner(u, r)*dx + factor_d1*inner(u, r)*dx \
# +(1.0-alpha_f)*inner(sigma(u), grad(r))*dx

# L =  factor_m1*inner(r, u0)*dx + factor_m2*inner(r, v0)*dx \
# + factor_m3*inner(r, a0)*dx \
# + factor_d1*inner(r, u0)*dx + factor_d2*inner(r, v0)*dx \
# + factor_d3*inner(r, a0)*dx \
# - alpha_f*inner(grad(r), sigma(u0))*dx \
# + inner(r, f)*dx + (1.0-alpha_f)*inner(r, p)*dss(3) + alpha_f*inner(r, p0)*dss(3)
a = rho * inner(dv, r) * dx
L =  - dt * inner(grad(r), sigma(v0))*dx \
+ dt * inner(r, f)*dx + dt * inner(r, p0)*dss(3)

# Set up boundary condition at left end
zero = Constant((0.0, 0.0))
bc = DirichletBC(V, zero, left)

# FIXME: This demo needs some improved commenting

# Time-stepping
dv = Function(V)
vtk_file = File("elasticity.pvd")
while t <= T:

    t += dt
    print("Time: ", t)

    p.t = t
    p0.t = t

    solve(a == L, dv, bc)
    update(dv, u0, v0, dt)
    # Save solution to VTK format
    vtk_file << v0

# Plot solution
plot(v0, mode="displacement")
plt.show()
