"""Plot solutions for Stokes flow at nodal coordinates."""


from __future__ import division
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import warnings
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# suppress FEniCS output to terminal
set_log_active(False)
warnings.filterwarnings("ignore")


"""Load data in .h5 file format."""

path = "output/data/%d.h5" % int(input("\nData file number >> "))

mesh = Mesh()
hdf5 = HDF5File(mpi_comm_world(), path, "r")
hdf5.read(mesh, "mesh", False)
nd = mesh.geometry().dim()

m = hdf5.attributes("mesh")
time = m["current time"]  # time (hours)
step = m["current step"]  # timestep
print("\nStep: " + str(step))
print("Time: " + str(time) + " hours\n")


"""Define function spaces."""

S1 = FunctionSpace(mesh, "CG", 1)  # first order scalar space
S2 = FunctionSpace(mesh, "CG", 2)  # second order scalar space
V1 = VectorFunctionSpace(mesh, "CG", 1)  # first order vector space
V2 = VectorFunctionSpace(mesh, "CG", 2)  # second order vector space

deg_quad = 2
scalar_quad = FiniteElement("Quadrature", cell=mesh.ufl_cell(),
                            degree=deg_quad, quad_scheme="default")
SQ = FunctionSpace(mesh, scalar_quad)
xQ1, xQ2 = SQ.tabulate_dof_coordinates().reshape((-1, nd)).T
form_params = {"quadrature_degree": deg_quad}


"""Extract information from data files as FEniCS functions."""

u = Function(V2)
p = Function(S1)
pHD = Function(SQ)
dmg = Function(SQ)
ddmg = Function(S1)

hdf5.read(u, "velocity")
hdf5.read(p, "pressure")
hdf5.read(pHD, "hydraulic pressure")
hdf5.read(dmg, "damage")
hdf5.read(ddmg, "damage increment")

hdf5.close()


"""Material and domain properties."""

rho_ice = 917  # density of ice (kg/m^3)
rho_H2O = 1020  # density of seawater (kg/m^3)
grav = 9.81  # gravity acceleration (m/s**2)
temp = -10 + 273  # temperature (K)
B0 = 2.207e-3  # viscosity coefficient (kPa * yr**(1/3))
B0 *= 1e3  # convert to (Pa * yr**(1/3))
B0 *= (365*24)**(1/3)  # convert to (Pa * hour**(1/3))
BT = B0*np.exp(3155/temp - 0.16612/(273.39 - temp)**1.17)

alpha = 0.21
beta = 0.63
Dcr, Dmax = 0.6, 0.99

x1, x2 = mesh.coordinates().T
L = float(max(x1) - min(x1))
H = float(max(x2) - min(x2))


"""Define constitutive and kinematic relationships."""


def D(u):
    """Symmetric gradient operator."""
    return sym(nabla_grad(u))


def DII(u):
    """Second strain invariant."""
    return (0.5*(D(u)[0, 0]**2 + D(u)[1, 1]**2) + D(u)[0, 1]**2)


def eta(u, n=3, gam=1e-14):
    """Nonlinear viscosity."""
    return 0.5*BT*(DII(u) + gam)**((1 - n)/2/n)


"""Compute stresses and store them in numpy arrays."""

tau = 2*eta(u)*D(u)
t11 = project(tau[0, 0], S1).compute_vertex_values()
t22 = project(tau[1, 1], S1).compute_vertex_values()
t33 = np.zeros(len(t11))
t12 = project(tau[0, 1], S1).compute_vertex_values()

# convert stresses to kPa
for t in [t11, t22, t33, t12]:
    t /= 1e3

# damage and damage increment
dmg = np.clip(project(dmg, S1, form_compiler_parameters={
    "quadrature_degree": deg_quad}).compute_vertex_values(), 0, Dmax)
ddmg = np.clip(ddmg.compute_vertex_values(), 0, np.inf)

# pore pressure
pHD = np.clip(project(pHD, S1, form_compiler_parameters={
    "quadrature_degree": deg_quad}).compute_vertex_values(), 0, np.inf)
pHD /= 1e3  # convert to kPa

# effective pressure
prs = p.compute_vertex_values()
prs /= 1e3

# effective Cauchy stress
s11, s22, s33, s12 = t11 - prs, t22 - prs, t33 - prs, t12

I1 = s11 + s22 + s33  # effective I1 invariant
I1_true = (1 - dmg)*I1 - 3*pHD  # true I1 invariant
J2 = 0.5*(t11**2 + t22**2 + t33**2) + t12**2  # effective J2 invariant
vms = np.sqrt(3*J2)  # effective von Mises stress

lam1 = s33
lam2 = 0.5*(s11 + s22 + np.sqrt(s11**2 - 2*s11*s22 + 4*(s12**2) + s22**2))
mps = np.fmax(lam1, lam2)  # effective max principal stress

# effective Hayhurst stress
chi = alpha*mps + beta*vms + (1 - alpha - beta)*I1
#chip = np.clip(chi, 0, np.clip)  # positive Hayhurst stress
chip = np.clip(chi, 0, None)  # positive Hayhurst stress

# velocity
u1, u2 = u.split(deepcopy=True)
u1 = u1.compute_vertex_values()
u2 = u2.compute_vertex_values()

# stress in fully damaged elements equal to pore pressure
for f in [t11, t22, t33, t12, s11, s22, s33, s12, prs, I1, J2, vms, mps, chi]:
    f[dmg > Dcr] = -pHD[dmg > Dcr]


"""Plot data."""

tri = np.asarray([cell.entities(0) for cell in cells(mesh)])

label_fs = 24  # font size for labels
tick_fs = 18  # font size for axes
title_fs = 32  # fonr size for title

jet = cm.jet
rainbow = cm.rainbow
redblue = cm.coolwarm
viridis = cm.viridis
plasma = cm.plasma
magma = cm.magma


def save_plot(name, data, filename):
    """Plot data and save as a PNG file."""
    path = "output/figs/%s.png" % (name if not filename else filename)
    try:
        fig = plt.figure(figsize=(24, 24))
        plt.title(name, fontsize=title_fs, y=1.04)
        ax = fig.add_subplot(111)
        f0, f1 = min(data), max(data)
        df = f1 - f0
        ct = (f0, f0 + 0.25*df, f0 + 0.5*df, f0 + 0.75*df, f0 + df)
        plt.xlabel("$\mathregular{x_1}$", fontsize=label_fs)
        plt.ylabel("$\mathregular{x_2}$", fontsize=label_fs)
        plt.ylim(0, H - (H % 25) + 50)
        plt.xlim(0, L - (L % 25) + 50)
        plot = plt.tricontourf(x1, x2, tri, data, 360, cmap=rainbow, vmin=f0,
                               vmax=f1)
        plt.triplot(x1, x2, tri, "k-", alpha=0.5, lw=0.75)
        plt.axes().set_aspect("equal")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.25, pad=0.5)
        cbar = plt.colorbar(plot, ticks=ct, cax=cax)
        cbar.ax.tick_params(labelsize=tick_fs)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        print("Saved " + path)
    except:
        print("Failed to save " + path)


def greek(letter, sub=None, sup=None, eff=True):
    """Write Greek letters with subscripts and superscripts."""
    if eff:
        letter = "\widetilde{%s}" % letter
    if sub:
        letter = letter+"_{%s}" % str(sub)
    if sup:
        letter = letter+"^{%s}" % str(sup)
    return "$\mathregular{%s}$" % letter


"""Plot figures."""

save_plot("D", dmg, filename="DMG")
save_plot(greek("\Delta D", eff=False), ddmg, filename="dDMG")
save_plot(greek("D * p", sub="H", eff=False), pHD, filename="pHD")
save_plot(greek("\sigma", sub="11"), s11, filename="S11")
save_plot(greek("\sigma", sub="22"), s22, filename="S22")
save_plot(greek("\sigma", sub="12"), s12, filename="S12")
save_plot(greek("\\tau", sub="11"), t11, filename="T11")
save_plot(greek("p"), prs, filename="P")
save_plot(greek("\chi"), chi, filename="HHS")
save_plot(greek("\sigma", sub="v"), vms, filename="MISES")
save_plot(greek("\sigma", sup="(1)"), mps, filename="MPS")
save_plot(greek("u", sub="1", eff=False), u1, filename="U1")
save_plot(greek("u", sub="2", eff=False), u2, filename="U2")

print("\nPlotting complete.\n")
