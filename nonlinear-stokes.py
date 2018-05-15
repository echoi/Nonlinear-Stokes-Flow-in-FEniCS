"""Run an updated-Lagrangian Stokes flow simulation with:
    - nonlinear viscosity
    - incompressibility
    - poro-mechanics formulation for hydraulic fracture
"""


from __future__ import division
from fenics import *
from mpi4py import MPI
import numpy as np
import time


# suppress FEniCS output to terminal
set_log_active(False)


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


"""Define function spaces."""

S1 = FunctionSpace(mesh, "CG", 1)  # first order scalar space
S2 = FunctionSpace(mesh, "CG", 2)  # second order scalar space
V1 = VectorFunctionSpace(mesh, "CG", 1)  # first order vector space
V2 = VectorFunctionSpace(mesh, "CG", 2)  # second order vector space
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)  # first order scalar element
P2 = VectorElement("CG", mesh.ufl_cell(), 2)  # second order vector element
V = FunctionSpace(mesh, MixedElement([P2, P1]))  # mixed finite element


"""Quadrature elements and function spaces."""

deg_quad = 2
scalar_quad = FiniteElement("Quadrature",
                            cell=mesh.ufl_cell(),
                            degree=deg_quad, quad_scheme="default")
vector_quad = VectorElement("Quadrature", 
                            cell=mesh.ufl_cell(),
                            degree=deg_quad, quad_scheme="default")
SQ = FunctionSpace(mesh, scalar_quad)  # quadrature points in scalar space
VQ = FunctionSpace(mesh, vector_quad)  # quadrature points in vector space
form_params = {"quadrature_degree": deg_quad}


"""Coordinates of nodes on initial mesh configuration."""
if nd == 2:
    X1, X2 = S1.tabulate_dof_coordinates().reshape((-1, nd)).T  # coordinates
elif nd == 3:
    X1, X2, X3 = S1.tabulate_dof_coordinates().reshape((-1, nd)).T  # coordinates
n_local = len(X1)  # number of coordinates on local process
n_global = S1.dim()  # number of coordinates in global system


"""Coordinates of quadrature points on initial mesh configuration."""
if nd == 2:
    XQ1, XQ2 = SQ.tabulate_dof_coordinates().reshape((-1, nd)).T  # coordinates
elif nd == 3:
    XQ1, XQ2, XQ3 = SQ.tabulate_dof_coordinates().reshape((-1, nd)).T  # coordinates
nQ_local = len(XQ1)  # number of quadrature points on local process
nQ_global = SQ.dim()  # number of quadrature points in global system


class left_edge(SubDomain):
    """Boundary on the left domain edge."""
    def inside(self, x, on_boundary): return near(x[0], 0) and on_boundary

class right_edge(SubDomain):
    """Boundary on the right domain edge."""
    def inside(self, x, on_boundary): return near(x[0], L) and on_boundary

class bottom_edge(SubDomain):
    """Boundary on the bottom domain edge."""
    def inside(self, x, on_boundary): return near(x[1], 0) and on_boundary

class top_edge(SubDomain):
    """Boundary on the top domain edge."""
    def inside(self, x, on_boundary): return near(x[1], H) and on_boundary

class back_edge(SubDomain):
    """Boundary on the back domain edge."""
    def inside(self, x, on_boundary): return near(x[2], 0) and on_boundary

class front_edge(SubDomain):
    """Boundary on the front domain edge."""
    def inside(self, x, on_boundary): return near(x[2], W) and on_boundary

""" Define boundaries and boundary conditions. """

left = left_edge()
right = right_edge()
bottom = bottom_edge()
top = top_edge()
if nd == 3:
    back = back_edge()
    front = front_edge()

boundaries = FacetFunction("size_t", mesh, 0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
bottom.mark(boundaries, 3)
top.mark(boundaries, 4)
if nd == 3:
    back.mark(boundaries, 5)
    front.mark(boundaries, 6)
ds = Measure("ds", subdomain_data=boundaries)

free_slip_left = DirichletBC(V.sub(0).sub(0), Constant(0), left)
free_slip_bottom = DirichletBC(V.sub(0).sub(1), Constant(0), bottom)
BC = [free_slip_left, free_slip_bottom]


"""Define loading functions."""


class hydrostatic(Expression):
    """Hydrostatic pressure class (applied as a Neumann BC)."""
    def __init__(self, h=0, **kwargs):
        self.h = float(h)  # water level

    def eval(self, value, x):
        value[0] = -rho_H2O*grav*max(self.h - x[1], 0)

    def value_shape(self):
        return ()


def bodyforce(dmg, y, h=0):
    """Gravity loading as a body force. Fully failed points have
        no density unless they are filled with water. Then they
        have the same density as water.
    """
    b = Function(VQ)  # body force as vector function
    by = Function(SQ)  # y-component of vector function
    ice_points = dmg.vector()[:] < Dcr  # ice material points
    H2O_points = (dmg.vector()[:] > Dcr)*(y <= h)  # water material points
    by.vector()[ice_points] = -rho_ice * grav
    by.vector()[H2O_points] = -rho_H2O * grav
    assign(b.sub(1), by)
    return b


def pore_pressure(dmg, y, h):
    """Hydraulic pressure from poro-mechanics formulation."""
    pHD = Function(SQ)
    pHD.vector()[:] = rho_H2O*grav*np.fmax(h - y, 0)*dmg.vector().get_local()
    return pHD


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


""" Define damage functions. """


def c_bar(dmg=None):
    """Implicit gradient damage constant term."""
    c = Function(SQ)
    if dmg:
        c.vector()[dmg.vector()[:] < Dcr] = 0.5*(lc**2)
    else:
        c.vector()[:] = 0.5*(lc**2)
    return c


def update_psi(dmg):
    """Function for enforcing incompressibility."""
    psi = Function(SQ)
    psi.vector()[:] = 1
    failed_points = dmg.vector()[:] > Dcr  # failed material points
    psi.vector()[failed_points] = 1e-32
    return psi


def surface_crevasse_level(dmg, y):
    """dynamically compute water level in the surface crevasse"""
    cs = 0
    if hs > 0:
        x2_dmg = y[dmg.vector().get_local() > Dcr]
        if len(x2_dmg) > 0:
            x2_dmg_local_max = min(max(x2_dmg), H)
            x2_dmg_local_min = max(min(x2_dmg), 0)
        else:
            x2_dmg_local_min = H
            x2_dmg_local_max = 0
        x2_dmg_max = comm.allreduce(x2_dmg_local_max, op=MPI.MAX)
        x2_dmg_min = comm.allreduce(x2_dmg_local_min, op=MPI.MIN)
        cs = float(x2_dmg_max - x2_dmg_min)*hs + x2_dmg_min
    return cs


"""Define damage function."""

dmg = Function(SQ)
dmg.vector()[(L/2 - 5 < XQ1)*(XQ1 < L/2 + 5)*(H - 10 < XQ2)] = Dmax


"""Initial guess for Picard iterations."""

uk = Function(V2)  # velocity
pk = Function(S1)  # pressure


comm.Barrier()


"""Main time loop."""

while t_elapsed <= t_total:

    time_counter += 1
    # get current configuration coordinates
    x1, x2 = S1.tabulate_dof_coordinates().reshape((-1, nd)).T
    xQ1, xQ2 = SQ.tabulate_dof_coordinates().reshape((-1, nd)).T

    u, p = TrialFunctions(V)  # trial functions in (V2, S1) space
    v, q = TestFunctions(V)  # test functions in (V2, S1) space

    # hydraulic pressure in surface crevasse
    cs = surface_crevasse_level(dmg=dmg, y=xQ2)  # height of water column
    pHD = pore_pressure(dmg=dmg, y=xQ2, h=cs)  # hydraulic pressure

    # define loading terms
    b_grav = bodyforce(dmg=dmg, y=xQ2, h=cs)  # gravity for ice and water
    b_hw = hydrostatic(h=hw, degree=1)  # terminus pressure

    # normal function to mesh
    nhat = FacetNormal(mesh)

    # incompressibility terms
    penalty = False
    psi = update_psi(dmg)

    # define variational form
    LHS = (inner(D(v), 2*(1 - dmg)*eta(uk)*D(u)) - (1 - dmg)*div(v)*p
           + psi*q*div(u))*dx
    if penalty:
        LHS += 1e12*inner(div(u), psi*div(v))*dx  # penalty term
    RHS = inner(v, b_grav)*dx  # ice and water gravity
    if hs > 0:
        RHS += inner(div(v), pHD)*dx  # hydraulic pressure in damage zone
    if hw > 0:
        RHS += inner(v, b_hw*nhat)*ds(2)  # terminus pressure

    """ Picard iterations. """

    eps_local = 1  # local error norm
    eps_global = 1  # global error norm
    tol = 1e-4  # error tolerance
    picard_count = 0  # iteration count
    picard_max = 50  # maximum iterations
    w = Function(V)  # empty function to dump solution

    while abs(eps_global) > tol and picard_count < picard_max:

        # solve the variational form
        solve(LHS == RHS, w, BC, form_compiler_parameters=form_params)
        u, p = w.split(deepcopy=True)
        u1, u2 = u.split(deepcopy=True)

        # compute error norms
        u1k, u2k = uk.split(deepcopy=True)
        diff1 = u1.vector().get_local() - u1k.vector().get_local()
        diff2 = u2.vector().get_local() - u2k.vector().get_local()
        diffp = p.vector().get_local() - pk.vector().get_local()
        eps1 = np.linalg.norm(diff1)/np.linalg.norm(u1.vector().get_local())
        eps2 = np.linalg.norm(diff2)/np.linalg.norm(u2.vector().get_local())
        epsp = np.linalg.norm(diffp)/np.linalg.norm(p.vector().get_local())

        # update solution for next iteration
        assign(uk, u)
        assign(pk, p)

        comm.Barrier()

        # obtain the max error on the local process
        eps_local = max(eps1, eps2, epsp)

        # obtain the max error on all processes
        eps_global = comm.allreduce(eps_local, op=MPI.MAX)

        # update iteration count
        picard_count += 1

    if rank == 0:
        with open(output_dir + "output/details/" + name + "_simulation_log.txt",
                  "a") as sim_log:
            sim_log.write("\nTime step "
                          "%d: %g hours\n" % (time_counter, t_elapsed))
            if picard_count < picard_max:
                sim_log.write("Convergence after "
                              "%d Picard iterations.\n" % picard_count)
            else:
                sim_log.write("WARNING: no convergence after "
                              "%d Picard iterations!\n" % picard_count)

    """ Generate numpy arrays from output. """

    # build effective deviatoric stress tensor
    tau = 2*eta(u)*D(u)
    t11 = project(tau[0, 0], SQ,
                  form_compiler_parameters=form_params).vector().get_local()
    t22 = project(tau[1, 1], SQ,
                  form_compiler_parameters=form_params).vector().get_local()
    t33 = np.zeros(nQ_local)
    t12 = project(tau[0, 1], SQ,
                  form_compiler_parameters=form_params).vector().get_local()

    dmg0 = dmg.vector().get_local()  # damage from previous time step
    prs = interpolate(p, SQ).vector().get_local()  # effective pressure

    # effective Cauchy stress
    s11, s22, s33, s12 = t11 - prs, t22 - prs, t33 - prs, t12

    I1 = s11 + s22 + s33  # effective I1 invariant
    I1_true = (1 - dmg0)*I1 - 3*pHD.vector().get_local()  # true I1 invariant
    J2 = 0.5*(t11**2 + t22**2 + t33**2) + t12**2  # effective J2 invariant
    vms = np.sqrt(3*J2)  # effective von Mises stress

    lam1 = s33
    lam2 = 0.5*(s11 + s22 + np.sqrt(s11**2 - 2*s11*s22 + 4*(s12**2) + s22**2))
    mps = np.fmax(lam1, lam2)  # effective max principal stress

    # effective Hayhurst stress
    chi = alpha*mps + beta*vms + (1 - alpha - beta)*I1

    """ Compute the local damage rate and increment. """

    # local damage rate
    dDloc_dt = np.zeros(nQ_local)
    for i in range(nQ_local):
        if (I1[i] > 0 and chi[i] > 0 and dmg0[i] < Dcr):
            k = min(k1 + k2*abs(I1_true[i]/1e6), 30)
            dDloc_dt[i] = B*((chi[i]/1e6)**r)/((1 - dmg0[i])**k)

    # select time increment on each local process
    if max(dDloc_dt) > 0:
        Delta_t_local = float(max_Delta_D/max(dDloc_dt))
        Delta_t_local = min(max_Delta_t*3600, Delta_t_local)
    else:
        Delta_t_local = max_Delta_t*3600

    # obtain the minimum time increment from all processes
    comm.Barrier()
    Delta_t = comm.allreduce(Delta_t_local, op=MPI.MIN)

    # compute the damage increment
    Delta_Dloc = Function(SQ)
    Delta_Dloc.vector()[:] = dDloc_dt*Delta_t

    """ Compute the nonlocal damage increment using implicit gradient. """

    # trial and test functions
    Delta_D = TrialFunction(S1)
    z = TestFunction(S1)

    # define variational form
    LHS = (inner(z, Delta_D)
           + c_bar(dmg)*inner(nabla_grad(z), nabla_grad(Delta_D)))*dx
    RHS = inner(z, Delta_Dloc)*dx

    # solve for the nonlocal damage increment
    w = Function(S1)  # empty function to dump solution
    solve(LHS == RHS, w, form_compiler_parameters=form_params)

    # interpolate nonlocal damage increment at quadrature points
    wq = np.clip(interpolate(w, SQ).vector().get_local(), 0, max_Delta_D)

    # update damage and psi UFL functions
    dmg = Function(SQ)
    if t_elapsed >= t_delay_dmg:
        dmgv = np.clip(dmg0+wq, 0, Dmax)  # 0 < damage < Dmax
        dmgv[dmgv > Dcr] = Dmax
        dmg.vector()[:] = dmgv

    """ Output data to mesh file. """

    Delta_t /= 3600  # convert time increment to hours

    if ((time_counter - 1) % output_increment == 0
            or (t_elapsed + Delta_t) >= t_total):
        data = {
            "velocity": u,
            "pressure": p,
            "damage": dmg,
            "damage increment": w,
            "hydraulic pressure": pHD,
            }
        write_hdf5(time_counter, mesh, data, t_elapsed)
        if rank == 0:
            print("Generated %d.h5" % time_counter)
            with open(output_dir
                      + "output/details/" + name + "_time_log.txt", "a") as time_log:
                time_log.write("%g, %g\n" % (time_counter, t_elapsed))

    """ Updated Lagrangian implementation. """

    # split velocity into components in S1 space
    u1, u2 = u.split(deepcopy=True)
    u1 = interpolate(u1, S1).vector().get_local()
    u2 = interpolate(u2, S1).vector().get_local()

    # compute the displacement increment vector Delta_u
    Delta_u1 = Function(S1)
    Delta_u2 = Function(S1)
    ind1 = x1 > 0  # indices of coordinates where x1 > 0
    ind2 = x2 > 0  # indices of coordinates where x2 > 0
    Delta_u1.vector()[ind1] = u1[ind1]*Delta_t
    Delta_u2.vector()[ind2] = u2[ind2]*Delta_t
    Delta_u = Function(V1)
    assign(Delta_u.sub(0), Delta_u1)
    assign(Delta_u.sub(1), Delta_u2)

    # move the mesh, update coordinates
    ALE.move(mesh, Delta_u)

    # update elapsed time
    t_elapsed += Delta_t

""" Finish output and close files. """

if rank == 0:
    finish = time.clock()
    with open(output_dir
              + "output/details/" + name + "_simulation_log.txt", "a") as sim_log:
        sim_log.write("-"*64+"\n\n")
        sim_log.write("Start time:             %g\n" % start)
        sim_log.write("Finish time:            %g\n" % finish)
        sim_log.write("Total simulation time:  %g hours" % ((finish
                      - start)/3600))
