from firedrake import *
from semismoothQVIs import *
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)

mesh = UnitSquareMesh(50,50,diagonal="crossed")
dx = dx(mesh)
Vu = FunctionSpace(mesh, "CG", 1)
VT = FunctionSpace(mesh, "CG", 1)


# Generalized Thermoforming QVI parameters
k = 1.0
Phi0 = Function(VT)
x = SpatialCoordinate(mesh)
Phi0.interpolate(1.0 - 2*Max(Max(1/2-x[0], x[0]-1/2),  Max(1/2-x[1], x[1]-1/2)))
Psi0 = Phi0.copy()

phi = Function(VT)
phi.interpolate(sin(np.pi*x[0])*sin(np.pi*x[1]))
psi = phi.copy()
f = Function(Vu)
f.assign(Constant(25))

# Define the g and dg functions
def g(s):
    return conditional(le(s,0.0), 
                       Constant(1/5),
                       conditional(ge(s,1.0), Constant(0.0), (1-s)/5)
            )
def dg(s):
    return conditional(le(s,0.0), 
                       Constant(0.0),
                       conditional(ge(s,1.0), Constant(0.0), Constant(-1/5))
            )
    
qvi = GeneralizedThermoformingQVI(dx, k, Phi0, phi, Psi0, psi, g, dg, f, Vu, VT)

u0 = Function(Vu)
u0.assign(Constant(0.0))
T0 = Function(VT)
T0.assign(Constant(1.0))


(zhs1, h1_1, its_1, is_SSN) = qvi.semismoothnewton(u0, T0, max_its=18, out_tol=1e-11, phi_tol=1e-12, my_tol=1e-11, PF=True, globalization=True, show_inner_trace=False)
u,T=zhs1[-1]
info_green("SSN Convergence: %s"%h1_1)
np.savetxt("test3_ssn.log", h1_1)

(zhs2, h1_2, its_2) = qvi.fixed_point(u0, T0, max_its=18, out_tol=1e-11, phi_tol=1e-12, my_tol=1e-11, PF=True, show_inner_trace=False)
info_green("FP Convergence: %s"%h1_2)
np.savetxt("test3_fp.log", h1_2)


###
# Plotting
###

# Convergence plot
plt.close()
plt.plot(range(len(h1_1)), h1_1, marker="v", label="SSN"); 
plt.plot(range(len(h1_2)), h1_2, "--", marker="o", label="Fixed Point");
plt.yscale('log')
plt.grid(True)
plt.legend(loc=0); 
plt.xlabel(r"Iterations", fontsize = 20)
plt.ylabel(r"$\Vert R(u_i) \Vert_{H^1(\Omega)}$")
plt.title(r"Convergence", fontsize = 20)
plt.savefig("convergence.pdf", bbox_inches = "tight", pad_inches=0.05)


# Save solutions as PVD files
mould, deform = Function(VT), Function(VT)
mould.interpolate(Phi0 + phi * T)
deform.interpolate(phi * T)

mould.rename("Mould")
deform.rename("Mould Deformation")
Phi0.rename("Original mould")
u.rename("Membrane")
T.rename("Temperature")

pvd = File("output.pvd")
pvd.write(u, T, mould, deform, Phi0)


# Plot 1D slice via matplotlib
ux = np.zeros(100)
mouldx = np.zeros(100)
deformx = np.zeros(100)
omouldx = np.zeros(100)
Tx = np.zeros(100)
xx = np.linspace(0,1-1e-10,100)
for i in range(len(xx)):
    ux[i] = u((xx[i],0.5))
    mouldx[i] = mould((xx[i],0.5))
    deformx[i] = deform((xx[i],0.5))
    omouldx[i] = Phi0((xx[i],0.5))
    Tx[i] = T((xx[i],0.5))

plt.close()
plt.plot(xx, ux, label="Membrane"); 
plt.plot(xx, mouldx, "--", label="Mould"); 
plt.plot(xx, omouldx, "--", label="Original mould"); 
plt.plot(xx, Tx, label="Temperature"); 
plt.legend(loc=0); 
plt.xlabel(r"$x_1$", fontsize = 20)
plt.title(r"Slice at $x_2=1/2$", fontsize = 20)
plt.savefig("slice.pdf", bbox_inches = "tight", pad_inches=0.05)