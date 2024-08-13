from firedrake import *
from semismoothQVIs import *
import matplotlib.pyplot as plt
import numpy as np

mesh = UnitIntervalMesh(1000)
dx = dx(mesh)
Vu = FunctionSpace(mesh, "CG", 1)
VT = FunctionSpace(mesh, "CG", 1)


# Define the interpolated functions
Phi0 = Function(VT)
x = SpatialCoordinate(mesh)
Phi0.interpolate(conditional(lt(abs(x[0] - 0.5), 0.25), Constant(0), abs(x[0] - 0.5) - 0.25))
Psi0 = Function(VT)
Psi0.assign(Constant(1.0))
alpha1, alpha2 = 1e-2, 1e-2
phi = Function(VT)
phi.interpolate(sin(np.pi*x[0])/alpha1)
psi = phi.copy()
f = Function(Vu)
f.interpolate(np.pi**2*sin(np.pi * x[0])+100*conditional(lt(-abs(x[0]-0.625)+0.125,0.0),Constant(0),-abs(x[0] - 0.625)+0.125))

u1, T1 = Function(Vu), Function(VT)
u1.interpolate(sin(np.pi*x[0]))
T1.assign(Constant(alpha1))



plt.plot(f.dat.data)
plt.savefig("f.png")


# Define the g and dg functions
def g(s):
    return conditional(le(s,1.0), alpha1 + atan((1-s)/ (2*alpha2)), alpha1 + atan((1-s)/alpha2))

def dg(s):
    return conditional(le(s,1.0), -2*alpha2 / (4*alpha2**2+(1-s)**2), -alpha2/(alpha2**2+(1-s)**2))
    
qvi = GeneralizedThermoformingQVI(dx, 1.0, Phi0, phi, Psi0, psi, g, dg, f, Vu, VT)

u0 = Function(Vu)
u0.assign(Constant(0.0))
T0 = Function(VT)
T0.assign(Constant(5.0))
# (zhs1, h1_1, its_1) = qvi.fixed_point(u0, T0, max_its=18, out_tol=1e-11, in_tol=1e-11, rho0=1, PF=True, show_inner_trace=False)

v = TestFunction(Vu)
a = inner(grad(u0), grad(v))*dx - inner(f,v)*dx
solve(a==0, u0, bcs=DirichletBC(Vu, Constant(0.0), "on_boundary"))

(zhs1, h1_1, its_1, is_SSN) = qvi.semismoothnewton(u0, T0, max_its=18, out_tol=1e-11, in_tol=1e-11, PF=True, show_inner_trace=False)
u,T=zhs1[-1]

err = norm(u-u1, norm_type="H1")
info_blue("H1-error: %s"%err)
info_green("Convergence: %s"%h1_1)

plt.close()
plt.plot(u.dat.data)
plt.savefig("u.pdf")

(zhs1, h1_1, its_1) = qvi.fixed_point(u0, T0, max_its=18, out_tol=1e-11, in_tol=1e-11, PF=True, show_inner_trace=False)