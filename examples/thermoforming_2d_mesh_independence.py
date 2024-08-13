from firedrake import *
from semismoothQVIs import *
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)


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
    


its_1 = np.zeros((7, 4))
h1s = []
for (n, i) in zip([25,50,100,150,200,250,300], range(7)): # 
    info_red("Considering n=%s"%n)
    mesh = UnitSquareMesh(n,n,diagonal="crossed")
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

    u0 = Function(Vu)
    u0.assign(Constant(0.0))
    T0 = Function(VT)
    T0.assign(Constant(1.0))

    qvi = GeneralizedThermoformingQVI(dx, k, Phi0, phi, Psi0, psi, g, dg, f, Vu, VT)

    (zhs1, h1_1, it_1, is_SSN) = qvi.semismoothnewton(u0, T0, max_its=18, out_tol=1e-11, phi_tol=1e-12, my_tol=1e-11, PF=True, globalization=False, show_inner_trace=False)
  
    its_1[i,:] = it_1
    np.savetxt("convergence.log", its_1.astype(int), fmt='%i', delimiter=",")
    h1s.append(h1_1)