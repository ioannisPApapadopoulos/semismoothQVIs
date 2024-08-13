from firedrake import *
from .mlogging import info, info_red, info_blue, info_green
import numpy as np

class GeneralizedThermoformingQVI(object):
    def __init__(self, dx, k, Phi0, phi, Psi0, psi, g, dg, f, Uu, UT):
        self.dx = dx
        self.k = k
        self.Phi0 = Phi0
        self.phi = phi
        self.Psi0 = Psi0
        self.psi = psi
        self.g = g
        self.dg = dg
        self.f = f
        self.Vu = Uu
        self.VT = UT

    def Xinner(self, u, v):
        return assemble(inner(grad(u), grad(v)) * self.dx)

    def Xnorm(self, u):
        return np.sqrt(self.Xinner(u, u))

    def _inner_X(self, u, h):
        return assemble(inner(grad(u), grad(h)) * self.dx)

    def _dpB(self, h, u, ur):
        return ur[1] / ur[0] * (h - (self._inner_X(u, h) * h) * inner(u, ur) / ur[0]**2)
    
    def au_form(self, uh, v, s, Th):
        return (inner(grad(uh), grad(v)) + inner(s(uh - self.Phi0 - self.phi * Th), v)) * self.dx - inner(self.f, v) * self.dx

    def au0_form(self, uh, v):
        return inner(grad(uh), grad(v)) * self.dx - inner(self.f, v) * self.dx
    
    def aT_form(self, Th, R, uh):
        return (inner(grad(Th), grad(R)) + self.k * inner(Th, R)) * self.dx - inner(self.g(self.Psi0 + self.psi * Th - uh), R) * self.dx
        

    def asn_form(self, z, w, R, A):
        du, xi, mu = split(z)
        v, zeta, q = split(w)
        return inner(R, v)*self.dx + inner(Constant(1e-100)*A, q)*self.dx + inner(Constant(1e-100)*du, v)*self.dx

    def jsn_form(self, zt, w, uh, Th):
        du, xi, mu = split(zt)
        v, zeta, q = split(w)
        
        A_phi = inner(grad(xi), grad(zeta)) + self.k * inner(xi, zeta) - inner((self.dg(self.Psi0 + self.psi * Th - uh)) * (self.psi * xi), zeta)
        jsn = (inner(du, v) - inner(self.phi * xi, v) - inner(mu, v) 
               + A_phi + inner((self.dg(self.Psi0 + self.psi * Th - uh)) * du, zeta)
               + inner(grad(mu), grad(q)) + self.phi * inner(grad(xi), grad(q)) + xi * inner(grad(self.phi), grad(q))
               ) * self.dx
        return jsn       
    # def amy_form(self, ds):
    #     du, xi, mu = TrialFunctions(self.Vu)
    #     v, zeta, q = TestFunctions(self.Vu)
    #     uh = Function(self.Vu)
    #     Th = Function(self.VT)
        
    #     A_phi = inner(grad(xi), grad(zeta)) + self.k * xi * zeta - (self.dg(self.Psi0 + self.psi * Th - uh)) * (self.psi * xi) * zeta
    #     AS = inner(grad(mu), grad(q)) + ds * (uh - self.Phi0 - self.phi * Th) * mu * q
        
    #     jmy = (du * v - mu * v + A_phi + (self.dg * (self.Psi0 + self.psi * Th - uh)) * du * zeta + 
    #            AS - ds(uh - self.Phi0 - self.phi * Th) * self.phi * xi * q) * self.dx
        
        # return None, jmy
    
    def sigma(self, u, rho):
        return conditional(lt(u, 0.0), Constant(0.0), conditional(gt(u,rho), u-rho/2, u**2/(2*rho)))

    def dsigma(self, u, rho):
        return conditional(lt(u, 0.0), Constant(0.0), conditional(gt(u,rho), Constant(1.0), u/rho))

    def solve_Phi(self, uh, T0=Constant(0.0), phi_tol=1e-10, show_trace=True):
        VT = self.VT
        
        T = Function(VT)
        T.assign(T0)
        R = TestFunction(VT)
        aT = self.aT_form(T, R, uh)
        nvp = NonlinearVariationalProblem(aT, T)

        sp = {"ksp_type":"none", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": 500, "snes_atol": phi_tol, "snes_rtol": 0.0, "snes_stol": 0.0, "snes_max_it": 5000, "snes_linesearch_type": "bt",}
        if show_trace:
            sp["snes_monitor"] = None
            sp["snes_converged_reason"] = None
        nvs = NonlinearVariationalSolver(nvp, solver_parameters=sp, options_prefix="")
        nvs.solve() 
        return T, nvs.snes.its
    
    def moreau_yosida_it(self, T, u0=None, rho=1e-5, my_tol=1e-10, max_iter=400, show_trace=True):
        s_r = lambda u: self.sigma(u, rho) / rho

        Vu = self.Vu
        uh = Function(Vu)
        if u0:
            uh.assign(u0)
        v = TestFunction(Vu)

        au = self.au_form(uh, v, s_r, T)
        
        nvp = NonlinearVariationalProblem(au, uh, bcs=DirichletBC(Vu, Constant(0.0), "on_boundary"))
        
        sp = {"ksp_type":"none", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": 500, "snes_atol": my_tol, "snes_rtol": 0.0, "snes_stol": 1e-3, "snes_max_it": 5000, "snes_linesearch_type": "l2",}
        if show_trace:
            sp["snes_monitor"] = None
            sp["snes_converged_reason"] = None
        nvs = NonlinearVariationalSolver(nvp, solver_parameters=sp, options_prefix="")
        nvs.solve() 
        return uh, nvs.snes.its

    def path_following_S(self, Ti, u0=None, rho0=1, rho_min=1e-6, max_its=20, my_tol=1e-10, show_trace=True):
        Vu = self.Vu
        uh = Function(Vu)
        if u0:
            uh.assign(u0)
        
        total_its = 0
        for i in range(max_its):
            rho = rho0 * 10**(-i)
            if show_trace:
                info("Considering rho = %s."%rho)
            
            uh, it = self.moreau_yosida_it(Ti, u0=uh, rho=rho, my_tol=my_tol, show_trace=show_trace)
            total_its += it
            
            if rho < rho_min:
                break
        
        return uh, total_its

    def bm_S(self ,Ti, u0=None, tol=1e-10, show_trace=True):        
        
        Vu = self.Vu
        uh = Function(Vu)
        if u0:
            uh.assign(u0)
        v = TestFunction(Vu)
        au0 = self.au0_form(uh, v)

        lb, ub = Function(Vu), Function(Vu)
        lb.assign(Constant(-1e10))
        ub.interpolate(self.Phi0 + self.phi * Ti)
        nvp = NonlinearVariationalProblem(au0, uh, bcs=DirichletBC(Vu, Constant(0.0), "on_boundary"))
        sp = {"snes_type": "vinewtonrsls", "ksp_type":"none", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": 500, "snes_atol": tol, "snes_rtol": 0.0, "snes_stol": 0.0,"snes_max_it": 5000,"snes_linesearch_type": "l2",}
        if show_trace:
            sp["snes_monitor"] = None
            sp["snes_converged_reason"] = None
        nvs = NonlinearVariationalSolver(nvp, solver_parameters=sp, options_prefix="")
        nvs.solve(bounds=(lb,ub))
        return uh, nvs.snes.its
    
    def inner_solve(self, u, T_, phi_tol, my_tol, hik_tol, PF, FS, rho0, rho_min, newton_its, pf_its, hik_its, show_trace=True):
        if show_trace:
            info("Solve for T.")
        # T, it = self.solve_Phi(u,in_tol=in_tol,show_trace=show_trace)
        T, it = self.solve_Phi(u, T0=T_,phi_tol=phi_tol,show_trace=show_trace)
        newton_its+=it

        if PF:
            if show_trace:
                info("Path-following MY for u.")
            S, it = self.path_following_S(T, u0=u, my_tol=my_tol, rho0=rho0, rho_min=rho_min, show_trace=show_trace) 
            pf_its+=it
        else:
            S = u


        if FS:
            if show_trace:
                info("BM feasibility step for u.")
            S, it = self.bm_S(T, u0=S, tol=hik_tol, show_trace=show_trace) 
            hik_its+=it

        return u, T, S, newton_its, pf_its, hik_its
    
    def fixed_point(self, ui, Ti, max_its=20, min_its=0, out_tol=1e-10, phi_tol=1e-10, my_tol=1e-10, hik_tol=1e-10, PF=True, FS=True, rho0=1, rho_min=1e-6, show_trace=True, show_inner_trace=True):

        h1_1, zhs = [], []
        newton_its, pf_its, hik_its, outer_its =  0, 0, 0, 0
        zhs.append([ui, Ti])

        while outer_its < max_its:

            ui, Tᵢ, uB, newton_its, pf_its, hik_its = self.inner_solve(ui, Ti, phi_tol, my_tol, hik_tol, PF, FS, rho0, rho_min, newton_its, pf_its, hik_its, show_trace=show_inner_trace)

            err = norm(uB-ui, norm_type="H1")
            h1_1.append(err)
            ui = uB.copy()
    
            outer_its += 1
            if show_trace:
                info("Fixed point: Iteration %s, ‖R(uᵢ)‖ = %s." %(outer_its, h1_1[-1]))

            zhs.append([ui, Ti])
            if h1_1[-1] <= out_tol and outer_its >= min_its:
                break

        its = (outer_its, newton_its, pf_its, hik_its)
        return (zhs, h1_1, its)

    def semismoothnewton(self, ui, Ti, max_its=20, out_tol=1e-10, phi_tol=1e-10, my_tol=1e-10, hik_tol=1e-10, globalization=False, rho_min=1e-6, PF=True, FS=True, show_trace=True, show_inner_trace=True):

        if FS == False: 
            info_red("Are you sure you want FS=false? This will prevent superlinear convergence.")
        Vu = self.Vu
        VT = self.VT

        # V = MixedElement([Vu, VT, Vu])
        V = Vu * VT * Vu
        zt = TrialFunction(V)
        w = TestFunction(V)
        Ah = Function(Vu)

        zhs, h1s = [], []
        zhs.append([ui, Ti])
        h1c, outer_its, hik_its, pf_its, newton_its = 1,0,0,0,0
        is_SSN = []

        ui, Ti, uB, newton_its, pf_its, hik_its = self.inner_solve(ui, Ti, phi_tol, my_tol, hik_tol, PF, FS, 1, rho_min, newton_its, pf_its, hik_its, show_trace=show_inner_trace)
        h1c = norm(uB-ui,norm_type="H1")
        h1s.append(h1c)

        inf = 1e100

        while outer_its < max_its and h1c > out_tol:

            z = Function(V)
            R = ui - uB
            
            """
            This is a trick to compute the SSN step in the Firedrake interface by leveraging
            PETSc's vinewtonrsls. It also works in parallel.
            """
            Ah.interpolate(self.Phi0 + self.phi * Ti - uB)
            A = conditional(le(Ah, 0.0), 1.0, 0.0)

            asn = self.asn_form(z, w, R, A)
            jsn = self.jsn_form(zt,w, ui, Ti)

            z.split()[2].interpolate(A)

            nvp = NonlinearVariationalProblem(asn, z, J=jsn, bcs=[DirichletBC(V.sub(0), Constant(0.0), "on_boundary"), DirichletBC(V.sub(2), Constant(0.0), "on_boundary")])
            sp = {"snes_type": "vinewtonrsls", "ksp_type":"none", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": 500, "snes_rtol": 1e20, "snes_max_it": 1}
            nvs = NonlinearVariationalSolver(nvp, solver_parameters=sp, options_prefix="")

            lb, ub = Function(V), Function(V)
            for i in range(0,2):
                lb.split()[i].interpolate(Constant(-inf))
                ub.split()[i].interpolate(Constant(+inf))
            lb.split()[2].interpolate(conditional(le(Ah, 0.0), Constant(0.0), Constant(-inf)))
            ub.split()[2].interpolate(conditional(le(Ah, 0.0), Constant(0.0), Constant(+inf)))
            nvs.solve(bounds=(lb,ub))         

            ###### End of trick
    
            uN = Function(Vu)
            uN.assign(ui + z.split()[0])
            uN, TN, SN, newton_its, pf_its, hik_its = self.inner_solve(uN, Ti, phi_tol, my_tol, hik_tol, PF, FS, 1e-2, rho_min, newton_its, pf_its, hik_its, show_trace=show_inner_trace)

            h1N = norm(uN-SN, norm_type="H1")

            if globalization:
                uB, TB, SB, newton_its, pf_its, hik_its = self.inner_solve(uB, Ti, phi_tol, my_tol, hik_tol, PF, FS, 1e-2, rho_min, newton_its, pf_its, hik_its, show_trace=show_inner_trace)
                h1B = norm(uB-SB, norm_type="H1")
                if h1B < h1N:
                    if show_trace:
                        info_blue("H¹ norms = (%s, %s), uB superior." %(h1B, h1N))
                    h1c, ui, Ti, uB = h1B, uB, TB, SB
                    is_SSN.append(False)
                elif h1N < h1B:
                    if show_trace:
                        info_green("H¹ norms = (%s, %s), uN superior." %(h1B, h1N))
                    h1c, ui, Ti, uB = h1N, uN, TN, SN
                    is_SSN.append(True)

            else:
                h1c, ui, Ti, uB = h1N, uN, TN, SN
                is_SSN.append(True)

            h1s.append(h1c)
            zhs.append([ui, Ti])
  
            outer_its += 1
            if show_trace:
                info("Semismooth Newton: Iteration %s, ‖R(uᵢ)‖ = %s" %(outer_its, h1c))
                # print("Semismooth Newton: Iteration %s, ‖R(uᵢ)‖ = %s, stepsize: 1.0" %(outer_its, h1c))
            # if ls_α < 1e-10:
            #     print("Stepsize below 1e-10, terminating SSN.\n")
            #     break
        zhs.append([uB, Ti])
        its = (outer_its, newton_its, pf_its, hik_its)
        # is = (is_proj, is_xBs)
        return (zhs, h1s, its, is_SSN)#, is)