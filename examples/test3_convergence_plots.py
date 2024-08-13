import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)


h1_1 = np.loadtxt("test3_ssn.log", unpack=True)
h1_2 = np.loadtxt("test3_FP.log", unpack=True)

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
