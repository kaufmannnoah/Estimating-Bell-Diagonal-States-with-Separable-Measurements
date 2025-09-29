import numpy as np
import matplotlib as mpl
from mpmath import *
import matplotlib.pyplot as plt
import rsmf

# Plotting parameters
formatter = rsmf.setup(r"\documentclass[twocolumn, amssymb, nobibnotes, aps, pra, superscriptaddress, 10pt]{revtex4-2}")
mpl.rcParams["legend.edgecolor"]    = "black"
mpl.rcParams["legend.fancybox"]     = True
mpl.rcParams["legend.borderpad"]    = 0.5
mpl.rcParams["legend.handlelength"] = .75
mpl.rcParams["lines.linewidth"]     = 2
mpl.rcParams["lines.markersize"]    = 4
mpl.rcParams["text.usetex"]         = False
est = ['Bayesian mean estimation','direct inversion']
markers = ['D', 's']
c_est = ['#009E73', 'plum']

# Create figure
fig = formatter.figure(aspect_ratio=1/1.618, wide=False)
axs = fig.add_subplot(1, 1, 1)

# Import Data: select bayesian and direct inversion + Hilbert-Schmidt norm + Parity check meas.
data_name = "data_figure_2-3.npy"
n_sample = 1000 # dataset contains 4000 samples
HS = np.load(data_name)[[1, 5], 0, 0, :, :, :] #[estimator][nmeas][sample]
HS = HS[:, [1, 2]] #[estimator][nmeas][sample]
HS = HS[:, :, :, :n_sample] #[estimator][nmeas][sample]
n_meas = [np.arange(3, 31, 3, dtype= int), np.arange(1, 31, 1, dtype= int)]

# Plotting
for i in [0, 1]:
    for j in [0, 1]:
        temp = HS[j][i][:len(n_meas[i])]
        HS_std = np.std(1 - temp, axis=1) / np.sqrt(n_sample)
        axs.errorbar(n_meas[i], np.average(temp, axis=1), yerr= HS_std, c= c_est[i], lw=0, ls= "", marker= markers[j], alpha=1, zorder= 9-3*i)
        axs.errorbar(n_meas[i], np.average(temp, axis=1), yerr= HS_std, c= 'black', lw= 0.5, ls= "",  ms= 0, alpha=1, capsize= 1.5, zorder= 10-3*i)

axs.errorbar(10, 10, yerr= 0.02, c= 'black', lw=0, ls= "", marker= 's', alpha=1, label= "direct inversion")
axs.errorbar(10, 10, yerr= 0.02, c= 'black', lw=0, ls= "", marker= 'D', alpha=1, label= "Bayesian mean est.")
axs.errorbar(10, 10, yerr= 0.02, c= 'white', lw=0, ls= "", marker= 'D', alpha=1, label= " ")
axs.errorbar(10, 10, yerr= 0.02, c= c_est[1], lw=0, ls= "", marker= 'o', alpha=1, label= "random parity check")
axs.errorbar(10, 10, yerr= 0.02, c= c_est[0], lw=0, ls= "", marker= 'o', alpha=1, label= "ordered parity check")

# Analytical Solutions
th_sq = 2 / 5
def fun(N):
    return (2/3)**N * ((1 - th_sq)*(1/2 * N * hyp3f2(1, 1, 1 - N, 2, 2, -1/2) - 1) + 3/4)
x = np.linspace(0.7, 30.7, 1000)
y= [fun(N) for N in x]
axs.plot(x, y, c=c_est[1],  ls= '-', alpha= 0.6, zorder= 1)
axs.plot(x, y, c= 'white',  ls= '-', alpha= 0.1, zorder= 2)
x = np.linspace(2.95, 30.7, 1000)
axs.plot(x, 9/(5*x), c= c_est[0], ls= '-', alpha =0.6, zorder= 1)
axs.plot(x, 9/(5*x), c= 'white', ls= '-', alpha =0.1, zorder= 2)
x = np.linspace(0.7, 30.7, 1000)
axs.plot(x, (x+3)/(5*(x/3+2)**2), c= c_est[0], ls= ':', dash_capstyle= 'round', alpha =0.6, zorder= 1)
axs.plot(x, (x+3)/(5*(x/3+2)*2), c= 'white', ls= ':', dash_capstyle= 'round', alpha =0.1, zorder= 2)

# Axis
axs.set_xlim(0, 31)
axs.set_ylim(0, 0.625)
axs.set_xticks(np.arange(0, 31, 12, dtype= int))
axs.set_yticks([0, 0.25, 0.5])
legend = axs.legend(loc='upper right', labelspacing = 0.26)
legend.get_frame().set_linewidth(0.5)
axs.grid()
axs.set_axisbelow(True)
axs.set_xlabel(r'number of measurements, $N$')
axs.set_ylabel(r'average risk, $r(\pi, \hat{\rho})$')

# Output
fig.tight_layout()
plt.savefig("figure_3.pdf", format="pdf", bbox_inches='tight')