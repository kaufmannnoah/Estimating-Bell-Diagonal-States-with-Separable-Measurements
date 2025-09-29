import numpy as np
import matplotlib as mpl
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
c_est = ['#56B4E9', '#CC79A7']

# Create figure
fig = formatter.figure(aspect_ratio=1/1.618, wide=False)
axs = fig.add_subplot(1, 1, 1)

# Import Data: select bayesian and direct inversion + Hilbert-Schmidt norm + Bell state meas.
data_name = "data_figure_2-3.npy"
n_sample = 1000 # dataset contains 4000 samples
HS = np.load(data_name)[[1, 5], 0, 0, 0, :, :n_sample] #[estimator][nmeas][sample]
HS = HS[:, :, :n_sample] #[estimator][nmeas][sample]
n_meas = np.arange(1, 31, 1, dtype= int)

# Bayesian mean estimation
HS_std = np.std(HS[0], axis=1) / np.sqrt(n_sample)
axs.errorbar(n_meas[::4], np.average(HS[0], axis=1)[::4], yerr= HS_std[::4], c= c_est[0], lw=0, ls= "", marker= markers[0], alpha=1, label= est[0], zorder= 3)
axs.errorbar(n_meas[::4], np.average(HS[0], axis=1)[::4], yerr= HS_std[::4], c= 'black', lw= 1, ls= "",  ms= 0, alpha=1, capsize= 1.5, zorder= 4)
axs.errorbar(n_meas[2::4], np.average(HS[0], axis=1)[2::4], yerr= HS_std[2::4], c= c_est[0], lw=0, ls= "", marker= markers[0], alpha=1, zorder= 7)
axs.errorbar(n_meas[2::4], np.average(HS[0], axis=1)[2::4], yerr= HS_std[2::4], c= 'black', lw= 1, ls= "",  ms= 0, alpha=1, capsize= 1.5, zorder= 8)

# direct inversion
HS_std = np.std(HS[1], axis=1) / np.sqrt(n_sample)
axs.errorbar(n_meas[::2], np.average(HS[1], axis=1)[::2], yerr= HS_std[::2], c= c_est[1], lw=0, ls= "", marker= markers[1], alpha=1, label= est[1], zorder= 5)
axs.errorbar(n_meas[::2], np.average(HS[1], axis=1)[::2], yerr= HS_std[::2], c= 'black', lw= 1, ls= "",  ms= 0, alpha=1, capsize= 1.5, zorder= 6)

# Analytical Solutions
x = np.linspace(1, 30, 1000)
axs.plot(x, 3/(5*x), c= c_est[1], ls= '-', alpha = 0.5, zorder= 1)
axs.plot(x, 3/(5*x), c= 'white', ls= '-', alpha = 0.1, zorder= 2)
x = np.linspace(0.4, 30, 1000)
axs.plot(x, 3/(5*(x+4)), c= c_est[0], ls= '-', alpha = 0.5, zorder= 1)
axs.plot(x, 3/(5*(x+4)), c= 'white', ls= '-', alpha = 0.1, zorder= 2)

# Axis
axs.set_xlim(0, 31)
axs.set_ylim(0, 0.625)
axs.set_xticks(np.arange(0, 31, 12, dtype= int))
axs.set_yticks([0, 0.25, 0.5])
legend = axs.legend(loc='upper right', labelspacing = 0.26)
legend.get_frame().set_linewidth(0.5)
axs.grid()
axs.set_axisbelow(True)
axs.set_xlabel(r'number of Bell measurements, $N$')
axs.set_ylabel(r'average risk, $r(\pi, \hat{\rho})$')

# Output
fig.tight_layout()
plt.savefig("figure_2.pdf", format="pdf", bbox_inches='tight')
