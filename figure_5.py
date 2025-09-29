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

recon = ['Bayesian mean estimation', 'maximum likelihood', 'direct inversion']
recon_l = ['BME', 'MLE', 'direct inv.']
meas = ['Bell state meas.', 'parity checks','MUB', 'Pauli meas.', 'Haar rand.', 'sep. Haar rand.']
c_meas = ['#009E73', 'plum', '#0072B2', '#D55E00', '#56B4E9', '#E69F00']
markers_fid = 'D'
markers_HS = 'o'
z_order = [3, 3, 4, 4, 3, 3, 3]

# Create figure
fig = formatter.figure(aspect_ratio=1/1.5, wide=False)
ax = fig.add_subplot(1, 1, 1)

# Import Data: select BME + Hilbert-Schmidt, Infidelity
n_meas = np.arange(1, 31, 1, dtype= int)
data_name = "data_figure_4-5.npy"
n_sample = 1000
HS = np.load(data_name)[1, 0, 0] #[estimator][meas][nmeas][sample]
Fid = 1 - np.load(data_name)[0, 0, 0] #[estimator][meas][nmeas][sample]
n_meas = [np.arange(3, 91, 3, dtype= int), np.arange(3, 91, 3, dtype= int), np.arange(5, 91, 5, dtype= int), np.arange(9, 91, 9, dtype= int), np.arange(3, 91, 3, dtype= int), np.arange(3, 91, 3, dtype= int)]

# Plotting
for i in range(4): #range(len(meas)):
    ax.plot(n_meas[i], np.average(Fid[i][:len(n_meas[i])], axis=1), c= c_meas[i], ls= "", marker= markers_fid, alpha=1, zorder= z_order[i])
    ax.plot(n_meas[i], np.average(HS[i][:len(n_meas[i])], axis=1), c= c_meas[i], ls= "", marker= markers_HS, alpha=1, zorder= z_order[i])

# Axis
xticks = [0, 25, 50, 75]
lims = [0, 0.22]
yticks = [0, 0.1, 0.2]
ax.set_xlim(0, 92)
ax.set_ylim(*lims)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.grid()
ax.set_xlabel(r'number of measurements, $N$')
ax.set_ylabel(r'average risk, $r(\pi, \hat{\rho})$')

# First legend for the colored lines
legend1 = ax.legend(
    handles=[
        ax.plot([], [], ls= "", marker= markers_fid, c=c_meas[0], label= meas[0])[0],
        ax.plot([], [], ls= "", marker= markers_fid, c=c_meas[1], label= meas[1])[0],
        ax.plot([], [], ls= "", marker= markers_fid, c=c_meas[2], label= meas[2])[0],
        ax.plot([], [], ls= "", marker= markers_fid, c=c_meas[3], label= meas[3])[0]
    ],
    loc= 'upper right',
    labelspacing=0.26,
)
ax.add_artist(legend1)
legend1.get_frame().set_linewidth(0.5)

# Second legend for the dashed and solid lines
legend2 = ax.legend(
    handles=[
        ax.plot([], [], ls= "", marker= markers_fid, c='gray', label= 'infidelity')[0],
        ax.plot([], [], ls= "", marker= markers_HS, c='gray', label= 'HS distance')[0]
    ],
    loc='upper left',
    labelspacing=0.26,
)
legend2.get_frame().set_linewidth(0.5)

# Save figure
fig.tight_layout()
plt.savefig("figure_5.pdf", format="pdf", bbox_inches='tight')